#!/usr/bin/env python3
"""Full-model, independently LR-tuned Qwen3-4B a-loss clip ablation.

The eight tuning curves are:

* W4A16 and W4A4KV4;
* full-block and single-linear Block-GD;
* a_loss_ratio 1.0 and 0.95.

Every curve uses the corrected full formal profile and is released in small,
agent-reviewed LR batches.  Tuning candidates never save checkpoints.  Once
all noise/bracket gates pass, exactly one formal checkpoint is produced for
each curve at its independently selected LR.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.realq_fullmodel_retune_20260817 import campaign as c  # noqa: E402
from experiments.realq_fullmodel_retune_20260817 import campaign_v3 as v3  # noqa: E402
from tools import realq_auto_tune as tuner  # noqa: E402


ABLATION_ID = "realq-qwen3-4b-aloss-clip-independent-lr-20260817-v1"
OUTPUT_ROOT = (
    REPO_ROOT.parent
    / "experiment_data"
    / "realq_qwen3_4b_aloss_clip_independent_lr_20260817"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
SELECTION_PATH = OUTPUT_ROOT / "selections.json"
FORMAL_MANIFEST_PATH = OUTPUT_ROOT / "manifests" / "formal_once.json"
SETTINGS = ("qwen3-4b_w4a16", "qwen3-4b_w4a4kv4")
RATIOS = (1.0, 0.95)
MAX_LAUNCHES_PER_GROUP = 20
MAX_BRACKET_DEX = 0.30
DOCUMENT_INPUTS = (
    "docs/REALQ_Llama31_Qwen3_20组量化评测_20260808.md",
    "调参方法.md",
    "对比实验方法.md",
)
SAFE_REPLICATE_RE = re.compile(r"[A-Za-z0-9_.-]+")
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
    re.IGNORECASE,
)


class AblationError(RuntimeError):
    pass


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AblationError(f"JSON root must be an object: {path}")
    return value


def _ratio_text(value: float) -> str:
    if value == 1.0:
        return "1.0"
    if value == 0.95:
        return "0.95"
    raise AblationError(f"unsupported a_loss_ratio: {value!r}")


def _ratio_slug(value: float) -> str:
    return _ratio_text(value).replace(".", "p")


def _parse_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise AblationError(f"invalid a_loss_ratio: {value!r}") from exc
    _ratio_text(ratio)
    return ratio


def _group_id(branch: str, config: str, ratio: float) -> str:
    return f"{branch}__{config}__aloss_{_ratio_slug(ratio)}"


def _groups() -> list[dict[str, Any]]:
    rows = [
        {
            "group": _group_id(branch, config, ratio),
            "branch": branch,
            "config": config,
            "a_loss_ratio": ratio,
            "schedule": c._schedule_for(config),
        }
        for config in SETTINGS
        for branch in c.BRANCH_VALUES
        for ratio in RATIOS
    ]
    if len(rows) != 8 or len({row["group"] for row in rows}) != 8:
        raise AblationError("a-loss ablation must contain eight unique groups")
    return rows


def _flag_map(command: Sequence[str]) -> dict[str, str]:
    if len(command) < 3 or command[1:3] != ["-m", "realq.ptq"]:
        raise AblationError("command must invoke python -m realq.ptq")
    if (len(command) - 3) % 2:
        raise AblationError("command is not strict flag/value form")
    values: dict[str, str] = {}
    for index in range(3, len(command), 2):
        flag, value = command[index], command[index + 1]
        if not flag.startswith("--") or flag in values:
            raise AblationError(f"invalid or duplicate command flag: {flag}")
        values[flag] = value
    return values


def _validate_command(
    command: Sequence[str],
    *,
    branch: str,
    config: str,
    ratio: float,
    lr: float | None,
    checkpoint: bool,
) -> None:
    values = _flag_map(command)
    low_activation = config.endswith("_w4a4kv4")
    expected = {
        "--dataset": "wikitext2",
        "--eval_datasets": "wikitext2",
        "--seed": "1",
        "--rotation_seed": "0",
        "--refresh_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--eval_seq_len": "2048",
        "--w_bits": "4",
        "--w_groupsize": "128",
        "--w_asym": "false",
        "--w_clip": "true",
        "--num_groups": "4",
        "--percdamp": "0.01",
        "--blocksize": "128",
        "--act_order": "true",
        "--group_parallel_quant": "rank",
        "--global_loss_bsz": "4",
        "--saliency_clip_percentile": "0.99",
        "--grad_hessian_topk": "-1",
        "--grad_clip": "1.0",
        "--grad_lr_layer_schedule": "none" if low_activation else "cosine",
        "--grad_lr_layer_base_ratio": "0.01",
        "--backward_samples": "32",
        "--backward_bsz": "32",
        "--final_layer_backward_bsz": "32",
        "--a_loss_ratio": _ratio_text(ratio),
        "--a_loss_clip_scope": "local_backward_chunk",
        "--bsz": "128",
        "--hessian_accum_bsz": "64",
        "--fsdp": "false",
        "--fsdp_cpu_offload": "false",
        "--cpu_master": "false",
        "--a_bits": "4" if low_activation else "16",
        "--a_groupsize": "-1",
        "--a_asym": "false",
        "--a_clip_ratio": "0.9" if low_activation else "1.0",
        "--k_bits": "4" if low_activation else "16",
        "--k_groupsize": "-1",
        "--k_asym": "false",
        "--k_clip_ratio": "0.9" if low_activation else "1.0",
        "--v_bits": "4" if low_activation else "16",
        "--v_groupsize": "-1",
        "--v_asym": "false",
        "--v_clip_ratio": "0.9" if low_activation else "1.0",
        "--act_quant_aware_gptq": "true" if low_activation else "false",
        "--k_cache_quant_aware_gptq": "true" if low_activation else "false",
        "--loss_slide_window": "true",
        "--final_layer_grad_lr": "1e-05",
        "--kl_topk": "-1",
        "--rotate": "true",
        "--quantizer_inner_fastpath": "true",
        "--w_clip_search_impl": "symmetric_union_exact",
        "--fisher_fp32_cache": "true",
        "--act_order_stitch_impl": "prefix_q_trailing_w_exact",
        "--w_clip_update_impl": "where_out",
        "--w_group_param_layout": "compact",
        "--prepared_clamp_bound_cache": "true",
        "--triton_column_block": "true",
        "--fused_block_adam": "true",
        "--attention_backend": "sdpa",
        "--full_block_refresh": c.BRANCH_VALUES[branch],
        "--skip_eval": "false",
        "--skip_kl_ppl_eval": "false",
        "--lm_eval": "false",
        "--reasoning_eval": "false",
        "--exit_after_precompute": "false",
        "--require_static_cache_hit": "true",
        "--require_reference_cache_hit": "true",
    }
    if lr is not None:
        expected["--grad_lr"] = c._stable_float(lr)
    for flag, wanted in expected.items():
        actual = values.get(flag)
        if actual != wanted:
            raise AblationError(
                f"{branch}/{config}/a={ratio} {flag}: expected {wanted}, got {actual}"
            )
    if "--quant_stop_layer" in values:
        raise AblationError("a-loss ablation must quantize the complete model")
    has_checkpoint = "--save_qmodel_path" in values
    if has_checkpoint != checkpoint:
        raise AblationError(
            f"checkpoint flag mismatch: expected {checkpoint}, got {has_checkpoint}"
        )


def _source_snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": c._file_sha256(path),
    }


def _verify_content_snapshot(item: Mapping[str, Any]) -> None:
    path = Path(str(item["path"]))
    if not path.is_file():
        raise AblationError(f"snapshot input disappeared: {path}")
    if path.stat().st_size != int(item["size_bytes"]):
        raise AblationError(f"snapshot size changed: {path}")
    if c._file_sha256(path) != item["sha256"]:
        raise AblationError(f"snapshot hash changed: {path}")


def _load_v3_plan() -> dict[str, Any]:
    # V3 intentionally bootstraps by mutating the shared base module.  Keep
    # this consumer side-effect free so importing/running the ablation cannot
    # alter another campaign or its tests in the same Python interpreter.
    saved = {
        "CAMPAIGN_ID": c.CAMPAIGN_ID,
        "OUTPUT_ROOT": c.OUTPUT_ROOT,
        "PLAN_PATH": c.PLAN_PATH,
        "INITIAL_LRS": c.INITIAL_LRS,
        "CODE_INPUTS": c.CODE_INPUTS,
        "_build_plan": c._build_plan,
    }
    try:
        v3._bootstrap()
        value = _read_json(v3.PLAN_PATH)
        c._verify_plan(value)
        return value
    finally:
        for name, previous in saved.items():
            setattr(c, name, previous)


def _normalized_command(command: Sequence[str], *ignored: str) -> dict[str, str]:
    values = _flag_map(command)
    for flag in ignored:
        values.pop(flag, None)
    return values


def _build_plan_body() -> dict[str, Any]:
    tuning_plan = _load_v3_plan()
    rendered: dict[str, dict[str, Any]] = {}
    for row in _groups():
        branch = str(row["branch"])
        config = str(row["config"])
        ratio = float(row["a_loss_ratio"])
        source = tuning_plan["configurations"][f"{branch}/{config}"]
        command = list(source["source_command"])
        c._set_arg(command, "--a_loss_ratio", _ratio_text(ratio))
        c._set_arg(command, "--skip_eval", "false")
        c._set_arg(command, "--skip_kl_ppl_eval", "false")
        c._set_arg(command, "--lm_eval", "false")
        c._set_arg(command, "--reasoning_eval", "false")
        c._set_arg(command, "--require_static_cache_hit", "true")
        c._set_arg(command, "--require_reference_cache_hit", "true")
        c._set_arg(command, "--output_dir", "<RUNTIME_OUTPUT_DIR>")
        c._set_arg(command, "--exp", "qwen3_4b_aloss_clip_independent_lr")
        c._remove_arg(command, "--save_qmodel_path")
        _validate_command(
            command,
            branch=branch,
            config=config,
            ratio=ratio,
            lr=float(c._arg_value(command, "--grad_lr")),
            checkpoint=False,
        )
        cache = tuning_plan["cache_snapshots"][f"{branch}/qwen3-4b"]
        rendered[str(row["group"])] = {
            **row,
            "source_command": command,
            "source_command_sha256": c._canonical_sha256(command),
            "source_log_snapshot": source["source_log_snapshot"],
            "cache_snapshot": cache,
            "paired_cache_key": source["paired_cache_key"],
        }

    for config in SETTINGS:
        for branch in c.BRANCH_VALUES:
            pair = [
                rendered[_group_id(branch, config, ratio)]["source_command"]
                for ratio in RATIOS
            ]
            if _normalized_command(pair[0], "--a_loss_ratio") != _normalized_command(
                pair[1], "--a_loss_ratio"
            ):
                raise AblationError(
                    f"ratio pair differs outside a_loss_ratio: {branch}/{config}"
                )
        for ratio in RATIOS:
            pair = [
                rendered[_group_id(branch, config, ratio)]["source_command"]
                for branch in c.BRANCH_VALUES
            ]
            if _normalized_command(pair[0], "--full_block_refresh") != _normalized_command(
                pair[1], "--full_block_refresh"
            ):
                raise AblationError(
                    f"branch pair differs outside full_block_refresh: {config}/a={ratio}"
                )

    body: dict[str, Any] = {
        "ablation_id": ABLATION_ID,
        "output_root": str(OUTPUT_ROOT),
        "source_v3_plan": {
            "path": str(v3.PLAN_PATH),
            "sha256": c._file_sha256(v3.PLAN_PATH),
            "protocol_fingerprint": tuning_plan["protocol_fingerprint"],
        },
        "documents": [
            _source_snapshot(REPO_ROOT / relative) for relative in DOCUMENT_INPUTS
        ],
        "code": _source_snapshot(Path(__file__).resolve()),
        "protocol": {
            "purpose": (
                "compare a_loss_ratio=1.0 and 0.95 only after independently "
                "tuning each full-model curve"
            ),
            "primary_metric": "full-model WikiText2 Exact KL",
            "secondary_metric": "full-model WikiText2 PPL",
            "settings": list(SETTINGS),
            "branches": dict(c.BRANCH_VALUES),
            "ratios": list(RATIOS),
            "max_launches_per_group": MAX_LAUNCHES_PER_GROUP,
            "candidate_release": (
                "agent-reviewed low-to-high immutable batches; start at zero, "
                "then 1e-7, extend only from observed results"
            ),
            "high_side_worse_points": 2,
            "local_bracket_max_dex": MAX_BRACKET_DEX,
            "top_two_same-configuration_replicates": (
                "2 when bit-identical, otherwise 3"
            ),
            "two_percent_plateau_is_selection_gate": False,
            "formal_quantizations_per_selected_group": 1,
            "fallback_rule": (
                "if ratio winners are mixed or otherwise ambiguous, choose 1.0"
            ),
            "determinism": {
                "seed": 1,
                "rotation_seed": 0,
                "refresh_seed": 0,
                "REALQ_DETERMINISTIC_SDPA": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": "0",
                "NVIDIA_TF32_OVERRIDE": "0",
            },
        },
        "groups": rendered,
    }
    body["plan_fingerprint"] = c._canonical_sha256(body)
    return body


def _write_plan(_: argparse.Namespace) -> int:
    body = _build_plan_body()
    if PLAN_PATH.exists():
        existing = _read_json(PLAN_PATH)
        comparable = dict(existing)
        comparable.pop("created_at", None)
        if comparable != body:
            raise AblationError("existing a-loss plan differs from frozen inputs")
    else:
        c._atomic_json(PLAN_PATH, {**body, "created_at": _utc_now()})
    print(PLAN_PATH)
    return 0


def _load_plan() -> dict[str, Any]:
    value = _read_json(PLAN_PATH)
    comparable = dict(value)
    comparable.pop("created_at", None)
    fingerprint = comparable.pop("plan_fingerprint", None)
    if c._canonical_sha256(comparable) != fingerprint:
        raise AblationError("a-loss plan fingerprint mismatch")
    if value.get("plan_fingerprint") != _build_plan_body()["plan_fingerprint"]:
        raise AblationError("a-loss plan no longer matches frozen inputs")
    for item in value["documents"]:
        _verify_content_snapshot(item)
    _verify_content_snapshot(value["code"])
    return value


def _parse_trial(value: str) -> dict[str, Any]:
    parts = value.split("/")
    if len(parts) not in {4, 5}:
        raise AblationError(
            "--trial must be branch/config/ratio/lr[/replicate], got " + repr(value)
        )
    branch, config, ratio_text, lr_text = parts[:4]
    ratio = _parse_ratio(ratio_text)
    try:
        lr = float(lr_text)
    except ValueError as exc:
        raise AblationError(f"invalid LR: {lr_text!r}") from exc
    c._stable_float(lr)
    trial: dict[str, Any] = {
        "group": _group_id(branch, config, ratio),
        "branch": branch,
        "config": config,
        "a_loss_ratio": ratio,
        "lr": lr,
        "schedule": c._schedule_for(config),
        "stage": "agent_directed_full_model_tuning",
    }
    if len(parts) == 5:
        replicate = parts[4]
        if not SAFE_REPLICATE_RE.fullmatch(replicate):
            raise AblationError(f"invalid replicate key: {replicate!r}")
        trial["replicate"] = replicate
    return trial


def _trial_id(trial: Mapping[str, Any]) -> str:
    payload = {
        "group": trial["group"],
        "lr": c._stable_float(float(trial["lr"])),
    }
    digest = c._canonical_sha256(payload)[:12]
    identity = f"{trial['group']}__lr_{digest}"
    if trial.get("replicate") is not None:
        identity += f"__rep_{trial['replicate']}"
    return identity


def _validate_trial(plan: Mapping[str, Any], trial: Mapping[str, Any]) -> None:
    group = str(trial.get("group"))
    if group not in plan["groups"]:
        raise AblationError(f"trial references unknown group: {group}")
    expected = plan["groups"][group]
    for key in ("branch", "config", "a_loss_ratio", "schedule"):
        if trial.get(key) != expected[key]:
            raise AblationError(f"trial {group} has mismatched {key}")
    c._stable_float(float(trial["lr"]))
    _trial_id(trial)


def _existing_launch_count(group: str) -> int:
    root = OUTPUT_ROOT / "trials"
    if not root.is_dir():
        return 0
    count = 0
    for spec_path in root.glob("*/spec.json"):
        try:
            if _read_json(spec_path).get("group") == group:
                count += 1
        except (AblationError, OSError, json.JSONDecodeError):
            continue
    return count


def _charged_trial_ids() -> set[str]:
    root = OUTPUT_ROOT / "trials"
    if not root.is_dir():
        return set()
    charged: set[str] = set()
    for spec_path in root.glob("*/spec.json"):
        try:
            identity = _read_json(spec_path).get("identity")
        except (AblationError, OSError, json.JSONDecodeError):
            continue
        if isinstance(identity, str):
            charged.add(identity)
    return charged


def _validate_manifest(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if manifest.get("ablation_id") != ABLATION_ID:
        raise AblationError("manifest ablation id mismatch")
    if manifest.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise AblationError("manifest plan fingerprint mismatch")
    trials = manifest.get("trials")
    if not isinstance(trials, list) or not trials:
        raise AblationError("manifest needs a non-empty trials list")
    identities: set[str] = set()
    charged = _charged_trial_ids()
    additions: dict[str, int] = defaultdict(int)
    for trial in trials:
        if not isinstance(trial, dict):
            raise AblationError("manifest trial must be an object")
        _validate_trial(plan, trial)
        identity = _trial_id(trial)
        if identity in identities:
            raise AblationError(f"duplicate trial identity: {identity}")
        identities.add(identity)
        # A worker may validate the immutable manifest after another worker
        # has already charged one of its rows.  Count only not-yet-charged
        # identities so parallel startup cannot double-charge the hard cap.
        if identity not in charged:
            additions[str(trial["group"])] += 1
    for group, count in additions.items():
        if _existing_launch_count(group) + count > MAX_LAUNCHES_PER_GROUP:
            raise AblationError(f"20-launch hard cap would be exceeded: {group}")


def _write_manifest(name: str, rationale: str, trials: list[dict[str, Any]]) -> Path:
    plan = _load_plan()
    manifest = {
        "ablation_id": ABLATION_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "name": name,
        "rationale": rationale,
        "created_at": _utc_now(),
        "trials": trials,
    }
    _validate_manifest(plan, manifest)
    path = OUTPUT_ROOT / "manifests" / f"{name}.json"
    if path.exists():
        raise AblationError(f"manifest already exists: {path}")
    c._atomic_json(path, manifest)
    return path


def _make_zero_manifest(args: argparse.Namespace) -> int:
    trials = []
    for group in _groups():
        trials.append({**group, "lr": 0.0, "stage": "zero_endpoint"})
    path = _write_manifest(
        args.name,
        "first immutable batch: one full-model grad_lr=0 endpoint per independent curve",
        trials,
    )
    print(path)
    return 0


def _make_directed_manifest(args: argparse.Namespace) -> int:
    plan = _load_plan()
    trials = [_parse_trial(value) for value in args.trial]
    for trial in trials:
        if float(trial["lr"]) > 0:
            rows = _result_rows(str(trial["group"]))
            if not any(
                row.get("status") == "succeeded" and float(row["lr"]) == 0.0
                for row in rows
            ):
                raise AblationError(
                    f"positive LR released before successful zero endpoint: {trial['group']}"
                )
        _validate_trial(plan, trial)
    path = _write_manifest(args.name, args.rationale, trials)
    print(path)
    return 0


def _runtime_command(
    plan: Mapping[str, Any],
    group: Mapping[str, Any],
    *,
    lr: float,
    output_dir: Path,
    checkpoint_path: Path | None,
) -> list[str]:
    command = list(group["source_command"])
    c._set_arg(command, "--grad_lr", c._stable_float(lr))
    c._set_arg(command, "--output_dir", str(output_dir))
    c._set_arg(
        command,
        "--exp",
        "qwen3_4b_aloss_clip_formal"
        if checkpoint_path is not None
        else "qwen3_4b_aloss_clip_tuning",
    )
    c._remove_arg(command, "--save_qmodel_path")
    if checkpoint_path is not None:
        command.extend(("--save_qmodel_path", str(checkpoint_path)))
    _validate_command(
        command,
        branch=str(group["branch"]),
        config=str(group["config"]),
        ratio=float(group["a_loss_ratio"]),
        lr=lr,
        checkpoint=checkpoint_path is not None,
    )
    return command


def _worker_env(gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    for key in c.TORCH_DISTRIBUTED_ENV:
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES=gpu,
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        REALQ_DETERMINISTIC_SDPA="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONHASHSEED="0",
        NVIDIA_TF32_OVERRIDE="0",
    )
    return environment


def _verify_group_inputs(group: Mapping[str, Any]) -> None:
    c._verify_snapshot(group["source_log_snapshot"])
    for category in ("tokens", "static", "reference"):
        for item in group["cache_snapshot"][category]:
            c._verify_snapshot(item)


def _claim_trial(
    plan: Mapping[str, Any],
    trial_dir: Path,
    spec: Mapping[str, Any],
) -> bool:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT_ROOT / ".launch_registry.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        result_path = trial_dir / "result.json"
        if result_path.exists():
            return False
        spec_path = trial_dir / "spec.json"
        if spec_path.exists():
            raise AblationError(
                f"trial already charged without result; use a new replicate: {trial_dir.name}"
            )
        group = str(spec["group"])
        if _existing_launch_count(group) >= int(
            plan["protocol"]["max_launches_per_group"]
        ):
            raise AblationError(f"20-launch hard cap reached: {group}")
        trial_dir.mkdir(parents=True, exist_ok=False)
        c._atomic_json(spec_path, spec)
    return True


def _execute(
    *,
    command: Sequence[str],
    log_path: Path,
    gpu: str,
    header: Mapping[str, Any],
) -> tuple[int, float, str, str]:
    started_at = _utc_now()
    started = time.monotonic()
    with log_path.open("wb") as handle:
        handle.write(
            (
                f"[{started_at}] command={json.dumps(command, ensure_ascii=False)}\n"
                f"[{started_at}] run={json.dumps(header, ensure_ascii=False)}\n"
            ).encode("utf-8")
        )
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=_worker_env(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        returncode = process.wait()
    return returncode, time.monotonic() - started, started_at, _utc_now()


def _run_trial(plan: Mapping[str, Any], trial: Mapping[str, Any], gpu: str) -> int:
    group = plan["groups"][str(trial["group"])]
    _verify_group_inputs(group)
    identity = _trial_id(trial)
    trial_dir = OUTPUT_ROOT / "trials" / identity
    command = _runtime_command(
        plan,
        group,
        lr=float(trial["lr"]),
        output_dir=trial_dir / "realq_output",
        checkpoint_path=None,
    )
    inventory = c._gpu_inventory(gpu)
    spec = {
        "ablation_id": ABLATION_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "identity": identity,
        "group": trial["group"],
        "branch": trial["branch"],
        "config": trial["config"],
        "a_loss_ratio": trial["a_loss_ratio"],
        "schedule": trial["schedule"],
        "lr": float(trial["lr"]),
        "lr_exact": c._stable_float(float(trial["lr"])),
        "replicate": trial.get("replicate"),
        "stage": trial.get("stage"),
        "command": command,
        "command_sha256": c._canonical_sha256(command),
        "cuda_visible_devices": gpu,
        "gpu": inventory,
        "hostname": socket.gethostname(),
        "created_at": _utc_now(),
    }
    if not _claim_trial(plan, trial_dir, spec):
        result = _read_json(trial_dir / "result.json")
        return 0 if result.get("status") == "succeeded" else 1
    log_path = trial_dir / "execution.log"
    returncode, elapsed, started_at, finished_at = _execute(
        command=command,
        log_path=log_path,
        gpu=gpu,
        header={"identity": identity, "plan_fingerprint": plan["plan_fingerprint"]},
    )
    result: dict[str, Any] = {
        **spec,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "log_path": str(log_path),
        "log_sha256": c._file_sha256(log_path),
    }
    if returncode == 0:
        try:
            kl, ppl = tuner.parse_exact_metric(
                log_path.read_text(encoding="utf-8", errors="replace"), "wikitext2"
            )
        except tuner.TunerError as exc:
            result.update(status="failed", failure_class="metric_parse", error=str(exc))
        else:
            result.update(status="succeeded", kl=kl, ppl=ppl)
    else:
        tail = log_path.read_bytes()[-4 * 1024 * 1024 :].decode(errors="replace")
        result.update(
            status="failed",
            failure_class="oom" if OOM_RE.search(tail) else "non_oom",
            error=f"child exited with status {returncode}",
        )
    c._atomic_json(trial_dir / "result.json", result)
    return 0 if result["status"] == "succeeded" else 1


def _run_worker(args: argparse.Namespace) -> int:
    plan = _load_plan()
    manifest = _read_json(args.manifest.expanduser().resolve())
    _validate_manifest(plan, manifest)
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise AblationError("invalid worker index/count")
    selected = [
        trial
        for index, trial in enumerate(manifest["trials"])
        if index % args.worker_count == args.worker_index
    ]
    failures = 0
    for trial in selected:
        identity = _trial_id(trial)
        print(f"[{_utc_now()}] start {identity} on cuda:{args.cuda_id}", flush=True)
        status = _run_trial(plan, trial, args.cuda_id)
        print(f"[{_utc_now()}] finish {identity} status={status}", flush=True)
        failures += int(status != 0)
    return int(failures != 0)


def _result_rows(group: str | None = None) -> list[dict[str, Any]]:
    root = OUTPUT_ROOT / "trials"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.glob("*/result.json")):
        row = _read_json(path)
        if group is None or row.get("group") == group:
            rows.append({**row, "result_path": str(path)})
    return rows


def _aggregate(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "succeeded":
            grouped[float(row["lr"])].append(row)
    output = []
    for lr, candidates in sorted(grouped.items()):
        values = [float(row["kl"]) for row in candidates]
        median = float(statistics.median(values))
        output.append(
            {
                "lr": lr,
                "lr_exact": c._stable_float(lr),
                "median_kl": median,
                "min_kl": min(values),
                "max_kl": max(values),
                "range": max(values) - min(values),
                "mad": float(
                    statistics.median(abs(value - median) for value in values)
                ),
                "replicates": len(values),
                "results": [
                    {
                        "identity": row["identity"],
                        "kl": row["kl"],
                        "ppl": row["ppl"],
                        "gpu_uuid": row.get("gpu", {}).get("uuid"),
                        "path": row["result_path"],
                    }
                    for row in candidates
                ],
            }
        )
    return output


def _analyze_group(group: Mapping[str, Any]) -> dict[str, Any]:
    rows = _result_rows(str(group["group"]))
    aggregate = _aggregate(rows)
    output: dict[str, Any] = {
        **group,
        "launches": len(rows),
        "remaining_launch_budget": MAX_LAUNCHES_PER_GROUP - len(rows),
        "failures": [
            {
                "identity": row.get("identity"),
                "lr": row.get("lr"),
                "failure_class": row.get("failure_class"),
            }
            for row in rows
            if row.get("status") != "succeeded"
        ],
        "aggregates": aggregate,
        "ready": False,
        "suggestions": [],
    }
    if not aggregate:
        output["reason"] = "no successful Exact-KL trial"
        return output

    ranked = sorted(aggregate, key=lambda item: (item["median_kl"], item["lr"]))
    best = ranked[0]
    top_two = ranked[:2]
    positive = [item for item in aggregate if item["lr"] > 0]
    lower_worse = [
        item
        for item in positive
        if item["lr"] < best["lr"] and item["median_kl"] > best["median_kl"]
    ]
    upper_worse = [
        item
        for item in positive
        if item["lr"] > best["lr"] and item["median_kl"] > best["median_kl"]
    ]
    bracket = None
    if best["lr"] > 0 and lower_worse and upper_worse:
        lower = max(lower_worse, key=lambda item: item["lr"])
        upper = min(upper_worse, key=lambda item: item["lr"])
        bracket = {
            "lower_lr": lower["lr"],
            "best_lr": best["lr"],
            "upper_lr": upper["lr"],
            "width_dex": math.log10(upper["lr"] / lower["lr"]),
        }
    for item in top_two:
        item["required_replicates"] = 2 if item["range"] == 0 else 3
    repeat_gate = len(top_two) == 2 and all(
        item["replicates"] >= item["required_replicates"] for item in top_two
    )
    high_gate = len(upper_worse) >= 2
    if best["lr"] == 0:
        bracket_gate = len(
            [item for item in positive if item["median_kl"] > best["median_kl"]]
        ) >= 2
    else:
        bracket_gate = bracket is not None and bracket["width_dex"] <= MAX_BRACKET_DEX
    noise_floor = max((item["range"] for item in top_two), default=0.0)
    top_gap = (
        abs(top_two[1]["median_kl"] - top_two[0]["median_kl"])
        if len(top_two) == 2
        else None
    )
    noise_tie = top_gap is not None and top_gap <= noise_floor
    selected_lr = None
    if high_gate and bracket_gate and repeat_gate:
        selected_lr = min(item["lr"] for item in top_two) if noise_tie else best["lr"]

    existing_lrs = {item["lr"] for item in aggregate}
    suggestions: list[dict[str, Any]] = []
    if not high_gate:
        if not positive:
            proposed, reason = 1e-7, "first_positive_candidate"
        elif best["lr"] == 0:
            proposed = min(item["lr"] for item in positive) / 10
            reason = "refine_zero_boundary"
        else:
            proposed = max(item["lr"] for item in positive) * 10
            reason = "extend_high_until_two_measured_worse_points"
        if proposed not in existing_lrs:
            suggestions.append({"reason": reason, "lr": proposed})
    if best["lr"] > 0 and not lower_worse:
        proposed = best["lr"] / math.sqrt(10)
        if proposed not in existing_lrs:
            suggestions.append({"reason": "obtain_measured_low_side", "lr": proposed})
    if best["lr"] > 0 and lower_worse and upper_worse and not bracket_gate:
        lower = max(lower_worse, key=lambda item: item["lr"])["lr"]
        upper = min(upper_worse, key=lambda item: item["lr"])["lr"]
        left = math.log10(best["lr"] / lower)
        right = math.log10(upper / best["lr"])
        proposed = (
            math.sqrt(lower * best["lr"])
            if left >= right
            else math.sqrt(best["lr"] * upper)
        )
        if proposed not in existing_lrs:
            suggestions.append({"reason": "refine_log_bracket", "lr": proposed})
    if high_gate and bracket_gate and not repeat_gate:
        for item in top_two:
            if item["replicates"] < item["required_replicates"]:
                suggestions.append(
                    {
                        "reason": "repeat_top_two_for_noise_gate",
                        "lr": item["lr"],
                        "replicate": f"confirm{item['replicates']}",
                    }
                )
    suggestions = suggestions[: max(0, output["remaining_launch_budget"])]
    output.update(
        {
            "best": best,
            "top_two": top_two,
            "two_percent_plateau": [
                item["lr"]
                for item in aggregate
                if item["median_kl"] <= best["median_kl"] * 1.02
            ],
            "upper_worse_lrs": [item["lr"] for item in upper_worse],
            "high_side_gate": high_gate,
            "bracket": bracket,
            "bracket_gate": bracket_gate,
            "repeat_gate": repeat_gate,
            "noise_floor_range": noise_floor,
            "top_two_gap": top_gap,
            "noise_tie": noise_tie,
            "selected_lr": selected_lr,
            "ready": selected_lr is not None,
            "suggestions": suggestions,
            "reason": (
                "all selection gates passed"
                if selected_lr is not None
                else "one or more high-side/bracket/repeat gates remain"
            ),
        }
    )
    return output


def _analysis() -> dict[str, Any]:
    plan = _load_plan()
    rows = [_analyze_group(group) for group in plan["groups"].values()]
    return {
        "ablation_id": ABLATION_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "generated_at": _utc_now(),
        "counts": {
            "groups": len(rows),
            "ready": sum(bool(row["ready"]) for row in rows),
            "launches": sum(int(row["launches"]) for row in rows),
            "failures": sum(len(row["failures"]) for row in rows),
        },
        "rows": rows,
    }


def _status(args: argparse.Namespace) -> int:
    value = _analysis()
    if args.output:
        c._atomic_json(args.output.expanduser().resolve(), value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _freeze(_: argparse.Namespace) -> int:
    plan = _load_plan()
    analysis = _analysis()
    pending = [row["group"] for row in analysis["rows"] if not row["ready"]]
    if pending:
        raise AblationError(f"cannot freeze; groups not ready: {pending}")
    body = {
        "ablation_id": ABLATION_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "selection_protocol": {
            "primary_metric": "WikiText2 Exact KL",
            "max_bracket_dex": MAX_BRACKET_DEX,
            "minimum_top_two_replicates": 2,
            "non_bit_identical_replicates": 3,
            "high_side_worse_points": 2,
            "two_percent_plateau_is_selection_gate": False,
        },
        "rows": [
            {
                "group": row["group"],
                "branch": row["branch"],
                "config": row["config"],
                "a_loss_ratio": row["a_loss_ratio"],
                "selected_lr": row["selected_lr"],
                "best": row["best"],
                "top_two": row["top_two"],
                "bracket": row["bracket"],
                "upper_worse_lrs": row["upper_worse_lrs"],
                "launches": row["launches"],
            }
            for row in analysis["rows"]
        ],
    }
    body["selection_fingerprint"] = c._canonical_sha256(body)
    body["created_at"] = _utc_now()
    if SELECTION_PATH.exists():
        raise AblationError(f"selection already exists: {SELECTION_PATH}")
    c._atomic_json(SELECTION_PATH, body)
    print(SELECTION_PATH)
    return 0


def _load_selection(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = _read_json(SELECTION_PATH)
    comparable = dict(value)
    fingerprint = comparable.pop("selection_fingerprint", None)
    comparable.pop("created_at", None)
    if c._canonical_sha256(comparable) != fingerprint:
        raise AblationError("selection fingerprint mismatch")
    if value.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise AblationError("selection plan mismatch")
    expected = set(plan["groups"])
    observed = {str(row["group"]) for row in value["rows"]}
    if observed != expected:
        raise AblationError("selection group coverage mismatch")
    return value


def _make_formal_manifest(_: argparse.Namespace) -> int:
    plan = _load_plan()
    selection = _load_selection(plan)
    trials = [
        {
            "group": row["group"],
            "selected_lr": row["selected_lr"],
        }
        for row in selection["rows"]
    ]
    value = {
        "ablation_id": ABLATION_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "selection_fingerprint": selection["selection_fingerprint"],
        "formal_quantizations_per_group": 1,
        "created_at": _utc_now(),
        "trials": trials,
    }
    if FORMAL_MANIFEST_PATH.exists():
        raise AblationError(f"formal manifest already exists: {FORMAL_MANIFEST_PATH}")
    c._atomic_json(FORMAL_MANIFEST_PATH, value)
    print(FORMAL_MANIFEST_PATH)
    return 0


def _run_formal_one(
    plan: Mapping[str, Any], selection: Mapping[str, Any], row: Mapping[str, Any], gpu: str
) -> int:
    group_id = str(row["group"])
    group = plan["groups"][group_id]
    _verify_group_inputs(group)
    directory = OUTPUT_ROOT / "formal" / group_id
    result_path = directory / "result.json"
    if result_path.exists():
        return int(_read_json(result_path).get("status") != "succeeded")
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / "checkpoint" / "quantized.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    command = _runtime_command(
        plan,
        group,
        lr=float(row["selected_lr"]),
        output_dir=directory / "realq_output",
        checkpoint_path=checkpoint,
    )
    inventory = c._gpu_inventory(gpu)
    spec = {
        "ablation_id": ABLATION_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "selection_fingerprint": selection["selection_fingerprint"],
        "group": group_id,
        "branch": group["branch"],
        "config": group["config"],
        "a_loss_ratio": group["a_loss_ratio"],
        "selected_lr": row["selected_lr"],
        "command": command,
        "command_sha256": c._canonical_sha256(command),
        "gpu": inventory,
        "hostname": socket.gethostname(),
        "created_at": _utc_now(),
    }
    c._atomic_json(directory / "spec.json", spec)
    log_path = directory / "execution.log"
    returncode, elapsed, started_at, finished_at = _execute(
        command=command,
        log_path=log_path,
        gpu=gpu,
        header={"group": group_id, "formal": True},
    )
    result: dict[str, Any] = {
        **spec,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "log_path": str(log_path),
        "log_sha256": c._file_sha256(log_path),
    }
    if returncode == 0 and checkpoint.is_file():
        try:
            kl, ppl = tuner.parse_exact_metric(
                log_path.read_text(encoding="utf-8", errors="replace"), "wikitext2"
            )
        except tuner.TunerError as exc:
            result.update(status="failed", failure_class="metric_parse", error=str(exc))
        else:
            result.update(
                status="succeeded",
                kl=kl,
                ppl=ppl,
                checkpoint={
                    "path": str(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                },
            )
    else:
        tail = log_path.read_bytes()[-4 * 1024 * 1024 :].decode(errors="replace")
        result.update(
            status="failed",
            failure_class="oom" if OOM_RE.search(tail) else "non_oom",
            error=(
                f"child exited with status {returncode}"
                if returncode
                else "checkpoint missing after successful child exit"
            ),
        )
    c._atomic_json(result_path, result)
    return int(result["status"] != "succeeded")


def _run_formal_worker(args: argparse.Namespace) -> int:
    plan = _load_plan()
    selection = _load_selection(plan)
    manifest = _read_json(FORMAL_MANIFEST_PATH)
    if manifest.get("selection_fingerprint") != selection["selection_fingerprint"]:
        raise AblationError("formal manifest selection mismatch")
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise AblationError("invalid worker index/count")
    failures = 0
    for index, row in enumerate(manifest["trials"]):
        if index % args.worker_count != args.worker_index:
            continue
        print(f"[{_utc_now()}] formal start {row['group']}", flush=True)
        failures += _run_formal_one(plan, selection, row, args.cuda_id)
        print(f"[{_utc_now()}] formal finish {row['group']}", flush=True)
    return int(failures != 0)


def _formal_status(args: argparse.Namespace) -> int:
    plan = _load_plan()
    rows = []
    for group_id, group in plan["groups"].items():
        path = OUTPUT_ROOT / "formal" / group_id / "result.json"
        if path.is_file():
            rows.append(_read_json(path))
    comparisons = []
    for config in SETTINGS:
        for branch in c.BRANCH_VALUES:
            by_ratio = {
                float(row["a_loss_ratio"]): row
                for row in rows
                if row.get("config") == config
                and row.get("branch") == branch
                and row.get("status") == "succeeded"
            }
            comparison: dict[str, Any] = {"config": config, "branch": branch}
            if set(by_ratio) == set(RATIOS):
                one, clipped = by_ratio[1.0], by_ratio[0.95]
                comparison.update(
                    complete=True,
                    lr_1=one["selected_lr"],
                    lr_0p95=clipped["selected_lr"],
                    kl_1=one["kl"],
                    kl_0p95=clipped["kl"],
                    delta_kl_0p95_minus_1=float(clipped["kl"]) - float(one["kl"]),
                    ppl_1=one["ppl"],
                    ppl_0p95=clipped["ppl"],
                    delta_ppl_0p95_minus_1=float(clipped["ppl"]) - float(one["ppl"]),
                )
            else:
                comparison["complete"] = False
            comparisons.append(comparison)
    complete = all(row["complete"] for row in comparisons)
    if complete:
        clipped_kl_wins = sum(row["delta_kl_0p95_minus_1"] < 0 for row in comparisons)
        clipped_ppl_nonworse = all(
            row["delta_ppl_0p95_minus_1"] <= 0 for row in comparisons
        )
        recommendation = (
            0.95
            if clipped_kl_wins == len(comparisons) and clipped_ppl_nonworse
            else 1.0
        )
        reason = (
            "0.95 wins Exact KL everywhere and does not worsen PPL"
            if recommendation == 0.95
            else "mixed/ambiguous outcome; apply the user-specified fallback to 1.0"
        )
    else:
        recommendation, reason = None, "formal results incomplete"
    payload = {
        "ablation_id": ABLATION_ID,
        "generated_at": _utc_now(),
        "counts": {
            "formal_results": len(rows),
            "succeeded": sum(row.get("status") == "succeeded" for row in rows),
            "expected": 8,
        },
        "comparisons": comparisons,
        "recommended_a_loss_ratio": recommendation,
        "recommendation_reason": reason,
    }
    if args.output:
        c._atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_plan = subparsers.add_parser("write-plan")
    write_plan.set_defaults(handler=_write_plan)
    zero = subparsers.add_parser("make-zero-manifest")
    zero.add_argument("--name", default="batch01_zero_endpoints")
    zero.set_defaults(handler=_make_zero_manifest)
    directed = subparsers.add_parser("make-directed-manifest")
    directed.add_argument("--name", required=True)
    directed.add_argument("--rationale", required=True)
    directed.add_argument("--trial", action="append", required=True)
    directed.set_defaults(handler=_make_directed_manifest)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--cuda-id", required=True)
    worker.set_defaults(handler=_run_worker)
    status = subparsers.add_parser("status")
    status.add_argument("--output", type=Path)
    status.set_defaults(handler=_status)
    freeze = subparsers.add_parser("freeze")
    freeze.set_defaults(handler=_freeze)
    formal = subparsers.add_parser("make-formal-manifest")
    formal.set_defaults(handler=_make_formal_manifest)
    formal_worker = subparsers.add_parser("run-formal-worker")
    formal_worker.add_argument("--worker-index", type=int, required=True)
    formal_worker.add_argument("--worker-count", type=int, required=True)
    formal_worker.add_argument("--cuda-id", required=True)
    formal_worker.set_defaults(handler=_run_formal_worker)
    formal_status = subparsers.add_parser("formal-status")
    formal_status.add_argument("--output", type=Path)
    formal_status.set_defaults(handler=_formal_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        AblationError,
        c.CampaignError,
        tuner.TunerError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"aloss-clip-ablation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
