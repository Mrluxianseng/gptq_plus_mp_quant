#!/usr/bin/env python3
"""Run15-optimized full-model LR retuning for both REAL-Q branches.

V4/V5 used strict math SDPA and globally disabled cuBLAS TF32.  That protocol
is scientifically self-consistent, but it does not execute the final optimized
path documented in ``REALQ_QWEN3_FUSIONS_AND_FISHER_TF32_20260808.md``.  V6 is
an immutable restart: deterministic FA4, scoped Fisher/Hessian TF32, and every
retained run15 quantization optimization are explicit in every command.

Because FA4 changes the BF16 teacher slightly, V6 first creates one shared
static-Fisher/reference-logit cache per model from the same frozen calibration
tokens.  Neither branch may reuse the old SDPA producer artifacts.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_fullmodel_retune_20260817 import campaign_v2 as v2


CAMPAIGN_ID = "realq-fullmodel-two-branch-retune-20260818-v6-run15"
OUTPUT_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_fullmodel_retune_20260818_v6_run15"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
V4_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_fullmodel_retune_20260817_v4"
)
V4_PLAN_PATH = V4_ROOT / "plan.json"
V5_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_fullmodel_retune_20260818_v5_q32_memory"
)
V5_PLAN_PATH = V5_ROOT / "plan.json"
CACHE_ROOT = OUTPUT_ROOT / "shared_cache"
MODULE_PATH = (
    "experiments/realq_fullmodel_retune_20260817/campaign_v6_run15.py"
)
FIRST_LRS = (0.0,)

RUN15_FLAGS = {
    "--attention_backend": "flash_attention_4",
    "--hessian_tf32": "true",
    "--quantizer_inner_fastpath": "true",
    "--w_clip_search_impl": "symmetric_union_exact",
    "--fisher_fp32_cache": "true",
    "--act_order_stitch_impl": "prefix_q_trailing_w_exact",
    "--w_clip_update_impl": "where_out",
    "--w_group_param_layout": "compact",
    "--prepared_clamp_bound_cache": "true",
    "--triton_column_block": "true",
    "--fused_block_adam": "true",
}

RUN15_ENV_OVERRIDES = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
}
RUN15_ENV_UNSET = ("REALQ_DETERMINISTIC_SDPA", "NVIDIA_TF32_OVERRIDE")

RUN15_CODE_INPUTS = (
    "realq/attention.py",
    "realq/akv.py",
    "realq/quant/triton_column_block.py",
    "realq/quant/triton_refresh_stitch.py",
    "realq/refresh/triton_block_adam.py",
    "realq/refresh/triton_fisher.py",
    "utils/triton_qwen3_fusions.py",
    MODULE_PATH,
)


def _load_verified_plan(path: Path) -> dict[str, Any]:
    plan = base._read_json(path)
    stable = dict(plan)
    fingerprint = stable.pop("protocol_fingerprint", None)
    stable.pop("created_at", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise base.CampaignError(f"source plan fingerprint mismatch: {path}")
    return plan


def _model_for_config(config: str) -> str:
    try:
        return next(
            model for model in base.MODEL_SLUGS if config.startswith(f"{model}_")
        )
    except StopIteration as exc:
        raise base.CampaignError(f"unknown model for config: {config}") from exc


def _cache_paths(model: str) -> tuple[Path, Path]:
    root = CACHE_ROOT / model
    return root / "static", root / "runtime"


def _configure_run15_command(
    command: list[str], *, branch: str, config: str
) -> list[str]:
    model = _model_for_config(config)
    static_root, runtime_root = _cache_paths(model)
    base._set_or_append_arg(
        command, "--full_block_refresh", base.BRANCH_VALUES[branch]
    )
    base._set_or_append_arg(command, "--a_loss_ratio", "1")
    base._set_or_append_arg(
        command,
        "--hessian_accum_bsz",
        "32" if model == "qwen3-32b" else "64",
    )
    for flag, value in RUN15_FLAGS.items():
        base._set_or_append_arg(command, flag, value)
    base._set_or_append_arg(command, "--static_cache_path", str(static_root))
    base._set_or_append_arg(command, "--cache_dir", str(runtime_root))
    base._set_or_append_arg(command, "--require_static_cache_hit", "true")
    base._set_or_append_arg(command, "--require_reference_cache_hit", "true")
    base._remove_arg(command, "--quant_stop_layer")
    return command


def _validate_full_profile(
    command: Sequence[str], *, branch: str, config: str
) -> None:
    expected = {
        "--dataset": "wikitext2",
        "--eval_datasets": "wikitext2",
        "--seed": "1",
        "--rotation_seed": "0",
        "--refresh_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--eval_seq_len": "2048",
        "--w_groupsize": "128",
        "--blocksize": "128",
        "--backward_samples": "32",
        "--backward_bsz": "32",
        "--loss_slide_window": "true",
        "--full_block_refresh": base.BRANCH_VALUES[branch],
        "--grad_lr_layer_schedule": base._schedule_for(config),
        "--a_loss_ratio": "1",
        "--rotate": "true",
        "--require_static_cache_hit": "true",
        "--require_reference_cache_hit": "true",
        **RUN15_FLAGS,
    }
    for flag, wanted in expected.items():
        actual = base._arg_value(command, flag)
        if actual != wanted:
            raise base.CampaignError(
                f"run15 {branch}/{config} {flag}: expected {wanted}, got {actual}"
            )
    wanted_hessian = "32" if config.startswith("qwen3-32b_") else "64"
    if base._arg_value(command, "--hessian_accum_bsz") != wanted_hessian:
        raise base.CampaignError(
            f"run15 {branch}/{config} must keep hessian_accum_bsz={wanted_hessian}"
        )
    if base._arg_indices(command, "--quant_stop_layer"):
        raise base.CampaignError("full-model trial must not contain --quant_stop_layer")
    model = _model_for_config(config)
    static_root, runtime_root = _cache_paths(model)
    if base._arg_value(command, "--static_cache_path") != str(static_root):
        raise base.CampaignError("run15 trial points at a non-V6 static cache")
    if base._arg_value(command, "--cache_dir") != str(runtime_root):
        raise base.CampaignError("run15 trial points at a non-V6 reference cache")


def _bootstrap() -> None:
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PLAN_PATH = PLAN_PATH
    base.INITIAL_LRS = FIRST_LRS
    base._a_loss_ratio_for = lambda _config: 1.0
    base._a_loss_ratio_text = lambda _config: "1"
    base.WORKER_ENV_OVERRIDES = dict(RUN15_ENV_OVERRIDES)
    base.WORKER_ENV_UNSET = RUN15_ENV_UNSET
    base._validate_full_profile = _validate_full_profile
    inputs = tuple(path for path in base.CODE_INPUTS if path != MODULE_PATH)
    base.CODE_INPUTS = tuple(dict.fromkeys((*inputs, *RUN15_CODE_INPUTS)))
    base._build_plan = _build_plan


def _representative_config(model: str) -> str:
    return f"{model}_w4a16"


def _cache_marker(model: str) -> Path:
    return CACHE_ROOT / model / "producer_success.json"


def _snapshot_files(paths: Sequence[Path], *, content_hash: bool) -> list[dict[str, Any]]:
    return [base._snapshot_path(path, content_hash=content_hash) for path in paths]


def _cache_snapshot(model: str, v4_plan: Mapping[str, Any]) -> dict[str, Any]:
    marker_path = _cache_marker(model)
    marker = base._read_json(marker_path)
    if marker.get("status") != "succeeded" or marker.get("model") != model:
        raise base.CampaignError(f"run15 cache producer incomplete: {marker_path}")
    if marker.get("command_sha256") != base._canonical_sha256(marker["command"]):
        raise base.CampaignError(f"cache producer command hash mismatch: {marker_path}")

    static_root, runtime_root = _cache_paths(model)
    static_files = sorted(path for path in static_root.rglob("*") if path.is_file())
    reference_files = sorted(
        path for path in (runtime_root / "ref_logits").rglob("*") if path.is_file()
    )
    if not static_files or not reference_files:
        raise base.CampaignError(f"run15 cache files are incomplete for {model}")

    source = v4_plan["configurations"][
        f"full_block/{_representative_config(model)}"
    ]["source_command"]
    token_root = Path(base._arg_value(source, "--tokens_cache_path"))
    token_files = sorted(path for path in token_root.rglob("*") if path.is_file())
    if len(token_files) != 1:
        raise base.CampaignError(
            f"expected one frozen calibration-token file for {model}, got {len(token_files)}"
        )
    return {
        "tokens": _snapshot_files(token_files, content_hash=True),
        "static": _snapshot_files(static_files, content_hash=False),
        "reference": _snapshot_files(reference_files, content_hash=False),
        "producer_success": base._snapshot_path(marker_path, content_hash=True),
    }


def _build_plan() -> dict[str, Any]:
    v4_plan = _load_verified_plan(V4_PLAN_PATH)
    configurations: dict[str, Any] = {}
    paired_profile_audit: dict[str, Any] = {}
    cache_snapshots: dict[str, Any] = {}

    for model in base.MODEL_SLUGS:
        snapshot = _cache_snapshot(model, v4_plan)
        for branch in base.BRANCH_VALUES:
            cache_snapshots[f"{branch}/{model}"] = copy.deepcopy(snapshot)

    for config in base.CONFIG_IDS:
        normalized: dict[str, dict[str, str]] = {}
        for branch in base.BRANCH_VALUES:
            key = f"{branch}/{config}"
            row = copy.deepcopy(v4_plan["configurations"][key])
            command = _configure_run15_command(
                list(row["source_command"]), branch=branch, config=config
            )
            _validate_full_profile(command, branch=branch, config=config)
            row.update(
                source_command=command,
                a_loss_ratio=1.0,
                optimization_profile="20260808_run15_copy_cast_index",
                historical_v4_result_only=True,
            )
            configurations[key] = row
            flags = v2._flag_map(command)
            flags.pop("--full_block_refresh")
            normalized[branch] = flags
        if normalized["full_block"] != normalized["single_linear"]:
            raise base.CampaignError(f"run15 paired command mismatch: {config}")
        paired_profile_audit[config] = {
            "only_branch_difference": "--full_block_refresh=true/false",
            "normalized_command_sha256": base._canonical_sha256(
                normalized["full_block"]
            ),
        }

    retired = []
    for label, path in (("V4", V4_PLAN_PATH), ("V5-Q32", V5_PLAN_PATH)):
        if path.is_file():
            retired.append(
                {"label": label, "path": str(path), "sha256": base._file_sha256(path)}
            )
    body = {
        "campaign_id": CAMPAIGN_ID,
        "supersedes_campaign_id": v4_plan["campaign_id"],
        "supersedes_reason": (
            "V4/V5 forced math SDPA and NVIDIA_TF32_OVERRIDE=0, while the "
            "documented final optimized path requires deterministic FA4 and "
            "scoped Fisher/Hessian TF32"
        ),
        "output_root": str(OUTPUT_ROOT),
        "branches": dict(base.BRANCH_VALUES),
        "config_ids": list(base.CONFIG_IDS),
        "initial_lrs": list(FIRST_LRS),
        "max_launches_per_branch_config": 20,
        "determinism": {
            "seed": 1,
            "rotation_seed": 0,
            "refresh_seed": 0,
            "torch_deterministic_algorithms": True,
            "fa4_deterministic_argument": True,
            "cublas_workspace_config": ":4096:8",
            "pythonhashseed": "0",
            "realq_deterministic_sdpa": "unset; no SDPA is used",
            "nvidia_tf32_override": "unset; scoped code controls TF32",
        },
        "optimization_profile": {
            "reference": "20260808_run15_copy_cast_index_layer5",
            "attention_backend": "flash_attention_4==4.0.0b25",
            "fisher_tf32": "all multiplication TF32 input / FP32 accumulate",
            "hessian_tf32": True,
            "qwen3_qk_rmsnorm_rope_fusion": True,
            "qwen3_swiglu_fusion": True,
            "retained_flags": dict(RUN15_FLAGS),
            "qwen3_32b_only_memory_exception": "hessian_accum_bsz=32",
        },
        "selection_protocol": {
            **copy.deepcopy(v4_plan["selection_protocol"]),
            "candidate_release": (
                "agent-reviewed low-to-high full-model points; every LR is "
                "retuned because the numerical protocol changed"
            ),
            "old_results": "retired as incompatible protocol evidence",
        },
        "calibration_token_hashes": copy.deepcopy(
            v4_plan["calibration_token_hashes"]
        ),
        "cache_snapshots": cache_snapshots,
        "paired_profile_audit": paired_profile_audit,
        "retired_protocol_plans": retired,
        "code_snapshot": base._code_snapshot(),
        "configurations": configurations,
    }
    body["protocol_fingerprint"] = base._canonical_sha256(body)
    body["created_at"] = base._utc_now()
    return body


def _cache_command(model: str, attempt_dir: Path) -> list[str]:
    v4_plan = _load_verified_plan(V4_PLAN_PATH)
    config = _representative_config(model)
    source = v4_plan["configurations"][f"full_block/{config}"]
    command = _configure_run15_command(
        list(source["source_command"]), branch="full_block", config=config
    )
    base._set_arg(command, "--grad_lr", "0")
    base._set_arg(command, "--skip_eval", "false")
    base._set_arg(command, "--skip_kl_ppl_eval", "false")
    base._set_arg(command, "--lm_eval", "false")
    base._set_arg(command, "--reasoning_eval", "false")
    base._set_arg(command, "--exit_after_precompute", "true")
    base._set_arg(command, "--require_static_cache_hit", "false")
    base._set_arg(command, "--require_reference_cache_hit", "false")
    base._set_arg(command, "--output_dir", str(attempt_dir / "realq_output"))
    base._set_arg(command, "--exp", "run15_cache_producer")
    base._remove_arg(command, "--save_qmodel_path")
    return command


def _prepare_cache(args: argparse.Namespace) -> int:
    model = str(args.model)
    if model not in base.MODEL_SLUGS:
        raise base.CampaignError(f"unknown model: {model}")
    root = CACHE_ROOT / model
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".producer.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise base.CampaignError(f"cache producer already active: {model}") from exc
        marker_path = _cache_marker(model)
        if marker_path.is_file():
            marker = base._read_json(marker_path)
            if marker.get("status") != "succeeded":
                raise base.CampaignError(f"invalid cache marker: {marker_path}")
            print(marker_path)
            return 0

        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        attempt_dir = root / "attempts" / stamp
        attempt_dir.mkdir(parents=True, exist_ok=False)
        command = _cache_command(model, attempt_dir)
        environment = base._worker_environment(str(args.cuda_id))
        gpu = base._gpu_inventory(str(args.cuda_id))
        started_at = base._utc_now()
        log_path = attempt_dir / "execution.log"
        started = time.monotonic()
        with log_path.open("wb") as handle:
            handle.write(
                (
                    f"[{started_at}] command={json.dumps(command, ensure_ascii=False)}\n"
                    f"[{started_at}] gpu={json.dumps(gpu, ensure_ascii=False)}\n"
                    f"[{started_at}] environment_set={json.dumps(RUN15_ENV_OVERRIDES, sort_keys=True)}\n"
                    f"[{started_at}] environment_unset={json.dumps(RUN15_ENV_UNSET)}\n"
                ).encode("utf-8")
            )
            process = subprocess.Popen(
                command,
                cwd=base.REPO_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            returncode = process.wait()
        result = {
            "campaign_id": CAMPAIGN_ID,
            "stage": "run15-shared-cache-producer",
            "model": model,
            "status": "succeeded" if returncode == 0 else "failed",
            "returncode": returncode,
            "started_at": started_at,
            "finished_at": base._utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "gpu": gpu,
            "command": command,
            "command_sha256": base._canonical_sha256(command),
            "execution_environment_contract": {
                "set": dict(sorted(RUN15_ENV_OVERRIDES.items())),
                "unset": sorted(RUN15_ENV_UNSET),
            },
            "log_path": str(log_path),
        }
        base._atomic_json(attempt_dir / "result.json", result)
        if returncode != 0:
            return 1
        static_root, runtime_root = _cache_paths(model)
        static_files = sorted(path for path in static_root.rglob("*") if path.is_file())
        reference_files = sorted(
            path for path in (runtime_root / "ref_logits").rglob("*") if path.is_file()
        )
        if not static_files or not reference_files:
            raise base.CampaignError(
                f"cache producer returned zero but files are incomplete: {model}"
            )
        success = {
            **result,
            "static_files": _snapshot_files(static_files, content_hash=False),
            "reference_files": _snapshot_files(reference_files, content_hash=False),
        }
        base._atomic_json(marker_path, success)
        print(marker_path)
        return 0


def _make_zero_manifest(args: argparse.Namespace) -> int:
    plan = base._read_json(PLAN_PATH)
    base._verify_plan(plan)
    trials = [
        {
            "branch": branch,
            "config": config,
            "schedule": base._schedule_for(config),
            "lr": 0.0,
            "stage": "V6-run15-full-model-zero",
        }
        for config in base.CONFIG_IDS
        for branch in base.BRANCH_VALUES
    ]
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "name": args.name,
        "rationale": (
            "fresh physical-zero endpoints for all 40 run15 branch/config "
            "curves; no V4/V5 metric is numerically adopted"
        ),
        "created_at": base._utc_now(),
        "trials": trials,
    }
    base._validate_manifest(plan, manifest)
    path = OUTPUT_ROOT / "manifests" / f"{args.name}.json"
    if path.exists():
        raise base.CampaignError(f"manifest already exists: {path}")
    base._atomic_json(path, manifest)
    print(path)
    return 0


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    producer = subparsers.add_parser("prepare-cache")
    producer.add_argument("--model", required=True, choices=base.MODEL_SLUGS)
    producer.add_argument("--cuda-id", required=True)
    producer.set_defaults(handler=_prepare_cache)
    zero = subparsers.add_parser("make-zero-manifest")
    zero.add_argument("--name", required=True)
    zero.set_defaults(handler=_make_zero_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"prepare-cache", "make-zero-manifest"}:
        args = _custom_parser().parse_args(arguments)
        try:
            return int(args.handler(args))
        except (
            base.CampaignError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            print(f"fullmodel-retune-v6-run15: {exc}", file=sys.stderr)
            return 2
    return base.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
