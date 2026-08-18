#!/usr/bin/env python3
"""Fail-closed full-model LR retuning for both REAL-Q Block-GD branches.

This module deliberately separates *suggesting* trials from launching them.
The agent writes a small, audited manifest after inspecting the preceding
batch; workers only execute trials already present in that manifest.  Final
LR selection is never made from the old shallow-layer proxy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import realq_auto_tune as tuner  # noqa: E402


CAMPAIGN_ID = "realq-fullmodel-two-branch-retune-20260817-v1"
OUTPUT_ROOT = (
    REPO_ROOT.parent / "experiment_data" / "realq_fullmodel_retune_20260817"
)
SOURCE_ROOTS = {
    "full_block": REPO_ROOT.parent
    / "experiment_data"
    / "realq_20group_20260808",
    "single_linear": REPO_ROOT.parent
    / "experiment_data"
    / "realq_linear_refresh_20group_20260811",
}
BRANCH_VALUES = {"full_block": "true", "single_linear": "false"}
MODEL_SLUGS = (
    "qwen3-0.6b",
    "llama31-8b-instruct",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-32b",
)
QUANT_SLUGS = ("w4a16", "w4a4kv4", "w3a16", "w2a16")
CONFIG_IDS = tuple(
    f"{model}_{quant}" for model in MODEL_SLUGS for quant in QUANT_SLUGS
)
INITIAL_LRS = (0.0, 1e-7, 1e-6, 1e-5, 1e-4)
MAX_LAUNCHES_PER_BRANCH_CONFIG = 20
PLAN_PATH = OUTPUT_ROOT / "plan.json"
COMMAND_RE = re.compile(r"command=(\[.*\])$")
SAFE_REPLICATE_RE = re.compile(r"[A-Za-z0-9_.-]+")
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
    re.IGNORECASE,
)
TORCH_DISTRIBUTED_ENV = tuner.TORCH_DISTRIBUTED_ENV
WORKER_ENV_OVERRIDES = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "REALQ_DETERMINISTIC_SDPA": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "NVIDIA_TF32_OVERRIDE": "0",
}
WORKER_ENV_UNSET: tuple[str, ...] = ()
CODE_INPUTS = (
    "realq/config.py",
    "realq/pipeline.py",
    "realq/ptq.py",
    "realq/quant/realq_layer.py",
    "realq/refresh/block_gd.py",
    "realq/refresh/fisher_loss.py",
    "realq/refresh/kl_loss.py",
    "realq/runner/layer_loop.py",
    "utils/hadamard_utils.py",
    "utils/reproducibility.py",
    "utils/rotation_utils.py",
    "tools/realq_auto_tune.py",
    "experiments/realq_fullmodel_retune_20260817/campaign.py",
)


class CampaignError(RuntimeError):
    pass


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


def _source_command(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in range(32):
            line = handle.readline()
            if not line:
                break
            match = COMMAND_RE.search(line.strip())
            if match:
                value = json.loads(match.group(1))
                if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    raise CampaignError(f"invalid source command: {path}")
                return value
    raise CampaignError(f"source command header not found: {path}")


def _arg_indices(command: Sequence[str], flag: str) -> list[int]:
    return [index for index, item in enumerate(command) if item == flag]


def _arg_value(command: Sequence[str], flag: str) -> str:
    indices = _arg_indices(command, flag)
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise CampaignError(f"expected one valued {flag}, found {len(indices)}")
    return command[indices[0] + 1]


def _set_arg(command: list[str], flag: str, value: str) -> None:
    indices = _arg_indices(command, flag)
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise CampaignError(f"expected one valued {flag}, found {len(indices)}")
    command[indices[0] + 1] = value


def _set_or_append_arg(command: list[str], flag: str, value: str) -> None:
    indices = _arg_indices(command, flag)
    if len(indices) > 1:
        raise CampaignError(f"expected at most one valued {flag}, found {len(indices)}")
    if indices:
        if indices[0] + 1 >= len(command):
            raise CampaignError(f"missing value after {flag}")
        command[indices[0] + 1] = value
    else:
        command.extend((flag, value))


def _remove_arg(command: list[str], flag: str) -> None:
    indices = _arg_indices(command, flag)
    if len(indices) > 1:
        raise CampaignError(f"expected at most one {flag}, found {len(indices)}")
    if indices:
        index = indices[0]
        if index + 1 >= len(command):
            raise CampaignError(f"missing value after {flag}")
        del command[index : index + 2]


def _stable_float(value: float) -> str:
    if not math.isfinite(value) or value < 0:
        raise CampaignError(f"LR must be finite and non-negative, got {value!r}")
    return format(value, ".17g")


def _trial_id(trial: Mapping[str, Any]) -> str:
    branch = str(trial["branch"])
    config = str(trial["config"])
    schedule = str(trial["schedule"])
    lr_text = _stable_float(float(trial["lr"]))
    digest = _canonical_sha256(
        {"branch": branch, "config": config, "schedule": schedule, "lr": lr_text}
    )[:12]
    identity = f"{branch}__{config}__{schedule}__lr_{digest}"
    replicate = trial.get("replicate")
    if replicate is None:
        return identity
    replicate_text = str(replicate)
    if not SAFE_REPLICATE_RE.fullmatch(replicate_text):
        raise CampaignError(f"invalid replicate key: {replicate!r}")
    return f"{identity}__rep_{replicate_text}"


def _command_source_from_success(root: Path, config_id: str) -> tuple[Path, dict[str, Any]]:
    success_path = root / "runs" / config_id / "formal_success.json"
    success = _read_json(success_path)
    attempts = success.get("formal_attempts")
    if not isinstance(attempts, list):
        raise CampaignError(f"formal_attempts missing: {success_path}")
    successful = [
        item
        for item in attempts
        if isinstance(item, dict) and int(item.get("returncode", -1)) == 0
    ]
    if not successful:
        raise CampaignError(f"no successful formal attempt: {success_path}")
    log_path = Path(str(successful[-1]["log"]))
    if not log_path.is_file():
        raise CampaignError(f"successful source log missing: {log_path}")
    return log_path, success


def _schedule_for(config_id: str) -> str:
    return "none" if config_id.endswith("_w4a4kv4") else "cosine"


def _a_loss_ratio_for(config_id: str) -> float:
    return 0.95 if config_id.startswith("qwen3-4b_") else 1.0


def _a_loss_ratio_text(config_id: str) -> str:
    return "0.95" if config_id.startswith("qwen3-4b_") else "1.0"


def _snapshot_path(path: Path, *, content_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    value = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        value["sha256"] = _file_sha256(path)
    return value


def _cache_snapshot(command: Sequence[str]) -> dict[str, Any]:
    token_root = Path(_arg_value(command, "--tokens_cache_path"))
    static_root = Path(_arg_value(command, "--static_cache_path"))
    runtime_root = Path(_arg_value(command, "--cache_dir"))
    token_files = sorted(path for path in token_root.rglob("*") if path.is_file())
    static_files = sorted(path for path in static_root.rglob("*") if path.is_file())
    reference_files = sorted(
        path for path in (runtime_root / "ref_logits").rglob("*") if path.is_file()
    )
    if len(token_files) != 1:
        raise CampaignError(
            f"expected exactly one calibration-token cache under {token_root}, "
            f"found {len(token_files)}"
        )
    if not static_files or not reference_files:
        raise CampaignError("static/reference cache is incomplete")
    return {
        "tokens": [_snapshot_path(token_files[0], content_hash=True)],
        # These files are multi-GB.  Cache keys, exact path, size, and mtime are
        # frozen and verified before every launch; require_*_cache_hit prevents
        # a worker from silently regenerating them.
        "static": [_snapshot_path(path, content_hash=False) for path in static_files],
        "reference": [
            _snapshot_path(path, content_hash=False) for path in reference_files
        ],
    }


def _code_snapshot() -> dict[str, Any]:
    files = []
    for relative in CODE_INPUTS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise CampaignError(f"code input missing: {path}")
        files.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        )
    return {"files": files, "sha256": _canonical_sha256(files)}


def _validate_full_profile(command: Sequence[str], *, branch: str, config: str) -> None:
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
        "--full_block_refresh": BRANCH_VALUES[branch],
        "--grad_lr_layer_schedule": _schedule_for(config),
        "--a_loss_ratio": _a_loss_ratio_text(config),
        "--rotate": "true",
        "--attention_backend": "sdpa",
    }
    for flag, wanted in expected.items():
        actual = _arg_value(command, flag)
        if actual != wanted:
            raise CampaignError(f"{branch}/{config} {flag}: expected {wanted}, got {actual}")
    wanted_hessian = "32" if config.startswith("qwen3-32b_") else "64"
    if _arg_value(command, "--hessian_accum_bsz") != wanted_hessian:
        raise CampaignError(
            f"{branch}/{config} must keep hessian_accum_bsz={wanted_hessian}"
        )
    if _arg_indices(command, "--quant_stop_layer"):
        raise CampaignError("full-model trial must not contain --quant_stop_layer")


def _build_plan() -> dict[str, Any]:
    configurations: dict[str, Any] = {}
    cache_snapshots: dict[str, Any] = {}
    calibration_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for config_id in CONFIG_IDS:
        model_slug = next(
            model for model in MODEL_SLUGS if config_id.startswith(f"{model}_")
        )
        for branch, source_root in SOURCE_ROOTS.items():
            branch_log, success = _command_source_from_success(source_root, config_id)
            old_result = {
                "selected_lr": float(success["selected_lr"]),
                "selected_kl": success.get("selected_kl"),
                "formal_global_loss_bsz": int(success["formal_global_loss_bsz"]),
                "source_success": str(source_root / "runs" / config_id / "formal_success.json"),
                "source_log": str(branch_log),
            }
            command = _source_command(branch_log)
            # The 2026-08-08 full-block campaign predates this CLI flag and
            # relied on the then-current default.  Current mainline defaults
            # to single-linear refresh, so both branches must be explicit.
            _set_or_append_arg(command, "--full_block_refresh", BRANCH_VALUES[branch])
            _set_arg(command, "--global_loss_bsz", str(old_result["formal_global_loss_bsz"]))
            _set_arg(command, "--grad_lr_layer_schedule", _schedule_for(config_id))
            _set_arg(command, "--a_loss_ratio", _a_loss_ratio_text(config_id))
            _validate_full_profile(command, branch=branch, config=config_id)
            cache_key = f"{branch}/{model_slug}"
            if cache_key not in cache_snapshots:
                cache_snapshots[cache_key] = _cache_snapshot(command)
                calibration_hashes[model_slug][branch] = cache_snapshots[cache_key][
                    "tokens"
                ][0]["sha256"]
            key = f"{branch}/{config_id}"
            configurations[key] = {
                "branch": branch,
                "config": config_id,
                "model": model_slug,
                "schedule": _schedule_for(config_id),
                "a_loss_ratio": _a_loss_ratio_for(config_id),
                "global_loss_bsz": old_result["formal_global_loss_bsz"],
                "old_proxy_result": old_result,
                "source_log_snapshot": _snapshot_path(branch_log, content_hash=True),
                "source_command": command,
            }
    for model_slug, hashes in calibration_hashes.items():
        if set(hashes) != set(BRANCH_VALUES):
            raise CampaignError(f"missing branch calibration hash for {model_slug}: {hashes}")
        if len(set(hashes.values())) != 1:
            raise CampaignError(
                f"branch calibration token tensors differ for {model_slug}: {hashes}"
            )
    body = {
        "campaign_id": CAMPAIGN_ID,
        "output_root": str(OUTPUT_ROOT),
        "branches": BRANCH_VALUES,
        "config_ids": list(CONFIG_IDS),
        "initial_lrs": list(INITIAL_LRS),
        "max_launches_per_branch_config": MAX_LAUNCHES_PER_BRANCH_CONFIG,
        "determinism": {
            "seed": 1,
            "rotation_seed": 0,
            "refresh_seed": 0,
            "realq_deterministic_sdpa": True,
            "cublas_workspace_config": ":4096:8",
            "pythonhashseed": "0",
            "nvidia_tf32_override": "0",
        },
        "selection_protocol": {
            "primary_metric": "wikitext2_exact_kl",
            "initial_log_grid": list(INITIAL_LRS),
            "high_side_requirement": "two successful higher-LR points worse than the incumbent; OOM/infra failures do not count",
            "local_bracket_max_dex": 0.30,
            "top_candidate_min_replicates": 2,
            "tie_rule": "median/MAD or range noise gate; if tied, select the lower LR",
            "two_percent_rule": "reporting-only near-optimal plateau, never a convergence gate",
            "formal_profile_only": True,
        },
        "code_snapshot": _code_snapshot(),
        "calibration_token_hashes": calibration_hashes,
        "cache_snapshots": cache_snapshots,
        "configurations": configurations,
    }
    body["protocol_fingerprint"] = _canonical_sha256(body)
    body["created_at"] = _utc_now()
    return body


def _verify_snapshot(item: Mapping[str, Any]) -> None:
    path = Path(str(item["path"]))
    stat = path.stat()
    if stat.st_size != int(item["size_bytes"]) or stat.st_mtime_ns != int(item["mtime_ns"]):
        raise CampaignError(f"frozen input metadata changed: {path}")
    expected_hash = item.get("sha256")
    if expected_hash is not None and _file_sha256(path) != expected_hash:
        raise CampaignError(f"frozen input content changed: {path}")


def _verify_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError("wrong campaign id")
    stable = dict(plan)
    fingerprint = stable.pop("protocol_fingerprint", None)
    stable.pop("created_at", None)
    if _canonical_sha256(stable) != fingerprint:
        raise CampaignError("plan protocol fingerprint mismatch")
    if _code_snapshot()["sha256"] != plan["code_snapshot"]["sha256"]:
        raise CampaignError("relevant code changed after plan freeze")


def _write_plan(_: argparse.Namespace) -> int:
    candidate = _build_plan()
    if PLAN_PATH.exists():
        existing = _read_json(PLAN_PATH)
        _verify_plan(existing)
        if existing["protocol_fingerprint"] != candidate["protocol_fingerprint"]:
            raise CampaignError("a different immutable plan already exists")
        print(PLAN_PATH)
        return 0
    _atomic_json(PLAN_PATH, candidate)
    print(PLAN_PATH)
    return 0


def _configuration(plan: Mapping[str, Any], trial: Mapping[str, Any]) -> Mapping[str, Any]:
    key = f"{trial['branch']}/{trial['config']}"
    try:
        return plan["configurations"][key]
    except KeyError as exc:
        raise CampaignError(f"unknown branch/config: {key}") from exc


def _validate_trial(plan: Mapping[str, Any], trial: Mapping[str, Any]) -> None:
    configuration = _configuration(plan, trial)
    schedule = str(trial.get("schedule"))
    if schedule != configuration["schedule"]:
        raise CampaignError(
            f"schedule for {trial['branch']}/{trial['config']} must be "
            f"{configuration['schedule']}, got {schedule}"
        )
    _stable_float(float(trial["lr"]))
    _trial_id(trial)


def _validate_manifest(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError("manifest campaign id mismatch")
    if manifest.get("protocol_fingerprint") != plan["protocol_fingerprint"]:
        raise CampaignError("manifest protocol fingerprint mismatch")
    trials = manifest.get("trials")
    if not isinstance(trials, list) or not trials:
        raise CampaignError("manifest must contain a non-empty trials list")
    identities = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise CampaignError("each manifest trial must be an object")
        _validate_trial(plan, trial)
        identity = _trial_id(trial)
        if identity in identities:
            raise CampaignError(f"duplicate trial identity in manifest: {identity}")
        identities.add(identity)


def _existing_launch_count(branch: str, config: str) -> int:
    count = 0
    trial_root = OUTPUT_ROOT / "trials"
    if not trial_root.is_dir():
        return 0
    for spec_path in trial_root.glob("*/spec.json"):
        try:
            spec = _read_json(spec_path)
        except (CampaignError, OSError, json.JSONDecodeError):
            continue
        if spec.get("branch") == branch and spec.get("config") == config:
            count += 1
    return count


def _initial_manifest(args: argparse.Namespace) -> int:
    plan = _read_json(PLAN_PATH)
    _verify_plan(plan)
    trials = []
    for lr in INITIAL_LRS:
        for config in CONFIG_IDS:
            for branch in BRANCH_VALUES:
                trials.append(
                    {
                        "branch": branch,
                        "config": config,
                        "schedule": _schedule_for(config),
                        "lr": lr,
                        "stage": "F2-global-bracket",
                    }
                )
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "name": args.name,
        "rationale": "branch-independent four-decade full-model Exact-KL bracket; old proxy optima are not used as final selectors",
        "created_at": _utc_now(),
        "trials": trials,
    }
    _validate_manifest(plan, manifest)
    path = OUTPUT_ROOT / "manifests" / f"{args.name}.json"
    if path.exists() and _read_json(path) != manifest:
        # created_at naturally differs; an existing manifest is immutable.
        raise CampaignError(f"manifest already exists: {path}")
    if not path.exists():
        _atomic_json(path, manifest)
    print(path)
    return 0


def _parse_trial_text(value: str) -> dict[str, Any]:
    # branch/config/lr[/replicate], e.g. full_block/qwen3-4b_w3a16/3.16e-5/r1
    parts = value.split("/")
    if len(parts) not in {3, 4}:
        raise CampaignError(
            "--trial must be branch/config/lr[/replicate], got " + repr(value)
        )
    branch, config, lr_text = parts[:3]
    trial: dict[str, Any] = {
        "branch": branch,
        "config": config,
        "schedule": _schedule_for(config),
        "lr": float(lr_text),
        "stage": "F2-agent-directed",
    }
    if len(parts) == 4:
        trial["replicate"] = parts[3]
    return trial


def _directed_manifest(args: argparse.Namespace) -> int:
    plan = _read_json(PLAN_PATH)
    _verify_plan(plan)
    trials = [_parse_trial_text(value) for value in args.trial]
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "name": args.name,
        "rationale": args.rationale,
        "created_at": _utc_now(),
        "trials": trials,
    }
    _validate_manifest(plan, manifest)
    by_configuration: dict[tuple[str, str], int] = defaultdict(int)
    for trial in trials:
        by_configuration[(str(trial["branch"]), str(trial["config"]))] += 1
    for (branch, config), additions in by_configuration.items():
        if _existing_launch_count(branch, config) + additions > MAX_LAUNCHES_PER_BRANCH_CONFIG:
            raise CampaignError(f"20-launch hard cap would be exceeded: {branch}/{config}")
    path = OUTPUT_ROOT / "manifests" / f"{args.name}.json"
    if path.exists():
        raise CampaignError(f"manifest already exists: {path}")
    _atomic_json(path, manifest)
    print(path)
    return 0


def _build_trial_command(
    plan: Mapping[str, Any], trial: Mapping[str, Any], trial_dir: Path
) -> list[str]:
    configuration = _configuration(plan, trial)
    command = list(configuration["source_command"])
    _set_arg(command, "--grad_lr", _stable_float(float(trial["lr"])))
    _set_arg(command, "--skip_eval", "false")
    _set_arg(command, "--skip_kl_ppl_eval", "false")
    _set_arg(command, "--lm_eval", "false")
    _set_arg(command, "--reasoning_eval", "false")
    _set_arg(command, "--require_static_cache_hit", "true")
    _set_arg(command, "--require_reference_cache_hit", "true")
    _set_arg(command, "--output_dir", str(trial_dir / "realq_output"))
    _set_arg(command, "--exp", "fullmodel_two_branch_retune")
    _remove_arg(command, "--save_qmodel_path")
    _validate_full_profile(
        command, branch=str(trial["branch"]), config=str(trial["config"])
    )
    return command


def _gpu_inventory(cuda_id: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={cuda_id}",
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    index, name, uuid = [item.strip() for item in result.stdout.strip().split(",", 2)]
    return {"index": index, "name": name, "uuid": uuid}


def _worker_environment(cuda_id: str) -> dict[str, str]:
    """Build the audited child environment for one isolated GPU worker.

    Derivative campaigns may replace ``WORKER_ENV_OVERRIDES`` and
    ``WORKER_ENV_UNSET`` before freezing their plan.  Keeping that policy in a
    named hook avoids copying the trial lifecycle merely to select a faster,
    independently deterministic attention backend.
    """

    environment = os.environ.copy()
    for key in TORCH_DISTRIBUTED_ENV:
        environment.pop(key, None)
    for key in WORKER_ENV_UNSET:
        environment.pop(key, None)
    environment.update(WORKER_ENV_OVERRIDES)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_id
    return environment


def _claim_trial(
    plan: Mapping[str, Any], trial: Mapping[str, Any], trial_dir: Path, spec: Mapping[str, Any]
) -> bool:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    campaign_lock = OUTPUT_ROOT / ".launch_registry.lock"
    with campaign_lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        result_path = trial_dir / "result.json"
        if result_path.exists():
            return False
        spec_path = trial_dir / "spec.json"
        if spec_path.exists():
            existing = _read_json(spec_path)
            if existing != spec:
                raise CampaignError(f"trial spec mismatch: {spec_path}")
            raise CampaignError(
                f"trial was already charged but has no result; use a new replicate id: {trial_dir.name}"
            )
        branch, config = str(trial["branch"]), str(trial["config"])
        if _existing_launch_count(branch, config) >= int(
            plan["max_launches_per_branch_config"]
        ):
            raise CampaignError(f"20-launch hard cap reached: {branch}/{config}")
        trial_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(spec_path, spec)
        return True


def _verify_trial_inputs(plan: Mapping[str, Any], trial: Mapping[str, Any]) -> None:
    configuration = _configuration(plan, trial)
    _verify_snapshot(configuration["source_log_snapshot"])
    cache = plan["cache_snapshots"][
        f"{trial['branch']}/{configuration['model']}"
    ]
    for category in ("tokens", "static", "reference"):
        for item in cache[category]:
            _verify_snapshot(item)


def _run_trial(
    plan: Mapping[str, Any], trial: Mapping[str, Any], *, cuda_id: str
) -> int:
    _verify_trial_inputs(plan, trial)
    identity = _trial_id(trial)
    trial_dir = OUTPUT_ROOT / "trials" / identity
    result_path = trial_dir / "result.json"
    command = _build_trial_command(plan, trial, trial_dir)
    gpu = _gpu_inventory(cuda_id)
    environment = _worker_environment(cuda_id)
    spec = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "identity": identity,
        "branch": trial["branch"],
        "config": trial["config"],
        "schedule": trial["schedule"],
        "lr": float(trial["lr"]),
        "lr_exact": _stable_float(float(trial["lr"])),
        "replicate": trial.get("replicate"),
        "stage": trial.get("stage"),
        "command": command,
        "command_sha256": _canonical_sha256(command),
        "cuda_visible_devices": cuda_id,
        "gpu": gpu,
        "execution_environment_contract": {
            "set": dict(sorted(WORKER_ENV_OVERRIDES.items())),
            "unset": sorted(WORKER_ENV_UNSET),
        },
        "created_at": _utc_now(),
    }
    claimed = _claim_trial(plan, trial, trial_dir, spec)
    if not claimed:
        result = _read_json(result_path)
        return 0 if result.get("status") == "succeeded" else 1

    log_path = trial_dir / "execution.log"
    started_at = _utc_now()
    started = time.monotonic()
    with log_path.open("wb") as handle:
        handle.write(
            (
                f"[{started_at}] command={json.dumps(command, ensure_ascii=False)}\n"
                f"[{started_at}] gpu={json.dumps(gpu, ensure_ascii=False)}\n"
                f"[{started_at}] protocol_fingerprint={plan['protocol_fingerprint']}\n"
            ).encode("utf-8")
        )
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        returncode = process.wait()
    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        **spec,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "log_path": str(log_path),
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
        tail = log_path.read_bytes()[-4 * 1024 * 1024 :].decode(
            "utf-8", errors="replace"
        )
        oom = bool(OOM_RE.search(tail))
        result.update(
            status="failed",
            failure_class="oom" if oom else "infrastructure_or_algorithm",
            error=f"child exited with status {returncode}",
        )
    _atomic_json(result_path, result)
    log_path.chmod(0o644)
    return 0 if result["status"] == "succeeded" else 1


def _run_worker(args: argparse.Namespace) -> int:
    plan = _read_json(PLAN_PATH)
    _verify_plan(plan)
    manifest = _read_json(args.manifest.expanduser().resolve())
    _validate_manifest(plan, manifest)
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise CampaignError("invalid worker index/count")
    selected = [
        trial
        for index, trial in enumerate(manifest["trials"])
        if index % args.worker_count == args.worker_index
    ]
    failures = 0
    for trial in selected:
        identity = _trial_id(trial)
        print(f"[{_utc_now()}] start {identity} on cuda:{args.cuda_id}", flush=True)
        status = _run_trial(plan, trial, cuda_id=args.cuda_id)
        print(f"[{_utc_now()}] finish {identity} status={status}", flush=True)
        failures += int(status != 0)
    return 1 if failures else 0


def _iter_results() -> Iterable[dict[str, Any]]:
    trial_root = OUTPUT_ROOT / "trials"
    if not trial_root.is_dir():
        return
    for result_path in sorted(trial_root.glob("*/result.json")):
        yield _read_json(result_path)


def _analysis_rows() -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in _iter_results():
        groups[(str(result["branch"]), str(result["config"]))].append(result)
    rows = []
    for branch in BRANCH_VALUES:
        for config in CONFIG_IDS:
            results = groups[(branch, config)]
            successes = [row for row in results if row.get("status") == "succeeded"]
            by_lr: dict[float, list[float]] = defaultdict(list)
            for row in successes:
                by_lr[float(row["lr"])].append(float(row["kl"]))
            aggregate = []
            for lr, values in sorted(by_lr.items()):
                ordered = sorted(values)
                median = ordered[len(ordered) // 2] if len(ordered) % 2 else sum(ordered[len(ordered)//2-1:len(ordered)//2+1]) / 2
                aggregate.append(
                    {
                        "lr": lr,
                        "median_kl": median,
                        "min_kl": min(values),
                        "max_kl": max(values),
                        "noise_range": max(values) - min(values),
                        "replicates": len(values),
                    }
                )
            best = min(aggregate, key=lambda row: (row["median_kl"], row["lr"])) if aggregate else None
            plateau = []
            high_side = []
            recommendation: dict[str, Any] | None = None
            if best is not None:
                plateau = [
                    row["lr"]
                    for row in aggregate
                    if row["median_kl"] <= best["median_kl"] * 1.02
                ]
                high_side = [
                    row["lr"]
                    for row in aggregate
                    if row["lr"] > best["lr"] and row["median_kl"] > best["median_kl"]
                ]
                positive = [row for row in aggregate if row["lr"] > 0]
                if positive and best["lr"] == positive[-1]["lr"]:
                    recommendation = {
                        "action": "extend_high",
                        "suggested_lr": positive[-1]["lr"] * 10,
                    }
                elif best["lr"] == 0 and positive:
                    recommendation = {
                        "action": "extend_low",
                        "suggested_lr": positive[0]["lr"] / 10,
                    }
                elif best["lr"] > 0:
                    lower = max((row["lr"] for row in positive if row["lr"] < best["lr"]), default=None)
                    upper = min((row["lr"] for row in positive if row["lr"] > best["lr"]), default=None)
                    candidates = []
                    if lower:
                        candidates.append(math.sqrt(lower * best["lr"]))
                    if upper:
                        candidates.append(math.sqrt(best["lr"] * upper))
                    recommendation = {
                        "action": "refine_log_bracket",
                        "suggested_lrs": candidates,
                    }
            rows.append(
                {
                    "branch": branch,
                    "config": config,
                    "launches": len(results),
                    "successes": len(successes),
                    "failures": len(results) - len(successes),
                    "aggregates": aggregate,
                    "best": best,
                    "two_percent_plateau": plateau,
                    "worse_high_side_points": high_side,
                    "has_two_high_side_points": len(high_side) >= 2,
                    "recommendation": recommendation,
                }
            )
    return rows


def _analyze(args: argparse.Namespace) -> int:
    plan = _read_json(PLAN_PATH)
    _verify_plan(plan)
    rows = _analysis_rows()
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "generated_at": _utc_now(),
        "rows": rows,
    }
    if args.output:
        _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _summarize_manifest(args: argparse.Namespace) -> int:
    plan = _read_json(PLAN_PATH)
    _verify_plan(plan)
    manifest = _read_json(args.manifest.expanduser().resolve())
    _validate_manifest(plan, manifest)
    rows = []
    for trial in manifest["trials"]:
        identity = _trial_id(trial)
        result_path = OUTPUT_ROOT / "trials" / identity / "result.json"
        if result_path.exists():
            result = _read_json(result_path)
            rows.append(
                {
                    "identity": identity,
                    "branch": trial["branch"],
                    "config": trial["config"],
                    "lr": trial["lr"],
                    "replicate": trial.get("replicate"),
                    "status": result.get("status"),
                    "failure_class": result.get("failure_class"),
                    "kl": result.get("kl"),
                    "ppl": result.get("ppl"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "gpu_uuid": result.get("gpu", {}).get("uuid"),
                }
            )
        else:
            rows.append(
                {
                    "identity": identity,
                    "branch": trial["branch"],
                    "config": trial["config"],
                    "lr": trial["lr"],
                    "replicate": trial.get("replicate"),
                    "status": "pending",
                }
            )
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_plan = subparsers.add_parser("write-plan")
    write_plan.set_defaults(handler=_write_plan)
    initial = subparsers.add_parser("make-initial-manifest")
    initial.add_argument("--name", default="f2_global_bracket_v1")
    initial.set_defaults(handler=_initial_manifest)
    directed = subparsers.add_parser("make-directed-manifest")
    directed.add_argument("--name", required=True)
    directed.add_argument("--rationale", required=True)
    directed.add_argument("--trial", action="append", required=True)
    directed.set_defaults(handler=_directed_manifest)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--cuda-id", required=True)
    worker.set_defaults(handler=_run_worker)
    summarize = subparsers.add_parser("summarize-manifest")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.set_defaults(handler=_summarize_manifest)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(handler=_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        CampaignError,
        tuner.TunerError,
        ValueError,
        KeyError,
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"fullmodel-retune-campaign: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
