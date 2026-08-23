#!/usr/bin/env python3
"""Isolate non-attention run15 fast paths with one shared SDPA static cache.

The preceding all-on/all-off experiment intentionally compared the complete
production paths, so its checkpoints mixed deterministic FA4 versus SDPA
teacher/Fisher artifacts with the remaining execution optimizations.  This
follow-up keeps the large-difference Qwen3-4B W3 configuration, frozen LR and
all scientific inputs, but changes the all-fast checkpoint producer to SDPA
and forces it to consume the exact legacy SDPA static cache used by all-off.
Both checkpoints are then scored with the pre-existing canonical SDPA
reference cache.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.realq_allopts_loss_ablation_20260821 import runner as prior
from experiments.realq_fullmodel_retune_20260817 import campaign as base
from tools import realq_auto_tune as tuner


CONFIG = "qwen3-4b_w3a16"
MODEL = "qwen3-4b"
OUTPUT_ROOT = (
    REPO_ROOT.parent
    / "experiment_data"
    / "realq_sdpa_fastpath_isolation_20260822_v1"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
CHECKPOINT = OUTPUT_ROOT / "checkpoint" / "quantized.pt"
LOCK_ROOT = prior.LOCK_ROOT
SOURCE_FILES = (
    "experiments/realq_sdpa_fastpath_isolation_20260822/runner.py",
    "experiments/realq_allopts_loss_ablation_20260821/runner.py",
    "realq/config.py",
    "realq/pipeline.py",
    "realq/quant/realq_layer.py",
    "realq/refresh/block_gd.py",
    "realq/refresh/fisher_loss.py",
    "realq/runner/layer_loop.py",
    "utils/hadamard_utils.py",
    "utils/quant_utils.py",
    "utils/rotation_utils.py",
)


class IsolationError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_row() -> Mapping[str, Any]:
    _, rows = prior._formal_rows()
    return rows[CONFIG]


def _source_snapshot() -> dict[str, Any]:
    files = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        files.append(
            {
                "path": relative,
                "sha256": base._file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"files": files, "sha256": base._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    row = _source_row()
    all_off_result = prior._stage_dir(CONFIG, "canonical_all_off") / "result.json"
    all_off_checkpoint = prior._all_off_checkpoint(CONFIG)
    for path in (all_off_result, all_off_checkpoint):
        if not path.is_file() or path.stat().st_size <= 0:
            raise IsolationError(f"required prior artifact missing: {path}")
    body = {
        "schema_version": 1,
        "experiment_id": "realq-sdpa-fastpath-isolation-20260822-v1",
        "config": CONFIG,
        "model": MODEL,
        "selected_lr": row["selected_lr"],
        "source_command_sha256": row["command_sha256"],
        "prior_plan": {
            "path": str(prior.PLAN_PATH),
            "sha256": base._file_sha256(prior.PLAN_PATH),
            "fingerprint": _read(prior.PLAN_PATH)["plan_fingerprint"],
        },
        "shared_sdpa_static_cache": str(
            prior.LEGACY_CACHE_ROOT / MODEL / "static"
        ),
        "canonical_sdpa_cache": str(prior.CANONICAL_CACHE_ROOT / MODEL),
        "all_off_checkpoint": {
            "path": str(all_off_checkpoint),
            "size_bytes": all_off_checkpoint.stat().st_size,
        },
        "all_off_canonical_result": {
            "path": str(all_off_result),
            "sha256": base._file_sha256(all_off_result),
        },
        "fast_on_delta_from_run15": {
            "attention_backend": "sdpa",
            "static_cache": "legacy/current-code deterministic-SDPA cache",
            "environment_set_REALQ_DETERMINISTIC_SDPA": "1",
        },
        "fixed_contract": {
            "same_model_tokens_seed_rotation_refresh_seed": True,
            "same_lr_schedule_and_full_block_scope": True,
            "same_sdpa_static_cache_as_all_off": True,
            "same_canonical_sdpa_reference_as_prior_pair": True,
            "all_remaining_public_run15_fastpaths_stay_on": True,
            "NVIDIA_TF32_OVERRIDE_unset_for_fast_on": True,
        },
        "source_snapshot": _source_snapshot(),
    }
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def init_plan() -> dict[str, Any]:
    plan = _build_plan()
    if PLAN_PATH.is_file():
        if _read(PLAN_PATH) != plan:
            raise IsolationError("existing isolation plan differs from inputs")
        return plan
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    base._atomic_json(PLAN_PATH, plan)
    return plan


def _verify_plan() -> tuple[dict[str, Any], Mapping[str, Any]]:
    plan = _read(PLAN_PATH)
    stable = dict(plan)
    fingerprint = stable.pop("plan_fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise IsolationError("isolation plan fingerprint mismatch")
    if _build_plan() != plan:
        raise IsolationError("isolation plan inputs or source files drifted")
    return plan, _source_row()


def _set(command: list[str], flag: str, value: str) -> None:
    prior._set(command, flag, value)


def _drop(command: list[str], flag: str) -> None:
    prior._drop(command, flag)


def _fast_on_quant_command(row: Mapping[str, Any]) -> list[str]:
    command = list(map(str, row["command"]))
    static = prior.LEGACY_CACHE_ROOT / MODEL / "static"
    _set(command, "--attention_backend", "sdpa")
    _set(command, "--static_cache_path", str(static))
    _set(command, "--cache_dir", str(OUTPUT_ROOT / "runtime"))
    _set(command, "--require_static_cache_hit", "true")
    _set(command, "--require_reference_cache_hit", "false")
    _set(command, "--skip_eval", "true")
    _set(command, "--skip_kl_ppl_eval", "true")
    _set(command, "--lm_eval", "false")
    _set(command, "--reasoning_eval", "false")
    _set(command, "--save_qmodel_path", str(CHECKPOINT))
    _set(command, "--output_dir", str(OUTPUT_ROOT / "quant" / "realq_output"))
    _set(command, "--exp", "sdpa_same_cache_fast_on_qwen3_4b_w3")
    return command


def _canonical_command(row: Mapping[str, Any]) -> list[str]:
    # The all-off canonical command already selects the legacy eager evaluator
    # and requires a hit on the shared deterministic-SDPA reference cache.
    command = prior._canonical_eval_command(row, CONFIG, "all_off")
    _set(command, "--load_qmodel_path", str(CHECKPOINT))
    _set(
        command,
        "--output_dir",
        str(OUTPUT_ROOT / "canonical_fast_on" / "realq_output"),
    )
    _set(command, "--exp", "canonical_sdpa_fast_on_qwen3_4b_w3")
    return command


def _validate_commands(
    row: Mapping[str, Any], quant: Sequence[str], canonical: Sequence[str]
) -> None:
    source = prior._flags(list(map(str, row["command"])))
    quant_flags = prior._flags(quant)
    mutable = {
        "--attention_backend",
        "--static_cache_path",
        "--cache_dir",
        "--require_static_cache_hit",
        "--require_reference_cache_hit",
        "--save_qmodel_path",
        "--output_dir",
        "--exp",
    }
    for flag, value in source.items():
        if flag not in mutable and quant_flags.get(flag) != value:
            raise IsolationError(f"fast-on command drifted {flag}")
    for flag, expected in prior.ALL_ON_FLAGS.items():
        if flag == "--attention_backend":
            expected = "sdpa"
        if quant_flags.get(flag) != expected:
            raise IsolationError(
                f"fast-on {flag}={quant_flags.get(flag)!r}, expected {expected!r}"
            )
    expected_static = str(prior.LEGACY_CACHE_ROOT / MODEL / "static")
    if quant_flags.get("--static_cache_path") != expected_static:
        raise IsolationError("fast-on is not using the all-off SDPA static cache")
    canonical_flags = prior._flags(canonical)
    if canonical_flags.get("--attention_backend") != "sdpa":
        raise IsolationError("canonical evaluator is not SDPA")
    if canonical_flags.get("--load_qmodel_path") != str(CHECKPOINT):
        raise IsolationError("canonical evaluator is not loading fast-on checkpoint")
    if canonical_flags.get("--require_reference_cache_hit") != "true":
        raise IsolationError("canonical evaluator may regenerate its reference")


def _fast_on_environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    for key in base.TORCH_DISTRIBUTED_ENV:
        environment.pop(key, None)
    environment.pop("NVIDIA_TF32_OVERRIDE", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "REALQ_DETERMINISTIC_SDPA": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": os.pathsep.join(
                value
                for value in (str(REPO_ROOT), environment.get("PYTHONPATH", ""))
                if value
            ),
        }
    )
    return environment


def _execute(
    *,
    stage: str,
    command: list[str],
    environment: Mapping[str, str],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    directory = OUTPUT_ROOT / stage
    result_path = directory / "result.json"
    if result_path.is_file():
        result = _read(result_path)
        if result.get("status") == "succeeded":
            return result
        raise IsolationError(f"existing stage failed: {result_path}")
    if directory.exists() or directory.is_symlink():
        raise IsolationError(f"refusing incomplete stage directory: {directory}")
    directory.mkdir(parents=True)
    log_path = directory / "execution.log"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "stage": stage,
        "hostname": socket.gethostname(),
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "command_sha256": base._canonical_sha256(command),
        "started_at": base._utc_now(),
    }
    base._atomic_json(directory / "manifest.json", manifest)
    started = time.monotonic()
    with log_path.open("xb") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {
        **manifest,
        "status": "failed",
        "returncode": completed.returncode,
        "finished_at": base._utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log_path),
        "log_sha256": base._file_sha256(log_path),
    }
    if completed.returncode == 0:
        if stage == "quant":
            if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size <= 0:
                raise IsolationError("quantization returned zero without checkpoint")
            result.update(
                checkpoint=str(CHECKPOINT),
                checkpoint_size_bytes=CHECKPOINT.stat().st_size,
            )
        else:
            kl, ppl = tuner.parse_exact_metric(
                log_path.read_text(encoding="utf-8", errors="replace"),
                "wikitext2",
            )
            result.update(exact_kl=kl, ppl=ppl)
        result["status"] = "succeeded"
    base._atomic_json(result_path, result)
    if result["status"] != "succeeded":
        raise IsolationError(f"stage failed: {stage}")
    return result


def execute(gpu: int) -> dict[str, Any]:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise IsolationError("experiment must run inside a Canoe debug pod")
    if gpu < 0 or gpu > 7:
        raise IsolationError(f"invalid physical GPU: {gpu}")
    plan, row = _verify_plan()
    quant = _fast_on_quant_command(row)
    canonical = _canonical_command(row)
    _validate_commands(row, quant, canonical)
    lock = LOCK_ROOT / hostname / f"gpu{gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _execute(
            stage="quant",
            command=quant,
            environment=_fast_on_environment(gpu),
            plan=plan,
        )
        fast = _execute(
            stage="canonical_fast_on",
            command=canonical,
            environment=prior._legacy_environment(gpu),
            plan=plan,
        )
    off = _read(Path(plan["all_off_canonical_result"]["path"]))
    fast_kl = float(fast["exact_kl"])
    off_kl = float(off["exact_kl"])
    summary = {
        "schema_version": 1,
        "status": "succeeded",
        "config": CONFIG,
        "selected_lr": plan["selected_lr"],
        "fast_on_same_sdpa_cache": {
            "exact_kl": fast_kl,
            "ppl": fast["ppl"],
        },
        "all_off_same_sdpa_cache": {
            "exact_kl": off_kl,
            "ppl": off["ppl"],
        },
        "off_minus_fast_on_kl": off_kl - fast_kl,
        "off_vs_fast_on_relative_kl": (
            (off_kl - fast_kl) / fast_kl if fast_kl != 0.0 else None
        ),
        "finished_at": base._utc_now(),
        "plan_fingerprint": plan["plan_fingerprint"],
    }
    base._atomic_json(OUTPUT_ROOT / "pair_result.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-plan", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    try:
        if args.init_plan:
            print(json.dumps(init_plan(), indent=2, sort_keys=True), flush=True)
            return 0
        if args.physical_gpu is None:
            parser.error("--physical-gpu is required unless --init-plan")
        print(json.dumps(execute(args.physical_gpu), indent=2, sort_keys=True))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
