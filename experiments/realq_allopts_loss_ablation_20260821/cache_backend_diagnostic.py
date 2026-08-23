#!/usr/bin/env python3
"""Regenerate a current-code deterministic-SDPA RealQ static cache.

This diagnostic isolates the Stage-0 attention backend from the run15 fast
paths.  It derives the command from the immutable run15 FA4 producer receipt,
changes only the backend/output/evaluation fields, and compares deterministic
samples against both the historical SDPA and run15 FA4 cache artifacts.
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
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.realq_allopts_loss_ablation_20260821 import (
    compare_static_cache,
)
from experiments.realq_fullmodel_retune_20260817 import campaign as base


DATA_ROOT = REPO_ROOT.parent / "experiment_data"
OUTPUT_ROOT = DATA_ROOT / "realq_static_backend_diagnostic_20260821_v1"
PLAN_PATH = OUTPUT_ROOT / "plan.json"
SOURCE_RECEIPT = (
    DATA_ROOT
    / "realq_fullmodel_retune_20260818_v6_run15"
    / "shared_cache/qwen3-4b/producer_success.json"
)
OLD_SDPA_CACHE = (
    DATA_ROOT
    / "realq_20group_20260808/shared_cache/qwen3-4b/static"
    / "Qwen3-4B_wikitext2_n256_sl2048_e306d5b89c27_world1_rank0.pt"
)
RUN15_FA4_CACHE = (
    DATA_ROOT
    / "realq_fullmodel_retune_20260818_v6_run15/shared_cache/qwen3-4b/static"
    / "Qwen3-4B_wikitext2_n256_sl2048_e306d5b89c27_world1_rank0.pt"
)
LOCK_ROOT = DATA_ROOT / "_fair20_physical_gpu_locks_20260821"
SOURCE_FILES = (
    "experiments/realq_allopts_loss_ablation_20260821/cache_backend_diagnostic.py",
    "experiments/realq_allopts_loss_ablation_20260821/compare_static_cache.py",
    "realq/ptq.py",
    "realq/attention.py",
    "realq/precompute/cache.py",
    "realq/precompute/static_e2e.py",
    "realq/precompute/hooks.py",
    "utils/reproducibility.py",
)


class BackendDiagnosticError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flags(command: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 3
    while index < len(command):
        flag = command[index]
        if not flag.startswith("--") or index + 1 >= len(command):
            raise BackendDiagnosticError(f"malformed command near {command[index:]}")
        if flag in result:
            raise BackendDiagnosticError(f"duplicate command flag: {flag}")
        result[flag] = command[index + 1]
        index += 2
    return result


def _set(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) == 1:
        command[positions[0] + 1] = value
    elif not positions:
        command.extend((flag, value))
    else:
        raise BackendDiagnosticError(f"duplicate command flag: {flag}")


def _source_command() -> tuple[dict[str, Any], list[str]]:
    receipt = _read(SOURCE_RECEIPT)
    if receipt.get("status") != "succeeded":
        raise BackendDiagnosticError("run15 producer receipt is not successful")
    command = list(map(str, receipt["command"]))
    if base._canonical_sha256(command) != receipt.get("command_sha256"):
        raise BackendDiagnosticError("run15 producer command hash mismatch")
    values = _flags(command)
    expected = {
        "--model": str(REPO_ROOT / "modelzoo/Qwen3/Qwen3-4B"),
        "--dataset": "wikitext2",
        "--seed": "1",
        "--rotation_seed": "0",
        "--refresh_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--global_loss_bsz": "4",
        "--hessian_accum_bsz": "64",
        "--rotate": "true",
        "--attention_backend": "flash_attention_4",
        "--hessian_tf32": "true",
        "--grad_lr": "0",
        "--exit_after_precompute": "true",
    }
    for flag, value in expected.items():
        if values.get(flag) != value:
            raise BackendDiagnosticError(
                f"run15 producer drifted {flag}: {values.get(flag)!r} != {value!r}"
            )
    return receipt, command


def _diagnostic_command() -> list[str]:
    _, command = _source_command()
    stage = OUTPUT_ROOT / "sdpa_current_code"
    overrides = {
        "--attention_backend": "sdpa",
        "--static_cache_path": str(stage / "static"),
        "--cache_dir": str(stage / "runtime"),
        "--output_dir": str(stage / "realq_output"),
        "--exp": "current_code_deterministic_sdpa_cache",
        "--skip_eval": "true",
        "--skip_kl_ppl_eval": "true",
        "--lm_eval": "false",
        "--reasoning_eval": "false",
        "--require_static_cache_hit": "false",
    }
    for flag, value in overrides.items():
        _set(command, flag, value)
    values = _flags(command)
    if values["--attention_backend"] != "sdpa":
        raise BackendDiagnosticError("diagnostic command did not select SDPA")
    return command


def _source_snapshot() -> dict[str, Any]:
    files = [
        {
            "path": relative,
            "sha256": base._file_sha256(REPO_ROOT / relative),
            "size_bytes": (REPO_ROOT / relative).stat().st_size,
        }
        for relative in SOURCE_FILES
    ]
    return {"files": files, "sha256": base._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    receipt, _ = _source_command()
    for path in (OLD_SDPA_CACHE, RUN15_FA4_CACHE):
        if not path.is_file() or path.stat().st_size <= 0:
            raise BackendDiagnosticError(f"comparison cache is missing: {path}")
    body = {
        "schema_version": 1,
        "experiment_id": "realq-static-backend-diagnostic-20260821-v1",
        "source_producer_receipt": {
            "path": str(SOURCE_RECEIPT),
            "sha256": base._file_sha256(SOURCE_RECEIPT),
            "command_sha256": receipt["command_sha256"],
        },
        "source_snapshot": _source_snapshot(),
        "command": _diagnostic_command(),
        "comparison_caches": {
            "historical_deterministic_sdpa": {
                "path": str(OLD_SDPA_CACHE),
                "size_bytes": OLD_SDPA_CACHE.stat().st_size,
            },
            "run15_deterministic_fa4": {
                "path": str(RUN15_FA4_CACHE),
                "size_bytes": RUN15_FA4_CACHE.stat().st_size,
            },
        },
        "environment_contract": {
            "same_as_run15_except": {
                "REALQ_DETERMINISTIC_SDPA": "1",
                "NVIDIA_TF32_OVERRIDE": "0",
            },
            "note": (
                "realq.ptq keeps CUDA matmul TF32 disabled during Stage 0; "
                "hessian_tf32 remains frozen true but only affects later quantization"
            ),
        },
    }
    body["command_sha256"] = base._canonical_sha256(body["command"])
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def init_plan() -> dict[str, Any]:
    plan = _build_plan()
    if PLAN_PATH.is_file():
        if _read(PLAN_PATH) != plan:
            raise BackendDiagnosticError("existing diagnostic plan drifted")
        return plan
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    base._atomic_json(PLAN_PATH, plan)
    return plan


def _verify_plan() -> dict[str, Any]:
    plan = _read(PLAN_PATH)
    stable = dict(plan)
    fingerprint = stable.pop("plan_fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise BackendDiagnosticError("diagnostic plan fingerprint mismatch")
    if plan != _build_plan():
        raise BackendDiagnosticError("diagnostic plan source or inputs drifted")
    return plan


def _environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    for key in base.TORCH_DISTRIBUTED_ENV:
        environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
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


def _compact(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        kind: {
            key: comparison[kind][key]
            for key in (
                "sampled_values",
                "unequal_fraction",
                "rms",
                "left_rms",
                "right_rms",
                "relative_rms_to_left",
                "relative_rms_to_right",
                "cosine",
                "max_abs",
                "max_abs_location",
            )
        }
        for kind in ("fisher", "saliency")
    }


def execute(gpu: int) -> dict[str, Any]:
    if socket.gethostname() != "j-zogxxxduju-master-0":
        raise BackendDiagnosticError("diagnostic is pinned to the node1 pod")
    if gpu != 1:
        raise BackendDiagnosticError("diagnostic is pinned to physical GPU1")
    plan = _verify_plan()
    stage = OUTPUT_ROOT / "sdpa_current_code"
    result_path = stage / "result.json"
    if result_path.is_file():
        result = _read(result_path)
        if result.get("status") == "succeeded":
            return result
        raise BackendDiagnosticError("existing diagnostic result failed")
    if stage.exists():
        raise BackendDiagnosticError(f"refusing incomplete stage: {stage}")
    stage.mkdir(parents=True)
    command = list(map(str, plan["command"]))
    log_path = stage / "execution.log"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "hostname": socket.gethostname(),
        "physical_gpu": gpu,
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "command_sha256": base._canonical_sha256(command),
        "started_at": base._utc_now(),
    }
    base._atomic_json(stage / "manifest.json", manifest)
    lock = LOCK_ROOT / socket.gethostname() / f"gpu{gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        with log_path.open("xb") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=_environment(gpu),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    result = {
        **manifest,
        "status": "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": base._utc_now(),
        "log": str(log_path),
        "log_sha256": base._file_sha256(log_path),
    }
    if completed.returncode == 0:
        cache_files = tuple((stage / "static").glob("*.pt"))
        if len(cache_files) != 1:
            raise BackendDiagnosticError(
                f"expected one regenerated cache, found {len(cache_files)}"
            )
        regenerated = cache_files[0]
        old_comparison = compare_static_cache.compare(OLD_SDPA_CACHE, regenerated)
        fa4_comparison = compare_static_cache.compare(RUN15_FA4_CACHE, regenerated)
        base._atomic_json(stage / "old_sdpa_vs_regenerated.json", old_comparison)
        base._atomic_json(stage / "run15_fa4_vs_regenerated.json", fa4_comparison)
        result.update(
            status="succeeded",
            regenerated_cache={
                "path": str(regenerated),
                "size_bytes": regenerated.stat().st_size,
                "sha256": base._file_sha256(regenerated),
            },
            historical_sdpa_vs_regenerated=_compact(old_comparison),
            run15_fa4_vs_regenerated=_compact(fa4_comparison),
        )
    base._atomic_json(result_path, result)
    if result["status"] != "succeeded":
        raise BackendDiagnosticError("SDPA cache producer failed")
    return result


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
