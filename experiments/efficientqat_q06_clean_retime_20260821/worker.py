#!/usr/bin/env python3
"""Replay Q06 W4/W3/W2 under the byte-equivalent pack/thread contract."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import traceback
from typing import Any

from experiments.efficientqat_weightonly_15group_20260821 import (
    worker as parent_worker,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO_ROOT / "experiments/efficientqat_weightonly_15group_20260821/plan.json"
)
EXPECTED_PLAN_SHA256 = (
    "ceffe64189f07a5753a4feffa21553d8c802fa4c798af645ca6199b0fa271032"
)
PACK_SOURCE = REPO_ROOT / "EfficientQAT/quantize/int_linear_real.py"
EXPECTED_PACK_SOURCE_SHA256 = (
    "d259af6def9db43766ecd704a6fc03f04913c8156d81d530f62777fe5defc2dc"
)
RUN_IDS = ("EQ15-Q06-W4", "EQ15-Q06-W3", "EQ15-Q06-W2")
OUTPUT_ROOT = (
    REPO_ROOT.parent
    / "experiment_data/efficientqat_q06_clean_retime_20260821_v2"
)
TARGET_HOST = "j-zogxxxduju-master-0"
TARGET_GPU = 5
LOCK_PATH = (
    REPO_ROOT.parent
    / "experiment_data/_fair20_physical_gpu_locks_20260821"
    / TARGET_HOST
    / f"gpu{TARGET_GPU}.lock"
)
YAQA_PLAN = REPO_ROOT / "experiments/yaqa_wclip_fair20_20260821/plan.json"
EXPECTED_YAQA_PLAN_SHA256 = (
    "a7bbd0db640b6d87edbb286f640d744fa13c66c2adccc102d16be9253145bda1"
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class CleanReplayError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CleanReplayError(f"JSON root is not an object: {path}")
    return value


def _source_result(run_id: str, plan: dict[str, Any], run: dict[str, Any]) -> Path:
    parent = Path(plan["output_root"]) / run["output_subdir"]
    if run_id == "EQ15-Q06-W2":
        return parent / "quantization_result.json"
    recovery = (
        REPO_ROOT.parent
        / "experiment_data/efficientqat_weightonly_15group_20260821_v4_recovery"
        / run["output_subdir"]
    )
    return recovery / "quantization_result.json"


def _run_one(
    plan: dict[str, Any], original_run: dict[str, Any]
) -> dict[str, Any]:
    from experiments.efficientqat_compare import run_one

    run = copy.deepcopy(original_run)
    run.update(
        canoe_pod=TARGET_HOST,
        job_id="j-zogxxxduju",
        physical_gpu=TARGET_GPU,
        output_subdir=f"runs/{original_run['run_id']}",
    )
    effective = copy.deepcopy(plan)
    effective["output_root"] = str(OUTPUT_ROOT)
    effective["calibration_contract"] = copy.deepcopy(
        plan["calibrations"][run["calibration"]]
    )
    runtime = parent_worker.verify_runtime(plan, effective, run)
    run_dir = OUTPUT_ROOT / run["output_subdir"]
    if run_dir.exists() or run_dir.is_symlink():
        raise CleanReplayError(f"immutable clean replay exists: {run_dir}")
    run_dir.mkdir(parents=True)
    logger = run_one._configure_logging(run_dir)
    source_result_path = _source_result(original_run["run_id"], plan, original_run)
    source_result = read_object(source_result_path)
    source_model = Path(source_result["materialized_checkpoint"]) / "model.safetensors"
    if source_result.get("status") != "quantization_succeeded" or not source_model.is_file():
        raise CleanReplayError(f"source result is incomplete: {source_result_path}")
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "hostname": socket.gethostname(),
        "physical_gpu": TARGET_GPU,
        "source_plan": str(PLAN_PATH),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "source_run": original_run,
        "retime_run": run,
        "runtime": runtime,
        "source_sha256": parent_worker.source_hashes(),
        "pack_source_sha256": parent_worker.sha256_file(PACK_SOURCE),
        "thread_environment": THREAD_ENVIRONMENT,
        "source_result": str(source_result_path.resolve()),
        "source_result_sha256": parent_worker.sha256_file(source_result_path),
        "source_quantization_gpu_hours": float(
            source_result["quantization_gpu_hours"]
        ),
        "accounting_policy": (
            "exclude source timing contaminated by 12x144-thread CPU packing; "
            "use clean Block-AP+E2E timing only after checkpoint byte identity"
        ),
    }
    parent_worker.write_json_atomic(run_dir / "clean_replay_manifest.json", manifest)
    try:
        block_dir, block_stage = run_one.run_block_ap(
            effective, run, run_dir, "cuda:0", logger
        )
        e2e_dir, e2e_stage = run_one.run_e2e_qp(
            effective, run, run_dir, "cuda:0", block_dir, logger
        )
        materialized_dir, materialize_stage = run_one.run_materialize(
            effective, run, run_dir, e2e_dir
        )
        clean_model = materialized_dir / "model.safetensors"
        source_model_sha = parent_worker.sha256_file(source_model)
        clean_model_sha = parent_worker.sha256_file(clean_model)
        if clean_model_sha != source_model_sha:
            raise CleanReplayError(
                f"materialized checkpoint differs: {clean_model_sha} != {source_model_sha}"
            )
        gpu_seconds = float(
            block_stage["gpu_seconds"] + e2e_stage["gpu_seconds"]
        )
        result = {
            **manifest,
            "status": "succeeded",
            "ended_at": now(),
            "stages": {
                "block_ap": block_stage,
                "e2e_qp": e2e_stage,
                "materialize": materialize_stage,
            },
            "clean_quantization_gpu_seconds": gpu_seconds,
            "clean_quantization_gpu_hours": gpu_seconds / 3600.0,
            "clean_materialized_checkpoint": str(materialized_dir),
            "source_model_safetensors_sha256": source_model_sha,
            "clean_model_safetensors_sha256": clean_model_sha,
            "checkpoint_byte_identity": True,
        }
        parent_worker.write_json_atomic(run_dir / "clean_replay_result.json", result)
        return result
    except Exception as exc:
        parent_worker.write_json_atomic(
            run_dir / "failure.json",
            {
                **manifest,
                "status": "failed",
                "ended_at": now(),
                "error_type": type(exc).__qualname__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def execute() -> dict[str, Any]:
    if socket.gethostname() != TARGET_HOST:
        raise CleanReplayError(f"clean replay must run on {TARGET_HOST}")
    if parent_worker.sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256:
        raise CleanReplayError("source plan SHA256 mismatch")
    pack_sha = parent_worker.sha256_file(PACK_SOURCE)
    if pack_sha != EXPECTED_PACK_SOURCE_SHA256:
        raise CleanReplayError(f"pack source SHA256 mismatch: {pack_sha}")
    if any(os.environ.get(key) != value for key, value in THREAD_ENVIRONMENT.items()):
        raise CleanReplayError("bounded CPU thread environment is missing")
    plan = read_object(PLAN_PATH)
    by_id = {run["run_id"]: run for run in plan["runs"]}
    if any(run_id not in by_id for run_id in RUN_IDS):
        raise CleanReplayError("Q06 source run set is incomplete")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise CleanReplayError(f"immutable output root exists: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with LOCK_PATH.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        for run_id in RUN_IDS:
            results.append(_run_one(plan, by_id[run_id]))
    summary = {
        "schema_version": 1,
        "status": "succeeded",
        "completed_at": now(),
        "source_plan": str(PLAN_PATH),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "pack_source_sha256": EXPECTED_PACK_SOURCE_SHA256,
        "thread_environment": THREAD_ENVIRONMENT,
        "runs": [
            {
                "run_id": result["source_run"]["run_id"],
                "source_quantization_gpu_hours": result[
                    "source_quantization_gpu_hours"
                ],
                "clean_quantization_gpu_hours": result[
                    "clean_quantization_gpu_hours"
                ],
                "checkpoint_byte_identity": result["checkpoint_byte_identity"],
                "result": str(
                    OUTPUT_ROOT
                    / result["retime_run"]["output_subdir"]
                    / "clean_replay_result.json"
                ),
            }
            for result in results
        ],
    }
    parent_worker.write_json_atomic(OUTPUT_ROOT / "summary.json", summary)
    return summary


def launch_yaqa_gpu5_queue() -> dict[str, Any]:
    if parent_worker.sha256_file(YAQA_PLAN) != EXPECTED_YAQA_PLAN_SHA256:
        raise CleanReplayError("YAQA plan SHA256 mismatch")
    plan = read_object(YAQA_PLAN)
    python = Path(plan["venv_python"])
    venv_root = python.parent.parent
    torch_lib = venv_root / "lib/python3.12/site-packages/torch/lib"
    for stage_id in ("YQ-Q4-W3", "YQ-Q4-W2"):
        stage_dir = Path(plan["output_root"]) / "stages" / stage_id
        if stage_dir.exists() or stage_dir.is_symlink():
            raise CleanReplayError(f"YAQA stage is already claimed: {stage_dir}")
    log = (
        Path(plan["output_root"])
        / "queue_logs"
        / TARGET_HOST
        / "gpu5.after_efficient_q06_clean_retime_v2.log"
    )
    if log.exists() or log.is_symlink():
        raise CleanReplayError(f"YAQA relaunch log exists: {log}")
    command = [
        str(python),
        "-u",
        "-m",
        "experiments.yaqa_wclip_fair20_20260821.worker",
        "--plan-file",
        str(YAQA_PLAN),
        "--expected-plan-sha256",
        EXPECTED_YAQA_PLAN_SHA256,
        "--physical-gpu",
        str(TARGET_GPU),
    ]
    env = os.environ.copy()
    env.update(
        VIRTUAL_ENV=str(venv_root),
        PATH=f"{python.parent}:{env.get('PATH', '')}",
        LD_LIBRARY_PATH=f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}",
        CUDA_VISIBLE_DEVICES=str(TARGET_GPU),
        PYTHONPATH=":".join(
            (
                str(REPO_ROOT),
                str(REPO_ROOT / "YAQA_wclip"),
                str(REPO_ROOT / "YAQA_wclip/hessian_llama"),
            )
        ),
        PYTHONHASHSEED="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()
    receipt = {
        "schema_version": 1,
        "status": "launched",
        "launched_at": now(),
        "pid": process.pid,
        "command": command,
        "log": str(log),
    }
    parent_worker.write_json_atomic(
        OUTPUT_ROOT / "yaqa_gpu5_relaunch_receipt.json", receipt
    )
    return receipt


def main() -> int:
    failure = None
    try:
        print(json.dumps(execute(), indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "ended_at": now(),
            "error_type": type(exc).__qualname__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        parent_worker.write_json_atomic(OUTPUT_ROOT / "failure.json", failure)
        traceback.print_exc()
    finally:
        print(
            json.dumps(launch_yaqa_gpu5_queue(), indent=2, sort_keys=True),
            flush=True,
        )
    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
