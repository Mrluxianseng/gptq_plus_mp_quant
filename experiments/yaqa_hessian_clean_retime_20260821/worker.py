#!/usr/bin/env python3
"""Retimestamp Q06 A16 Hessian on idle GPU5, then restore its YAQA queue."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import traceback
from typing import Any

from experiments.yaqa_wclip_fair20_20260821 import worker as yaqa_worker


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "experiments/yaqa_wclip_fair20_20260821/plan.json"
EXPECTED_PLAN_SHA256 = (
    "a7bbd0db640b6d87edbb286f640d744fa13c66c2adccc102d16be9253145bda1"
)
SOURCE_STAGE_ID = "YH-Q06-A16"
RETIME_STAGE_ID = "YH-Q06-A16-CLEAN-RETIME"
RETIME_ROOT = (
    REPO_ROOT.parent
    / "experiment_data/yaqa_hessian_clean_retime_20260821_v1"
)
TARGET_HOST = "j-zogxxxduju-master-0"
TARGET_GPU = 5
LOCK_PATH = (
    REPO_ROOT.parent
    / "experiment_data/_fair20_physical_gpu_locks_20260821"
    / TARGET_HOST
    / f"gpu{TARGET_GPU}.lock"
)


class RetimeError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RetimeError(f"JSON root is not an object: {path}")
    return value


def _tensor_inventory(validation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in validation.get("tensors", []):
        relative = Path(row["path"]).name
        if relative in result:
            raise RetimeError(f"duplicate tensor basename: {relative}")
        result[relative] = {
            key: row[key]
            for key in (
                "bytes",
                "diagonal_mean",
                "dimension",
                "dtype",
                "numel",
                "sha256",
            )
        }
    return result


def _normalized_manifest(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop("save_path", None)
    return result


def execute() -> dict[str, Any]:
    if socket.gethostname() != TARGET_HOST:
        raise RetimeError(f"retime must run on {TARGET_HOST}")
    plan_sha = yaqa_worker.sha256_file(PLAN_PATH)
    if plan_sha != EXPECTED_PLAN_SHA256:
        raise RetimeError(f"plan SHA256 mismatch: {plan_sha}")
    plan = read_object(PLAN_PATH)
    if yaqa_worker.sha256_file(plan["source_plan"]) != plan["source_plan_sha256"]:
        raise RetimeError("source plan SHA256 mismatch")
    source = read_object(Path(plan["source_plan"]))
    matches = [s for s in plan["stages"] if s["stage_id"] == SOURCE_STAGE_ID]
    if len(matches) != 1:
        raise RetimeError("source stage does not resolve exactly once")
    source_stage = matches[0]
    source_dir = Path(plan["output_root"]) / "stages" / SOURCE_STAGE_ID
    source_receipt_path = source_dir / "stage_receipt.json"
    source_receipt = read_object(source_receipt_path)
    if source_receipt.get("status") != "succeeded":
        raise RetimeError("source Hessian stage is not successful")
    source_validation = read_object(Path(source_receipt["hessian_validation"]))
    source_timing = source_receipt["timing"]
    if source_timing.get("status") != "completed":
        raise RetimeError("source Hessian timing is incomplete")

    stage = copy.deepcopy(source_stage)
    stage.update(
        stage_id=RETIME_STAGE_ID,
        canoe_pod=TARGET_HOST,
        job_id="j-zogxxxduju",
        physical_gpu=TARGET_GPU,
        queue_order=0,
    )
    replay_plan = copy.deepcopy(plan)
    replay_plan["output_root"] = str(RETIME_ROOT)
    replay_dir = RETIME_ROOT / "stages" / RETIME_STAGE_ID
    if replay_dir.exists() or replay_dir.is_symlink():
        raise RetimeError(f"immutable retime directory exists: {replay_dir}")

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yaqa_worker.wait_until_gpu_free(TARGET_GPU, 15)
        receipt = yaqa_worker.run_hessian(
            PLAN_PATH,
            plan_sha,
            replay_plan,
            source,
            Path(plan["preflight_path"]),
            stage,
        )

    replay_validation = read_object(Path(receipt["hessian_validation"]))
    source_manifest = source_validation.get("manifest", {})
    replay_manifest = replay_validation.get("manifest", {})
    source_inventory = _tensor_inventory(source_validation)
    replay_inventory = _tensor_inventory(replay_validation)
    if (
        source_validation.get("status") != "validated"
        or replay_validation.get("status") != "validated"
        or _normalized_manifest(source_manifest)
        != _normalized_manifest(replay_manifest)
        or source_inventory != replay_inventory
    ):
        raise RetimeError("clean replay is not tensor-byte-identical to source")

    result = {
        "schema_version": 1,
        "status": "succeeded",
        "reason": "source timing overlapped 12x144-thread CPU packing oversubscription",
        "started_from_source_receipt": str(source_receipt_path.resolve()),
        "source_receipt_sha256": yaqa_worker.sha256_file(source_receipt_path),
        "source_stage": source_stage,
        "source_hessian_gpu_hours": float(source_receipt["hessian_gpu_hours"]),
        "source_gpu_seconds": float(source_timing["gpu_seconds"]),
        "retime_stage": stage,
        "retime_receipt": str(
            (replay_dir / "stage_receipt.json").resolve()
        ),
        "retime_receipt_sha256": yaqa_worker.sha256_file(
            replay_dir / "stage_receipt.json"
        ),
        "clean_hessian_gpu_hours": float(receipt["hessian_gpu_hours"]),
        "clean_gpu_seconds": float(receipt["timing"]["gpu_seconds"]),
        "tensor_count": len(replay_inventory),
        "tensor_byte_identity": True,
        "completed_at": now(),
        "accounting_policy": (
            "use clean timing for the shared A16 producer; retain source tensors "
            "and exclude the contended timing from the fair GPU-hour table"
        ),
    }
    yaqa_worker.write_json_atomic(RETIME_ROOT / "retime_result.json", result)
    return result


def launch_original_gpu5_queue() -> dict[str, Any]:
    plan = read_object(PLAN_PATH)
    python = Path(plan["venv_python"])
    venv_root = python.parent.parent
    torch_lib = venv_root / "lib/python3.12/site-packages/torch/lib"
    log = (
        Path(plan["output_root"])
        / "queue_logs"
        / TARGET_HOST
        / "gpu5.after_clean_retime.log"
    )
    if log.exists() or log.is_symlink():
        raise RetimeError(f"YAQA relaunch log exists: {log}")
    for stage_id in ("YQ-Q4-W3", "YQ-Q4-W2"):
        directory = Path(plan["output_root"]) / "stages" / stage_id
        if directory.exists() or directory.is_symlink():
            raise RetimeError(f"YAQA GPU5 stage is already claimed: {directory}")
    command = [
        str(python),
        "-u",
        "-m",
        "experiments.yaqa_wclip_fair20_20260821.worker",
        "--plan-file",
        str(PLAN_PATH),
        "--expected-plan-sha256",
        EXPECTED_PLAN_SHA256,
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
    payload = {
        "schema_version": 1,
        "status": "launched",
        "launched_at": now(),
        "pid": process.pid,
        "command": command,
        "log": str(log),
    }
    yaqa_worker.write_json_atomic(
        RETIME_ROOT / "yaqa_gpu5_relaunch_receipt.json", payload
    )
    return payload


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
        yaqa_worker.write_json_atomic(RETIME_ROOT / "failure.json", failure)
        traceback.print_exc()
    finally:
        print(
            json.dumps(launch_original_gpu5_queue(), indent=2, sort_keys=True),
            flush=True,
        )
    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
