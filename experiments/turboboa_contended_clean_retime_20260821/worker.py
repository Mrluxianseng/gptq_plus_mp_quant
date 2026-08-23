#!/usr/bin/env python3
"""Replay contended TurboBOA runs and require checkpoint content identity."""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping

from experiments.turboboa_fair20_20260821 import worker as parent


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT.parent / "experiment_data"
PLAN_PATH = REPO_ROOT / "experiments/turboboa_fair20_20260821/plan.json"
EXPECTED_PLAN_SHA256 = (
    "48ad6e6ca3df8c3af2194e09632e5b4db7380e9b2fabf414f666551db3d1e0cc"
)
OUTPUT_ROOT = DATA_ROOT / "turboboa_contended_clean_retime_20260821_v1"
TARGET_HOST = "j-zogxxxduju-master-0"
RUNS_BY_GPU = {
    5: ("TB20-Q4-W2", "TB20-L8-W4"),
    7: (
        "TB20-Q06-W4A4",
        "TB20-Q06-W4",
        "TB20-Q06-W3",
        "TB20-Q06-W2",
    ),
}
CONFIGURATION_OUTPUT_PATH_KEYS = {
    "cache_dir",
    "results_path",
    "save_qmodel_path",
}
EXPECTED_SOURCES = {
    "experiments/turboboa_fair20_20260821/worker.py": (
        "036b2b13ba55ce2efdb328e1e075e2ccf4eaa1dbf2723c4aa73e879243e20232"
    ),
    "turboBOA/main.py": (
        "c4a3cc3220fd8edd16143f8555a29fedcd5a2b6dd8ccdaf0c573cdc9997b3642"
    ),
    "utils/checkpoint_utils.py": (
        "cfb2d887937877baba298d3183fd6c990a6f164cee55bfb9b32cbb4d9ec634e3"
    ),
}
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class CleanRetimeError(RuntimeError):
    """The replay contract or checkpoint identity failed."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CleanRetimeError(f"JSON root is not an object: {path}")
    return value


def source_run_dir(plan: Mapping[str, Any], run: Mapping[str, Any]) -> Path:
    return (
        Path(plan["output_root"])
        / "runs"
        / str(run["model"])
        / str(run["setting"]).lower()
        / str(run["run_id"])
    )


def verify_static_contract(
    physical_gpu: int,
    run_group_gpu: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if socket.gethostname() != TARGET_HOST:
        raise CleanRetimeError(f"clean retime must run on {TARGET_HOST}")
    if run_group_gpu not in RUNS_BY_GPU:
        raise CleanRetimeError(f"unsupported clean-retime run group: GPU{run_group_gpu}")
    if physical_gpu < 0 or physical_gpu > 7:
        raise CleanRetimeError(f"unsupported physical GPU: {physical_gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise CleanRetimeError("CUDA_VISIBLE_DEVICES does not match physical GPU")
    if any(os.environ.get(key) != value for key, value in THREAD_ENVIRONMENT.items()):
        raise CleanRetimeError("bounded CPU thread environment is missing")
    if parent.sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256:
        raise CleanRetimeError("TurboBOA plan SHA256 mismatch")
    for relative, expected in EXPECTED_SOURCES.items():
        actual = parent.sha256_file(REPO_ROOT / relative)
        if actual != expected:
            raise CleanRetimeError(f"source SHA256 mismatch: {relative}: {actual}")
    plan = read_object(PLAN_PATH)
    if parent.sha256_file(plan["source_plan"]) != plan["source_plan_sha256"]:
        raise CleanRetimeError("source comparison plan SHA256 mismatch")
    source = read_object(Path(plan["source_plan"]))
    by_id = {run["run_id"]: run for run in plan["runs"]}
    expected_ids = RUNS_BY_GPU[run_group_gpu]
    if any(run_id not in by_id for run_id in expected_ids):
        raise CleanRetimeError("clean-retime run set is absent from source plan")
    for run_id in expected_ids:
        run = by_id[run_id]
        directory = source_run_dir(plan, run)
        receipt_path = directory / "run_receipt.json"
        result_path = directory / "result.json"
        checkpoint_path = directory / "qmodel.pt"
        if not receipt_path.is_file() or not result_path.is_file() or not checkpoint_path.is_file():
            raise CleanRetimeError(f"source run is incomplete: {run_id}")
        receipt = read_object(receipt_path)
        result = read_object(result_path)
        if (
            receipt.get("status") != "quantization_succeeded"
            or Path(receipt.get("result", "")).resolve() != result_path.resolve()
            or Path(receipt.get("checkpoint", "")).resolve() != checkpoint_path.resolve()
            or parent.sha256_file(result_path) != receipt.get("result_sha256")
        ):
            raise CleanRetimeError(f"source receipt binding failed: {run_id}")
        parent.validate_result(result, run, source)
    return plan, source


def _primitive_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "model"}


def normalized_configuration(result: Mapping[str, Any]) -> dict[str, Any]:
    configuration = result.get("configuration")
    if not isinstance(configuration, Mapping):
        raise CleanRetimeError("result configuration is not a mapping")
    return {
        key: value
        for key, value in configuration.items()
        if key not in CONFIGURATION_OUTPUT_PATH_KEYS
    }


def compare_checkpoints(source_path: Path, clean_path: Path) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    source_payload = torch.load(
        source_path, map_location="cpu", weights_only=True, mmap=True
    )
    clean_payload = torch.load(
        clean_path, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(source_payload, Mapping) or not isinstance(clean_payload, Mapping):
        raise CleanRetimeError("checkpoint payload is not a mapping")
    if _primitive_checkpoint_payload(source_payload) != _primitive_checkpoint_payload(clean_payload):
        raise CleanRetimeError("checkpoint primitive manifests differ")
    source_state = source_payload.get("model")
    clean_state = clean_payload.get("model")
    if not isinstance(source_state, Mapping) or not isinstance(clean_state, Mapping):
        raise CleanRetimeError("checkpoint model state is not a mapping")
    if tuple(source_state) != tuple(clean_state):
        raise CleanRetimeError("checkpoint state_dict key order differs")
    tensor_bytes = 0
    tensor_count = 0
    for name, source_tensor in source_state.items():
        clean_tensor = clean_state[name]
        if not isinstance(source_tensor, torch.Tensor) or not isinstance(clean_tensor, torch.Tensor):
            raise CleanRetimeError(f"non-tensor state value: {name}")
        if (
            source_tensor.dtype != clean_tensor.dtype
            or source_tensor.shape != clean_tensor.shape
            or source_tensor.stride() != clean_tensor.stride()
            or not torch.equal(source_tensor, clean_tensor)
        ):
            raise CleanRetimeError(f"checkpoint tensor differs: {name}")
        tensor_count += 1
        tensor_bytes += source_tensor.numel() * source_tensor.element_size()
    del source_payload, clean_payload, source_state, clean_state
    gc.collect()
    return {
        "checkpoint_content_identity": True,
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
        "source_checkpoint_bytes": source_path.stat().st_size,
        "clean_checkpoint_bytes": clean_path.stat().st_size,
    }


def run_one(
    plan: dict[str, Any],
    source: dict[str, Any],
    original_run: dict[str, Any],
    physical_gpu: int,
    run_ids: tuple[str, ...],
) -> dict[str, Any]:
    run = copy.deepcopy(original_run)
    run.update(
        canoe_pod=TARGET_HOST,
        job_id="j-zogxxxduju",
        physical_gpu=physical_gpu,
        queue_order=run_ids.index(run["run_id"]),
    )
    effective_plan = copy.deepcopy(plan)
    effective_plan["output_root"] = str(OUTPUT_ROOT)
    run_dir = source_run_dir(effective_plan, run)
    source_dir = source_run_dir(plan, original_run)
    source_receipt_path = source_dir / "run_receipt.json"
    source_result_path = source_dir / "result.json"
    source_checkpoint_path = source_dir / "qmodel.pt"
    source_receipt = read_object(source_receipt_path)
    source_result = read_object(source_result_path)

    if run_dir.exists() or run_dir.is_symlink():
        success_path = run_dir / "clean_replay_result.json"
        if success_path.is_file():
            success = read_object(success_path)
            if (
                success.get("status") != "succeeded"
                or success.get("source_run", {}).get("run_id") != run["run_id"]
                or not success.get("checkpoint_content_identity")
            ):
                raise CleanRetimeError(f"invalid existing clean replay: {success_path}")
            return success

        failure_path = run_dir / "failure.json"
        manifest_path = run_dir / "clean_replay_manifest.json"
        result_path = run_dir / "result.json"
        checkpoint_path = run_dir / "qmodel.pt"
        if not all(
            path.is_file()
            for path in (failure_path, manifest_path, result_path, checkpoint_path)
        ):
            raise CleanRetimeError(f"immutable partial clean replay exists: {run_dir}")
        failure = read_object(failure_path)
        manifest = read_object(manifest_path)
        if (
            failure.get("error_type") != "CleanRetimeError"
            or failure.get("error") != "source and clean result configurations differ"
            or manifest.get("source_run", {}).get("run_id") != run["run_id"]
            or manifest.get("source_plan_sha256") != EXPECTED_PLAN_SHA256
            or manifest.get("source_result_sha256")
            != parent.sha256_file(source_result_path)
            or manifest.get("source_receipt_sha256")
            != parent.sha256_file(source_receipt_path)
        ):
            raise CleanRetimeError(f"existing failure is not recoverable: {failure_path}")
        result = read_object(result_path)
        parent.validate_result(result, run, source)
        if normalized_configuration(result) != normalized_configuration(source_result):
            raise CleanRetimeError(
                "source and clean configurations differ beyond output paths"
            )
        identity = compare_checkpoints(source_checkpoint_path, checkpoint_path)
        record = {
            **manifest,
            "status": "succeeded",
            "ended_at": now(),
            "clean_result": str(result_path),
            "clean_result_sha256": parent.sha256_file(result_path),
            "clean_checkpoint": str(checkpoint_path),
            "clean_quantization_gpu_hours": float(
                result["gpu_hours"]["algorithm_including_precompute_excluding_eval"]
            ),
            "source_runtime_seconds": source_result["runtime_seconds"],
            "clean_runtime_seconds": result["runtime_seconds"],
            "recovered_from_false_configuration_gate": str(failure_path),
            "ignored_configuration_keys": sorted(CONFIGURATION_OUTPUT_PATH_KEYS),
            **identity,
        }
        parent.write_json_atomic(success_path, record)
        return record

    run_dir.mkdir(parents=True)
    command = parent.command_for_run(effective_plan, source, run, run_dir)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "reason": "source timing overlapped EfficientQAT 12x144-thread CPU packing",
        "hostname": socket.gethostname(),
        "physical_gpu": physical_gpu,
        "source_plan": str(PLAN_PATH),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "source_run": original_run,
        "retime_run": run,
        "source_receipt": str(source_receipt_path),
        "source_receipt_sha256": parent.sha256_file(source_receipt_path),
        "source_result": str(source_result_path),
        "source_result_sha256": parent.sha256_file(source_result_path),
        "source_quantization_gpu_hours": float(source_receipt["quantization_gpu_hours"]),
        "thread_environment": THREAD_ENVIRONMENT,
        "source_sha256": EXPECTED_SOURCES,
        "command": command,
        "accounting_policy": (
            "exclude source timing contaminated by CPU oversubscription; use clean "
            "algorithm timing only after checkpoint tensor/content identity"
        ),
    }
    parent.write_json_atomic(run_dir / "clean_replay_manifest.json", manifest)
    log_path = run_dir / "run.log"
    try:
        with log_path.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise CleanRetimeError(f"TurboBOA clean replay exited {completed.returncode}")
        result_path = run_dir / "result.json"
        checkpoint_path = run_dir / "qmodel.pt"
        if not result_path.is_file() or not checkpoint_path.is_file():
            raise CleanRetimeError("clean replay did not publish result/checkpoint")
        result = read_object(result_path)
        parent.validate_result(result, run, source)
        if normalized_configuration(result) != normalized_configuration(source_result):
            raise CleanRetimeError(
                "source and clean configurations differ beyond output paths"
            )
        identity = compare_checkpoints(source_checkpoint_path, checkpoint_path)
        clean_hours = float(
            result["gpu_hours"]["algorithm_including_precompute_excluding_eval"]
        )
        record = {
            **manifest,
            "status": "succeeded",
            "ended_at": now(),
            "clean_result": str(result_path),
            "clean_result_sha256": parent.sha256_file(result_path),
            "clean_checkpoint": str(checkpoint_path),
            "clean_quantization_gpu_hours": clean_hours,
            "source_runtime_seconds": source_result["runtime_seconds"],
            "clean_runtime_seconds": result["runtime_seconds"],
            "ignored_configuration_keys": sorted(CONFIGURATION_OUTPUT_PATH_KEYS),
            **identity,
        }
        parent.write_json_atomic(run_dir / "clean_replay_result.json", record)
        return record
    except Exception as exc:
        parent.write_json_atomic(
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


def execute(
    physical_gpu: int,
    *,
    run_group_gpu: int,
    check_only: bool,
) -> dict[str, Any]:
    plan, source = verify_static_contract(physical_gpu, run_group_gpu)
    run_ids = RUNS_BY_GPU[run_group_gpu]
    if check_only:
        return {
            "schema_version": 1,
            "status": "check_succeeded",
            "hostname": socket.gethostname(),
            "physical_gpu": physical_gpu,
            "run_group_gpu": run_group_gpu,
            "run_ids": run_ids,
        }
    by_id = {run["run_id"]: run for run in plan["runs"]}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = (
        DATA_ROOT
        / "_fair20_physical_gpu_locks_20260821"
        / TARGET_HOST
        / f"gpu{physical_gpu}.lock"
    )
    results = []
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        for run_id in run_ids:
            results.append(
                run_one(plan, source, by_id[run_id], physical_gpu, run_ids)
            )
    summary = {
        "schema_version": 1,
        "status": "succeeded",
        "completed_at": now(),
        "hostname": socket.gethostname(),
        "physical_gpu": physical_gpu,
        "run_group_gpu": run_group_gpu,
        "source_plan": str(PLAN_PATH),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "thread_environment": THREAD_ENVIRONMENT,
        "runs": [
            {
                "run_id": result["source_run"]["run_id"],
                "source_quantization_gpu_hours": result["source_quantization_gpu_hours"],
                "clean_quantization_gpu_hours": result["clean_quantization_gpu_hours"],
                "checkpoint_content_identity": result["checkpoint_content_identity"],
                "result": str(
                    source_run_dir(
                        {"output_root": str(OUTPUT_ROOT)}, result["retime_run"]
                    )
                    / "clean_replay_result.json"
                ),
            }
            for result in results
        ],
    }
    parent.write_json_atomic(
        OUTPUT_ROOT / f"summary_gpu{physical_gpu}_group{run_group_gpu}.json", summary
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--run-group-gpu", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                execute(
                    args.physical_gpu,
                    run_group_gpu=(
                        args.physical_gpu
                        if args.run_group_gpu is None
                        else args.run_group_gpu
                    ),
                    check_only=args.check_only,
                ),
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
