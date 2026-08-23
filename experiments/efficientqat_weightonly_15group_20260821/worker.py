#!/usr/bin/env python3
"""Run one immutable EfficientQAT symmetric weight-only quantization job."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


class WorkerError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_contract(args: argparse.Namespace):
    plan_path = Path(args.plan_file).resolve()
    plan_sha = sha256_file(plan_path)
    if plan_sha != args.expected_plan_sha256:
        raise WorkerError("plan SHA256 mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    matches = [run for run in plan["runs"] if run["run_id"] == args.run_id]
    if len(matches) != 1:
        raise WorkerError(f"run_id must resolve once: {args.run_id}")
    run = matches[0]
    if socket.gethostname() != run["canoe_pod"]:
        raise WorkerError(
            f"wrong pod: {socket.gethostname()} != {run['canoe_pod']}"
        )
    if int(args.physical_gpu) != int(run["physical_gpu"]):
        raise WorkerError("physical GPU differs from the plan")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(run["physical_gpu"]):
        raise WorkerError("CUDA_VISIBLE_DEVICES differs from the plan")
    effective = copy.deepcopy(plan)
    effective["calibration_contract"] = copy.deepcopy(
        plan["calibrations"][run["calibration"]]
    )
    preflight_path = Path(plan["preflight_path"])
    if not preflight_path.is_file():
        raise WorkerError(f"preflight is missing: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "succeeded"
        or preflight.get("plan_sha256") != plan_sha
    ):
        raise WorkerError("preflight does not bind the current plan")
    return plan_path, plan_sha, plan, effective, run, preflight_path


def verify_runtime(plan, effective, run) -> dict[str, Any]:
    import torch

    runtime = plan["runtime_contract"]
    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "lm_eval": importlib.metadata.version("lm_eval"),
    }
    for key, expected in runtime["versions"].items():
        if versions[key] != expected:
            raise WorkerError(f"runtime version mismatch for {key}: {versions[key]}")
    if torch.cuda.device_count() != 1:
        raise WorkerError("each worker must see exactly one GPU")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < int(runtime["minimum_free_gpu_bytes"]):
        raise WorkerError(
            f"GPU is not free enough: {free_bytes} < {runtime['minimum_free_gpu_bytes']}"
        )
    model = plan["models"][run["model"]]
    if sha256_file(Path(model["path"]) / "config.json") != model["config_sha256"]:
        raise WorkerError("model config identity changed")
    artifact = effective["calibration_contract"]["token_artifact"]
    if sha256_file(artifact["path"]) != artifact["sha256"]:
        raise WorkerError("calibration artifact identity changed")
    determinism = runtime["determinism"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != determinism[
        "cublas_workspace_config"
    ]:
        raise WorkerError("CUBLAS deterministic workspace is not installed")
    if os.environ.get("PYTHONHASHSEED") != str(determinism["python_hash_seed"]):
        raise WorkerError("PYTHONHASHSEED is not installed")
    return {
        "versions": versions,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free_bytes_at_start": int(free_bytes),
        "gpu_total_bytes": int(total_bytes),
        "token_artifact_sha256": artifact["sha256"],
        "determinism_environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
        },
    }


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        REPO_ROOT / "experiments/efficientqat_compare/run_one.py",
        REPO_ROOT / "experiments/efficientqat_compare/data.py",
        REPO_ROOT / "experiments/efficientqat_compare/compat.py",
        REPO_ROOT / "experiments/efficientqat_compare/materialize.py",
        REPO_ROOT / "EfficientQAT/quantize/block_ap.py",
        REPO_ROOT / "EfficientQAT/quantize/int_linear_fake.py",
        REPO_ROOT / "EfficientQAT/quantize/int_linear_real.py",
        REPO_ROOT / "EfficientQAT/quantize/quantizer.py",
        REPO_ROOT / "EfficientQAT/quantize/utils.py",
    ]
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan_path, plan_sha, plan, effective, run, preflight_path = load_contract(args)
    output_root = Path(plan["output_root"])
    lock_path = (
        output_root / "_gpu_locks" / run["job_id"] / f"gpu{run['physical_gpu']}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkerError(f"GPU lock is already held: {lock_path}") from exc
        runtime = verify_runtime(plan, effective, run)
        run_dir = output_root / run["output_subdir"]
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise WorkerError(f"immutable run directory already exists: {run_dir}") from exc

        from experiments.efficientqat_compare import run_one

        logger = run_one._configure_logging(run_dir)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "started_at": now(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "plan_file": str(plan_path),
            "plan_sha256": plan_sha,
            "preflight": str(preflight_path),
            "run": run,
            "runtime": runtime,
            "source_sha256": source_hashes(),
            "protocol": {
                "method": "EfficientQAT Block-AP + E2E-QP",
                "weight_quantizer": "signed-symmetric uniform scalar G128",
                "activation_kv": "BF16 identity",
                "rotation": False,
                "calibration": "exact shared 256x2048 WikiText-2 artifact",
            },
        }
        write_json_atomic(run_dir / "run_manifest.json", manifest)
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
            gpu_seconds = float(
                block_stage["gpu_seconds"] + e2e_stage["gpu_seconds"]
            )
            result = {
                **manifest,
                "status": "quantization_succeeded",
                "ended_at": now(),
                "stages": {
                    "block_ap": block_stage,
                    "e2e_qp": e2e_stage,
                    "materialize": materialize_stage,
                },
                "quantization_gpu_seconds": gpu_seconds,
                "quantization_gpu_hours": gpu_seconds / 3600.0,
                "materialized_checkpoint": str(materialized_dir),
                "evaluation_status": "pending",
            }
            write_json_atomic(run_dir / "quantization_result.json", result)
            logger.info(
                "QUANTIZATION_SUCCEEDED gpu_hours=%.9f checkpoint=%s",
                result["quantization_gpu_hours"],
                materialized_dir,
            )
            return result
        except Exception as exc:
            failure = {
                **manifest,
                "status": "failed",
                "ended_at": now(),
                "error_type": type(exc).__qualname__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json_atomic(run_dir / "failure.json", failure)
            raise
        finally:
            gc.collect()
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        execute(args)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        print(f"worker_wall_seconds={time.monotonic() - started:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
