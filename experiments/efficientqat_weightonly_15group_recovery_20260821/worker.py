#!/usr/bin/env python3
"""Recover one v3 EfficientQAT run without repeating successful Block-AP."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_PLAN = (
    REPO_ROOT / "experiments/efficientqat_weightonly_15group_20260821/plan.json"
)
EXPECTED_PARENT_PLAN_SHA256 = (
    "ceffe64189f07a5753a4feffa21553d8c802fa4c798af645ca6199b0fa271032"
)
EXPECTED_FAILED_RUN_ONE_SHA256 = (
    "c976e7bba151eb4366abdacbbd020f00d5c93aee7d6606de6bb7713dedcff908"
)
RECOVERY_ROOT = (
    REPO_ROOT.parent
    / "experiment_data/efficientqat_weightonly_15group_20260821_v4_recovery"
)
PHYSICAL_LOCK_ROOT = (
    REPO_ROOT.parent / "experiment_data/_fair20_physical_gpu_locks_20260821"
)
EXPECTED_ERROR = (
    "positive scale projection contract failed: "
    "9.999999747378752e-05 < 0.0001"
)


class RecoveryError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON root must be an object: {source}")
    return value


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    os.chmod(destination, 0o644)


def now() -> str:
    from datetime import timezone

    return datetime.now(timezone.utc).isoformat()


def _run_contract(run_id: str, physical_gpu: int):
    if sha256_file(PARENT_PLAN) != EXPECTED_PARENT_PLAN_SHA256:
        raise RecoveryError("parent EfficientQAT plan changed")
    plan = read_json(PARENT_PLAN)
    matches = [run for run in plan["runs"] if run["run_id"] == run_id]
    if len(matches) != 1:
        raise RecoveryError(f"run_id must resolve exactly once: {run_id}")
    run = matches[0]
    if socket.gethostname() != run["canoe_pod"]:
        raise RecoveryError("recovery worker is on the wrong pod")
    if physical_gpu != int(run["physical_gpu"]):
        raise RecoveryError("recovery worker is on the wrong physical GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RecoveryError("CUDA_VISIBLE_DEVICES differs from the contract")
    effective = copy.deepcopy(plan)
    effective["calibration_contract"] = copy.deepcopy(
        plan["calibrations"][run["calibration"]]
    )
    return plan, effective, run


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("launcher.py").resolve(),
        REPO_ROOT / "experiments/efficientqat_compare/run_one.py",
        REPO_ROOT / "experiments/efficientqat_compare/materialize.py",
        REPO_ROOT
        / "experiments/efficientqat_weightonly_15group_20260821/worker.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths
    }


def _tree_stat_identity(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        stat = path.stat()
        records.append(
            {
                "relative": str(path.relative_to(root)),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
            }
        )
    if not records:
        raise RecoveryError(f"Block-AP artifact has no files: {root}")
    return records


def _superseded_e2e_seconds(block_stage: dict[str, Any], failure: dict[str, Any]) -> float:
    try:
        block_end = datetime.fromisoformat(block_stage["ended_at"])
        failure_end = datetime.fromisoformat(failure["ended_at"])
        return max(0.0, (failure_end - block_end).total_seconds())
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _recover(
    plan: dict[str, Any],
    effective: dict[str, Any],
    run: dict[str, Any],
    parent_dir: Path,
    failure: dict[str, Any],
) -> dict[str, Any]:
    from experiments.efficientqat_compare import run_one
    from experiments.efficientqat_weightonly_15group_20260821 import (
        worker as parent_worker,
    )

    if failure.get("status") != "failed" or failure.get("error") != EXPECTED_ERROR:
        raise RecoveryError(
            f"parent failure is not the audited FP32 gate incident: {failure.get('error')}"
        )
    if failure.get("plan_sha256") != EXPECTED_PARENT_PLAN_SHA256:
        raise RecoveryError("parent failure is bound to another plan")
    if (
        failure.get("source_sha256", {}).get(
            "experiments/efficientqat_compare/run_one.py"
        )
        != EXPECTED_FAILED_RUN_ONE_SHA256
    ):
        raise RecoveryError("parent failure source is not the audited implementation")
    patched_run_one_sha = sha256_file(
        REPO_ROOT / "experiments/efficientqat_compare/run_one.py"
    )
    if patched_run_one_sha == EXPECTED_FAILED_RUN_ONE_SHA256:
        raise RecoveryError("representable-threshold fix is not installed")

    block_stage_path = parent_dir / "block_ap/stage.json"
    block_stage = read_json(block_stage_path)
    block_dir = Path(block_stage.get("output", "")).resolve()
    if (
        block_stage.get("status") != "succeeded"
        or block_stage.get("stage") != "block_ap"
        or block_stage.get("weight_bits") != run["w_bits"]
        or block_dir != (parent_dir / "block_ap/packed_model").resolve()
        or not block_dir.is_dir()
    ):
        raise RecoveryError("parent Block-AP success gate failed")
    before = _tree_stat_identity(block_dir)

    # Reuse the original runtime gate: exact model/token hashes, one visible
    # L20C, deterministic environment, and the same minimum-free-memory bound.
    runtime = parent_worker.verify_runtime(plan, effective, run)
    recovery_dir = RECOVERY_ROOT / run["output_subdir"]
    recovery_dir.parent.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "run": run,
        "parent_plan": str(PARENT_PLAN),
        "parent_plan_sha256": EXPECTED_PARENT_PLAN_SHA256,
        "parent_run_dir": str(parent_dir),
        "parent_failure": str((parent_dir / "failure.json").resolve()),
        "parent_failure_sha256": sha256_file(parent_dir / "failure.json"),
        "parent_block_stage": str(block_stage_path.resolve()),
        "parent_block_stage_sha256": sha256_file(block_stage_path),
        "parent_block_artifact_stat_identity": before,
        "runtime": runtime,
        "source_sha256": _source_hashes(),
        "recovery_policy": {
            "reason": "FP32 representable 1e-4 post-projection gate false positive",
            "reused_stage": "successful parent Block-AP",
            "rerun_stages": ["E2E-QP", "materialize"],
            "training_configuration_changed": False,
            "algorithm_gpu_hour_accounting": "parent Block-AP + recovered E2E-QP",
            "superseded_failed_e2e_excluded": True,
        },
    }
    write_json_atomic(recovery_dir / "recovery_manifest.json", manifest)
    logger = run_one._configure_logging(recovery_dir)
    try:
        e2e_dir, e2e_stage = run_one.run_e2e_qp(
            effective,
            run,
            recovery_dir,
            "cuda:0",
            block_dir,
            logger,
        )
        if _tree_stat_identity(block_dir) != before:
            raise RecoveryError("parent Block-AP artifact changed during recovery")
        materialized_dir, materialize_stage = run_one.run_materialize(
            effective,
            run,
            recovery_dir,
            e2e_dir,
        )
        gpu_seconds = float(block_stage["gpu_seconds"]) + float(
            e2e_stage["gpu_seconds"]
        )
        result = {
            **manifest,
            "status": "quantization_succeeded",
            "ended_at": now(),
            "stages": {
                "block_ap": {
                    **block_stage,
                    "reused_from_parent": True,
                    "source_stage_sha256": sha256_file(block_stage_path),
                },
                "e2e_qp": e2e_stage,
                "materialize": materialize_stage,
            },
            "quantization_gpu_seconds": gpu_seconds,
            "quantization_gpu_hours": gpu_seconds / 3600.0,
            "materialized_checkpoint": str(materialized_dir),
            "superseded_failed_e2e_wall_seconds": _superseded_e2e_seconds(
                block_stage, failure
            ),
            "evaluation_status": "pending",
        }
        write_json_atomic(recovery_dir / "quantization_result.json", result)
        return result
    except Exception as exc:
        write_json_atomic(
            recovery_dir / "failure.json",
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


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan, effective, run = _run_contract(args.run_id, args.physical_gpu)
    parent_dir = Path(plan["output_root"]) / run["output_subdir"]
    parent_success = parent_dir / "quantization_result.json"
    parent_failure = parent_dir / "failure.json"
    lock = (
        PHYSICAL_LOCK_ROOT
        / socket.gethostname()
        / f"gpu{args.physical_gpu}.lock"
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        while not parent_success.is_file() and not parent_failure.is_file():
            print(
                f"{now()} waiting_for_parent run_id={run['run_id']} "
                f"gpu={args.physical_gpu}",
                flush=True,
            )
            time.sleep(args.poll_seconds)
        if parent_success.is_file():
            return {
                "schema_version": 1,
                "status": "not_needed_parent_succeeded",
                "run": run,
                "parent_success": str(parent_success.resolve()),
                "observed_at": now(),
            }
        failure = read_json(parent_failure)
        return _recover(plan, effective, run, parent_dir, failure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args), indent=2, sort_keys=True))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
