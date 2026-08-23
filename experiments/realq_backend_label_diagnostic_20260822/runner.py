#!/usr/bin/env python3
"""Run paired REAL-Q Stage-0 FA4/SDPA label captures on idle GPUs."""

from __future__ import annotations

import argparse
from array import array
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

from experiments.realq_allopts_loss_ablation_20260821 import (
    cache_backend_diagnostic as source_diagnostic,
)
from experiments.realq_fullmodel_retune_20260817 import campaign as base


DATA_ROOT = REPO_ROOT.parent / "experiment_data"
OUTPUT_ROOT = DATA_ROOT / "realq_backend_label_diagnostic_20260822_v1"
PLAN_PATH = OUTPUT_ROOT / "plan.json"
LOCK_ROOT = DATA_ROOT / "_fair20_physical_gpu_locks_20260821"
SOURCE_RECEIPT_SHA256 = (
    "2405c6c6fa339e27e3b5d02f26564481c96c8f36e58d3231210ed533d047d2aa"
)
HOST = "j-zogxxxduju-master-0"
ARMS = {
    "sdpa": {"backend": "sdpa", "physical_gpu": 0},
    "fa4": {"backend": "flash_attention_4", "physical_gpu": 2},
}
SOURCE_FILES = (
    "experiments/realq_backend_label_diagnostic_20260822/capture_driver.py",
    "experiments/realq_backend_label_diagnostic_20260822/runner.py",
    "experiments/realq_allopts_loss_ablation_20260821/cache_backend_diagnostic.py",
    "realq/ptq.py",
    "realq/attention.py",
    "realq/precompute/cache.py",
    "realq/precompute/static_e2e.py",
    "realq/precompute/labels.py",
    "realq/precompute/hooks.py",
    "utils/reproducibility.py",
)


class LabelDiagnosticError(RuntimeError):
    """A paired label-diagnostic safety gate failed."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LabelDiagnosticError(f"JSON root is not an object: {path}")
    return value


def _flags(command: Sequence[str]) -> dict[str, str]:
    return source_diagnostic._flags(command)


def _set(command: list[str], flag: str, value: str) -> None:
    source_diagnostic._set(command, flag, value)


def _command(arm: str) -> list[str]:
    if arm not in ARMS:
        raise LabelDiagnosticError(f"unknown arm: {arm}")
    _receipt, command = source_diagnostic._source_command()
    if command[1:3] != ["-m", "realq.ptq"]:
        raise LabelDiagnosticError("source command entry point changed")
    command[2] = (
        "experiments.realq_backend_label_diagnostic_20260822.capture_driver"
    )
    stage = OUTPUT_ROOT / arm
    overrides = {
        "--attention_backend": str(ARMS[arm]["backend"]),
        # Force a real Stage-0 execution but avoid writing another 5.2 GB cache.
        "--static_cache_path": "",
        "--cache_dir": str(stage / "runtime"),
        "--output_dir": str(stage / "realq_output"),
        "--exp": f"realq_backend_label_capture_{arm}",
        "--require_static_cache_hit": "false",
    }
    for flag, value in overrides.items():
        _set(command, flag, value)
    values = _flags(command)
    if (
        values["--grad_hessian_topk"] != "-1"
        or values["--global_loss_bsz"] != "4"
        or values["--nsamples"] != "256"
        or values["--seq_len"] != "2048"
        or values["--rotate"] != "true"
        or values["--exit_after_precompute"] != "true"
    ):
        raise LabelDiagnosticError("source scientific contract drifted")
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
    if base._file_sha256(source_diagnostic.SOURCE_RECEIPT) != SOURCE_RECEIPT_SHA256:
        raise LabelDiagnosticError("source producer receipt SHA256 changed")
    source_receipt, source_command = source_diagnostic._source_command()
    body = {
        "schema_version": 1,
        "experiment_id": "realq-backend-label-diagnostic-20260822-v1",
        "hostname": HOST,
        "source_producer_receipt": {
            "path": str(source_diagnostic.SOURCE_RECEIPT),
            "sha256": SOURCE_RECEIPT_SHA256,
            "command_sha256": source_receipt["command_sha256"],
        },
        "source_command_sha256": base._canonical_sha256(source_command),
        "source_snapshot": _source_snapshot(),
        "arms": {
            arm: {**definition, "command": _command(arm)}
            for arm, definition in ARMS.items()
        },
        "paired_contract": {
            "only_numerical_difference": "attention_backend",
            "model": "Qwen3-4B",
            "calibration": "same frozen WikiText2 256x2048 token artifact",
            "seeds": {"calibration": 1, "rotation": 0},
            "global_loss_bsz": 4,
            "captured_tensor": "production deterministic categorical labels",
            "capture_changes_loss": False,
        },
    }
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def init_plan() -> dict[str, Any]:
    plan = _build_plan()
    if PLAN_PATH.is_file():
        if _read(PLAN_PATH) != plan:
            raise LabelDiagnosticError("existing plan drifted")
        return plan
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base._atomic_json(PLAN_PATH, plan)
    return plan


def _verify_plan() -> dict[str, Any]:
    plan = _read(PLAN_PATH)
    stable = dict(plan)
    fingerprint = stable.pop("plan_fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise LabelDiagnosticError("plan fingerprint mismatch")
    if plan != _build_plan():
        raise LabelDiagnosticError("plan inputs or source changed")
    return plan


def _environment(gpu: int, arm: str, capture_root: Path) -> dict[str, str]:
    environment = source_diagnostic._environment(gpu)
    environment.update(
        REALQ_LABEL_CAPTURE_ROOT=str(capture_root),
        REALQ_LABEL_CAPTURE_BACKEND=str(ARMS[arm]["backend"]),
    )
    return environment


def _gpu_compute_pids(gpu: int) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        raise LabelDiagnosticError(result.stderr.strip())
    return [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]


def execute(arm: str, gpu: int) -> dict[str, Any]:
    if socket.gethostname() != HOST:
        raise LabelDiagnosticError("diagnostic child is on the wrong host")
    if arm not in ARMS or gpu != ARMS[arm]["physical_gpu"]:
        raise LabelDiagnosticError("arm/GPU mapping changed")
    plan = _verify_plan()
    stage = OUTPUT_ROOT / arm
    if stage.exists() or stage.is_symlink():
        raise LabelDiagnosticError(f"stage is not fresh: {stage}")
    stage.mkdir()
    command = list(map(str, plan["arms"][arm]["command"]))
    manifest = {
        "schema_version": 1,
        "status": "running",
        "arm": arm,
        "backend": ARMS[arm]["backend"],
        "hostname": HOST,
        "physical_gpu": gpu,
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "command_sha256": base._canonical_sha256(command),
        "started_at": base._utc_now(),
    }
    base._atomic_json(stage / "manifest.json", manifest)
    capture_root = stage / "capture"
    log_path = stage / "execution.log"
    lock_path = LOCK_ROOT / HOST / f"gpu{gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _gpu_compute_pids(gpu):
            raise LabelDiagnosticError(f"GPU {gpu} has an untracked process")
        with log_path.open("xb") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=_environment(gpu, arm, capture_root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    summary_path = capture_root / "capture_summary.json"
    result = {
        **manifest,
        "status": "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": base._utc_now(),
        "log": str(log_path),
        "log_sha256": base._file_sha256(log_path),
    }
    if completed.returncode == 0 and summary_path.is_file():
        summary = _read(summary_path)
        labels_path = Path(summary["labels"])
        if (
            summary.get("status") != "completed"
            or summary.get("backend") != ARMS[arm]["backend"]
            or summary.get("total_labels") != 256 * 2048
            or labels_path.stat().st_size != 256 * 2048 * 8
            or base._file_sha256(labels_path) != summary.get("labels_sha256")
        ):
            raise LabelDiagnosticError("capture terminal validation failed")
        result.update(
            status="succeeded",
            capture_summary=str(summary_path),
            capture_summary_sha256=base._file_sha256(summary_path),
            labels=str(labels_path),
            labels_sha256=summary["labels_sha256"],
        )
    base._atomic_json(stage / "result.json", result)
    if result["status"] != "succeeded":
        raise LabelDiagnosticError(f"label capture failed: {arm}")
    return result


def launch_arm(arm: str) -> dict[str, Any]:
    if socket.gethostname() != HOST:
        raise LabelDiagnosticError("diagnostic must be launched on node1")
    if arm not in ARMS:
        raise LabelDiagnosticError(f"unknown arm: {arm}")
    plan = _verify_plan()
    definition = ARMS[arm]
    stage = OUTPUT_ROOT / arm
    launch_path = OUTPUT_ROOT / f"launch_{arm}.json"
    if stage.exists() or launch_path.exists():
        raise LabelDiagnosticError(f"arm was already claimed: {arm}")
    command = [
        str(plan["arms"][arm]["command"][0]),
        "-u",
        "-m",
        "experiments.realq_backend_label_diagnostic_20260822.runner",
        "--child",
        "--arm",
        arm,
        "--physical-gpu",
        str(definition["physical_gpu"]),
    ]
    log_path = OUTPUT_ROOT / f"launcher_{arm}.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=source_diagnostic._environment(definition["physical_gpu"]),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch = {
        "schema_version": 1,
        "status": "launched",
        "arm": arm,
        "pid": process.pid,
        "physical_gpu": definition["physical_gpu"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "log": str(log_path),
        "launched_at": base._utc_now(),
    }
    base._atomic_json(launch_path, launch)
    return launch


def launch_all() -> dict[str, Any]:
    return {
        "status": "launched",
        "arms": {arm: launch_arm(arm) for arm in ARMS},
    }


def _load_i64(path: Path) -> array:
    values = array("q")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    return values


def _compare_values(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    if len(left) != len(right) or len(left) != 256 * 2048:
        raise LabelDiagnosticError("captured label lengths differ")
    per_sample = []
    first_mismatches = []
    total = 0
    for sample in range(256):
        start = sample * 2048
        count = 0
        for token in range(2048):
            index = start + token
            if left[index] != right[index]:
                count += 1
                total += 1
                if len(first_mismatches) < 64:
                    first_mismatches.append(
                        {
                            "sample": sample,
                            "token": token,
                            "sdpa": int(left[index]),
                            "fa4": int(right[index]),
                        }
                    )
        per_sample.append(count)
    changed = [value for value in per_sample if value]
    return {
        "total_labels": len(left),
        "unequal_labels": total,
        "unequal_fraction": total / len(left),
        "samples_with_any_change": len(changed),
        "sample_change_fraction": len(changed) / 256,
        "per_sample_unequal_min": min(per_sample),
        "per_sample_unequal_max": max(per_sample),
        "per_sample_unequal_mean": sum(per_sample) / 256,
        "per_sample_unequal_counts": per_sample,
        "first_mismatches": first_mismatches,
    }


def compare() -> dict[str, Any]:
    plan = _verify_plan()
    results = {arm: _read(OUTPUT_ROOT / arm / "result.json") for arm in ARMS}
    summaries = {
        arm: _read(Path(result["capture_summary"]))
        for arm, result in results.items()
    }
    for arm, result in results.items():
        if (
            result.get("status") != "succeeded"
            or result.get("plan_fingerprint") != plan["plan_fingerprint"]
            or base._file_sha256(result["capture_summary"])
            != result.get("capture_summary_sha256")
            or summaries[arm].get("records") != summaries["sdpa"].get("records")
        ):
            raise LabelDiagnosticError(f"unbound capture result: {arm}")
    metrics = _compare_values(
        _load_i64(Path(results["sdpa"]["labels"])),
        _load_i64(Path(results["fa4"]["labels"])),
    )
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "plan_fingerprint": plan["plan_fingerprint"],
        "scientific_difference": "attention_backend only",
        "arms": {
            arm: {
                "result": str(OUTPUT_ROOT / arm / "result.json"),
                "result_sha256": base._file_sha256(
                    OUTPUT_ROOT / arm / "result.json"
                ),
                "labels_sha256": results[arm]["labels_sha256"],
                "elapsed_seconds": results[arm]["elapsed_seconds"],
            }
            for arm in ARMS
        },
        "metrics": metrics,
        "finished_at": base._utc_now(),
    }
    base._atomic_json(OUTPUT_ROOT / "comparison.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-plan", action="store_true")
    parser.add_argument("--launch-all", action="store_true")
    parser.add_argument("--launch-arm", choices=tuple(ARMS))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    try:
        modes = sum(
            (
                args.init_plan,
                args.launch_all,
                args.launch_arm is not None,
                args.compare,
                args.child,
            )
        )
        if modes != 1:
            parser.error("select exactly one execution mode")
        if args.init_plan:
            result = init_plan()
        elif args.launch_all:
            result = launch_all()
        elif args.launch_arm is not None:
            result = launch_arm(args.launch_arm)
        elif args.compare:
            result = compare()
        else:
            if args.arm is None or args.physical_gpu is None:
                parser.error("--child requires --arm and --physical-gpu")
            result = execute(args.arm, args.physical_gpu)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
