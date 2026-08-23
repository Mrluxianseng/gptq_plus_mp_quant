#!/usr/bin/env python3
"""Split the remaining audited TurboBOA Llama recoveries across free GPUs.

The original recovery queue reserves all four suites and evaluates them
serially.  This helper is only valid after that queue process is dead.  It
preserves each queue-owned claim, the corrected loader, checkpoint identity,
seeds, and canonical suite implementation; only the physical scheduling is
changed.  Every split child takes the shared physical-GPU lock and publishes
an immutable scheduling receipt.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from typing import Any, Mapping

from . import common
from .recover_turboboa_llama import ALLOWED_EVAL_IDS
from . import recover_turboboa_llama_queue as original_queue


SPLIT_IDS = ALLOWED_EVAL_IDS[1:]
SPLIT_ROOT = (
    common.OUTPUT_ROOT / "turboboa_llama_contract_recovery" / "parallel_split"
)
KIND = "turboboa_llama_contract_recovery_parallel_split"


class SplitRecoveryError(RuntimeError):
    """The serial-to-parallel recovery handoff failed a safety gate."""


def _process_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _claim_path(eval_id: str) -> Path:
    return common.OUTPUT_ROOT / "evals" / eval_id / ".claim"


def _validated_owner(
    owner: Mapping[str, Any],
    *,
    eval_id: str,
    hostname: str,
    expected_queue_pid: int,
) -> None:
    expected = {
        "eval_id": eval_id,
        "hostname": hostname,
        "pid": expected_queue_pid,
        "kind": "turboboa_llama_contract_recovery_queue",
    }
    mismatch = {
        key: (owner.get(key), value)
        for key, value in expected.items()
        if owner.get(key) != value
    }
    if mismatch:
        raise SplitRecoveryError(f"queue claim owner mismatch: {mismatch}")


def _validate_dead_queue(expected_queue_pid: int) -> None:
    if expected_queue_pid <= 1:
        raise SplitRecoveryError("expected queue PID must be greater than one")
    if _process_exists(expected_queue_pid):
        raise SplitRecoveryError(
            f"original recovery queue is still live: pid={expected_queue_pid}"
        )


def _validate_suite(eval_id: str) -> dict[str, Any]:
    path = common.OUTPUT_ROOT / "evals" / eval_id / "suite_success.json"
    value = common.read_object(path)
    if (
        value.get("eval_id") != eval_id
        or value.get("status")
        != "generation_succeeded_official_humaneval_pending"
        or value.get("contract_recovery", {}).get("kind")
        != "turboboa_llama_manifest_alias_only"
    ):
        raise SplitRecoveryError(f"invalid recovered suite terminal: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": common.sha256_file(path),
    }


def cleanup_completed_claim(
    *, eval_id: str, expected_queue_pid: int
) -> dict[str, Any]:
    if eval_id != ALLOWED_EVAL_IDS[0]:
        raise SplitRecoveryError("only the completed first serial suite is cleanable")
    hostname = socket.gethostname()
    _validate_dead_queue(expected_queue_pid)
    terminal = _validate_suite(eval_id)
    claim = _claim_path(eval_id)
    owner_path = claim / "owner.json"
    owner = common.read_object(owner_path)
    _validated_owner(
        owner,
        eval_id=eval_id,
        hostname=hostname,
        expected_queue_pid=expected_queue_pid,
    )
    audit = SPLIT_ROOT / "completed_claim_audit" / eval_id
    if audit.exists() or audit.is_symlink():
        raise SplitRecoveryError(f"completed-claim audit exists: {audit}")
    audit.parent.mkdir(parents=True, exist_ok=True)
    owner_sha256 = common.sha256_file(owner_path)
    os.replace(claim, audit)
    receipt = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": KIND,
        "action": "move_completed_serial_claim_to_audit",
        "eval_id": eval_id,
        "hostname": hostname,
        "expected_dead_queue_pid": expected_queue_pid,
        "finished_at": common.now(),
        "suite_terminal": terminal,
        "claim_audit": str(audit.resolve()),
        "owner_sha256": owner_sha256,
    }
    common.atomic_json(SPLIT_ROOT / f"{eval_id}.claim_cleanup.json", receipt)
    return receipt


def execute(
    *, eval_id: str, physical_gpu: int, expected_queue_pid: int
) -> dict[str, Any]:
    if eval_id not in SPLIT_IDS:
        raise SplitRecoveryError(f"suite is not eligible for split recovery: {eval_id}")
    if physical_gpu < 0 or physical_gpu > 7:
        raise SplitRecoveryError("physical GPU must be in [0, 7]")
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise SplitRecoveryError("split recovery must run on a Canoe pod")
    _validate_dead_queue(expected_queue_pid)
    suite = common.OUTPUT_ROOT / "evals" / eval_id / "suite_success.json"
    if suite.exists() or suite.is_symlink():
        raise SplitRecoveryError(f"suite terminal already exists: {suite}")
    claim = _claim_path(eval_id)
    owner_path = claim / "owner.json"
    owner = common.read_object(owner_path)
    _validated_owner(
        owner,
        eval_id=eval_id,
        hostname=hostname,
        expected_queue_pid=expected_queue_pid,
    )

    plans = common.load_quant_plans()
    python = str(plans["efficientqat"]["venv_python"])
    environment = original_queue._runtime_environment(physical_gpu)
    original_queue._validate_runtime(python, environment)
    lock_path = (
        original_queue.LOCK_ROOT / hostname / f"gpu{physical_gpu}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = common.now()
    started = time.monotonic()
    print(f"{started_at} waiting_for_physical_gpu_lock={lock_path}", flush=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _validate_dead_queue(expected_queue_pid)
        owner = common.read_object(owner_path)
        _validated_owner(
            owner,
            eval_id=eval_id,
            hostname=hostname,
            expected_queue_pid=expected_queue_pid,
        )
        reassigned = {
            **owner,
            "pid": os.getpid(),
            "physical_gpu": physical_gpu,
            "kind": "turboboa_llama_contract_recovery_queue",
            "previous_queue_owner": owner,
            "split_kind": KIND,
            "reassigned_at": common.now(),
        }
        common.atomic_json(owner_path, reassigned)
        command = [
            python,
            "-u",
            "-m",
            "experiments.additional_methods_fair20_eval_20260821.recover_turboboa_llama",
            "--eval-id",
            eval_id,
        ]
        print(
            f"{common.now()} split_recovery_start eval_id={eval_id} "
            f"gpu={physical_gpu}",
            flush=True,
        )
        completed = subprocess.run(
            command,
            cwd=common.REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            raise SplitRecoveryError(
                f"split recovery failed: eval_id={eval_id} "
                f"returncode={completed.returncode}"
            )
        terminal = _validate_suite(eval_id)
        original_queue._release(claim)

    source = Path(__file__).resolve()
    receipt = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": KIND,
        "eval_id": eval_id,
        "hostname": hostname,
        "physical_gpu": physical_gpu,
        "expected_dead_queue_pid": expected_queue_pid,
        "started_at": started_at,
        "finished_at": common.now(),
        "wall_seconds": time.monotonic() - started,
        "suite_terminal": terminal,
        "source": {
            "path": str(source),
            "sha256": common.sha256_file(source),
        },
        "numerical_recovery": {
            "module": (
                "experiments.additional_methods_fair20_eval_20260821."
                "recover_turboboa_llama"
            ),
            "command": command,
            "scheduling_only_change": True,
        },
    }
    common.atomic_json(SPLIT_ROOT / f"{eval_id}.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return receipt


def launch(
    *, eval_id: str, physical_gpu: int, expected_queue_pid: int
) -> dict[str, Any]:
    hostname = socket.gethostname()
    plans = common.load_quant_plans()
    python = str(plans["efficientqat"]["venv_python"])
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = SPLIT_ROOT / f"{hostname}_gpu{physical_gpu}__{eval_id}.log"
    launch_path = log_path.with_suffix(".launch.json")
    if log_path.exists() or launch_path.exists():
        raise SplitRecoveryError(f"split launch already exists: {log_path}")
    command = [
        python,
        "-u",
        "-m",
        "experiments.additional_methods_fair20_eval_20260821.split_turboboa_llama_recovery",
        "--eval-id",
        eval_id,
        "--physical-gpu",
        str(physical_gpu),
        "--expected-queue-pid",
        str(expected_queue_pid),
        "--child",
    ]
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=common.REPO_ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {
        "schema_version": 1,
        "status": "launched",
        "kind": KIND,
        "eval_id": eval_id,
        "hostname": hostname,
        "physical_gpu": physical_gpu,
        "expected_dead_queue_pid": expected_queue_pid,
        "pid": process.pid,
        "launched_at": common.now(),
        "command": command,
        "log": str(log_path),
    }
    common.atomic_json(launch_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-id", choices=ALLOWED_EVAL_IDS)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--expected-queue-pid", type=int, required=True)
    parser.add_argument("--cleanup-completed-claim", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.cleanup_completed_claim:
            result = cleanup_completed_claim(
                eval_id=str(args.eval_id),
                expected_queue_pid=args.expected_queue_pid,
            )
        else:
            if args.eval_id is None or args.physical_gpu is None:
                raise SplitRecoveryError(
                    "--eval-id and --physical-gpu are required for recovery"
                )
            if args.child:
                result = execute(
                    eval_id=args.eval_id,
                    physical_gpu=args.physical_gpu,
                    expected_queue_pid=args.expected_queue_pid,
                )
            else:
                result = launch(
                    eval_id=args.eval_id,
                    physical_gpu=args.physical_gpu,
                    expected_queue_pid=args.expected_queue_pid,
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
