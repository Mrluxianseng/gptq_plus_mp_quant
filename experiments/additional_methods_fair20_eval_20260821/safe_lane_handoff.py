#!/usr/bin/env python3
"""Release a stopped evaluation parent only after its child succeeded safely.

Persistent evaluators hold a physical-GPU lock across suites.  For a planned
lane handoff, the parent can be SIGSTOP'ed while its current ``run_suite``
child continues.  This helper requires the child to have published the suite
terminal and left CUDA, reconstructs the small parent-owned attempt receipt,
releases the claim, and terminates only that stopped parent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from experiments.additional_methods_fair20_eval_20260821 import common


class HandoffError(RuntimeError):
    pass


def _state(pid: int) -> str | None:
    path = Path(f"/proc/{pid}/stat")
    if not path.is_file():
        return None
    fields = path.read_text(encoding="utf-8").split()
    return fields[2] if len(fields) >= 3 else None


def _cmdline(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        return ""
    return path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")


def _cuda_pids() -> set[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    }


def _utc_seconds(value: str) -> float:
    return dt.datetime.fromisoformat(value).timestamp()


def execute(*, parent_pid: int, child_pid: int, eval_id: str, gpu: int) -> dict:
    hostname = socket.gethostname()
    expected = (
        "experiments.additional_methods_fair20_eval_20260821.worker "
        f"--physical-gpu {gpu}"
    )
    parent_state = _state(parent_pid)
    if parent_state != "T" or expected not in _cmdline(parent_pid):
        raise HandoffError(
            f"parent is not the expected stopped worker: state={parent_state!r}"
        )
    child_state = _state(child_pid)
    if child_state not in {None, "Z"}:
        raise HandoffError(f"evaluation child is still live: state={child_state!r}")
    if child_pid in _cuda_pids():
        raise HandoffError("evaluation child is still registered on CUDA")

    suite_dir = common.OUTPUT_ROOT / "evals" / eval_id
    success_path = suite_dir / "suite_success.json"
    if not success_path.is_file():
        raise HandoffError("suite_success.json has not been published")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if success.get("eval_id") != eval_id or not str(success.get("status", "")).startswith(
        "generation_succeeded"
    ):
        raise HandoffError("suite terminal marker is not a generation success")

    attempts = sorted((suite_dir / "attempts").glob("attempt[0-9][0-9][0-9]"))
    if not attempts:
        raise HandoffError("evaluation attempt directory is missing")
    attempt = attempts[-1]
    manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    result_path = attempt / "result.json"
    if manifest.get("status") != "running" or manifest.get("eval_id") != eval_id:
        raise HandoffError("attempt manifest is not the expected running attempt")
    if result_path.exists():
        raise HandoffError("parent already published the attempt result")

    claim = suite_dir / ".claim"
    owner_path = claim / "owner.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if owner.get("pid") != parent_pid or owner.get("eval_id") != eval_id:
        raise HandoffError("claim is not owned by the stopped parent")

    finished_at = str(success["finished_at"])
    common.atomic_json(
        result_path,
        {
            **manifest,
            "status": "succeeded",
            "returncode": 0,
            "wall_seconds": _utc_seconds(finished_at)
            - _utc_seconds(str(manifest["started_at"])),
            "wall_seconds_source": "suite_finished_at_minus_attempt_started_at",
            "finished_at": finished_at,
            "safe_lane_handoff": True,
        },
    )
    owner_path.unlink()
    claim.rmdir()
    os.kill(parent_pid, signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _state(parent_pid) not in {None, "Z"}:
        time.sleep(0.05)

    receipt = {
        "schema_version": 1,
        "status": "succeeded",
        "hostname": hostname,
        "physical_gpu": gpu,
        "eval_id": eval_id,
        "suite_success": str(success_path),
        "suite_success_sha256": common.sha256_file(success_path),
        "attempt_result": str(result_path),
        "stopped_parent_pid": parent_pid,
        "completed_child_pid": child_pid,
        "parent_state_after_sigkill": _state(parent_pid),
        "finished_at": common.now(),
    }
    output = common.OUTPUT_ROOT / "lane_handoffs" / hostname
    output.mkdir(parents=True, exist_ok=True)
    common.atomic_json(output / f"gpu{gpu}__{eval_id}.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--child-pid", type=int, required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    receipt = execute(
        parent_pid=args.parent_pid,
        child_pid=args.child_pid,
        eval_id=args.eval_id,
        gpu=args.physical_gpu,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
