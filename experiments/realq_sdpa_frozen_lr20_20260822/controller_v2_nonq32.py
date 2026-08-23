#!/usr/bin/env python3
"""Continue the 32 non-Q32 V2b rows after Q32 capacity failures."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from experiments.realq_sdpa_frozen_lr20_20260822 import controller as core
from experiments.realq_sdpa_frozen_lr20_20260822 import controller_v2


EXPECTED_ROWS = 32


def _activate() -> None:
    controller_v2._activate()
    core._task_order = _task_order


def _task_order() -> list[tuple[str, str]]:
    return [
        (branch, config)
        for branch, config in core.formal._balanced_pairs()
        if not config.startswith("qwen3-32b_")
    ]


def _state() -> dict[str, int]:
    counts = {"succeeded": 0, "claimed": 0, "failed": 0, "pending": 0}
    for branch, config in _task_order():
        root = core._formal_dir(branch, config)
        if (root / "formal_success.json").is_file():
            counts["succeeded"] += 1
        elif (root / core.CONTROLLER_CLAIM).is_dir():
            counts["claimed"] += 1
        elif (root / core.TERMINAL_FAILURE).is_file():
            counts["failed"] += 1
        else:
            counts["pending"] += 1
    if sum(counts.values()) != EXPECTED_ROWS:
        raise core.c.CampaignError("non-Q32 controller matrix is not 32 rows")
    return counts


def _run_worker(args: argparse.Namespace) -> int:
    _activate()
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise core.c.CampaignError("controller must run inside a Canoe debug pod")
    if not 0 <= args.physical_gpu <= 7:
        raise core.c.CampaignError("physical GPU must be in [0, 7]")
    plan = core.formal._load_plan()
    gpu_lock = core.campaign.LOCK_ROOT / hostname / f"gpu{args.physical_gpu}.lock"
    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state = _state()
        if state["succeeded"] == EXPECTED_ROWS:
            return 0
        if state["failed"]:
            return 1
        claimed = core._claim_one(hostname, args.physical_gpu)
        if claimed is None:
            if not state["claimed"] and not state["pending"]:
                return 0
            time.sleep(core.POLL_SECONDS)
            continue
        branch, config, claim = claimed
        try:
            with gpu_lock.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                status = core.formal._run_one(
                    plan, branch, config, str(args.physical_gpu)
                )
            if status and core._attempt_count(branch, config) >= core.MAX_FORMAL_ATTEMPTS:
                core.c._atomic_json(
                    core._formal_dir(branch, config) / core.TERMINAL_FAILURE,
                    {
                        "campaign_id": core.campaign.CAMPAIGN_ID,
                        "formal_id": core.formal.FORMAL_ID,
                        "branch": branch,
                        "config": config,
                        "attempts": core._attempt_count(branch, config),
                        "hostname": hostname,
                        "physical_gpu": args.physical_gpu,
                        "failed_at": core.c._utc_now(),
                    },
                )
        except BaseException as exc:
            if core._attempt_count(branch, config) >= core.MAX_FORMAL_ATTEMPTS:
                core.c._atomic_json(
                    core._formal_dir(branch, config) / core.TERMINAL_FAILURE,
                    {
                        "campaign_id": core.campaign.CAMPAIGN_ID,
                        "formal_id": core.formal.FORMAL_ID,
                        "branch": branch,
                        "config": config,
                        "attempts": core._attempt_count(branch, config),
                        "hostname": hostname,
                        "physical_gpu": args.physical_gpu,
                        "error_type": type(exc).__qualname__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "failed_at": core.c._utc_now(),
                    },
                )
            else:
                print(
                    f"retryable failure {branch}/{config}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            shutil.rmtree(claim, ignore_errors=True)


def _status(_: argparse.Namespace) -> int:
    _activate()
    print(json.dumps({"scope": "non-qwen3-32b", "formal": _state()}, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--physical-gpu", required=True, type=int)
    worker.set_defaults(handler=_run_worker)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
