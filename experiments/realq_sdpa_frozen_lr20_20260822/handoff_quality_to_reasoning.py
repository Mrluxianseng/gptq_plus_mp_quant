#!/usr/bin/env python3
"""Fail-closed handoff from an eager quality lane to eager reasoning."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from typing import Any, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import (
    eager_quality_nonq32 as quality,
)
from experiments.realq_sdpa_frozen_lr20_20260822 import (
    eager_reasoning_nonq32 as reasoning,
)


HANDOFF_ID = "realq-sdpa-eager-quality-to-reasoning-handoff-20260823-v1"
POLL_SECONDS = 20


class HandoffError(RuntimeError):
    """The predecessor or terminal quality contract is invalid."""


def _cmdline(pid: int) -> str | None:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        return path.read_bytes().replace(b"\0", b" ").decode(errors="strict")
    except FileNotFoundError:
        return None


def _receipt_path(hostname: str, gpu: int) -> Path:
    return reasoning.OUTPUT_ROOT / "handoffs" / f"{hostname}_gpu{gpu}.json"


def _atomic_receipt(path: Path, value: dict[str, Any]) -> None:
    value["receipt_fingerprint"] = base._canonical_sha256(value)
    base._atomic_json(path, value)


def _run(args: argparse.Namespace) -> int:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise HandoffError("handoff must run in a debug pod")
    if not 0 <= args.physical_gpu <= 7:
        raise HandoffError("physical GPU must be in [0, 7]")
    quality_plan = quality._load_plan()
    reasoning_plan = reasoning._load_plan()
    path = _receipt_path(hostname, args.physical_gpu)
    if path.exists() or path.is_symlink():
        raise HandoffError(f"handoff receipt already exists: {path}")
    command = _cmdline(args.predecessor_pid)
    expected = (
        "experiments.realq_sdpa_frozen_lr20_20260822.eager_quality_nonq32 "
        f"run-worker --physical-gpu {args.physical_gpu}"
    )
    if command is None or expected not in command:
        raise HandoffError("predecessor PID is not the expected eager quality lane")
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        while _cmdline(args.predecessor_pid) is not None:
            time.sleep(POLL_SECONDS)
        state = quality._state()
        if state != {"succeeded": 31, "claimed": 0, "failed": 0, "pending": 0}:
            raise HandoffError(f"quality did not terminate successfully: {state}")
        quality._audit(argparse.Namespace(require_complete=True))
        receipt = {
            "schema_version": 1,
            "status": "launching_reasoning",
            "handoff_id": HANDOFF_ID,
            "hostname": hostname,
            "physical_gpu": args.physical_gpu,
            "predecessor_pid": args.predecessor_pid,
            "predecessor_command": command,
            "quality_plan_fingerprint": quality_plan["plan_fingerprint"],
            "reasoning_plan_fingerprint": reasoning_plan["plan_fingerprint"],
            "quality_terminal_state": state,
            "started_at": started_at,
            "reasoning_started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _atomic_receipt(path, receipt)
        return reasoning.main(
            ["run-worker", "--physical-gpu", str(args.physical_gpu)]
        )
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "handoff_id": HANDOFF_ID,
            "hostname": hostname,
            "physical_gpu": args.physical_gpu,
            "predecessor_pid": args.predecessor_pid,
            "predecessor_command": command,
            "quality_plan_fingerprint": quality_plan["plan_fingerprint"],
            "reasoning_plan_fingerprint": reasoning_plan["plan_fingerprint"],
            "started_at": started_at,
            "failed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error_type": type(exc).__qualname__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_receipt(path.with_suffix(".failure.json"), failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--predecessor-pid", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(_run(args))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
