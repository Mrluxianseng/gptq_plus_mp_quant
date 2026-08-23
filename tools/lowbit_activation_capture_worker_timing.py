#!/usr/bin/env python3
"""Capture a completed worker timing sidecar before a launcher can replace it.

This is a narrow recovery aid for Canoe environments that export ``RANK`` to
the outer ``torchrun`` launcher.  The formal timing hook then runs in both the
real worker and the launcher; the worker writes the valid interval first, and
the launcher can replace it at exit with a ``not_started`` sidecar.

The tool never edits the canonical sidecar.  It watches already-running
numerical worker PIDs and atomically preserves the first valid
``complete=true`` payload as ``phase_timing_rank0.worker_complete.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


CANONICAL_NAME = "phase_timing_rank0.json"
CAPTURED_NAME = "phase_timing_rank0.worker_complete.json"


class CaptureError(RuntimeError):
    """Raised when watcher input or captured evidence is invalid."""


def _read_cmdline(pid: int) -> list[str]:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise CaptureError(f"PID {pid} does not exist") from exc
    argv = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    if not argv:
        raise CaptureError(f"PID {pid} has an empty command line")
    return argv


def _option(argv: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1:
        raise CaptureError(
            f"command must contain exactly one {flag}; found {len(positions)}"
        )
    index = positions[0]
    if index + 1 >= len(argv):
        raise CaptureError(f"{flag} has no value")
    return argv[index + 1]


def _run_dir(argv: list[str]) -> Path:
    checkpoint = Path(_option(argv, "--save_qmodel_path"))
    if checkpoint.name != "model.pt" or checkpoint.parent.name != "checkpoint":
        raise CaptureError(
            "--save_qmodel_path must end in checkpoint/model.pt"
        )
    return checkpoint.parent.parent.resolve()


def _valid_complete_payload(raw: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        return None
    if payload.get("timing_status") != "complete":
        return None
    started = payload.get("started_monotonic_ns")
    ended = payload.get("ended_monotonic_ns")
    elapsed = payload.get("elapsed_seconds")
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(ended, bool)
        or not isinstance(ended, int)
        or ended <= started
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or float(elapsed) <= 0.0
    ):
        return None
    return payload


def _atomic_create(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        existing = path.read_bytes()
        if existing != raw:
            raise CaptureError(
                f"capture already exists with different bytes: {path}"
            )
        return
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _capture_once(run_dir: Path) -> dict[str, Any] | None:
    source = run_dir / CANONICAL_NAME
    try:
        raw = source.read_bytes()
    except OSError:
        # JFS can briefly expose a stale handle while the worker atomically
        # replaces the sidecar.  A later poll must retry the new inode.
        return None
    payload = _valid_complete_payload(raw)
    if payload is None:
        return None
    target = run_dir / CAPTURED_NAME
    _atomic_create(target, raw)
    return {
        "source": str(source),
        "captured": str(target),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "elapsed_seconds": float(payload["elapsed_seconds"]),
        "source_set_sha256": payload.get("source_set_sha256"),
        "spec_sha256": payload.get("spec_sha256"),
    }


def watch(pids: list[int], poll_seconds: float) -> dict[str, Any]:
    pending: dict[int, Path] = {}
    for pid in pids:
        if pid <= 0:
            raise CaptureError(f"PID must be positive: {pid}")
        argv = _read_cmdline(pid)
        pending[pid] = _run_dir(argv)

    captured: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    while pending:
        for pid, run_dir in tuple(pending.items()):
            item = _capture_once(run_dir)
            if item is not None:
                item["pid"] = pid
                captured.append(item)
                del pending[pid]
                print(
                    json.dumps(
                        {"event": "captured", **item},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            if not _pid_exists(pid):
                failures.append(
                    {
                        "pid": pid,
                        "run_dir": str(run_dir),
                        "error": "worker exited before a complete sidecar was captured",
                    }
                )
                del pending[pid]
        if pending:
            time.sleep(poll_seconds)
    return {
        "schema_version": 1,
        "captured": captured,
        "failures": failures,
        "ok": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pid",
        action="append",
        required=True,
        type=int,
        help="already-running numerical worker PID; repeat as needed",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.02,
        help="poll interval while awaiting the worker sidecar",
    )
    args = parser.parse_args()
    if not 0.005 <= args.poll_seconds <= 1.0:
        parser.error("--poll-seconds must be in [0.005, 1.0]")
    try:
        report = watch(args.pid, args.poll_seconds)
    except CaptureError as exc:
        print(f"capture error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
