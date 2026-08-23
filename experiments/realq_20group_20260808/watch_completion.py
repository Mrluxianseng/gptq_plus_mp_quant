#!/usr/bin/env python3
"""Wait for every atomic campaign marker, then run the final audit once."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.realq_20group_20260808 import audit
from experiments.realq_20group_20260808 import campaign as c


def completion_counts(root: Path) -> dict[str, int]:
    return {
        "formal": sum(
            (c.run_root(root, run) / "formal_success.json").is_file()
            for run in c.RUNS
        ),
        "generation": sum(
            (
                c.run_root(root, run)
                / "reasoning"
                / task
                / "generation_success.json"
            ).is_file()
            for run in c.RUNS
            for task in c.TASKS
        ),
        "official": sum(
            (
                c.run_root(root, run)
                / "reasoning"
                / "humaneval_plus"
                / "official_eval"
                / "official_success.json"
            ).is_file()
            for run in c.RUNS
        ),
    }


def expected_counts() -> dict[str, int]:
    return {
        "formal": len(c.RUNS),
        "generation": len(c.RUNS) * len(c.TASKS),
        "official": len(c.RUNS),
    }


def wait_and_audit(root: Path, poll_seconds: float) -> dict[str, Any]:
    expected = expected_counts()
    previous: dict[str, int] | None = None
    while True:
        counts = completion_counts(root)
        if counts != previous:
            print(
                json.dumps(
                    {
                        "observed_at": c.now(),
                        "counts": counts,
                        "expected": expected,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            previous = counts
        if counts == expected:
            break
        time.sleep(poll_seconds)

    payload = audit.audit_campaign(root)
    print(
        json.dumps(
            {
                "audited_at": payload["audited_at"],
                "status": payload["status"],
                "counts": payload["counts"],
                "audit_fingerprint": payload["audit_fingerprint"],
                "output": str(root / "final_audit.json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", default=str(c.DEFAULT_ROOT))
    result.add_argument("--poll-seconds", type=float, default=30.0)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.poll_seconds <= 0:
        raise c.CampaignError("poll interval must be positive")
    root = c.output_root(args.root)
    wait_and_audit(root, args.poll_seconds)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except c.CampaignError as exc:
        print(f"completion watcher error: {exc}", flush=True)
        raise SystemExit(2)
