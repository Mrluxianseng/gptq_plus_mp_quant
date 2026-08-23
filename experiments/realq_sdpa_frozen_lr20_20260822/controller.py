#!/usr/bin/env python3
"""Dynamic one-GPU worker for the SDPA cache/formal campaign.

Workers may start before every model cache exists.  They claim only formal
rows whose model-specific SDPA static cache has succeeded, then run the frozen
formal lifecycle while holding the cross-campaign physical-GPU lock.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal


POLL_SECONDS = 20
MAX_FORMAL_ATTEMPTS = 3
CONTROLLER_CLAIM = ".sdpa_controller_claim"
TERMINAL_FAILURE = "controller_terminal_failure.json"


def _task_order() -> list[tuple[str, str]]:
    # The inherited balanced order starts long models early and alternates
    # REALQ-F/REALQ-S, which keeps the two branches comparable in wall time.
    return formal._balanced_pairs()


def _formal_dir(branch: str, config: str) -> Path:
    return formal._formal_dir(branch, config)


def _attempt_count(branch: str, config: str) -> int:
    root = _formal_dir(branch, config)
    return len(list(root.glob("attempt[0-9][0-9][0-9]"))) if root.is_dir() else 0


def _cache_ready(config: str) -> bool:
    model = campaign._model_for_config(config)
    marker_path = campaign._cache_marker(model)
    if not marker_path.is_file():
        return False
    marker = c._read_json(marker_path)
    if (
        marker.get("status") != "succeeded"
        or marker.get("model") != model
        or marker.get("campaign_id") != campaign.CAMPAIGN_ID
        or marker.get("protocol_fingerprint")
        != c._read_json(campaign.PLAN_PATH)["protocol_fingerprint"]
    ):
        raise c.CampaignError(f"invalid SDPA cache marker: {marker_path}")
    return True


def _claim_one(hostname: str, physical_gpu: int) -> tuple[str, str, Path] | None:
    for branch, config in _task_order():
        root = _formal_dir(branch, config)
        if (root / "formal_success.json").is_file():
            continue
        if (root / TERMINAL_FAILURE).is_file():
            continue
        if not _cache_ready(config):
            continue
        root.mkdir(parents=True, exist_ok=True)
        claim = root / CONTROLLER_CLAIM
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        c._atomic_json(
            claim / "owner.json",
            {
                "campaign_id": campaign.CAMPAIGN_ID,
                "formal_id": formal.FORMAL_ID,
                "branch": branch,
                "config": config,
                "hostname": hostname,
                "physical_gpu": physical_gpu,
                "pid": os.getpid(),
                "claimed_at": c._utc_now(),
            },
        )
        return branch, config, claim
    return None


def _all_cache_producers_terminal() -> bool:
    for model in c.MODEL_SLUGS:
        if campaign._cache_marker(model).is_file():
            continue
        if campaign._cache_attempt_count(model) < 3:
            return False
    return True


def _campaign_state() -> dict[str, int]:
    counts = {"succeeded": 0, "claimed": 0, "failed": 0, "pending": 0}
    for branch, config in _task_order():
        root = _formal_dir(branch, config)
        if (root / "formal_success.json").is_file():
            counts["succeeded"] += 1
        elif (root / CONTROLLER_CLAIM).is_dir():
            counts["claimed"] += 1
        elif (root / TERMINAL_FAILURE).is_file():
            counts["failed"] += 1
        else:
            counts["pending"] += 1
    return counts


def _run_worker(args: argparse.Namespace) -> int:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise c.CampaignError("controller must run inside a Canoe debug pod")
    if not 0 <= args.physical_gpu <= 7:
        raise c.CampaignError("physical GPU must be in [0, 7]")
    plan = formal._load_plan()
    gpu_lock = campaign.LOCK_ROOT / hostname / f"gpu{args.physical_gpu}.lock"
    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state = _campaign_state()
        if state["succeeded"] == 40:
            return 0
        if state["failed"]:
            return 1
        claimed = _claim_one(hostname, args.physical_gpu)
        if claimed is None:
            if _all_cache_producers_terminal() and not state["claimed"]:
                return 1
            time.sleep(POLL_SECONDS)
            continue
        branch, config, claim = claimed
        try:
            with gpu_lock.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                status = formal._run_one(
                    plan, branch, config, str(args.physical_gpu)
                )
            if status and _attempt_count(branch, config) >= MAX_FORMAL_ATTEMPTS:
                c._atomic_json(
                    _formal_dir(branch, config) / TERMINAL_FAILURE,
                    {
                        "campaign_id": campaign.CAMPAIGN_ID,
                        "formal_id": formal.FORMAL_ID,
                        "branch": branch,
                        "config": config,
                        "attempts": _attempt_count(branch, config),
                        "hostname": hostname,
                        "physical_gpu": args.physical_gpu,
                        "failed_at": c._utc_now(),
                    },
                )
        except BaseException as exc:
            if _attempt_count(branch, config) >= MAX_FORMAL_ATTEMPTS:
                c._atomic_json(
                    _formal_dir(branch, config) / TERMINAL_FAILURE,
                    {
                        "campaign_id": campaign.CAMPAIGN_ID,
                        "formal_id": formal.FORMAL_ID,
                        "branch": branch,
                        "config": config,
                        "attempts": _attempt_count(branch, config),
                        "hostname": hostname,
                        "physical_gpu": args.physical_gpu,
                        "error_type": type(exc).__qualname__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "failed_at": c._utc_now(),
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
    payload: dict[str, Any] = {
        "campaign_id": campaign.CAMPAIGN_ID,
        "cache_producers": {
            model: (
                "succeeded"
                if campaign._cache_marker(model).is_file()
                else f"attempts={campaign._cache_attempt_count(model)}"
            )
            for model in c.MODEL_SLUGS
        },
        "formal": _campaign_state(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--physical-gpu", required=True, type=int)
    worker.set_defaults(handler=_run_worker)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
