#!/usr/bin/env python3
"""Noise-aware selector for V5 Q32 memory-equivalent retries.

The scientific curve consists of V4's zero endpoint plus V5 positive trials.
V4's failed first-positive attempt is also included as a charged failure, so
the global 20-launch cap cannot be reset by the execution-only derivative.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import campaign_v5_q32_memory as v5


v5._bootstrap()

from experiments.realq_fullmodel_retune_20260817 import selection as core  # noqa: E402


SELECTION_ID = "realq-fullmodel-two-branch-lr-selection-20260818-v5-q32-memory"
SELECTION_PATH = v5.OUTPUT_ROOT / "selections.json"
core.SELECTION_ID = SELECTION_ID
core.SELECTION_PATH = SELECTION_PATH
_V5_RESULT_ROWS = core._result_rows


def _v4_result(branch: str, config: str, lr: float) -> dict[str, Any]:
    trial = {
        "branch": branch,
        "config": config,
        "schedule": c._schedule_for(config),
        "lr": lr,
    }
    identity = c._trial_id(trial)
    path = v5.V4_ROOT / "trials" / identity / "result.json"
    result = c._read_json(path)
    if result.get("branch") != branch or result.get("config") != config:
        raise c.CampaignError(f"V4 evidence identity mismatch: {path}")
    if float(result.get("lr", -1)) != lr:
        raise c.CampaignError(f"V4 evidence LR mismatch: {path}")
    return {
        **result,
        "result_path": str(path),
        "result_sha256": c._file_sha256(path),
        "evidence_campaign": "V4",
    }


def _combined_result_rows(branch: str, config: str) -> list[dict[str, Any]]:
    rows = [_v4_result(branch, config, 0.0), _v4_result(branch, config, 1e-7)]
    rows.extend(_V5_RESULT_ROWS(branch, config))
    return rows


core._result_rows = _combined_result_rows


def _status(args: argparse.Namespace) -> int:
    payload = core.analyze_all()
    if args.output:
        c._atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _freeze(_: argparse.Namespace) -> int:
    plan = core._read_plan()
    analysis = core.analyze_all()
    not_ready = [
        f"{row['branch']}/{row['config']}"
        for row in analysis["rows"]
        if not row["ready"]
    ]
    if not_ready:
        raise c.CampaignError(
            f"cannot freeze; {len(not_ready)} Q32 groups are not ready: {not_ready}"
        )
    rows = [
        {
            "branch": row["branch"],
            "config": row["config"],
            "selected_lr": row["selected_lr"],
            "best": row["best"],
            "top_two": row["top_two"],
            "noise_floor_range": row["noise_floor_range"],
            "noise_tie": row["noise_tie"],
            "bracket": row["bracket"],
            "high_side_worse_lrs": row["high_side_worse_lrs"],
            "two_percent_plateau": row["two_percent_plateau"],
            "launches": row["launches"],
        }
        for row in analysis["rows"]
    ]
    code_paths = [Path(core.__file__).resolve(), Path(__file__).resolve()]
    body: dict[str, Any] = {
        "selection_id": SELECTION_ID,
        "campaign_id": v5.CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "plan": {"path": str(v5.PLAN_PATH), "sha256": c._file_sha256(v5.PLAN_PATH)},
        "selection_code": [
            {"path": str(path), "sha256": c._file_sha256(path)}
            for path in code_paths
        ],
        "protocol": {
            "primary_metric": "wikitext2 Exact KL",
            "a_loss_ratio": 1.0,
            "adopted_v4_zero_and_charged_failure": True,
            "max_total_launches_including_v4": 20,
            "max_bracket_dex": core.MAX_BRACKET_DEX,
            "top_candidate_same_branch_replicates": (
                "2 when bit-identical, otherwise 3"
            ),
            "high_side_worse_points": 2,
            "tie_break": "lower LR within observed repeat noise",
            "two_percent_plateau_is_selection_gate": False,
        },
        "rows": rows,
    }
    body["selection_fingerprint"] = c._canonical_sha256(body)
    body["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if SELECTION_PATH.exists():
        raise c.CampaignError(f"selection artifact already exists: {SELECTION_PATH}")
    c._atomic_json(SELECTION_PATH, body)
    print(SELECTION_PATH)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--output", type=Path)
    status.set_defaults(handler=_status)
    freeze = subparsers.add_parser("freeze")
    freeze.set_defaults(handler=_freeze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (c.CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"realq-fullmodel-selection-v5-q32-memory: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
