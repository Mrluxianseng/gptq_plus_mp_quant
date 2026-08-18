#!/usr/bin/env python3
"""Noise-aware LR selector for the deterministic-FA4 run15 V6 restart.

V6 intentionally does not adopt any numerical result from the retired
math-SDPA V4/V5 campaigns.  This wrapper binds the shared selection logic to
the immutable V6 plan/output tree and records the V6 entry point alongside
the selector implementation when the final 40-row decision is frozen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import campaign_v6_run15 as v6


v6._bootstrap()

from experiments.realq_fullmodel_retune_20260817 import selection as core  # noqa: E402


SELECTION_ID = "realq-fullmodel-two-branch-lr-selection-20260818-v6-run15"
SELECTION_PATH = v6.OUTPUT_ROOT / "selections.json"
core.SELECTION_ID = SELECTION_ID
core.SELECTION_PATH = SELECTION_PATH


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
            f"cannot freeze; {len(not_ready)} V6 groups have not passed all "
            f"gates: {not_ready}"
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
    code_paths = [
        Path(core.__file__).resolve(),
        Path(v6.__file__).resolve(),
        Path(__file__).resolve(),
    ]
    body: dict[str, Any] = {
        "selection_id": SELECTION_ID,
        "campaign_id": v6.CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "plan": {
            "path": str(v6.PLAN_PATH),
            "sha256": c._file_sha256(v6.PLAN_PATH),
        },
        "selection_code": [
            {"path": str(path), "sha256": c._file_sha256(path)}
            for path in code_paths
        ],
        "protocol": {
            "primary_metric": "wikitext2 Exact KL",
            "a_loss_ratio": 1.0,
            "attention_backend": "flash_attention_4==4.0.0b25",
            "fa4_deterministic_argument": True,
            "fisher_hessian_tf32_is_scoped": True,
            "retired_v4_v5_results_are_selection_inputs": False,
            "max_bracket_dex": core.MAX_BRACKET_DEX,
            "top_candidate_same_branch_replicates": (
                "2 when Exact KL repeats bit-identically; 3 when either of "
                "the first two executions differs"
            ),
            "high_side_worse_points": 2,
            "tie_break": (
                "lower LR when top-two gap <= observed same-branch repeat range"
            ),
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
        print(f"realq-fullmodel-selection-v6-run15: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
