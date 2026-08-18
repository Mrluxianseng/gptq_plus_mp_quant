#!/usr/bin/env python3
"""Freeze the 28 V4 curves used by the merged 40-row formal campaign.

Qwen3-32B uses the V5 memory-equivalent execution campaign, while Qwen3-4B
W4A16/W4A4KV4 uses the independently tuned ``a_loss_ratio=1`` campaign.  This
selector therefore freezes only the disjoint V4 remainder and cannot silently
adopt rows that have a newer authoritative source.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import selection_v4 as parent


SELECTION_ID = "realq-fullmodel-v4-subset28-selection-20260818-v1"
SELECTION_PATH = parent.v4.OUTPUT_ROOT / "selections_subset28.json"
Q4_EXTERNAL_CONFIGS = {"qwen3-4b_w4a16", "qwen3-4b_w4a4kv4"}


def _expected_keys() -> set[tuple[str, str]]:
    keys = {
        (branch, config)
        for branch in c.BRANCH_VALUES
        for config in c.CONFIG_IDS
        if not config.startswith("qwen3-32b_") and config not in Q4_EXTERNAL_CONFIGS
    }
    if len(keys) != 28:
        raise c.CampaignError(f"V4 subset must contain 28 rows, got {len(keys)}")
    return keys


def _subset_analysis() -> dict[str, Any]:
    analysis = parent.core.analyze_all()
    expected = _expected_keys()
    rows = [
        row
        for row in analysis["rows"]
        if (str(row["branch"]), str(row["config"])) in expected
    ]
    observed = {(str(row["branch"]), str(row["config"])) for row in rows}
    if observed != expected or len(rows) != 28:
        raise c.CampaignError("V4 subset analysis coverage mismatch")
    return {
        "selection_id": SELECTION_ID,
        "campaign_id": parent.v4.CAMPAIGN_ID,
        "protocol_fingerprint": analysis["protocol_fingerprint"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "counts": {
            "groups": len(rows),
            "ready": sum(bool(row["ready"]) for row in rows),
            "launches": sum(int(row["launches"]) for row in rows),
            "failures": sum(len(row["failures"]) for row in rows),
        },
        "rows": rows,
    }


def _status(args: argparse.Namespace) -> int:
    payload = _subset_analysis()
    if args.output:
        c._atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _selection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
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


def _build_selection() -> dict[str, Any]:
    plan = parent.core._read_plan()
    analysis = _subset_analysis()
    pending = [
        f"{row['branch']}/{row['config']}"
        for row in analysis["rows"]
        if not row["ready"]
    ]
    if pending:
        raise c.CampaignError(
            f"cannot freeze V4 subset; {len(pending)} groups are not ready: {pending}"
        )
    code_paths = [
        Path(parent.core.__file__).resolve(),
        Path(parent.__file__).resolve(),
        Path(__file__).resolve(),
    ]
    body: dict[str, Any] = {
        "selection_id": SELECTION_ID,
        "campaign_id": parent.v4.CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "plan": {
            "path": str(parent.v4.PLAN_PATH),
            "sha256": c._file_sha256(parent.v4.PLAN_PATH),
        },
        "selection_code": [
            {"path": str(path), "sha256": c._file_sha256(path)}
            for path in code_paths
        ],
        "coverage": {
            "rows": 28,
            "excluded_qwen3_32b_rows": 8,
            "excluded_qwen3_4b_w4_kv4_rows": 4,
        },
        "protocol": {
            "primary_metric": "WikiText2 Exact KL",
            "a_loss_ratio": 1.0,
            "max_bracket_dex": parent.core.MAX_BRACKET_DEX,
            "top_candidate_same_branch_replicates": (
                "2 when bit-identical, otherwise 3"
            ),
            "high_side_worse_points": 2,
            "tie_break": "lower LR within observed repeat noise",
            "two_percent_plateau_is_selection_gate": False,
        },
        "rows": [_selection_row(row) for row in analysis["rows"]],
    }
    body["selection_fingerprint"] = c._canonical_sha256(body)
    return body


def _freeze(_: argparse.Namespace) -> int:
    value = _build_selection()
    if SELECTION_PATH.exists():
        current = c._read_json(SELECTION_PATH)
        comparable = dict(current)
        comparable.pop("created_at", None)
        if comparable != value:
            raise c.CampaignError(f"existing V4 subset selection differs: {SELECTION_PATH}")
    else:
        c._atomic_json(
            SELECTION_PATH,
            {**value, "created_at": dt.datetime.now(dt.timezone.utc).isoformat()},
        )
    print(SELECTION_PATH)
    return 0


def load_selection() -> dict[str, Any]:
    value = c._read_json(SELECTION_PATH)
    if value.get("selection_id") != SELECTION_ID:
        raise c.CampaignError("wrong V4 subset selection id")
    comparable = dict(value)
    fingerprint = comparable.pop("selection_fingerprint", None)
    comparable.pop("created_at", None)
    if c._canonical_sha256(comparable) != fingerprint:
        raise c.CampaignError("V4 subset selection fingerprint mismatch")
    plan_ref = value.get("plan")
    if not isinstance(plan_ref, dict) or plan_ref.get("sha256") != c._file_sha256(
        parent.v4.PLAN_PATH
    ):
        raise c.CampaignError("V4 subset selection plan mismatch")
    for item in value.get("selection_code", []):
        path = Path(str(item.get("path", "")))
        if not path.is_file() or c._file_sha256(path) != item.get("sha256"):
            raise c.CampaignError(f"V4 subset selection code changed: {path}")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise c.CampaignError("V4 subset selection rows are invalid")
    observed = {(str(row.get("branch")), str(row.get("config"))) for row in rows}
    if observed != _expected_keys() or len(rows) != 28:
        raise c.CampaignError("V4 subset selection coverage mismatch")
    return value


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
        print(f"realq-v4-subset28-selection: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
