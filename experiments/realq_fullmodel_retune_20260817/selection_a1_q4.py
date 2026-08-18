#!/usr/bin/env python3
"""Freeze the four Qwen3-4B ``a_loss_ratio=1`` tuning curves.

The original clip-ablation selector covers both 1.0 and 0.95.  The ablation
was intentionally converged to 1.0 after the independently tuned evidence
showed mixed wins, so the stopped 0.95 curves must not block the four 1.0
curves from entering the merged 40-row formal campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import aloss_clip_ablation as a
from experiments.realq_fullmodel_retune_20260817 import campaign as c


SELECTION_ID = "realq-qwen3-4b-aloss-one-selection-20260818-v1"
SELECTION_PATH = a.OUTPUT_ROOT / "selections_a1_only.json"
RATIO = 1.0


def _selected_analysis() -> dict[str, Any]:
    analysis = a._analysis()
    rows = [
        row for row in analysis["rows"] if float(row["a_loss_ratio"]) == RATIO
    ]
    expected = {
        a._group_id(branch, config, RATIO)
        for branch in c.BRANCH_VALUES
        for config in a.SETTINGS
    }
    observed = {str(row["group"]) for row in rows}
    if len(rows) != 4 or observed != expected:
        raise a.AblationError("a=1 selector must cover exactly four Qwen3-4B groups")
    return {
        "selection_id": SELECTION_ID,
        "ablation_id": a.ABLATION_ID,
        "plan_fingerprint": analysis["plan_fingerprint"],
        "generated_at": a._utc_now(),
        "counts": {
            "groups": len(rows),
            "ready": sum(bool(row["ready"]) for row in rows),
            "launches": sum(int(row["launches"]) for row in rows),
            "failures": sum(len(row["failures"]) for row in rows),
        },
        "rows": rows,
    }


def _status(args: argparse.Namespace) -> int:
    value = _selected_analysis()
    if args.output:
        c._atomic_json(args.output.expanduser().resolve(), value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _selection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "group": row["group"],
        "branch": row["branch"],
        "config": row["config"],
        "a_loss_ratio": row["a_loss_ratio"],
        "selected_lr": row["selected_lr"],
        "best": row["best"],
        "top_two": row["top_two"],
        "bracket": row["bracket"],
        "upper_worse_lrs": row["upper_worse_lrs"],
        "launches": row["launches"],
    }


def _build_selection() -> dict[str, Any]:
    plan = a._load_plan()
    analysis = _selected_analysis()
    pending = [row["group"] for row in analysis["rows"] if not row["ready"]]
    if pending:
        raise a.AblationError(f"cannot freeze a=1 selection; groups not ready: {pending}")
    code_paths = [Path(a.__file__).resolve(), Path(__file__).resolve()]
    body: dict[str, Any] = {
        "selection_id": SELECTION_ID,
        "ablation_id": a.ABLATION_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "plan": {
            "path": str(a.PLAN_PATH),
            "sha256": c._file_sha256(a.PLAN_PATH),
        },
        "selection_code": [
            {"path": str(path), "sha256": c._file_sha256(path)}
            for path in code_paths
        ],
        "decision": {
            "a_loss_ratio": RATIO,
            "rule": "use 1.0 when independently tuned 1.0/0.95 evidence is mixed",
            "stopped_ratio": 0.95,
            "stopped_ratio_rows_are_not_selection_inputs": True,
        },
        "selection_protocol": {
            "primary_metric": "WikiText2 Exact KL",
            "max_bracket_dex": a.MAX_BRACKET_DEX,
            "minimum_top_two_replicates": 2,
            "non_bit_identical_replicates": 3,
            "high_side_worse_points": 2,
            "two_percent_plateau_is_selection_gate": False,
        },
        "rows": [_selection_row(row) for row in analysis["rows"]],
    }
    body["selection_fingerprint"] = c._canonical_sha256(body)
    return body


def _freeze(_: argparse.Namespace) -> int:
    value = _build_selection()
    if SELECTION_PATH.exists():
        current = a._read_json(SELECTION_PATH)
        comparable = dict(current)
        comparable.pop("created_at", None)
        if comparable != value:
            raise a.AblationError(f"existing a=1 selection differs: {SELECTION_PATH}")
    else:
        c._atomic_json(SELECTION_PATH, {**value, "created_at": a._utc_now()})
    print(SELECTION_PATH)
    return 0


def load_selection() -> dict[str, Any]:
    value = a._read_json(SELECTION_PATH)
    if value.get("selection_id") != SELECTION_ID:
        raise a.AblationError("wrong a=1 selection id")
    comparable = dict(value)
    fingerprint = comparable.pop("selection_fingerprint", None)
    comparable.pop("created_at", None)
    if c._canonical_sha256(comparable) != fingerprint:
        raise a.AblationError("a=1 selection fingerprint mismatch")
    plan_ref = value.get("plan")
    if not isinstance(plan_ref, dict) or plan_ref.get("sha256") != c._file_sha256(
        a.PLAN_PATH
    ):
        raise a.AblationError("a=1 selection plan mismatch")
    for item in value.get("selection_code", []):
        path = Path(str(item.get("path", "")))
        if not path.is_file() or c._file_sha256(path) != item.get("sha256"):
            raise a.AblationError(f"a=1 selection code changed: {path}")
    expected = {
        a._group_id(branch, config, RATIO)
        for branch in c.BRANCH_VALUES
        for config in a.SETTINGS
    }
    rows = value.get("rows")
    if not isinstance(rows, list) or {str(row.get("group")) for row in rows} != expected:
        raise a.AblationError("a=1 selection group coverage mismatch")
    if any(float(row.get("a_loss_ratio", -1)) != RATIO for row in rows):
        raise a.AblationError("a=1 selection contains another ratio")
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
    except (
        a.AblationError,
        c.CampaignError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"aloss-a1-selection: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
