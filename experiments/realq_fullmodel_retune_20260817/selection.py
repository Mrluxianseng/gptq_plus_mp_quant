#!/usr/bin/env python3
"""Noise-aware, log-space analysis and final LR gate for the 2026-08-17 run.

This program never launches a trial.  It emits evidence and suggestions for
agent review, and only freezes selections when every branch/config satisfies
the protocol recorded in ``plan.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c


SELECTION_ID = "realq-fullmodel-two-branch-lr-selection-20260817-v1"
SELECTION_PATH = c.OUTPUT_ROOT / "selections.json"
MAX_BRACKET_DEX = 0.30


def _read_plan() -> dict[str, Any]:
    plan = c._read_json(c.PLAN_PATH)
    c._verify_plan(plan)
    return plan


def _result_rows(branch: str, config: str) -> list[dict[str, Any]]:
    rows = []
    root = c.OUTPUT_ROOT / "trials"
    if not root.is_dir():
        return rows
    for result_path in sorted(root.glob("*/result.json")):
        result = c._read_json(result_path)
        if result.get("branch") != branch or result.get("config") != config:
            continue
        rows.append(
            {
                **result,
                "result_path": str(result_path),
                "result_sha256": c._file_sha256(result_path),
            }
        )
    return rows


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _aggregate(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "succeeded":
            grouped[float(row["lr"])].append(row)
    output = []
    for lr, candidates in sorted(grouped.items()):
        values = [float(row["kl"]) for row in candidates]
        med = _median(values)
        deviations = [abs(value - med) for value in values]
        output.append(
            {
                "lr": lr,
                "lr_exact": c._stable_float(lr),
                "median_kl": med,
                "min_kl": min(values),
                "max_kl": max(values),
                "range": max(values) - min(values),
                "mad": _median(deviations),
                "replicates": len(values),
                "result_refs": [
                    {
                        "identity": row["identity"],
                        "path": row["result_path"],
                        "sha256": row["result_sha256"],
                        "kl": float(row["kl"]),
                        "ppl": float(row["ppl"]),
                        "gpu_uuid": row.get("gpu", {}).get("uuid"),
                    }
                    for row in candidates
                ],
            }
        )
    return output


def _next_rep_key(item: Mapping[str, Any]) -> str:
    return f"confirm{int(item['replicates'])}"


def _analyze_one(branch: str, config: str) -> dict[str, Any]:
    rows = _result_rows(branch, config)
    aggregate = _aggregate(rows)
    failures = [row for row in rows if row.get("status") != "succeeded"]
    result: dict[str, Any] = {
        "branch": branch,
        "config": config,
        "launches": len(rows),
        "remaining_launch_budget": c.MAX_LAUNCHES_PER_BRANCH_CONFIG - len(rows),
        "failures": [
            {
                "identity": row.get("identity"),
                "lr": row.get("lr"),
                "failure_class": row.get("failure_class"),
                "error": row.get("error"),
            }
            for row in failures
        ],
        "aggregates": aggregate,
        "ready": False,
        "suggestions": [],
    }
    if not aggregate:
        result["reason"] = "no successful Exact-KL trial"
        return result

    ranked = sorted(aggregate, key=lambda item: (item["median_kl"], item["lr"]))
    best = ranked[0]
    top_two = ranked[:2]
    positive = [item for item in aggregate if item["lr"] > 0]
    lower_worse = [
        item
        for item in positive
        if item["lr"] < best["lr"] and item["median_kl"] > best["median_kl"]
    ]
    upper_worse = [
        item
        for item in positive
        if item["lr"] > best["lr"] and item["median_kl"] > best["median_kl"]
    ]
    high_side = upper_worse
    bracket = None
    if best["lr"] > 0 and lower_worse and upper_worse:
        lower = max(lower_worse, key=lambda item: item["lr"])
        upper = min(upper_worse, key=lambda item: item["lr"])
        bracket = {
            "lower_lr": lower["lr"],
            "best_lr": best["lr"],
            "upper_lr": upper["lr"],
            "width_dex": math.log10(upper["lr"] / lower["lr"]),
        }

    # Two executions are enough only when the deterministic stack reproduces
    # Exact KL bit-for-bit.  If either candidate differs across its first two
    # executions, require a third observation so a median/noise interval is
    # not inferred from a single pair by accident.
    for item in top_two:
        item["required_replicates"] = 2 if item["range"] == 0 else 3
    replicate_gate = len(top_two) == 2 and all(
        item["replicates"] >= item["required_replicates"] for item in top_two
    )
    high_gate = len(high_side) >= 2
    if best["lr"] == 0:
        # Zero is an independent control.  Two measured positive degradations
        # plus repeated top candidates are sufficient; log interpolation is
        # undefined at zero.
        bracket_gate = len(
            [item for item in positive if item["median_kl"] > best["median_kl"]]
        ) >= 2
    else:
        bracket_gate = bracket is not None and bracket["width_dex"] <= MAX_BRACKET_DEX

    plateau = [
        item["lr"]
        for item in aggregate
        if item["median_kl"] <= best["median_kl"] * 1.02
    ]
    noise_floor = max((item["range"] for item in top_two), default=0.0)
    top_gap = (
        abs(top_two[1]["median_kl"] - top_two[0]["median_kl"])
        if len(top_two) == 2
        else None
    )
    noise_tie = top_gap is not None and top_gap <= noise_floor
    selected_lr = None
    if replicate_gate and bracket_gate and high_gate:
        selected_lr = (
            min(item["lr"] for item in top_two) if noise_tie else best["lr"]
        )

    result.update(
        {
            "best": best,
            "top_two": top_two,
            "two_percent_plateau": plateau,
            "high_side_worse_lrs": [item["lr"] for item in high_side],
            "high_side_gate": high_gate,
            "bracket": bracket,
            "bracket_gate": bracket_gate,
            "replicate_gate": replicate_gate,
            "noise_floor_range": noise_floor,
            "top_two_gap": top_gap,
            "noise_tie": noise_tie,
            "selected_lr": selected_lr,
            "ready": selected_lr is not None,
        }
    )

    existing_lrs = {item["lr"] for item in aggregate}
    suggestions: list[dict[str, Any]] = []
    if not high_gate:
        if not positive:
            # The first reviewed batch contains only the branch-specific zero
            # endpoint.  Start at the documented smallest global candidate;
            # do not silently skip a decade to 1e-6.
            proposed = 1e-7
            reason = "first_positive_candidate"
        elif best["lr"] == 0:
            # A worse first positive point means the optimum may be on the
            # 0..first-positive boundary.  Obtain a second, *smaller* positive
            # observation instead of marching farther into the already-worse
            # high side.  Once two positive degradations are measured, the
            # physical-zero boundary gate is satisfied.
            proposed = min(item["lr"] for item in positive) / 10
            reason = "refine_physical_zero_boundary"
        else:
            proposed = max(item["lr"] for item in positive) * 10
            reason = "need_second_measured_high_side_point"
        if proposed not in existing_lrs:
            suggestions.append({"reason": reason, "lr": proposed})
    if best["lr"] > 0 and not lower_worse:
        proposed = best["lr"] / math.sqrt(10)
        if proposed not in existing_lrs:
            suggestions.append({"reason": "need_low_side_point", "lr": proposed})
    if best["lr"] > 0 and lower_worse and upper_worse and not bracket_gate:
        lower = max(lower_worse, key=lambda item: item["lr"])["lr"]
        upper = min(upper_worse, key=lambda item: item["lr"])["lr"]
        left_width = math.log10(best["lr"] / lower)
        right_width = math.log10(upper / best["lr"])
        proposed = (
            math.sqrt(lower * best["lr"])
            if left_width >= right_width
            else math.sqrt(best["lr"] * upper)
        )
        if proposed not in existing_lrs:
            suggestions.append({"reason": "refine_log_bracket", "lr": proposed})
    if bracket_gate and high_gate and not replicate_gate:
        for item in top_two:
            if item["replicates"] < item["required_replicates"]:
                suggestions.append(
                    {
                        "reason": "repeat_top_two_for_noise_gate",
                        "lr": item["lr"],
                        "replicate": _next_rep_key(item),
                    }
                )
    remaining = result["remaining_launch_budget"]
    result["suggestions"] = suggestions[: max(remaining, 0)]
    if suggestions and remaining <= 0:
        result["reason"] = "launch budget exhausted before protocol gates passed"
    elif not result["ready"]:
        result["reason"] = "one or more bracket/high-side/repeat gates remain"
    return result


def analyze_all() -> dict[str, Any]:
    plan = _read_plan()
    rows = [
        _analyze_one(branch, config)
        for config in c.CONFIG_IDS
        for branch in c.BRANCH_VALUES
    ]
    return {
        "selection_id": SELECTION_ID,
        "campaign_id": c.CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
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
    payload = analyze_all()
    if args.output:
        c._atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _freeze(_: argparse.Namespace) -> int:
    plan = _read_plan()
    analysis = analyze_all()
    not_ready = [
        f"{row['branch']}/{row['config']}" for row in analysis["rows"] if not row["ready"]
    ]
    if not_ready:
        raise c.CampaignError(
            f"cannot freeze; {len(not_ready)} groups have not passed all gates: {not_ready}"
        )
    rows = []
    for row in analysis["rows"]:
        rows.append(
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
        )
    body = {
        "selection_id": SELECTION_ID,
        "campaign_id": c.CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "plan": {"path": str(c.PLAN_PATH), "sha256": c._file_sha256(c.PLAN_PATH)},
        "selection_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": c._file_sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "primary_metric": "wikitext2 Exact KL",
            "max_bracket_dex": MAX_BRACKET_DEX,
            "minimum_top_candidate_replicates": 2,
            "high_side_worse_points": 2,
            "tie_break": "lower LR when top-two gap <= observed repeat range",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--output", type=Path)
    status.set_defaults(handler=_status)
    freeze = subparsers.add_parser("freeze")
    freeze.set_defaults(handler=_freeze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (c.CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"realq-fullmodel-selection: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
