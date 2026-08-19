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
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import campaign_v6_run15 as v6


v6._bootstrap()

from experiments.realq_fullmodel_retune_20260817 import selection as core  # noqa: E402


SELECTION_ID = "realq-fullmodel-two-branch-lr-selection-20260818-v6-run15"
SELECTION_PATH = v6.OUTPUT_ROOT / "selections.json"
core.SELECTION_ID = SELECTION_ID
core.SELECTION_PATH = SELECTION_PATH

# User-mandated full-model coarse sweep (2026-08-19).  Every branch/config
# must measure these exact points before the selector may consume any local
# bracket/refinement result.  Existing off-grid measurements remain valid
# evidence, but cannot by themselves satisfy this gate.
REQUIRED_COARSE_LRS = (5e-7, 1e-6, 3e-6, 7e-6, 1e-5, 3e-5, 5e-5)
CAP32_EXTENSION_CAMPAIGN_ID = (
    "realq-fullmodel-two-branch-retune-20260818-v6-cap32-extension-v1"
)
CAP32_EXTENSION_PLAN_PATH = v6.OUTPUT_ROOT / "cap32_extension" / "plan.json"
CAP32_EXTENSION_MODULE_PATH = (
    Path(__file__).resolve().parent / "campaign_v6_cap32_extension.py"
)
NUMERICAL_PARITY_KEYS = (
    "configurations",
    "cache_snapshots",
    "calibration_token_hashes",
    "determinism",
    "optimization_profile",
)


def _has_lr(measured: Sequence[float], wanted: float) -> bool:
    return any(
        math.isclose(value, wanted, rel_tol=1e-12, abs_tol=0.0)
        for value in measured
    )


def _verify_code_snapshot(plan: Mapping[str, Any]) -> None:
    frozen = plan["code_snapshot"]
    observed = []
    for item in frozen["files"]:
        path = c.REPO_ROOT / str(item["path"])
        if not path.is_file():
            raise c.CampaignError(f"execution code input missing: {path}")
        current = {
            "path": str(item["path"]),
            "size_bytes": path.stat().st_size,
            "sha256": c._file_sha256(path),
        }
        if current != item:
            raise c.CampaignError(f"execution code input changed: {path}")
        observed.append(current)
    if c._canonical_sha256(observed) != frozen["sha256"]:
        raise c.CampaignError("execution code snapshot aggregate mismatch")


def _read_canonical_plan(path: Path) -> dict[str, Any]:
    plan = c._read_json(path)
    stable = dict(plan)
    fingerprint = stable.pop("protocol_fingerprint", None)
    stable.pop("created_at", None)
    if c._canonical_sha256(stable) != fingerprint:
        raise c.CampaignError(f"execution plan fingerprint mismatch: {path}")
    _verify_code_snapshot(plan)
    return plan


def _execution_campaign_provenance() -> list[dict[str, Any]]:
    """Validate every charged/result artifact against V6 or its cap extension."""

    primary = core._read_plan()
    plans = {v6.CAMPAIGN_ID: primary}
    plan_paths = {v6.CAMPAIGN_ID: v6.PLAN_PATH}
    if CAP32_EXTENSION_PLAN_PATH.is_file():
        extension = _read_canonical_plan(CAP32_EXTENSION_PLAN_PATH)
        if extension.get("campaign_id") != CAP32_EXTENSION_CAMPAIGN_ID:
            raise c.CampaignError("unexpected cap32 extension campaign id")
        source_ref = extension.get("source_v6_plan", {})
        if source_ref != {
            "path": str(v6.PLAN_PATH),
            "sha256": c._file_sha256(v6.PLAN_PATH),
            "campaign_id": primary["campaign_id"],
            "protocol_fingerprint": primary["protocol_fingerprint"],
        }:
            raise c.CampaignError("cap32 extension source-plan reference mismatch")
        if int(extension.get("max_launches_per_branch_config", 0)) != 32:
            raise c.CampaignError("cap32 extension launch cap mismatch")
        for key in NUMERICAL_PARITY_KEYS:
            if extension.get(key) != primary.get(key):
                raise c.CampaignError(
                    f"cap32 extension changed numerical protocol field: {key}"
                )
        plans[CAP32_EXTENSION_CAMPAIGN_ID] = extension
        plan_paths[CAP32_EXTENSION_CAMPAIGN_ID] = CAP32_EXTENSION_PLAN_PATH

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"charged_specs": 0, "results": 0}
    )
    trial_root = v6.OUTPUT_ROOT / "trials"
    for filename, counter in (("spec.json", "charged_specs"), ("result.json", "results")):
        for artifact_path in sorted(trial_root.glob(f"*/{filename}")):
            artifact = c._read_json(artifact_path)
            campaign_id = str(artifact.get("campaign_id"))
            if campaign_id not in plans:
                raise c.CampaignError(
                    f"unrecognized execution campaign in {artifact_path}: {campaign_id}"
                )
            if artifact.get("protocol_fingerprint") != plans[campaign_id][
                "protocol_fingerprint"
            ]:
                raise c.CampaignError(
                    f"execution fingerprint mismatch: {artifact_path}"
                )
            command = artifact.get("command")
            if not isinstance(command, list) or artifact.get(
                "command_sha256"
            ) != c._canonical_sha256(command):
                raise c.CampaignError(f"execution command hash mismatch: {artifact_path}")
            c._validate_full_profile(
                command,
                branch=str(artifact["branch"]),
                config=str(artifact["config"]),
            )
            counts[campaign_id][counter] += 1

    return [
        {
            "campaign_id": campaign_id,
            "protocol_fingerprint": plan["protocol_fingerprint"],
            "plan": {
                "path": str(plan_paths[campaign_id]),
                "sha256": c._file_sha256(plan_paths[campaign_id]),
            },
            **counts[campaign_id],
        }
        for campaign_id, plan in plans.items()
    ]


def _apply_required_coarse_grid(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the user-required coarse-grid gate to core selector output."""

    for row in payload["rows"]:
        measured = [float(item["lr"]) for item in row.get("aggregates", [])]
        missing = [
            lr for lr in REQUIRED_COARSE_LRS if not _has_lr(measured, lr)
        ]
        row["required_coarse_grid"] = list(REQUIRED_COARSE_LRS)
        row["missing_coarse_lrs"] = missing
        row["coarse_grid_gate"] = not missing
        if not missing:
            continue

        # Preserve the deferred local actions for audit/debugging, but do not
        # expose them as launch suggestions until all seven coarse points are
        # measured.  This makes "coarse first, then refine" fail closed.
        row["deferred_local_suggestions"] = list(row.get("suggestions", []))
        row["suggestions"] = [
            {"reason": "complete_user_required_coarse_grid", "lr": lr}
            for lr in missing
        ]
        row["selected_lr"] = None
        row["ready"] = False
        row["reason"] = (
            "user-required seven-point coarse grid incomplete; local "
            "refinement and final selection are deferred"
        )

    payload["counts"]["ready"] = sum(
        bool(row["ready"]) for row in payload["rows"]
    )
    payload["required_coarse_grid"] = list(REQUIRED_COARSE_LRS)
    payload["coarse_grid_complete_groups"] = sum(
        bool(row["coarse_grid_gate"]) for row in payload["rows"]
    )
    return payload


def _analyze_all() -> dict[str, Any]:
    execution_campaigns = _execution_campaign_provenance()
    payload = _apply_required_coarse_grid(core.analyze_all())
    payload["execution_campaigns"] = execution_campaigns
    return payload


def _status(args: argparse.Namespace) -> int:
    payload = _analyze_all()
    if args.output:
        c._atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _freeze(_: argparse.Namespace) -> int:
    plan = core._read_plan()
    analysis = _analyze_all()
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
            "required_coarse_grid": row["required_coarse_grid"],
            "coarse_grid_gate": row["coarse_grid_gate"],
            "launches": row["launches"],
        }
        for row in analysis["rows"]
    ]
    code_paths = [
        Path(core.__file__).resolve(),
        Path(v6.__file__).resolve(),
        Path(__file__).resolve(),
    ]
    if CAP32_EXTENSION_MODULE_PATH.is_file():
        code_paths.append(CAP32_EXTENSION_MODULE_PATH)
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
        "execution_campaigns": analysis["execution_campaigns"],
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
            "required_full_model_coarse_lrs": list(REQUIRED_COARSE_LRS),
            "coarse_grid_must_complete_before_local_refinement": True,
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
