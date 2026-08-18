#!/usr/bin/env python3
"""Final-layer-aware paired-cache V3 full-model LR campaign.

V2 incorrectly required the two Block-GD branches to be identical at
``grad_lr=0``.  That is not a physical zero: the frozen experiment profile
keeps ``final_layer_grad_lr`` at 1e-5 (models <=4B) or 1e-6 (models >4B), so
the final transformer block still follows the branch-specific refresh path.

V3 retains V2's shared physical token/static/reference caches and common
``global_loss_bsz``.  It removes the invalid cross-branch equality gate and
uses same-branch repeats for the determinism/noise gate.  Candidate release is
agent-directed and low-to-high; the first immutable batch contains only the
branch-specific ``grad_lr=0`` endpoint.  The 1e-7 batch is written only after
all endpoint results have been inspected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_fullmodel_retune_20260817 import campaign_v2 as v2


CAMPAIGN_ID = "realq-fullmodel-two-branch-retune-20260817-v3-final-layer-aware"
OUTPUT_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_fullmodel_retune_20260817_v3"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
MODULE_PATH = "experiments/realq_fullmodel_retune_20260817/campaign_v3.py"
FIRST_LRS = (0.0,)


def _bootstrap() -> None:
    """Point the audited execution engine at V3 without mutating V1/V2 files."""

    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PLAN_PATH = PLAN_PATH
    base.INITIAL_LRS = FIRST_LRS
    inputs = tuple(
        path
        for path in base.CODE_INPUTS
        if path not in {v2.MODULE_PATH, MODULE_PATH}
    )
    base.CODE_INPUTS = (*inputs, v2.MODULE_PATH, MODULE_PATH)
    base._build_plan = _build_plan


def _build_plan() -> dict[str, Any]:
    # Reuse only V2's audited paired-cache construction.  Rewrite every
    # campaign/protocol identity before computing the immutable V3 fingerprint.
    body = v2._build_plan()
    body.pop("protocol_fingerprint", None)
    body.pop("created_at", None)
    body.update(
        {
            "campaign_id": CAMPAIGN_ID,
            "supersedes_campaign_id": v2.CAMPAIGN_ID,
            "supersedes_reason": (
                "V2 incorrectly treated grad_lr=0 as a cross-branch physical "
                "zero although final_layer_grad_lr remains nonzero"
            ),
            "output_root": str(OUTPUT_ROOT),
            "initial_lrs": list(FIRST_LRS),
            "selection_protocol": {
                "primary_metric": "wikitext2_exact_kl",
                "final_layer_grad_lr": (
                    "frozen by model size exactly as required by 调参方法.md; "
                    "it is not swept with grad_lr"
                ),
                "cross_branch_zero_equality_gate": False,
                "cross_branch_zero_equality_reason": (
                    "grad_lr=0 leaves the branch-specific final transformer "
                    "block refresh active via final_layer_grad_lr"
                ),
                "candidate_release": (
                    "agent-reviewed low-to-high batches; first batch contains "
                    "only grad_lr=0, followed by a separately frozen 1e-7 "
                    "batch; continue upward only while this group's Exact KL "
                    "still improves"
                ),
                "high_side_requirement": (
                    "two successful higher-LR points worse than the incumbent; "
                    "OOM/infra failures do not count"
                ),
                "local_bracket_max_dex": 0.30,
                "top_candidate_min_same_branch_replicates": 2,
                "tie_rule": (
                    "use observed same-branch repeat range; if top candidates "
                    "are tied within that range, select the lower LR"
                ),
                "two_percent_rule": (
                    "reporting-only near-optimal plateau, never a convergence gate"
                ),
                "formal_profile_only": True,
            },
            "v2_invalid_gate_evidence": {
                "full_block_kl": 4.375136375427246,
                "single_linear_kl": 4.753751754760742,
                "config": "qwen3-0.6b_w4a4kv4",
                "grad_lr": 0.0,
                "final_layer_grad_lr": 1e-5,
                "interpretation": (
                    "expected branch-specific final-layer update, not a "
                    "determinism failure"
                ),
            },
            "code_snapshot": base._code_snapshot(),
        }
    )
    body["protocol_fingerprint"] = base._canonical_sha256(body)
    body["created_at"] = base._utc_now()
    return body


def _make_first_manifest(args: argparse.Namespace) -> int:
    plan = base._read_json(PLAN_PATH)
    base._verify_plan(plan)
    trials = []
    # V2's balanced pair order mixes model sizes across the first 16 worker
    # lanes.  This manifest deliberately contains no higher LR: a separate
    # immutable batch is required after all endpoint results are inspected.
    for lr in FIRST_LRS:
        for branch, config in v2._balanced_pairs():
            trials.append(
                {
                    "branch": branch,
                    "config": config,
                    "schedule": base._schedule_for(config),
                    "lr": lr,
                    "stage": "F2-v3-low-to-high-batch-01",
                }
            )
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "name": args.name,
        "rationale": (
            "final-layer-aware first batch: grad_lr=0 is a branch-specific "
            "endpoint because final_layer_grad_lr remains active; no positive "
            "candidate is released in this manifest"
        ),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trials": trials,
    }
    base._validate_manifest(plan, manifest)
    path = OUTPUT_ROOT / "manifests" / f"{args.name}.json"
    if path.exists():
        raise base.CampaignError(f"manifest already exists: {path}")
    base._atomic_json(path, manifest)
    print(path)
    return 0


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    first = subparsers.add_parser("make-first-manifest")
    first.add_argument("--name", default="low_to_high_batch01")
    first.set_defaults(handler=_make_first_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "make-first-manifest":
        args = _custom_parser().parse_args(arguments)
        try:
            return int(args.handler(args))
        except (
            base.CampaignError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            print(f"fullmodel-retune-v3: {exc}", file=sys.stderr)
            return 2
    return base.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
