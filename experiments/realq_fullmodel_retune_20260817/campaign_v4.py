#!/usr/bin/env python3
"""Campaign V4: restore the 20-setting-specific all-one a-loss profile.

V3 correctly fixed paired physical caches and the final-layer-aware zero
endpoint, but inherited the generic Qwen3-4B ``a_loss_ratio=0.95`` rule.  The
frozen 2026-08-08 20-setting campaign explicitly requires 1.0 for every
model/config.  V4 changes that flag only, also freezes reference-cache hits in
the source commands, and retains agent-reviewed, full-model LR release.
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
from experiments.realq_fullmodel_retune_20260817 import campaign_v3 as v3


CAMPAIGN_ID = "realq-fullmodel-two-branch-retune-20260817-v4-all-aloss-one"
OUTPUT_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_fullmodel_retune_20260817_v4"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
MODULE_PATH = "experiments/realq_fullmodel_retune_20260817/campaign_v4.py"
FIRST_LRS = (0.0,)


def _all_one_ratio(_: str) -> float:
    return 1.0


def _all_one_ratio_text(_: str) -> str:
    return "1"


def _bootstrap() -> None:
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PLAN_PATH = PLAN_PATH
    base.INITIAL_LRS = FIRST_LRS
    base._a_loss_ratio_for = _all_one_ratio
    base._a_loss_ratio_text = _all_one_ratio_text
    inputs = tuple(
        path
        for path in base.CODE_INPUTS
        if path not in {v2.MODULE_PATH, v3.MODULE_PATH, MODULE_PATH}
    )
    base.CODE_INPUTS = (*inputs, v2.MODULE_PATH, v3.MODULE_PATH, MODULE_PATH)
    base._build_plan = _build_plan


def _build_plan() -> dict[str, Any]:
    body = v3._build_plan()
    body.pop("protocol_fingerprint", None)
    body.pop("created_at", None)

    for config in base.CONFIG_IDS:
        commands = {}
        for branch in base.BRANCH_VALUES:
            row = body["configurations"][f"{branch}/{config}"]
            row["a_loss_ratio"] = 1.0
            command = row["source_command"]
            base._set_arg(command, "--a_loss_ratio", "1")
            base._set_arg(command, "--require_static_cache_hit", "true")
            base._set_arg(command, "--require_reference_cache_hit", "true")
            base._validate_full_profile(command, branch=branch, config=config)
            commands[branch] = v2._flag_map(command)
        left = dict(commands["full_block"])
        right = dict(commands["single_linear"])
        if left.pop("--full_block_refresh") != "true":
            raise base.CampaignError(f"full branch flag mismatch: {config}")
        if right.pop("--full_block_refresh") != "false":
            raise base.CampaignError(f"single branch flag mismatch: {config}")
        if left != right:
            raise base.CampaignError(f"V4 paired command mismatch: {config}")
        if left["--a_loss_ratio"] != "1":
            raise base.CampaignError(f"V4 a-loss ratio mismatch: {config}")
        if left["--require_reference_cache_hit"] != "true":
            raise base.CampaignError(f"V4 reference cache gate mismatch: {config}")
        body["paired_profile_audit"][config][
            "normalized_command_sha256"
        ] = base._canonical_sha256(left)

    protocol = dict(body["selection_protocol"])
    protocol.update(
        {
            "a_loss_ratio": (
                "1.0 for all 40 branch/config curves, as frozen by the "
                "2026-08-08 20-setting campaign-specific requirements"
            ),
            "reference_cache_gate": (
                "both source and runtime commands require exact reference-cache hit"
            ),
            "candidate_release": (
                "agent-reviewed low-to-high zero-only sub-batches; no positive "
                "LR until all branch/config zero endpoints have been inspected"
            ),
        }
    )
    body.update(
        {
            "campaign_id": CAMPAIGN_ID,
            "supersedes_campaign_id": v3.CAMPAIGN_ID,
            "supersedes_reason": (
                "V3 used the generic Qwen3-4B a_loss_ratio=0.95 rule, but the "
                "specific 20-setting campaign froze a_loss_ratio=1 for all models"
            ),
            "output_root": str(OUTPUT_ROOT),
            "initial_lrs": list(FIRST_LRS),
            "selection_protocol": protocol,
            "code_snapshot": base._code_snapshot(),
        }
    )
    body["protocol_fingerprint"] = base._canonical_sha256(body)
    body["created_at"] = base._utc_now()
    return body


def _make_zero_subbatch(args: argparse.Namespace) -> int:
    plan = base._read_json(PLAN_PATH)
    base._verify_plan(plan)
    models = tuple(args.model)
    unknown = sorted(set(models) - set(base.MODEL_SLUGS))
    if unknown:
        raise base.CampaignError(f"unknown model slug(s): {unknown}")
    trials = []
    for branch, config in v2._balanced_pairs():
        model = next(
            slug for slug in base.MODEL_SLUGS if config.startswith(f"{slug}_")
        )
        if model not in models:
            continue
        trials.append(
            {
                "branch": branch,
                "config": config,
                "schedule": base._schedule_for(config),
                "lr": 0.0,
                "stage": "V4-zero-endpoint-subbatch",
            }
        )
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "name": args.name,
        "rationale": (
            "zero-only full-model subbatch for "
            + ",".join(models)
            + "; no positive LR is released"
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
    zero = subparsers.add_parser("make-zero-subbatch")
    zero.add_argument("--name", required=True)
    zero.add_argument("--model", action="append", required=True)
    zero.set_defaults(handler=_make_zero_subbatch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "make-zero-subbatch":
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
            print(f"fullmodel-retune-v4: {exc}", file=sys.stderr)
            return 2
    return base.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
