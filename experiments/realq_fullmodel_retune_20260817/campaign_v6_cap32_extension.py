#!/usr/bin/env python3
"""Audited launch-cap extension for the deterministic-FA4 V6 campaign.

The original V6 plan immutablely capped each branch/config at 20 launches.
The user-required seven-point exact coarse grid can consume that allowance on
curves that already accumulated exploratory trials, leaving no room for local
bracket refinement and top-two repeats.  This extension shares the V6 trial
tree and numerical commands, but freezes a new plan/fingerprint whose only
execution-policy change is a total per-curve launch cap of 32.

It never rewrites the V6 plan or existing results.  Since the output root is
shared, the base launch registry counts V6 and extension specs together when
enforcing the new total cap.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_fullmodel_retune_20260817 import campaign_v6_run15 as v6


CAMPAIGN_ID = "realq-fullmodel-two-branch-retune-20260818-v6-cap32-extension-v1"
OUTPUT_ROOT = v6.OUTPUT_ROOT
PLAN_PATH = OUTPUT_ROOT / "cap32_extension" / "plan.json"
MODULE_PATH = (
    "experiments/realq_fullmodel_retune_20260817/"
    "campaign_v6_cap32_extension.py"
)
MAX_TOTAL_LAUNCHES_PER_BRANCH_CONFIG = 32
REQUIRED_COARSE_LRS = (5e-7, 1e-6, 3e-6, 7e-6, 1e-5, 3e-5, 5e-5)

# Capture the unmodified verifier once.  It reads campaign globals at call
# time, so the extension bootstrap below can still reuse its canonical checks.
_BASE_VERIFY_PLAN = base._verify_plan


def _read_source_plan() -> dict[str, Any]:
    source = base._read_json(v6.PLAN_PATH)
    stable = dict(source)
    fingerprint = stable.pop("protocol_fingerprint", None)
    stable.pop("created_at", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise base.CampaignError("source V6 plan fingerprint mismatch")
    return source


def _source_plan_ref(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(v6.PLAN_PATH),
        "sha256": base._file_sha256(v6.PLAN_PATH),
        "campaign_id": source["campaign_id"],
        "protocol_fingerprint": source["protocol_fingerprint"],
    }


def _assert_numerical_parity(
    extension: Mapping[str, Any], source: Mapping[str, Any]
) -> None:
    if extension["configurations"] != source["configurations"]:
        raise base.CampaignError("extension changed one or more source commands")
    if extension["cache_snapshots"] != source["cache_snapshots"]:
        raise base.CampaignError("extension changed V6 cache snapshots")
    if extension["calibration_token_hashes"] != source["calibration_token_hashes"]:
        raise base.CampaignError("extension changed calibration-token hashes")
    if extension["determinism"] != source["determinism"]:
        raise base.CampaignError("extension changed deterministic seeds/options")
    if extension["optimization_profile"] != source["optimization_profile"]:
        raise base.CampaignError("extension changed the run15 optimization profile")


def _build_plan() -> dict[str, Any]:
    source = _read_source_plan()
    body = copy.deepcopy(source)
    body.pop("protocol_fingerprint", None)
    body.pop("created_at", None)
    body.update(
        {
            "campaign_id": CAMPAIGN_ID,
            "extends_campaign_id": source["campaign_id"],
            "extends_protocol_fingerprint": source["protocol_fingerprint"],
            "extension_reason": (
                "user-required exact seven-point coarse grid plus subsequent "
                "local refinement/repeats cannot be fail-closed under the "
                "original 20-launch orchestration cap"
            ),
            "extension_scope": (
                "only max_launches_per_branch_config changes; numerical "
                "commands, calibration/cache inputs, seeds, deterministic "
                "options, and optimization profile are byte-identical to V6"
            ),
            "max_launches_per_branch_config": (
                MAX_TOTAL_LAUNCHES_PER_BRANCH_CONFIG
            ),
            "initial_lrs": list(REQUIRED_COARSE_LRS),
            "source_v6_plan": _source_plan_ref(source),
            "code_snapshot": base._code_snapshot(),
        }
    )
    body["selection_protocol"] = {
        **copy.deepcopy(source["selection_protocol"]),
        "required_exact_coarse_lrs": list(REQUIRED_COARSE_LRS),
        "coarse_grid_before_local_refinement": True,
        "total_launch_cap_after_extension": (
            MAX_TOTAL_LAUNCHES_PER_BRANCH_CONFIG
        ),
    }
    _assert_numerical_parity(body, source)
    body["protocol_fingerprint"] = base._canonical_sha256(body)
    body["created_at"] = base._utc_now()
    return body


def _verify_extension_plan(plan: Mapping[str, Any]) -> None:
    _BASE_VERIFY_PLAN(plan)
    source = _read_source_plan()
    if plan.get("source_v6_plan") != _source_plan_ref(source):
        raise base.CampaignError("extension source-plan reference changed")
    if int(plan.get("max_launches_per_branch_config", 0)) != (
        MAX_TOTAL_LAUNCHES_PER_BRANCH_CONFIG
    ):
        raise base.CampaignError("extension launch cap is not 32")
    _assert_numerical_parity(plan, source)


def _bootstrap() -> None:
    # Reuse every V6 run15 execution hook, then change only orchestration
    # identity/plan/cap and extend the code snapshot with this module.
    v6._bootstrap()
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PLAN_PATH = PLAN_PATH
    base.INITIAL_LRS = REQUIRED_COARSE_LRS
    base.MAX_LAUNCHES_PER_BRANCH_CONFIG = MAX_TOTAL_LAUNCHES_PER_BRANCH_CONFIG
    base.CODE_INPUTS = tuple(dict.fromkeys((*base.CODE_INPUTS, MODULE_PATH)))
    base._build_plan = _build_plan
    base._verify_plan = _verify_extension_plan


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
