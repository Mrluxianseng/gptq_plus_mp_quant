#!/usr/bin/env python3
"""Reviewable next-batch proposals for the V3 full-model LR search.

``propose`` is read-only with respect to experiment launches: it requires all
charged trials to have terminal successful results, snapshots the current
noise-aware analysis, and writes an immutable candidate proposal.  ``freeze``
recomputes that analysis and turns an unchanged, agent-reviewed proposal into
an execution manifest.  Neither command launches a worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import campaign_v3 as v3

v3._bootstrap()

from experiments.realq_fullmodel_retune_20260817 import selection_v3 as selection_v3  # noqa: E402

core = selection_v3.core


PROPOSAL_ID = "realq-fullmodel-retune-v3-reviewed-batch-proposal"


def _stable_analysis() -> dict[str, Any]:
    value = core.analyze_all()
    return {
        "selection_id": value["selection_id"],
        "campaign_id": value["campaign_id"],
        "protocol_fingerprint": value["protocol_fingerprint"],
        "counts": value["counts"],
        "rows": value["rows"],
    }


def _terminal_trial_gate() -> None:
    root = v3.OUTPUT_ROOT / "trials"
    specs = sorted(root.glob("*/spec.json")) if root.is_dir() else []
    missing = [path.parent.name for path in specs if not (path.parent / "result.json").is_file()]
    if missing:
        raise c.CampaignError(
            f"cannot propose while {len(missing)} charged trials are nonterminal: {missing}"
        )
    failures = []
    for path in root.glob("*/result.json") if root.is_dir() else ():
        value = c._read_json(path)
        if value.get("status") != "succeeded":
            failures.append(
                f"{path.parent.name}:{value.get('failure_class', value.get('status'))}"
            )
    if failures:
        raise c.CampaignError(
            f"cannot propose before {len(failures)} failures are reviewed/retried: {failures}"
        )


def _proposal_path(name: str) -> Path:
    return v3.OUTPUT_ROOT / "proposals" / f"{name}.json"


def _proposal_trials(analysis: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    trials = []
    for row in analysis["rows"]:
        for suggestion in row["suggestions"]:
            trial = {
                "branch": row["branch"],
                "config": row["config"],
                "schedule": c._schedule_for(str(row["config"])),
                "lr": float(suggestion["lr"]),
                "stage": f"F2-v3-reviewed-{name}",
                "selection_reason": suggestion["reason"],
            }
            if suggestion.get("replicate") is not None:
                trial["replicate"] = suggestion["replicate"]
            trials.append(trial)
    identities = [c._trial_id(trial) for trial in trials]
    if len(identities) != len(set(identities)):
        raise c.CampaignError("analysis produced duplicate trial identities")
    if not trials:
        raise c.CampaignError("analysis produced no next-batch suggestions")
    return trials


def _propose(args: argparse.Namespace) -> int:
    _terminal_trial_gate()
    plan = c._read_json(v3.PLAN_PATH)
    c._verify_plan(plan)
    analysis = _stable_analysis()
    trials = _proposal_trials(analysis, args.name)
    code = [
        Path(core.__file__).resolve(),
        Path(selection_v3.__file__).resolve(),
        Path(__file__).resolve(),
    ]
    body: dict[str, Any] = {
        "proposal_id": PROPOSAL_ID,
        "campaign_id": v3.CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "name": args.name,
        "rationale": args.rationale,
        "analysis": analysis,
        "analysis_sha256": c._canonical_sha256(analysis),
        "code": [
            {"path": str(path), "sha256": c._file_sha256(path)} for path in code
        ],
        "trials": trials,
    }
    body["proposal_fingerprint"] = c._canonical_sha256(body)
    body["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path = _proposal_path(args.name)
    if path.exists():
        raise c.CampaignError(f"proposal already exists: {path}")
    c._atomic_json(path, body)
    print(path)
    return 0


def _load_proposal(path: Path) -> dict[str, Any]:
    value = c._read_json(path.expanduser().resolve())
    if value.get("proposal_id") != PROPOSAL_ID:
        raise c.CampaignError("proposal id mismatch")
    if value.get("campaign_id") != v3.CAMPAIGN_ID:
        raise c.CampaignError("proposal campaign mismatch")
    identity = dict(value)
    fingerprint = identity.pop("proposal_fingerprint", None)
    identity.pop("created_at", None)
    if c._canonical_sha256(identity) != fingerprint:
        raise c.CampaignError("proposal fingerprint mismatch")
    for item in value.get("code", []):
        source = Path(str(item.get("path", "")))
        if not source.is_file() or c._file_sha256(source) != item.get("sha256"):
            raise c.CampaignError(f"proposal code changed: {source}")
    return value


def _freeze(args: argparse.Namespace) -> int:
    _terminal_trial_gate()
    plan = c._read_json(v3.PLAN_PATH)
    c._verify_plan(plan)
    proposal = _load_proposal(args.proposal)
    current = _stable_analysis()
    if c._canonical_sha256(current) != proposal.get("analysis_sha256"):
        raise c.CampaignError("trial evidence changed after proposal review")
    trials = []
    for proposed in proposal["trials"]:
        # selection_reason is proposal evidence, not part of the execution
        # engine's numerical trial schema.
        trial = dict(proposed)
        trial.pop("selection_reason", None)
        trials.append(trial)
    manifest = {
        "campaign_id": v3.CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "name": proposal["name"],
        "rationale": proposal["rationale"],
        "proposal": {
            "path": str(args.proposal.expanduser().resolve()),
            "sha256": c._file_sha256(args.proposal.expanduser().resolve()),
            "proposal_fingerprint": proposal["proposal_fingerprint"],
        },
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trials": trials,
    }
    c._validate_manifest(plan, manifest)
    additions: dict[tuple[str, str], int] = {}
    for trial in trials:
        key = (str(trial["branch"]), str(trial["config"]))
        additions[key] = additions.get(key, 0) + 1
    for (branch, config), count in additions.items():
        if c._existing_launch_count(branch, config) + count > c.MAX_LAUNCHES_PER_BRANCH_CONFIG:
            raise c.CampaignError(f"20-launch hard cap would be exceeded: {branch}/{config}")
    path = v3.OUTPUT_ROOT / "manifests" / f"{proposal['name']}.json"
    if path.exists():
        raise c.CampaignError(f"manifest already exists: {path}")
    c._atomic_json(path, manifest)
    print(path)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    propose = subparsers.add_parser("propose")
    propose.add_argument("--name", required=True)
    propose.add_argument("--rationale", required=True)
    propose.set_defaults(handler=_propose)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--proposal", type=Path, required=True)
    freeze.set_defaults(handler=_freeze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (c.CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"realq-v3-reviewed-batch: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
