#!/usr/bin/env python3
"""Audit planned and observed cache parity with GPTAQ/GuidedQuant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v2 as formal


AUDIT_ID = "realq-sdpa-v2b-gptaq-guidedquant-cache-parity-20260822-v1"


def _planned_row_audit(
    campaign_plan: Mapping[str, Any], formal_plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = formal._plan_rows(formal_plan)
    audited: list[dict[str, Any]] = []
    for key in sorted(rows):
        branch, config = key.split("/", 1)
        model = campaign._model_for_config(config)
        contract = campaign_plan["calibration_and_reference_contracts"][model]
        command = list(rows[key]["command"])
        flags = formal._flags(command)
        token = Path(contract["tokens"]["path"])
        token_root = Path(flags["--tokens_cache_path"])
        if sorted(token_root.glob("*.pt")) != [token]:
            raise base.CampaignError(f"planned token cache drift: {key}")
        if flags.get("--cache_dir") != contract["reference_logits"]["runtime_root"]:
            raise base.CampaignError(f"planned reference cache drift: {key}")
        expected = {
            "--dataset": "wikitext2",
            "--seed": "1",
            "--rotation_seed": "0",
            "--refresh_seed": "0",
            "--nsamples": "256",
            "--seq_len": "2048",
            "--rotate": "true",
            "--attention_backend": "sdpa",
            "--require_static_cache_hit": "true",
            "--require_reference_cache_hit": "true",
        }
        for flag, wanted in expected.items():
            if flags.get(flag) != wanted:
                raise base.CampaignError(
                    f"planned cache/seed protocol drift {key} {flag}: {flags.get(flag)}"
                )
        audited.append(
            {
                "branch": branch,
                "config": config,
                "model": model,
                "command_sha256": base._canonical_sha256(command),
                "token_path": str(token),
                "token_archive_sha256": contract["tokens"]["archive_sha256"],
                "token_semantic_sha256": contract["tokens"]["semantic_sha256"],
                "reference_logits_path": contract["reference_logits"]["path"],
            }
        )
    if len(audited) != 40:
        raise base.CampaignError("planned cache audit matrix is not 40 rows")
    return audited


def _runtime_row_audit(
    campaign_plan: Mapping[str, Any], formal_plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for branch, config in formal._balanced_pairs():
        directory = formal._formal_dir(branch, config)
        success = directory / "formal_success.json"
        if not success.is_file():
            continue
        identity = formal._audit_one(formal_plan, branch, config)
        result_path = Path(base._read_json(success)["result"]["path"])
        result = base._read_json(result_path)
        log_path = Path(result["log"]["path"])
        text = log_path.read_text(encoding="utf-8", errors="strict")
        model = campaign._model_for_config(config)
        contract = campaign_plan["calibration_and_reference_contracts"][model]
        token_line = f"Loading tokens from {contract['tokens']['path']}"
        marker = base._read_json(campaign._cache_marker(model))
        static_path = marker["static_cache"]["path"]
        static_line = f"[realq.precompute] cache hit (rank 0): {static_path}"
        if token_line not in text:
            raise base.CampaignError(f"runtime token load not observed: {branch}/{config}")
        if static_line not in text:
            raise base.CampaignError(f"runtime static hit not observed: {branch}/{config}")
        audited.append(
            {
                "branch": branch,
                "config": config,
                "model": model,
                "formal_audit": identity,
                "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
                "log": {"path": str(log_path), "sha256": base._file_sha256(log_path)},
                "observed_exact_baseline_token_load": True,
                "observed_backend_isolated_static_cache_hit": True,
            }
        )
    return audited


def build(*, require_complete: bool) -> dict[str, Any]:
    campaign._bootstrap()
    formal._activate()
    campaign_plan = base._read_json(campaign.PLAN_PATH)
    campaign._verify_plan(campaign_plan)
    formal_plan = formal._load_plan()
    baseline_plan = campaign.v1._baseline_plan()
    # Recompute archive SHA, semantic SHA, tensor contract, size and mtime for
    # all five physical token/reference sources.
    contracts = campaign.v1._baseline_cache_contracts()
    if contracts != campaign_plan["calibration_and_reference_contracts"]:
        raise base.CampaignError("campaign cache contracts differ from baseline")
    planned = _planned_row_audit(campaign_plan, formal_plan)
    runtime = _runtime_row_audit(campaign_plan, formal_plan)
    if require_complete and len(runtime) != 40:
        raise base.CampaignError(f"runtime cache audit incomplete: {len(runtime)}/40")
    body: dict[str, Any] = {
        "audit_id": AUDIT_ID,
        "status": "complete" if len(runtime) == 40 else "partial",
        "campaign_id": campaign.CAMPAIGN_ID,
        "protocol_fingerprint": campaign_plan["protocol_fingerprint"],
        "formal_id": formal.FORMAL_ID,
        "formal_plan_fingerprint": formal_plan["formal_plan_fingerprint"],
        "baseline_plan": {
            "path": str(campaign.v1.BASELINE_PLAN_PATH),
            "sha256": base._file_sha256(campaign.v1.BASELINE_PLAN_PATH),
            "fingerprint": baseline_plan["fingerprint"],
        },
        "cache_contracts": contracts,
        "planned_rows": planned,
        "runtime_rows": runtime,
        "counts": {
            "planned": len(planned),
            "runtime_succeeded_and_observed": len(runtime),
        },
        "conclusion": (
            "Every audited REALQ row uses the exact GPTAQ/GuidedQuant token "
            "and reference-cache sources; successful rows also show the exact "
            "token load and backend-isolated SDPA static-cache hit in runtime logs."
        ),
    }
    body["audit_fingerprint"] = base._canonical_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        value = build(require_complete=args.require_complete)
        if args.output is not None:
            path = args.output.expanduser().resolve()
            base._atomic_json(path, value)
        print(
            json.dumps(
                {
                    "status": value["status"],
                    "counts": value["counts"],
                    "audit_fingerprint": value["audit_fingerprint"],
                    "output": str(args.output.expanduser().resolve()) if args.output else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
