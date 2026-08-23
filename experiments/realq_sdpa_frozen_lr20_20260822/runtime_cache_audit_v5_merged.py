#!/usr/bin/env python3
"""Audit baseline-cache parity for the final 32+8 SDPA checkpoint release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as formal


AUDIT_ID = "realq-sdpa-v5-merged-gptaq-guidedquant-cache-parity-20260822-v1"
AUDIT_PATH = campaign.OUTPUT_ROOT / "runtime_cache_audit_v5_merged.json"


def _contracts() -> dict[str, Any]:
    campaign._bootstrap()
    plan = base._read_json(campaign.PLAN_PATH)
    campaign._verify_plan(plan)
    observed = campaign.v1._baseline_cache_contracts()
    if observed != plan["calibration_and_reference_contracts"]:
        raise base.CampaignError("current physical caches differ from frozen baseline")
    return observed


def _planned_rows(
    plan: Mapping[str, Any], contracts: Mapping[str, Any]
) -> list[dict[str, Any]]:
    audited = []
    for key, row in sorted(formal._plan_rows(plan).items()):
        branch, config = key.split("/", 1)
        model = campaign._model_for_config(config)
        contract = contracts[model]
        command = list(row["command"])
        flags = formal._flags(command)
        token = Path(contract["tokens"]["path"])
        token_root = Path(flags["--tokens_cache_path"])
        if sorted(token_root.glob("*.pt")) != [token]:
            raise base.CampaignError(f"planned token cache drift: {key}")
        if flags.get("--cache_dir") != contract["reference_logits"]["runtime_root"]:
            raise base.CampaignError(f"planned reference-cache drift: {key}")
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
                    f"planned cache/seed drift {key} {flag}: {flags.get(flag)}"
                )
        audited.append(
            {
                "branch": branch,
                "config": config,
                "model": model,
                "formal_source": row["source"],
                "command_sha256": base._canonical_sha256(command),
                "token_path": str(token),
                "token_archive_sha256": contract["tokens"]["archive_sha256"],
                "token_semantic_sha256": contract["tokens"]["semantic_sha256"],
                "reference_logits_path": contract["reference_logits"]["path"],
                "reference_runtime_root": contract["reference_logits"]["runtime_root"],
            }
        )
    if len(audited) != 40:
        raise base.CampaignError("planned merged cache audit is not 40 rows")
    return audited


def _runtime_rows(
    plan: Mapping[str, Any], contracts: Mapping[str, Any]
) -> list[dict[str, Any]]:
    audited = []
    for branch, config in formal._balanced_pairs():
        directory = formal._formal_dir(branch, config)
        if not (directory / "formal_success.json").is_file():
            continue
        identity = formal._audit_one(plan, branch, config)
        result_path = Path(identity["result"]["path"])
        result = base._read_json(result_path)
        log_path = Path(result["log"]["path"])
        text = log_path.read_text(encoding="utf-8", errors="strict")
        model = campaign._model_for_config(config)
        contract = contracts[model]
        token_line = f"Loading tokens from {contract['tokens']['path']}"
        marker = base._read_json(campaign._cache_marker(model))
        static_path = marker["static_cache"]["path"]
        static_line = f"[realq.precompute] cache hit (rank 0): {static_path}"
        if token_line not in text:
            raise base.CampaignError(f"runtime token load not observed: {branch}/{config}")
        if static_line not in text:
            raise base.CampaignError(f"runtime static hit not observed: {branch}/{config}")
        flags = formal._flags(result["command"])
        if (
            flags.get("--cache_dir") != contract["reference_logits"]["runtime_root"]
            or flags.get("--require_reference_cache_hit") != "true"
        ):
            raise base.CampaignError(f"runtime reference binding drift: {branch}/{config}")
        audited.append(
            {
                "branch": branch,
                "config": config,
                "model": model,
                "formal_source": identity["source"],
                "formal_audit": identity,
                "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
                "log": {"path": str(log_path), "sha256": base._file_sha256(log_path)},
                "observed_exact_baseline_token_load": True,
                "observed_backend_isolated_static_cache_hit": True,
                "reference_cache_path_bound_and_miss_forbidden": True,
            }
        )
    return audited


def build(*, require_complete: bool) -> dict[str, Any]:
    contracts = _contracts()
    plan = formal._load_plan()
    planned = _planned_rows(plan, contracts)
    runtime = _runtime_rows(plan, contracts)
    if require_complete and len(runtime) != 40:
        raise base.CampaignError(f"runtime cache audit incomplete: {len(runtime)}/40")
    body: dict[str, Any] = {
        "audit_id": AUDIT_ID,
        "status": "complete" if len(runtime) == 40 else "partial",
        "formal_id": formal.FORMAL_ID,
        "formal_plan": {
            "path": str(formal.FORMAL_PLAN_PATH),
            "sha256": base._file_sha256(formal.FORMAL_PLAN_PATH),
            "fingerprint": plan["formal_plan_fingerprint"],
        },
        "baseline_plan": {
            "path": str(campaign.v1.BASELINE_PLAN_PATH),
            "sha256": base._file_sha256(campaign.v1.BASELINE_PLAN_PATH),
        },
        "cache_contracts": contracts,
        "planned_rows": planned,
        "runtime_rows": runtime,
        "counts": {
            "planned": len(planned),
            "runtime_succeeded_and_observed": len(runtime),
        },
        "conclusion": (
            "All planned rows bind the exact GPTAQ/GuidedQuant token and SDPA "
            "reference-cache physical sources; each successful formal row also "
            "shows that exact token load and its backend-isolated static hit."
        ),
    }
    body["audit_fingerprint"] = base._canonical_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, default=AUDIT_PATH)
    args = parser.parse_args(argv)
    try:
        value = build(require_complete=args.require_complete)
        output = args.output.expanduser().resolve()
        base._atomic_json(output, value)
        print(
            json.dumps(
                {
                    "status": value["status"],
                    "counts": value["counts"],
                    "audit_fingerprint": value["audit_fingerprint"],
                    "output": str(output),
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
