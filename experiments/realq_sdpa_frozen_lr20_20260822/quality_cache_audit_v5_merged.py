#!/usr/bin/env python3
"""Audit exact GPTAQ/GuidedQuant cache reuse in the merged SDPA quality runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as formal
from experiments.realq_sdpa_frozen_lr20_20260822 import quality_v5_merged as quality


AUDIT_ID = "realq-sdpa-v5-merged-quality-reference-cache-parity-20260823-v2"
AUDIT_PATH = campaign.OUTPUT_ROOT / "quality_cache_audit_v5_merged.json"
EXPECTED_REFERENCE_SHA256 = {
    "llama31-8b-instruct": (
        "91a2a553e8bf50818de7888055ee178110908916a3318ae12416bd4e57273fd6"
    ),
    "qwen3-0.6b": (
        "0b274dbe8f34f38459b0945273f78337db497cac5014a5d4d46411c364ddd10c"
    ),
    "qwen3-4b": (
        "82c47366b8f6ee2539b8d8b745c25c3b7271fed2cf0f88cb64af7239e331999d"
    ),
    "qwen3-8b": (
        "f68f031202fc0f92e51870dd97a13cdf983590745ab6a58275cdd65dfee5decb"
    ),
    "qwen3-32b": (
        "4a09b761e75ede67240988a0d137b581d0379621eae81f8af1f904f29e6be846"
    ),
}


def _contracts() -> dict[str, Any]:
    campaign._bootstrap()
    campaign_plan = base._read_json(campaign.PLAN_PATH)
    campaign._verify_plan(campaign_plan)
    observed = campaign.v1._baseline_cache_contracts()
    if observed != campaign_plan["calibration_and_reference_contracts"]:
        raise base.CampaignError("current physical caches differ from frozen baseline")
    baseline_plan = base._read_json(campaign.v1.BASELINE_PLAN_PATH)
    links = baseline_plan.get("cache_links")
    if not isinstance(links, dict) or set(links) != set(observed):
        raise base.CampaignError("baseline cache-link inventory drifted")
    enriched = json.loads(json.dumps(observed))
    for model, contract in enriched.items():
        link = links[model]
        token_source = Path(str(contract["tokens"]["path"]))
        token_target = Path(str(link["token"]["target"]))
        reference_source = Path(str(contract["reference_logits"]["path"]))
        reference_target = Path(str(link["reference_logits"]["target"]))
        if (
            str(token_source) != str(link["token"]["source"])
            or str(reference_source) != str(link["reference_logits"]["source"])
            or not token_target.is_symlink()
            or not reference_target.is_symlink()
            or not token_source.samefile(token_target)
            or not reference_source.samefile(reference_target)
        ):
            raise base.CampaignError(f"baseline physical cache alias drift: {model}")
        reference_sha256 = base._file_sha256(reference_source)
        if reference_sha256 != EXPECTED_REFERENCE_SHA256[model]:
            raise base.CampaignError(f"baseline reference full SHA drift: {model}")
        contract["tokens"]["baseline_legacy_target"] = str(token_target)
        contract["tokens"]["source_and_baseline_target_samefile"] = True
        contract["reference_logits"]["baseline_legacy_target"] = str(
            reference_target
        )
        contract["reference_logits"]["source_and_baseline_target_samefile"] = True
        contract["reference_logits"]["full_file_sha256"] = reference_sha256
    return enriched


def _validate_command(
    command: Sequence[str],
    *,
    key: str,
    model: str,
    contract: Mapping[str, Any],
) -> dict[str, str]:
    flags = formal._flags(command)
    token = Path(str(contract["tokens"]["path"]))
    token_root = Path(flags["--tokens_cache_path"])
    if sorted(token_root.glob("*.pt")) != [token]:
        raise base.CampaignError(f"quality token cache drift: {key}")
    expected = {
        "--dataset": "wikitext2",
        "--eval_datasets": "wikitext2",
        "--seed": "1",
        "--rotation_seed": "0",
        "--refresh_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--eval_seq_len": "2048",
        "--attention_backend": "sdpa",
        "--skip_eval": "false",
        "--skip_kl_ppl_eval": "false",
        "--lm_eval": "true",
        "--reasoning_eval": "false",
        "--require_reference_cache_hit": "true",
        "--cache_dir": str(contract["reference_logits"]["runtime_root"]),
    }
    for flag, wanted in expected.items():
        if flags.get(flag) != wanted:
            raise base.CampaignError(
                f"quality cache/protocol drift {key} {flag}: {flags.get(flag)}"
            )
    reference = Path(str(contract["reference_logits"]["path"]))
    stat = reference.stat()
    if (
        not reference.is_file()
        or reference.is_symlink()
        or stat.st_size != int(contract["reference_logits"]["size_bytes"])
        or stat.st_mtime_ns != int(contract["reference_logits"]["mtime_ns"])
    ):
        raise base.CampaignError(f"baseline reference cache changed: {model}")
    return flags


def _planned_rows(
    plan: Mapping[str, Any], contracts: Mapping[str, Any]
) -> list[dict[str, Any]]:
    audited = []
    for key, row in sorted(quality._rows(plan).items()):
        branch, config = key.split("/", 1)
        model = campaign._model_for_config(config)
        contract = contracts[model]
        command = list(row["command"])
        _validate_command(command, key=key, model=model, contract=contract)
        audited.append(
            {
                "branch": branch,
                "config": config,
                "model": model,
                "command_sha256": base._canonical_sha256(command),
                "token_path": contract["tokens"]["path"],
                "token_archive_sha256": contract["tokens"]["archive_sha256"],
                "token_semantic_sha256": contract["tokens"]["semantic_sha256"],
                "reference_logits_path": contract["reference_logits"]["path"],
                "reference_logits_full_file_sha256": contract["reference_logits"][
                    "full_file_sha256"
                ],
                "token_source_and_baseline_target_samefile": True,
                "reference_source_and_baseline_target_samefile": True,
                "reference_runtime_root": contract["reference_logits"]["runtime_root"],
                "reference_regeneration_forbidden": True,
            }
        )
    if len(audited) != 40:
        raise base.CampaignError("planned quality cache audit is not 40 rows")
    return audited


def _runtime_rows(
    plan: Mapping[str, Any], contracts: Mapping[str, Any]
) -> list[dict[str, Any]]:
    audited = []
    for branch, config in formal._balanced_pairs():
        directory = quality._quality_dir(branch, config)
        if not (directory / "quality_success.json").is_file():
            continue
        identity = quality._audit_one(plan, branch, config)
        result_path = Path(str(identity["result"]["path"]))
        result = base._read_json(result_path)
        command = list(result["command"])
        model = campaign._model_for_config(config)
        contract = contracts[model]
        _validate_command(
            command,
            key=f"{branch}/{config}",
            model=model,
            contract=contract,
        )
        log_path = Path(str(result["log"]["path"]))
        text = log_path.read_text(encoding="utf-8", errors="strict")
        reference = str(contract["reference_logits"]["path"])
        load_line = f"Loading reference logits for wikitext2 from {reference}"
        if load_line not in text:
            raise base.CampaignError(
                f"exact reference-cache hit not observed: {branch}/{config}"
            )
        if "Generating reference logits for wikitext2" in text:
            raise base.CampaignError(
                f"reference cache was regenerated: {branch}/{config}"
            )
        audited.append(
            {
                "branch": branch,
                "config": config,
                "model": model,
                "result": {
                    "path": str(result_path),
                    "sha256": base._file_sha256(result_path),
                },
                "log": {
                    "path": str(log_path),
                    "sha256": base._file_sha256(log_path),
                },
                "observed_exact_baseline_reference_hit": True,
                "observed_reference_regeneration": False,
            }
        )
    return audited


def build(*, require_complete: bool) -> dict[str, Any]:
    contracts = _contracts()
    plan = quality._load_plan()
    planned = _planned_rows(plan, contracts)
    runtime = _runtime_rows(plan, contracts)
    if require_complete and len(runtime) != 40:
        raise base.CampaignError(
            f"quality cache audit incomplete: {len(runtime)}/40"
        )
    body: dict[str, Any] = {
        "audit_id": AUDIT_ID,
        "status": "complete" if len(runtime) == 40 else "partial",
        "quality_id": quality.QUALITY_ID,
        "quality_plan": {
            "path": str(quality.QUALITY_PLAN_PATH),
            "sha256": base._file_sha256(quality.QUALITY_PLAN_PATH),
            "fingerprint": plan["quality_plan_fingerprint"],
        },
        "baseline_plan": {
            "path": str(campaign.v1.BASELINE_PLAN_PATH),
            "sha256": base._file_sha256(campaign.v1.BASELINE_PLAN_PATH),
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": base._file_sha256(Path(__file__).resolve()),
        },
        "cache_contracts": contracts,
        "planned_rows": planned,
        "runtime_rows": runtime,
        "counts": {
            "planned": len(planned),
            "runtime_succeeded_and_observed": len(runtime),
        },
        "conclusion": (
            "Every quality row binds the exact GPTAQ/GuidedQuant calibration-token "
            "and SDPA BF16 reference-cache physical sources. Every successful row "
            "must log an exact reference hit and must not regenerate the teacher cache."
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
