#!/usr/bin/env python3
"""Hash the five legacy BF16 teacher caches once for all 55 consumers."""

from __future__ import annotations

import json
from pathlib import Path

from . import common


DESTINATION = common.REFERENCE_MANIFEST_PATH


def build() -> dict:
    plans = common.load_quant_plans()
    references = {}
    for model in common.MODEL_ORDER:
        record = common.reference_cache_record(model)
        path = Path(record["source"])
        before = path.stat()
        digest = common.sha256_file(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
        ):
            raise common.EvaluationError(
                f"reference cache changed while hashing: {path}"
            )
        references[model] = {
            "path": str(path),
            "sha256": digest,
            **common.reference_stat_identity(path),
        }
    sources = {
        str(path.relative_to(common.REPO_ROOT)): common.sha256_file(path)
        for path in common.evaluation_source_files()
    }
    payload = {
        "schema_version": 2,
        "status": "prepared",
        "prepared_at": common.now(),
        "baseline_plan": {
            "path": str(common.BASELINE_PLAN),
            "sha256": common.EXPECTED_BASELINE_PLAN_SHA256,
        },
        "quantization_plans": {
            method: {
                "path": str(path),
                "sha256": common.EXPECTED_PLAN_SHA256[method],
            }
            for method, path in (
                ("efficientqat", common.EFFICIENTQAT_PLAN),
                ("turboboa", common.TURBOBOA_PLAN),
                ("yaqa_wclip", common.YAQA_PLAN),
            )
        },
        "protocol": {
            "reference_origin": (
                "exact GPTAQ/GuidedQuant/ResComp-C legacy SDPA BF16 caches"
            ),
            "wikitext2_sequence_length": 2048,
            "kl_direction": "KL(FP||quantized)",
            "kl_full_vocabulary_fp32": True,
            "paper_qa_tasks": list(common.PAPER_QA_TASKS),
            "lm_eval_version": "0.4.4",
            "lm_eval_batch_size": 32,
            "reasoning_protocol": "realq_zero_shot_v1",
            "reasoning_tasks": common.REASONING_TASKS,
        },
        "references": references,
        "sources": sources,
        "matrix_counts": {
            method: sum(spec.method == method for spec in common.iter_specs(plans))
            for method in common.EXPECTED_PLAN_SHA256
        },
    }
    payload["fingerprint"] = common.canonical_sha256(payload)
    return payload


def main() -> int:
    expected = build()
    if DESTINATION.exists():
        current = common.read_object(DESTINATION)
        # prepared_at is intentionally fixed by the first successful producer.
        comparison = dict(expected)
        comparison["prepared_at"] = current.get("prepared_at")
        comparison.pop("fingerprint", None)
        comparison["fingerprint"] = common.canonical_sha256(
            {key: value for key, value in comparison.items() if key != "fingerprint"}
        )
        if current != comparison:
            raise common.EvaluationError(
                f"existing reference manifest differs: {DESTINATION}"
            )
    else:
        common.atomic_json(DESTINATION, expected)
    print(json.dumps(common.read_object(DESTINATION), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
