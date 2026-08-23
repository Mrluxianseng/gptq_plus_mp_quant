#!/usr/bin/env python3
"""Build the fail-closed 55-suite release used by the master comparison table."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import traceback
from typing import Any, Mapping

from . import common


EXPECTED_REFERENCE_FINGERPRINT = (
    "3ecffa5d4cb97efb24aac4a668b6570d8e8193f4f4fa21f98825c2d9d238be05"
)
EFFICIENT_RETIME = (
    common.DATA_ROOT / "efficientqat_q06_clean_retime_20260821_v2"
)
TURBO_RETIME = (
    common.DATA_ROOT / "turboboa_contended_clean_retime_20260821_v1"
)
YAQA_RETIME = (
    common.DATA_ROOT / "yaqa_hessian_clean_retime_20260821_v1"
)
EFFICIENT_RETIME_IDS = {
    "EQ15-Q06-W4",
    "EQ15-Q06-W3",
    "EQ15-Q06-W2",
}
TURBO_RETIME_IDS = {
    "TB20-Q06-W4A4",
    "TB20-Q06-W4",
    "TB20-Q06-W3",
    "TB20-Q06-W2",
    "TB20-Q4-W2",
    "TB20-L8-W4",
}
YAQA_Q06_A16_IDS = {
    "YQ-Q06-W4",
    "YQ-Q06-W3",
    "YQ-Q06-W2",
}
COMPARISON_IDS = {
    ("qwen3-0.6b", "W4A16KV16"): "C01",
    ("qwen3-0.6b", "W4A4KV4"): "C02",
    ("qwen3-0.6b", "W3A16KV16"): "C03",
    ("qwen3-0.6b", "W2A16KV16"): "C04",
    ("llama31-8b-instruct", "W4A16KV16"): "C05",
    ("llama31-8b-instruct", "W4A4KV4"): "C06",
    ("llama31-8b-instruct", "W3A16KV16"): "C07",
    ("llama31-8b-instruct", "W2A16KV16"): "C08",
    ("qwen3-4b", "W4A16KV16"): "C09",
    ("qwen3-4b", "W4A4KV4"): "C10",
    ("qwen3-4b", "W3A16KV16"): "C11",
    ("qwen3-4b", "W2A16KV16"): "C12",
    ("qwen3-8b", "W4A16KV16"): "C13",
    ("qwen3-8b", "W4A4KV4"): "C14",
    ("qwen3-8b", "W3A16KV16"): "C15",
    ("qwen3-8b", "W2A16KV16"): "C16",
    ("qwen3-32b", "W4A16KV16"): "C17",
    ("qwen3-32b", "W4A4KV4"): "C18",
    ("qwen3-32b", "W3A16KV16"): "C19",
    ("qwen3-32b", "W2A16KV16"): "C20",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise common.EvaluationError(message)


def _read_bound(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(common.sha256_file(source) == expected_sha256, f"SHA changed: {source}")
    return common.read_object(source)


def _retime_gpu_hours(
    spec: common.EvalSpec,
    artifact: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    reported = float(artifact["quantization_gpu_hours"])
    if spec.run_id in EFFICIENT_RETIME_IDS:
        path = EFFICIENT_RETIME / "runs" / spec.run_id / "clean_replay_result.json"
        value = common.read_object(path)
        _require(
            value.get("checkpoint_byte_identity") is True
            and value.get("source_run", {}).get("run_id") == spec.run_id
            and value.get("source_plan_sha256")
            == common.EXPECTED_PLAN_SHA256["efficientqat"],
            f"invalid EfficientQAT clean retime: {spec.run_id}",
        )
        clean = float(value["clean_quantization_gpu_hours"])
        return clean, {
            "kind": "checkpoint_byte_identical_clean_retime",
            "path": str(path),
            "sha256": common.sha256_file(path),
            "reported_gpu_hours_excluded": reported,
        }

    if spec.run_id in TURBO_RETIME_IDS:
        path = (
            TURBO_RETIME
            / "runs"
            / spec.model
            / spec.setting.lower()
            / spec.run_id
            / "clean_replay_result.json"
        )
        value = common.read_object(path)
        _require(
            value.get("checkpoint_content_identity") is True
            and value.get("source_run", {}).get("run_id") == spec.run_id
            and value.get("source_plan_sha256")
            == common.EXPECTED_PLAN_SHA256["turboboa"],
            f"invalid TurboBOA clean retime: {spec.run_id}",
        )
        clean = float(value["clean_quantization_gpu_hours"])
        return clean, {
            "kind": "checkpoint_content_identical_clean_retime",
            "path": str(path),
            "sha256": common.sha256_file(path),
            "reported_gpu_hours_excluded": reported,
        }

    if spec.run_id in YAQA_Q06_A16_IDS:
        retime_path = YAQA_RETIME / "retime_result.json"
        retime = common.read_object(retime_path)
        _require(
            retime.get("status") == "succeeded"
            and retime.get("tensor_byte_identity") is True
            and retime.get("source_stage", {}).get("stage_id") == "YH-Q06-A16",
            "invalid YAQA clean Hessian retime",
        )
        receipt_path = Path(artifact["terminal"])
        receipt = common.read_object(receipt_path)
        clean_hessian = float(retime["clean_hessian_gpu_hours"])
        quant = float(receipt["quantization_gpu_hours_excluding_hessian"])
        clean = quant + clean_hessian / 3.0
        return clean, {
            "kind": "tensor_byte_identical_shared_hessian_clean_retime",
            "path": str(retime_path),
            "sha256": common.sha256_file(retime_path),
            "reported_gpu_hours_excluded": reported,
            "quantization_gpu_hours_excluding_hessian": quant,
            "clean_shared_hessian_gpu_hours": clean_hessian,
            "amortization_divisor": 3,
        }

    return reported, {
        "kind": "campaign_terminal",
        "path": artifact["terminal"],
        "sha256": artifact["terminal_sha256"],
    }


def _validate_suite(spec: common.EvalSpec) -> dict[str, Any]:
    suite_dir = common.OUTPUT_ROOT / "evals" / spec.eval_id
    suite_path = suite_dir / "suite_success.json"
    suite = common.read_object(suite_path)
    _require(
        suite.get("status") == "generation_succeeded_official_humaneval_pending"
        and suite.get("eval_id") == spec.eval_id
        and suite.get("method") == spec.method
        and suite.get("run_id") == spec.run_id
        and suite.get("model") == spec.model
        and suite.get("setting") == spec.setting
        and suite.get("reference_manifest_fingerprint")
        == EXPECTED_REFERENCE_FINGERPRINT,
        f"suite identity/status mismatch: {spec.eval_id}",
    )

    artifact = common.resolve_completed_artifact(spec)
    declared_artifact = suite.get("artifact", {})
    _require(
        declared_artifact.get("terminal_sha256") == artifact["terminal_sha256"]
        and declared_artifact.get("validation_sha256")
        == artifact["validation_sha256"],
        f"suite artifact binding mismatch: {spec.eval_id}",
    )

    quality_declared = suite.get("quality", {})
    quality = _read_bound(
        quality_declared.get("path", ""), quality_declared.get("sha256", "")
    )
    _require(
        quality.get("status") == "succeeded"
        and quality.get("eval_id") == spec.eval_id
        and quality.get("reference", {}).get("sha256")
        == common.load_reference_manifest()["references"][spec.model]["sha256"],
        f"quality identity mismatch: {spec.eval_id}",
    )
    quality_metrics = quality["metrics"]
    wiki = quality_metrics["wikitext2"]
    qa = quality_metrics["paper_qa"]
    _require(
        wiki.get("full_vocabulary_fp32") is True
        and wiki.get("kl_direction") == "KL(FP||quantized)"
        and qa.get("task_order") == list(common.PAPER_QA_TASKS)
        and set(qa.get("tasks", {})) == set(common.PAPER_QA_TASKS),
        f"quality protocol mismatch: {spec.eval_id}",
    )

    reasoning_metrics: dict[str, float] = {}
    for task in common.REASONING_TASKS:
        declared = suite.get("reasoning", {}).get(task, {})
        manifest = _read_bound(
            declared.get("manifest", ""), declared.get("manifest_sha256", "")
        )
        result = declared.get("result", {})
        _require(
            manifest.get("status") == "completed"
            and manifest.get("tasks") == [task]
            and manifest.get("generation", {}).get("protocol")
            == "realq_zero_shot_v1"
            and manifest.get("generation", {}).get("seed") == 1234
            and result.get("task") == task
            and result.get("num_examples") == common.REASONING_TASKS[task]["count"],
            f"reasoning protocol/coverage mismatch: {spec.eval_id}/{task}",
        )
        if task in {"gsm8k", "math_500"}:
            reasoning_metrics[task] = 100.0 * float(result["pass_at_1"])

    official_path = (
        suite_dir
        / "reasoning/humaneval_plus/official_eval/official_success.json"
    )
    official = common.read_object(official_path)
    _require(
        official.get("status") == "official_scored"
        and official.get("eval_id") == spec.eval_id
        and official.get("reference_manifest_fingerprint")
        == EXPECTED_REFERENCE_FINGERPRINT
        and official.get("base_total") == 164
        and official.get("plus_total") == 164,
        f"official HumanEval+ mismatch: {spec.eval_id}",
    )

    gpu_hours, accounting = _retime_gpu_hours(spec, artifact)
    numeric = (
        float(wiki["kl_raw"]),
        float(wiki["ppl"]),
        float(qa["acc_avg"]),
        reasoning_metrics["gsm8k"],
        reasoning_metrics["math_500"],
        100.0 * float(official["base_pass_at_1"]),
        100.0 * float(official["plus_pass_at_1"]),
        gpu_hours,
    )
    _require(
        all(math.isfinite(value) and value >= 0 for value in numeric),
        f"non-finite/negative release metric: {spec.eval_id}",
    )
    return {
        "comparison_id": COMPARISON_IDS[(spec.model, spec.setting)],
        "eval_id": spec.eval_id,
        "method": spec.method,
        "run_id": spec.run_id,
        "model": spec.model,
        "setting": spec.setting,
        "metrics": {
            "kl_raw": numeric[0],
            "ppl": numeric[1],
            "qa_avg": numeric[2],
            "gsm8k": numeric[3],
            "math_500": numeric[4],
            "humaneval_base": numeric[5],
            "humaneval_plus": numeric[6],
        },
        "quantization_gpu_hours": gpu_hours,
        "gpu_hour_accounting": accounting,
        "suite": str(suite_path),
        "suite_sha256": common.sha256_file(suite_path),
        "official": str(official_path),
        "official_sha256": common.sha256_file(official_path),
    }


def build(*, allow_incomplete: bool) -> dict[str, Any]:
    reference = common.load_reference_manifest()
    _require(
        reference.get("fingerprint") == EXPECTED_REFERENCE_FINGERPRINT,
        "reference fingerprint mismatch",
    )
    specs = common.iter_specs()
    _require(len(specs) == 55, "evaluation matrix is not 55 suites")
    rows = []
    missing = []
    for spec in specs:
        suite = common.OUTPUT_ROOT / "evals" / spec.eval_id / "suite_success.json"
        official = (
            common.OUTPUT_ROOT
            / "evals"
            / spec.eval_id
            / "reasoning/humaneval_plus/official_eval/official_success.json"
        )
        if not suite.is_file() or not official.is_file():
            missing.append(spec.eval_id)
            continue
        rows.append(_validate_suite(spec))
    if missing and not allow_incomplete:
        raise common.EvaluationError(
            f"release incomplete: {len(rows)}/55 complete; missing={missing}"
        )

    rows.sort(
        key=lambda row: (
            int(row["comparison_id"][1:]),
            {"efficientqat": 0, "turboboa": 1, "yaqa_wclip": 2}[row["method"]],
        )
    )
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        totals[row["method"]] += float(row["quantization_gpu_hours"])
        counts[row["method"]] += 1
    return {
        "schema_version": 1,
        "status": "complete" if not missing else "incomplete",
        "created_at": common.now(),
        "reference_manifest": str(common.REFERENCE_MANIFEST_PATH),
        "reference_manifest_sha256": common.sha256_file(
            common.REFERENCE_MANIFEST_PATH
        ),
        "reference_fingerprint": reference["fingerprint"],
        "expected_suite_count": 55,
        "completed_suite_count": len(rows),
        "missing_eval_ids": missing,
        "rows": rows,
        "gpu_hour_summary": {
            method: {
                "count": counts[method],
                "total": total,
                "mean": total / counts[method],
            }
            for method, total in sorted(totals.items())
        },
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": common.sha256_file(Path(__file__).resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        release = build(allow_incomplete=args.allow_incomplete)
        if args.output:
            output = Path(args.output).resolve()
            common.atomic_json(output, release)
        print(
            json.dumps(
                {
                    "status": release["status"],
                    "completed_suite_count": release["completed_suite_count"],
                    "missing_count": len(release["missing_eval_ids"]),
                    "gpu_hour_summary": release["gpu_hour_summary"],
                    "output": str(Path(args.output).resolve()) if args.output else None,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
