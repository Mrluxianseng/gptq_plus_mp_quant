#!/usr/bin/env python3
"""Bind and compare the completed SDPA-side and FA4-side layer-6 replays.

This is a CPU-only post-hoc analysis over immutable replay outcomes.  It
directly measures the trajectory effect with the same local implementation,
then measures the production-like SDPA-trajectory/math-VJP versus
FA4-trajectory/FA4-VJP endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.realq_fa4_q4_fa4side_tensor_replay_20260822 import common


OLD_ROOT = common.DATA_ROOT / "realq_fa4_q4_true_tensor_replay_20260822_v2"
NEW_ROOT = common.DATA_ROOT / "realq_fa4_q4_fa4side_tensor_replay_20260822_v1"
OUTPUT_ROOT = (
    common.DATA_ROOT / "realq_fa4_q4_fa4side_cross_analysis_20260822_v1"
)
OLD_PLAN_FINGERPRINT = common.SDPA_TRAJECTORY_PLAN_FINGERPRINT
NEW_PLAN_FINGERPRINT = (
    "b22c4ce9469172cf3590fe287da2c8229e01753a89aeb3bcda0924a24ff36399"
)
OLD_COMPARISON_SHA256 = (
    "54f5f25a6a9040754ff166c486eac17996d17b41ac58fb280930b17cd978d0f0"
)
NEW_COMPARISON_SHA256 = (
    "973a430b6e4208d6ae917307817dde4764b5ca692262a94195f5a4600c84062d"
)


def _load_json_bound(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or common.file_sha256(path) != expected_sha256:
        raise common.ReplayDiagnosticError(f"immutable JSON changed: {path}")
    return common.read_json(path)


def _load_outcome(
    root: Path, arm: str, plan_fingerprint: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    receipt_path = root / arm / "replay/replay_receipt.json"
    receipt = common.read_json(receipt_path)
    if (
        receipt.get("status") != "completed"
        or receipt.get("plan_fingerprint") != plan_fingerprint
        or receipt.get("arm") != arm
    ):
        raise common.ReplayDiagnosticError(f"replay receipt changed: {receipt_path}")
    record = receipt.get("records", {}).get("layer06/real_b4")
    if not isinstance(record, dict):
        raise common.ReplayDiagnosticError(f"real_b4 record missing: {receipt_path}")
    outcome_path = Path(str(record.get("path")))
    if (
        not outcome_path.is_file()
        or common.file_sha256(outcome_path) != record.get("serialization_sha256")
    ):
        raise common.ReplayDiagnosticError(f"outcome changed: {outcome_path}")
    outcome = torch.load(outcome_path, map_location="cpu", weights_only=True)
    if not isinstance(outcome, dict):
        raise common.ReplayDiagnosticError("outcome root is not a mapping")
    common.validate_outcome(outcome, batch=common.BATCH_SIZE)
    if (
        common.named_tensors_semantic_sha256(outcome)
        != record.get("semantic_sha256")
        or common.named_tensors_layout_sha256(outcome)
        != record.get("layout_semantic_sha256")
    ):
        raise common.ReplayDiagnosticError(f"outcome semantics changed: {outcome_path}")
    binding = {
        "arm": arm,
        "plan_fingerprint": plan_fingerprint,
        "receipt": str(receipt_path),
        "receipt_sha256": common.file_sha256(receipt_path),
        "outcome": str(outcome_path),
        "outcome_serialization_sha256": record["serialization_sha256"],
        "outcome_semantic_sha256": record["semantic_sha256"],
        "outcome_layout_semantic_sha256": record["layout_semantic_sha256"],
    }
    return outcome, binding


def _compare(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    fields = {
        field: common.full_tensor_metrics(left[field], right[field])
        for field in common.OUTPUT_FIELDS
    }
    samples = {
        str(sample): {
            "fields": {
                field: common.full_tensor_metrics(
                    left[field][sample : sample + 1],
                    right[field][sample : sample + 1],
                )
                for field in common.OUTPUT_FIELDS
            }
        }
        for sample in range(common.BATCH_SIZE)
    }
    return {
        "metrics_scope": "all tensor elements; aggregate and per sample",
        "fields": fields,
        "samples": samples,
    }


def main() -> int:
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise common.ReplayDiagnosticError(f"output root is not fresh: {OUTPUT_ROOT}")
    old_comparison_path = (
        OLD_ROOT / "comparison/comparison/comparison_receipt.json"
    )
    new_comparison_path = (
        NEW_ROOT / "comparison/comparison/comparison_receipt.json"
    )
    old_comparison = _load_json_bound(
        old_comparison_path, OLD_COMPARISON_SHA256
    )
    new_comparison = _load_json_bound(
        new_comparison_path, NEW_COMPARISON_SHA256
    )
    if (
        old_comparison.get("status") != "completed"
        or old_comparison.get("plan_fingerprint") != OLD_PLAN_FINGERPRINT
        or new_comparison.get("status") != "completed"
        or new_comparison.get("plan_fingerprint") != NEW_PLAN_FINGERPRINT
    ):
        raise common.ReplayDiagnosticError("comparison receipt contract changed")

    outcomes: dict[str, dict[str, torch.Tensor]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for name, root, arm, fingerprint in (
        ("sdpa_trajectory_math_vjp", OLD_ROOT, "math_sdpa", OLD_PLAN_FINGERPRINT),
        ("sdpa_trajectory_fa4_vjp", OLD_ROOT, "fa4_default", OLD_PLAN_FINGERPRINT),
        ("fa4_trajectory_math_vjp", NEW_ROOT, "math_sdpa", NEW_PLAN_FINGERPRINT),
        ("fa4_trajectory_fa4_vjp", NEW_ROOT, "fa4_default", NEW_PLAN_FINGERPRINT),
    ):
        outcomes[name], bindings[name] = _load_outcome(root, arm, fingerprint)

    comparisons: dict[str, Any] = {}
    for name, left_name, right_name, interpretation in (
        (
            "trajectory_effect_same_math_vjp",
            "sdpa_trajectory_math_vjp",
            "fa4_trajectory_math_vjp",
            "changes only captured Q/K/V/dO trajectory; local VJP is math-SDPA",
        ),
        (
            "trajectory_effect_same_fa4_vjp",
            "sdpa_trajectory_fa4_vjp",
            "fa4_trajectory_fa4_vjp",
            "changes only captured Q/K/V/dO trajectory; local VJP is FA4",
        ),
        (
            "production_backend_and_trajectory_endpoint",
            "sdpa_trajectory_math_vjp",
            "fa4_trajectory_fa4_vjp",
            "matches each production backend's own trajectory and local VJP",
        ),
    ):
        comparisons[name] = {
            "left": left_name,
            "right": right_name,
            "interpretation": interpretation,
            **_compare(outcomes[left_name], outcomes[right_name]),
        }

    receipt = {
        "schema_version": 1,
        "status": "completed",
        "kind": "qwen3_4b_layer6_cross_trajectory_actual_vjp_analysis",
        "source_file": str(Path(__file__).resolve()),
        "source_file_sha256": common.file_sha256(Path(__file__).resolve()),
        "old_comparison": {
            "path": str(old_comparison_path),
            "sha256": OLD_COMPARISON_SHA256,
            "plan_fingerprint": OLD_PLAN_FINGERPRINT,
        },
        "new_comparison": {
            "path": str(new_comparison_path),
            "sha256": NEW_COMPARISON_SHA256,
            "plan_fingerprint": NEW_PLAN_FINGERPRINT,
        },
        "source_trajectory_comparison": new_comparison["layers"]["6"][
            "source_trajectory_comparison"
        ],
        "outcomes": bindings,
        "comparisons": comparisons,
    }
    OUTPUT_ROOT.mkdir(parents=True)
    common.atomic_json(OUTPUT_ROOT / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": "completed",
                "receipt": str(OUTPUT_ROOT / "receipt.json"),
                "receipt_sha256": common.file_sha256(OUTPUT_ROOT / "receipt.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
