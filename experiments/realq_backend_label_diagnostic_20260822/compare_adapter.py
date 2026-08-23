#!/usr/bin/env python3
"""Finish the frozen paired-label comparison through a Path-only adapter.

The executed runner passed a string to an otherwise correct SHA helper whose
interface accepts ``Path``.  Both capture terminals were already successful;
this adapter binds their hashes and converts only that path-like argument at
comparison time.  It does not read model tensors or alter captured labels.
"""

from __future__ import annotations

import json
from pathlib import Path
import traceback
from typing import Any, Callable

from experiments.realq_backend_label_diagnostic_20260822 import runner
from experiments.realq_fullmodel_retune_20260817 import campaign as base


EXPECTED_RUNNER_SHA256 = (
    "740ebbb676632b6476c2d416bbaa7523de0042efe97cf33a9483b858b672871c"
)
EXPECTED_PLAN_SHA256 = (
    "c55587e1e2ee84a93889bd440709c5e9f2c3675d70af84395771e0ffe8e7eddc"
)
EXPECTED_PLAN_FINGERPRINT = (
    "d0287668fb41fe6f93e5a1bfc14b30a2e96ad7656ad7b6ea37e436dc663e0ddf"
)
EXPECTED_RESULTS = {
    "sdpa": "9634096ecf8bb50a13b07bed6a8460b05c87016d62bc7bf7912c86d20116d59b",
    "fa4": "d571ee6ca7a89eba6a4da6a85ce7a306b94237d5ff1d1ff9f3fa3aa3d302a83f",
}


class ComparisonAdapterError(RuntimeError):
    """The completed captures do not match the immutable adapter contract."""


def _path_adapter(
    original: Callable[[Path], str],
) -> Callable[[str | Path], str]:
    def adapted(value: str | Path) -> str:
        return original(value if isinstance(value, Path) else Path(value))

    return adapted


def execute() -> dict[str, Any]:
    source = Path(runner.__file__).resolve()
    if base._file_sha256(source) != EXPECTED_RUNNER_SHA256:
        raise ComparisonAdapterError("executed runner source SHA256 changed")
    if base._file_sha256(runner.PLAN_PATH) != EXPECTED_PLAN_SHA256:
        raise ComparisonAdapterError("frozen plan SHA256 changed")
    plan = json.loads(runner.PLAN_PATH.read_text(encoding="utf-8"))
    if plan.get("plan_fingerprint") != EXPECTED_PLAN_FINGERPRINT:
        raise ComparisonAdapterError("frozen plan fingerprint changed")
    for arm, expected in EXPECTED_RESULTS.items():
        path = runner.OUTPUT_ROOT / arm / "result.json"
        if base._file_sha256(path) != expected:
            raise ComparisonAdapterError(f"capture result changed: {arm}")
    comparison_path = runner.OUTPUT_ROOT / "comparison.json"
    receipt_path = runner.OUTPUT_ROOT / "comparison_adapter_receipt.json"
    if comparison_path.exists() or receipt_path.exists():
        raise ComparisonAdapterError("comparison terminal is not fresh")

    original = base._file_sha256
    base._file_sha256 = _path_adapter(original)
    try:
        comparison = runner.compare()
    finally:
        base._file_sha256 = original
    if (
        comparison.get("status") != "succeeded"
        or comparison.get("plan_fingerprint") != EXPECTED_PLAN_FINGERPRINT
        or comparison.get("metrics", {}).get("total_labels") != 256 * 2048
    ):
        raise ComparisonAdapterError("comparison terminal validation failed")
    receipt = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": "path_type_only_comparison_adapter",
        "numerical_contract_changed": False,
        "reason": "runner SHA helper received str instead of Path",
        "runner": str(source),
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "plan": str(runner.PLAN_PATH),
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "plan_fingerprint": EXPECTED_PLAN_FINGERPRINT,
        "capture_result_sha256": EXPECTED_RESULTS,
        "comparison": str(comparison_path),
        "comparison_sha256": original(comparison_path),
        "adapter_source": str(Path(__file__).resolve()),
        "adapter_source_sha256": original(Path(__file__).resolve()),
        "finished_at": base._utc_now(),
    }
    base._atomic_json(receipt_path, receipt)
    return {"comparison": comparison, "adapter_receipt": receipt}


def main() -> int:
    try:
        print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
