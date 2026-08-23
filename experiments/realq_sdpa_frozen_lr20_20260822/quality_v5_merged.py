#!/usr/bin/env python3
"""KL/PPL and ten-task QA for the final merged SDPA 40-checkpoint release."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_fullmodel_retune_20260817 import quality_v6_run15 as immutable
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as formal


core = immutable.core
QUALITY_ID = "realq-sdpa-frozen-run15-lr20-quality-20260822-v5-merged"
QUALITY_PLAN_PATH = campaign.OUTPUT_ROOT / "quality_plan_v5_merged.json"
QUALITY_AUDIT_PATH = campaign.OUTPUT_ROOT / "quality_final_audit_v5_merged.json"
_IMPLEMENTATION_PATH = immutable._IMPLEMENTATION_PATH
_BASE_BUILD_PLAN = immutable._BASE_BUILD_PLAN
_MERGE_ERROR = immutable._MERGE_ERROR
_FROZEN_FLAGS = dict(immutable._FROZEN_FLAGS)
_SELECTION_SHIM = SimpleNamespace(
    OUTPUT_ROOT=campaign.OUTPUT_ROOT,
    MERGE_ID=formal.MERGE_ID,
    FROZEN_FLAGS=_FROZEN_FLAGS,
    MergeError=_MERGE_ERROR,
)


def _code_snapshot() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        _IMPLEMENTATION_PATH,
        Path(formal.__file__).resolve(),
        Path(campaign.__file__).resolve(),
    )
    files = [{"path": str(path), "sha256": base._file_sha256(path)} for path in paths]
    return {"files": files, "sha256": base._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    _activate()
    body = _BASE_BUILD_PLAN()
    body.pop("quality_plan_fingerprint", None)
    body["code"] = _code_snapshot()
    body["protocol"] = {
        **body["protocol"],
        "attention_backend": "deterministic math-SDPA",
        "execution_environment_matches_formal": True,
        "formal_sources": "32 V2b non-Q32 + 8 exact tiled-SDPA Q32",
        "formal_calibration_tokens_are_exact_gptaq_guidedquant_files": True,
        "reference_logits_are_exact_gptaq_guidedquant_files": True,
        "quality_reference_cache_regeneration_forbidden": True,
    }
    body["quality_plan_fingerprint"] = base._canonical_sha256(body)
    return body


def _activate() -> None:
    formal._activate()
    core.formal = formal
    core.merged = _SELECTION_SHIM
    core.QUALITY_ID = QUALITY_ID
    core.QUALITY_PLAN_PATH = QUALITY_PLAN_PATH
    core.QUALITY_AUDIT_PATH = QUALITY_AUDIT_PATH
    core._build_plan = _build_plan


_activate()


def __getattr__(name: str) -> Any:
    return getattr(core, name)


def main(argv: Sequence[str] | None = None) -> int:
    _activate()
    return int(core.main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
