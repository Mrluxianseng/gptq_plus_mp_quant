#!/usr/bin/env python3
"""Three-task reasoning generation/scoring for SDPA V2b checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import reasoning_v6_run15 as immutable
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v2 as formal
from experiments.realq_sdpa_frozen_lr20_20260822 import selection_v2 as selection


core = immutable.core
REASONING_ID = "realq-sdpa-frozen-run15-lr20-reasoning-20260822-v2b-memory"
REASONING_PLAN_PATH = campaign.OUTPUT_ROOT / "reasoning_plan_v2b.json"
REASONING_AUDIT_PATH = campaign.OUTPUT_ROOT / "reasoning_final_audit_v2b.json"
_IMPLEMENTATION_PATH = immutable._IMPLEMENTATION_PATH
_BASE_BUILD_PLAN = immutable._BASE_BUILD_PLAN
_MERGE_ERROR = immutable._MERGE_ERROR
_FROZEN_FLAGS = dict(immutable._FROZEN_FLAGS)
_SELECTION_SHIM = SimpleNamespace(
    OUTPUT_ROOT=campaign.OUTPUT_ROOT,
    MERGE_ID=selection.SELECTION_ID,
    FROZEN_FLAGS=_FROZEN_FLAGS,
    MergeError=_MERGE_ERROR,
)


def _code_snapshot() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        _IMPLEMENTATION_PATH,
        Path(formal.__file__).resolve(),
        Path(selection.__file__).resolve(),
        Path(campaign.__file__).resolve(),
    )
    files = [{"path": str(path), "sha256": c._file_sha256(path)} for path in paths]
    return {"files": files, "sha256": c._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    _activate()
    body = _BASE_BUILD_PLAN()
    body.pop("reasoning_plan_fingerprint", None)
    body["code"] = _code_snapshot()
    body["protocol"] = {
        **body["protocol"],
        "attention_backend": "sdpa",
        "attention_backend_scope": "formal quantization and reasoning generation",
        "execution_environment_matches_v2b_formal": True,
        "formal_calibration_tokens_are_exact_gptaq_guidedquant_files": True,
    }
    body["reasoning_plan_fingerprint"] = c._canonical_sha256(body)
    return body


def _activate() -> None:
    formal._activate()
    core.formal = formal
    core.merged = _SELECTION_SHIM
    core.REASONING_ID = REASONING_ID
    core.REASONING_PLAN_PATH = REASONING_PLAN_PATH
    core.REASONING_AUDIT_PATH = REASONING_AUDIT_PATH
    core._build_plan = _build_plan


_activate()


def __getattr__(name: str) -> Any:
    return getattr(core, name)


def main(argv: Sequence[str] | None = None) -> int:
    _activate()
    return int(core.main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
