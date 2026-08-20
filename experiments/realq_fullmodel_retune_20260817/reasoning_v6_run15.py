#!/usr/bin/env python3
"""Three-task reasoning generation and official scoring for V6 run15."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import campaign_v6_run15 as v6
from experiments.realq_fullmodel_retune_20260817 import selection_v6_run15 as select
from experiments.realq_fullmodel_retune_20260817 import formal_v6_run15 as formal
from experiments.realq_fullmodel_retune_20260817 import reasoning_merged40 as core


REASONING_ID = "realq-fullmodel-two-branch-reasoning-20260820-v6-run15-sdpa-v2"
REASONING_PLAN_PATH = v6.OUTPUT_ROOT / "reasoning_plan.json"
REASONING_AUDIT_PATH = v6.OUTPUT_ROOT / "reasoning_final_audit.json"
_IMPLEMENTATION_PATH = Path(core.__file__).resolve()
_BASE_BUILD_PLAN = core._build_plan
_MERGE_ERROR = core.merged.MergeError
_FROZEN_FLAGS = {
    "--dataset": "wikitext2",
    "--eval_datasets": "wikitext2",
    "--seed": "1",
    "--rotation_seed": "0",
    "--refresh_seed": "0",
    "--nsamples": "256",
    "--seq_len": "2048",
    "--eval_seq_len": "2048",
    "--a_loss_clip_scope": "local_backward_chunk",
}
_SELECTION_SHIM = SimpleNamespace(
    OUTPUT_ROOT=v6.OUTPUT_ROOT,
    MERGE_ID=select.SELECTION_ID,
    FROZEN_FLAGS=_FROZEN_FLAGS,
    MergeError=_MERGE_ERROR,
)


def _code_snapshot() -> dict[str, Any]:
    files = [
        {"path": str(path), "sha256": c._file_sha256(path)}
        for path in (
            Path(__file__).resolve(),
            _IMPLEMENTATION_PATH,
            Path(formal.__file__).resolve(),
            Path(select.__file__).resolve(),
        )
    ]
    return {"files": files, "sha256": c._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    _activate()
    body = _BASE_BUILD_PLAN()
    body.pop("reasoning_plan_fingerprint", None)
    body["code"] = _code_snapshot()
    protocol = dict(body["protocol"])
    protocol.update(
        {
            "attention_backend": "sdpa",
            "attention_backend_scope": (
                "reasoning generation only; formal quantization remains "
                "flash_attention_4==4.0.0b25"
            ),
            "attention_backend_reason": (
                "lm-eval generation requires padded batches and KV-cache "
                "decoding; matches the prior 20-group reasoning protocol"
            ),
            "execution_environment_matches_v6_formal_except_attention_backend": True,
        }
    )
    body["protocol"] = protocol
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
