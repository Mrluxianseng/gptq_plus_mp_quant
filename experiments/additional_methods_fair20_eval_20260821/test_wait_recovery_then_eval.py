from __future__ import annotations

import json

import pytest

from experiments.additional_methods_fair20_eval_20260821 import common
from experiments.additional_methods_fair20_eval_20260821 import (
    wait_recovery_then_eval as delayed,
)
from experiments.additional_methods_fair20_eval_20260821.recover_turboboa_llama import (
    ALLOWED_EVAL_IDS,
)


def _write_terminal(root, eval_id: str, *, kind: str) -> None:
    path = root / "evals" / eval_id / "suite_success.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "eval_id": eval_id,
                "status": "generation_succeeded_official_humaneval_pending",
                "contract_recovery": {"kind": kind},
            }
        ),
        encoding="utf-8",
    )


def test_recovery_terminals_wait_until_all_four(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    assert delayed._recovery_terminals() is None
    for eval_id in ALLOWED_EVAL_IDS[:-1]:
        _write_terminal(
            tmp_path,
            eval_id,
            kind="turboboa_llama_manifest_alias_only",
        )
    assert delayed._recovery_terminals() is None
    _write_terminal(
        tmp_path,
        ALLOWED_EVAL_IDS[-1],
        kind="turboboa_llama_manifest_alias_only",
    )
    terminals = delayed._recovery_terminals()
    assert terminals is not None
    assert [value["eval_id"] for value in terminals] == list(ALLOWED_EVAL_IDS)
    assert all(len(value["sha256"]) == 64 for value in terminals)


def test_recovery_terminals_reject_wrong_recovery_kind(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    _write_terminal(tmp_path, ALLOWED_EVAL_IDS[0], kind="unexpected")
    with pytest.raises(delayed.SupervisorError, match="invalid recovery terminal"):
        delayed._recovery_terminals()
