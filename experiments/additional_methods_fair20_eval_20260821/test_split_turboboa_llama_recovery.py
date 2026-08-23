from __future__ import annotations

import json

import pytest

from experiments.additional_methods_fair20_eval_20260821 import common
from experiments.additional_methods_fair20_eval_20260821 import (
    split_turboboa_llama_recovery as split,
)
from experiments.additional_methods_fair20_eval_20260821.recover_turboboa_llama import (
    ALLOWED_EVAL_IDS,
)


def _owner(eval_id: str) -> dict[str, object]:
    return {
        "eval_id": eval_id,
        "hostname": "j-test-master-0",
        "pid": 12345,
        "physical_gpu": 3,
        "kind": "turboboa_llama_contract_recovery_queue",
    }


def test_split_matrix_excludes_only_completed_first_suite():
    assert split.SPLIT_IDS == ALLOWED_EVAL_IDS[1:]


def test_validated_owner_requires_exact_original_queue_identity():
    eval_id = split.SPLIT_IDS[0]
    split._validated_owner(
        _owner(eval_id),
        eval_id=eval_id,
        hostname="j-test-master-0",
        expected_queue_pid=12345,
    )
    for field, value in (
        ("eval_id", "unexpected"),
        ("hostname", "j-other-master-0"),
        ("pid", 99999),
        ("kind", "unexpected"),
    ):
        owner = _owner(eval_id)
        owner[field] = value
        with pytest.raises(split.SplitRecoveryError, match="owner mismatch"):
            split._validated_owner(
                owner,
                eval_id=eval_id,
                hostname="j-test-master-0",
                expected_queue_pid=12345,
            )


def test_dead_queue_gate_rejects_invalid_or_live_pid(monkeypatch):
    with pytest.raises(split.SplitRecoveryError, match="greater than one"):
        split._validate_dead_queue(1)
    monkeypatch.setattr(split, "_process_exists", lambda pid: pid == 12345)
    with pytest.raises(split.SplitRecoveryError, match="still live"):
        split._validate_dead_queue(12345)
    split._validate_dead_queue(54321)


def test_validate_suite_requires_recovery_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    eval_id = ALLOWED_EVAL_IDS[0]
    path = tmp_path / "evals" / eval_id / "suite_success.json"
    path.parent.mkdir(parents=True)
    value = {
        "eval_id": eval_id,
        "status": "generation_succeeded_official_humaneval_pending",
        "contract_recovery": {
            "kind": "turboboa_llama_manifest_alias_only"
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    terminal = split._validate_suite(eval_id)
    assert terminal["path"] == str(path.resolve())
    assert len(terminal["sha256"]) == 64

    value["contract_recovery"]["kind"] = "unexpected"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(split.SplitRecoveryError, match="invalid recovered suite"):
        split._validate_suite(eval_id)
