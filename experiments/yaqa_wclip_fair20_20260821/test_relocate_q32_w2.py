from __future__ import annotations

import pytest

from experiments.yaqa_wclip_fair20_20260821 import relocate_q32_w2 as relocate
from experiments.yaqa_wclip_fair20_20260821 import worker


def test_frozen_contract_selects_exact_q32_w2_stage():
    plan, _source, stage, previous, _preflight = relocate._load_contract()
    assert worker.sha256_file(relocate.PLAN_PATH) == relocate.EXPECTED_PLAN_SHA256
    assert stage["stage_id"] == "YQ-Q32-W2"
    assert stage["w_bits"] == 2
    assert stage["physical_gpu"] == relocate.SOURCE_GPU
    assert previous["stage_id"] == "YQ-Q32-W3"
    assert previous["queue_order"] == 0
    assert plan["stages"].index(previous) < plan["stages"].index(stage)


def test_redirected_base_environment_changes_only_source_gpu():
    calls = []

    def original(gpu):
        calls.append(gpu)
        return {"CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)}

    redirected = relocate._redirected_base_env(original)
    assert redirected(None)["CUDA_VISIBLE_DEVICES"] == ""
    assert redirected(relocate.SOURCE_GPU)["CUDA_VISIBLE_DEVICES"] == str(
        relocate.TARGET_GPU
    )
    assert calls == [None, relocate.TARGET_GPU]
    with pytest.raises(relocate.RelocationError, match="unexpected numerical"):
        redirected(7)


def test_patch_metadata_preserves_frozen_stage(monkeypatch, tmp_path):
    _plan, _source, stage, _previous, _preflight = relocate._load_contract()
    path = tmp_path / "receipt.json"
    worker.write_json_atomic(
        path,
        {
            "status": "succeeded",
            "hostname": relocate.TARGET_HOST,
            "physical_gpu": relocate.SOURCE_GPU,
            "stage": stage,
        },
    )
    scheduling = {"kind": relocate.KIND, "scheduling_only": True}
    value = relocate._patch_execution_metadata(
        path,
        stage=stage,
        relocation=scheduling,
    )
    assert value["stage"] == stage
    assert value["physical_gpu"] == relocate.TARGET_GPU
    assert value["scheduling_relocation"] == scheduling


def test_handoff_rejects_any_numerical_contract_change():
    plan, _source, stage, _previous, _preflight = relocate._load_contract()
    dependency_path = (
        relocate.Path(plan["output_root"])
        / "stages"
        / stage["hessian_id"]
        / "stage_receipt.json"
    )
    handoff = {
        "status": "prepared",
        "kind": relocate.KIND,
        "plan_sha256": relocate.EXPECTED_PLAN_SHA256,
        "stage": stage,
        "scheduling_only": True,
        "numerical_contract_changed": False,
        "helper_source_sha256": worker.sha256_file(relocate.Path(relocate.__file__)),
        "source": {
            "hostname": relocate.SOURCE_HOST,
            "physical_gpu": relocate.SOURCE_GPU,
            "queue_parent": {"state": "T (stopped)"},
        },
        "target": {
            "hostname": relocate.TARGET_HOST,
            "physical_gpu": relocate.TARGET_GPU,
        },
        "dependency": {
            "receipt": str(dependency_path),
            "receipt_sha256": worker.sha256_file(dependency_path),
        },
    }
    relocate._validate_handoff(handoff, plan=plan, stage=stage)
    handoff["numerical_contract_changed"] = True
    with pytest.raises(relocate.RelocationError, match="contract mismatch"):
        relocate._validate_handoff(handoff, plan=plan, stage=stage)
