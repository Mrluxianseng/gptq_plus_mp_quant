from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.additional_methods_fair20_eval_20260821 import common
from experiments.additional_methods_fair20_eval_20260821 import (
    recover_turboboa_llama as recovery,
)
from experiments.additional_methods_fair20_eval_20260821 import (
    recover_turboboa_llama_queue as queue,
)


def _spec(*, model: str = "llama31-8b-instruct"):
    return common.EvalSpec(
        method="turboboa",
        run_id="TB20-L8-W4A4",
        model=model,
        setting="W4A4KV4",
        w_bits=4,
        a_bits=4,
        k_bits=4,
        v_bits=4,
        quant_dir=Path("/tmp/not-used"),
    )


def _result_config():
    return {
        "llm_type": "Llama",
        "w_bits": 4,
        "group_size": 128,
        "w_sym": True,
        "w_asym": False,
        "w_clip": True,
        "w_method": "turboboa",
        "weight_quantizer": "realq_mse",
        "qparam_comput": "RealQ-MSE",
        "act_order_col": True,
        "act_order_row": False,
        "rotate": True,
        "rotation_seed": 0,
        "seed": 1,
        "global_seed": 1,
        "deterministic": True,
        "nsamples": 256,
        "seqlen": 2048,
        "a_bits": 4,
        "k_bits": 4,
        "v_bits": 4,
        "a_groupsize": -1,
        "k_groupsize": -1,
        "v_groupsize": -1,
        "a_asym": False,
        "k_asym": False,
        "v_asym": False,
        "a_clip_ratio": 0.9,
        "k_clip_ratio": 0.9,
        "v_clip_ratio": 0.9,
        "act_quant_aware_gptq": True,
        "k_cache_quant_aware_gptq": True,
        "qk_rmsnorm_hessian_mode": "mean_jacobian_kfac",
    }


def test_recovery_matrix_is_exactly_the_four_llama_turboboa_settings():
    assert recovery.ALLOWED_EVAL_IDS == (
        "turboboa__TB20-L8-W4A4",
        "turboboa__TB20-L8-W4",
        "turboboa__TB20-L8-W3",
        "turboboa__TB20-L8-W2",
    )


def test_quantization_contract_accepts_truthful_llama_alias(monkeypatch, tmp_path):
    path = tmp_path / "result.json"
    common.atomic_json(
        path,
        {
            "status": "quantized_only",
            "method": "TurboBoA",
            "configuration": _result_config(),
        },
    )
    artifact = {"validation": str(path)}
    result = recovery._validate_quantization_result(_spec(), artifact)
    assert result["configuration"]["w_method"] == "turboboa"


def test_quantization_contract_rejects_qwen_only_alias(monkeypatch, tmp_path):
    path = tmp_path / "result.json"
    cfg = _result_config()
    cfg["w_method"] = "turboboa_rmsnorm_mean_jacobian_kfac"
    common.atomic_json(
        path,
        {"status": "quantized_only", "method": "TurboBoA", "configuration": cfg},
    )
    with pytest.raises(recovery.RecoveryError, match="contract mismatch"):
        recovery._validate_quantization_result(_spec(), {"validation": str(path)})


def test_corrected_loader_has_one_literal_turboboa_method_expectation():
    tree = ast.parse(Path(recovery.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_turboboa_llama"
    )
    literals = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert literals.count("turboboa") == 1
    assert "turboboa_rmsnorm_mean_jacobian_kfac" not in literals


def test_corrected_loader_rejects_non_llama_before_importing_model(monkeypatch):
    with pytest.raises(recovery.RecoveryError, match="restricted"):
        recovery.load_turboboa_llama(
            _spec(model="qwen3-8b"),
            {},
            {},
            analyzer=SimpleNamespace(),
        )


def test_recovery_queue_environment_includes_audited_evalplus_runtime():
    environment = queue._runtime_environment(3)
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert str(queue.EVALPLUS_RUNTIME) in environment["PYTHONPATH"].split(":")


def test_recovery_queue_can_resume_only_its_dead_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    eval_id = recovery.ALLOWED_EVAL_IDS[0]
    claim = tmp_path / "evals" / eval_id / ".claim"
    claim.mkdir(parents=True)
    common.atomic_json(
        claim / "owner.json",
        {
            "eval_id": eval_id,
            "hostname": "j-test-master-0",
            "pid": 999_999_999,
            "physical_gpu": 3,
            "kind": "turboboa_llama_contract_recovery_queue",
            "claimed_at": "old",
        },
    )
    resumed = queue._claim(eval_id, "j-test-master-0", 3)
    assert resumed == claim
    owner = common.read_object(claim / "owner.json")
    assert owner["previous_pid"] == 999_999_999
    assert owner["pid"] > 0
