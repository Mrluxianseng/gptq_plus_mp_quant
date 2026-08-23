from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.turboboa_fair20_20260821 import worker


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "experiments/turboboa_fair20_20260821/plan.json"


def _plans():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    source_path = Path(plan["source_plan"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == plan[
        "source_plan_sha256"
    ]
    return plan, source


def test_plan_is_exact_five_by_four_matrix_with_contiguous_gpu_queues():
    plan, source = _plans()
    expected = {
        (model, setting)
        for model in source["models"]
        for setting in (
            "W4A4KV4",
            "W4A16KV16",
            "W3A16KV16",
            "W2A16KV16",
        )
    }
    assert len(plan["runs"]) == 20
    assert {(run["model"], run["setting"]) for run in plan["runs"]} == expected
    queues = {}
    for run in plan["runs"]:
        queues.setdefault((run["canoe_pod"], run["physical_gpu"]), []).append(
            run["queue_order"]
        )
    assert all(sorted(orders) == list(range(len(orders))) for orders in queues.values())


@pytest.mark.parametrize("setting,aware", [("W4A4KV4", True), ("W3A16KV16", False)])
def test_worker_command_freezes_shared_quantizer_tokens_seeds_and_akv(setting, aware, tmp_path):
    plan, source = _plans()
    run = next(run for run in plan["runs"] if run["setting"] == setting)
    command = worker.command_for_run(plan, source, run, tmp_path)
    joined = " ".join(command)
    assert "--weight_quantizer realq_mse" in joined
    assert "--qparam_comput RealQ-MSE" in joined
    assert "--group_size 128" in joined
    assert "--calib_tokens_file" in command
    assert "--calib_tokens_sha256" in command
    assert "--seed 1" in joined
    assert "--global_seed 1" in joined
    assert "--deterministic" in command
    assert "--rotate --rotation_seed 0" in joined
    assert "--save_qmodel_path" in command
    assert "--skip_eval" in command
    if aware:
        assert "--a_bits 4" in joined
        assert "--a_clip_ratio 0.9" in joined
        assert "--act_quant_aware_gptq" in command
        assert "--k_cache_quant_aware_gptq" in command
    else:
        assert "--a_bits 16" in joined
        assert "--a_clip_ratio 1.0" in joined
        assert "--no-act_quant_aware_gptq" in command
        assert "--no-k_cache_quant_aware_gptq" in command


def test_result_gate_requires_complete_deterministic_contract():
    plan, source = _plans()
    run = next(run for run in plan["runs"] if run["setting"] == "W4A4KV4")
    token = source["calibrations"][run["calibration"]]["token_artifact"]
    config = {
        "w_bits": 4,
        "group_size": 128,
        "w_sym": True,
        "weight_quantizer": "realq_mse",
        "qparam_comput": "RealQ-MSE",
        "block_v": True,
        "n_quant_rows": 16,
        "consider_dX": True,
        "alpha": 0.25,
        "adaptive_qparam": True,
        "refine_qparam": True,
        "n_iters": 1,
        "act_order_col": True,
        "act_order_row": False,
        "rotate": True,
        "rotation_seed": 0,
        "seed": 1,
        "global_seed": 1,
        "deterministic": True,
        "nsamples": 256,
        "seqlen": 2048,
        "calib_tokens_file": str(Path(token["path"]).resolve()),
        "calib_tokens_sha256": token["sha256"],
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
        "quant_stop_layer": None,
        "skip_eval": True,
        "determinism_contract": {
            "enabled": True,
            "global_seed": 1,
            "python_hash_seed": 1,
            "python_random_seed": 1,
            "numpy_random_seed": 1,
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
        },
    }
    payload = {
        "schema_version": 1,
        "status": "quantized_only",
        "metrics": None,
        "configuration": config,
    }
    worker.validate_result(payload, run, source)
    config["determinism_contract"] = dict(config["determinism_contract"])
    del config["determinism_contract"]["numpy_random_seed"]
    with pytest.raises(worker.WorkerError, match="determinism_contract"):
        worker.validate_result(payload, run, source)
