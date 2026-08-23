from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiments.yaqa_wclip_fair20_20260821 import worker


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "experiments/yaqa_wclip_fair20_20260821/plan.json"


def _plans():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    source_path = Path(plan["source_plan"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == plan[
        "source_plan_sha256"
    ]
    return plan, source


def test_plan_contains_exact_hessian_and_quantization_dag():
    plan, source = _plans()
    stages = plan["stages"]
    hessians = [stage for stage in stages if stage["kind"] == "hessian"]
    quantizations = [stage for stage in stages if stage["kind"] == "quantize"]
    assert len(stages) == 30
    assert len(hessians) == 10
    assert len(quantizations) == 20
    assert {(stage["model"], stage["hessian_mode"]) for stage in hessians} == {
        (model, mode) for model in source["models"] for mode in ("a16", "a4")
    }
    assert {(stage["model"], stage["setting"]) for stage in quantizations} == {
        (model, setting)
        for model in source["models"]
        for setting in (
            "W4A4KV4",
            "W4A16KV16",
            "W3A16KV16",
            "W2A16KV16",
        )
    }
    hessian_by_id = {stage["stage_id"]: stage for stage in hessians}
    for stage in quantizations:
        dependency = hessian_by_id[stage["hessian_id"]]
        assert dependency["model"] == stage["model"]
        assert dependency["calibration"] == stage["calibration"]
        assert dependency["hessian_mode"] == (
            "a4" if stage["setting"] == "W4A4KV4" else "a16"
        )


def test_gpu_queues_are_contiguous_and_q06_a16_uses_reserved_node0_gpu7():
    plan, _ = _plans()
    queues = defaultdict(list)
    for stage in plan["stages"]:
        queues[(stage["canoe_pod"], stage["physical_gpu"])].append(
            stage["queue_order"]
        )
    assert all(sorted(values) == list(range(len(values))) for values in queues.values())
    q06 = next(stage for stage in plan["stages"] if stage["stage_id"] == "YH-Q06-A16")
    assert q06["canoe_pod"] == "j-4mj21jb084-master-0"
    assert q06["physical_gpu"] == 7


def test_worker_environment_hides_or_selects_exactly_one_physical_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unexpected")
    gpu = worker.base_env(7)
    cpu = worker.base_env(None)
    assert gpu["CUDA_VISIBLE_DEVICES"] == "7"
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert str(ROOT / "YAQA_wclip") in gpu["PYTHONPATH"]
