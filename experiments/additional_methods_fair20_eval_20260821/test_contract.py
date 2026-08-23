from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import common, run_suite, worker


def test_frozen_matrix_has_exact_expected_cells():
    specs = common.iter_specs()
    by_method = {
        method: [spec for spec in specs if spec.method == method]
        for method in common.EXPECTED_PLAN_SHA256
    }
    assert {method: len(rows) for method, rows in by_method.items()} == {
        "efficientqat": 15,
        "turboboa": 20,
        "yaqa_wclip": 20,
    }
    assert all(spec.setting != "W4A4KV4" for spec in by_method["efficientqat"])
    for method in ("turboboa", "yaqa_wclip"):
        assert {
            (spec.model, spec.setting) for spec in by_method[method]
        } == {
            (model, setting)
            for model in common.MODEL_ORDER
            for setting in common.SETTING_ORDER
        }


def test_every_quant_lane_has_dependencies():
    plans = common.load_quant_plans()
    hosts = {
        run["canoe_pod"]
        for run in plans["turboboa"]["runs"]
    } | {
        stage["canoe_pod"]
        for stage in plans["yaqa_wclip"]["stages"]
    }
    for host in hosts:
        for gpu in range(8):
            dependencies = worker._lane_quant_dependencies(host, gpu, plans)
            assert dependencies, (host, gpu)
            assert len(dependencies) == len(set(dependencies))


def test_quality_gate_preserves_explicit_task_order(tmp_path: Path):
    spec = next(spec for spec in common.iter_specs() if spec.method == "turboboa")
    reference = "a" * 64
    tasks = {task: 50.0 for task in reversed(common.PAPER_QA_TASKS)}
    payload = {
        "schema_version": 1,
        "status": "succeeded",
        "eval_id": spec.eval_id,
        "reference": {"sha256": reference},
        "metrics": {
            "wikitext2": {"kl_raw": 0.1, "ppl": 10.0},
            "paper_qa": {
                "task_order": list(common.PAPER_QA_TASKS),
                "tasks": tasks,
                "acc_avg": 50.0,
            },
        },
    }
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    assert run_suite._validate_quality_result(path, spec, reference) == payload
    payload["metrics"]["paper_qa"]["task_order"] = sorted(common.PAPER_QA_TASKS)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(common.EvaluationError):
        run_suite._validate_quality_result(path, spec, reference)


def _reasoning_payload(task: str, status: str) -> dict:
    protocol = common.REASONING_TASKS[task]
    return {
        "schema_version": 1,
        "status": status,
        "tasks": [task],
        "generation": {
            "protocol": "realq_zero_shot_v1",
            "seed": 1234,
            "do_sample": False,
            "batch_size": protocol["batch_size"],
            "max_new_tokens": protocol["max_new_tokens"],
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "apply_chat_template": True,
            "enable_thinking": True,
        },
        "results": (
            [{"task": task, "num_examples": protocol["count"]}]
            if status == "completed"
            else []
        ),
    }


@pytest.mark.parametrize("status", ["running", "failed", "completed"])
def test_reasoning_contract_allows_controlled_resume(tmp_path: Path, status: str):
    task = "gsm8k"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_reasoning_payload(task, status)), encoding="utf-8")
    partial = run_suite._validate_reasoning_contract(
        path, task, require_completed=False
    )
    assert partial["status"] == status
    if status == "completed":
        run_suite._validate_reasoning_manifest(path, task)
    else:
        with pytest.raises(common.EvaluationError):
            run_suite._validate_reasoning_manifest(path, task)


def test_source_release_set_is_unique_and_present():
    assert len(common.EVALUATION_SOURCE_RELATIVE) == len(
        set(common.EVALUATION_SOURCE_RELATIVE)
    )
    assert tuple(
        str(path.relative_to(common.REPO_ROOT))
        for path in common.evaluation_source_files()
    ) == common.EVALUATION_SOURCE_RELATIVE


def test_reference_stat_gate_is_portable_across_mount_device_ids(tmp_path: Path):
    cache = tmp_path / "reference.cache"
    cache.write_bytes(b"frozen-reference")
    released = common.reference_stat_identity(cache)
    released["device"] += 1
    assert common.reference_stat_matches(cache, released)
    released["inode"] += 1
    assert not common.reference_stat_matches(cache, released)
