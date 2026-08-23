from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import pytest

from experiments.additional_methods_fair20_eval_20260821 import common, run_suite
from experiments.yaqa_q32_eval_fanout_20260822 import fanout


def test_schedule_covers_exactly_four_q32_yaqa_suites():
    assert set(fanout.SCHEDULE) == {
        "yaqa_wclip__YQ-Q32-W4A4",
        "yaqa_wclip__YQ-Q32-W4",
        "yaqa_wclip__YQ-Q32-W3",
        "yaqa_wclip__YQ-Q32-W2",
    }
    assert all(
        set(components) == set(fanout.COMPONENTS)
        for components in fanout.SCHEDULE.values()
    )
    for eval_id in fanout.SCHEDULE:
        spec = run_suite._find_spec(eval_id)
        assert spec.method == "yaqa_wclip"
        assert spec.model == "qwen3-32b"


def test_first_three_groups_are_disjoint_and_w2_reuses_three_w4a4_lanes():
    groups = {}
    for eval_id, components in fanout.SCHEDULE.items():
        groups[eval_id] = {
            (value["hostname"], value["physical_gpu"])
            for value in components.values()
        }
        assert len(groups[eval_id]) == 4
    first = [
        "yaqa_wclip__YQ-Q32-W4A4",
        "yaqa_wclip__YQ-Q32-W4",
        "yaqa_wclip__YQ-Q32-W3",
    ]
    assert all(
        not (groups[left] & groups[right])
        for left, right in combinations(first, 2)
    )
    w2 = groups["yaqa_wclip__YQ-Q32-W2"]
    w4a4 = groups["yaqa_wclip__YQ-Q32-W4A4"]
    assert len(w2 & w4a4) == 3
    assert (fanout.HOST0, 2) in w2


def test_reasoning_component_configs_are_the_frozen_protocol():
    artifact = {"kind": "hf_directory", "path": "/checkpoint"}
    for eval_id in fanout.SCHEDULE:
        spec = fanout._spec(eval_id)
        for task, protocol in common.REASONING_TASKS.items():
            cfg = run_suite._reasoning_config(
                spec, artifact, task, fanout.Path("/tmp/output")
            )
            assert cfg.reasoning_tasks == [task]
            assert cfg.reasoning_seed == 1234
            assert cfg.reasoning_protocol == "realq_zero_shot_v1"
            assert cfg.reasoning_do_sample is False
            assert cfg.reasoning_batch_size == protocol["batch_size"]
            assert cfg.reasoning_max_new_tokens == protocol["max_new_tokens"]


def test_plan_binds_frozen_reference_and_yaqa_plan():
    plan = fanout._build_plan()
    assert plan["reference_manifest"]["fingerprint"] == (
        fanout.EXPECTED_REFERENCE_FINGERPRINT
    )
    assert plan["yaqa_plan"]["sha256"] == common.EXPECTED_PLAN_SHA256[
        "yaqa_wclip"
    ]
    assert plan["protocol"]["numerical_contract_changed"] is False


def test_runtime_assignment_requires_exact_host_and_visible_gpu():
    eval_id = "yaqa_wclip__YQ-Q32-W4"
    component = "gsm8k"
    plan = {"schedule": fanout.SCHEDULE}
    assignment = fanout.SCHEDULE[eval_id][component]
    runtime = {
        "hostname": assignment["hostname"],
        "cuda_visible_devices": str(assignment["physical_gpu"]),
    }
    fanout._assert_runtime_assignment(
        runtime=runtime,
        eval_id=eval_id,
        component=component,
        physical_gpu=assignment["physical_gpu"],
        plan=plan,
    )
    with pytest.raises(fanout.FanoutError, match="assignment changed"):
        fanout._assert_runtime_assignment(
            runtime={**runtime, "cuda_visible_devices": "7"},
            eval_id=eval_id,
            component=component,
            physical_gpu=assignment["physical_gpu"],
            plan=plan,
        )


def test_component_attempt_treats_valid_atomic_receipt_as_terminal(
    monkeypatch, tmp_path
):
    eval_id = "yaqa_wclip__YQ-Q32-W4"
    component = "gsm8k"
    root = tmp_path / "component"
    monkeypatch.setattr(
        fanout,
        "_component_dir",
        lambda actual_eval_id, actual_component: root,
    )
    monkeypatch.setattr(fanout, "_worker_environment", lambda _gpu: {})

    def fake_run(*_args, **_kwargs):
        (root / "receipt.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(fanout.subprocess, "run", fake_run)
    monkeypatch.setattr(
        fanout,
        "_load_component_receipt",
        lambda actual_eval_id, actual_component, _plan: {
            "eval_id": actual_eval_id,
            "component": actual_component,
        },
    )
    succeeded, result = fanout._run_component_attempt(
        eval_id=eval_id,
        component=component,
        physical_gpu=3,
        python="python",
        plan={"plan_fingerprint": "test"},
    )
    assert succeeded is True
    assert result["status"] == "succeeded"
    assert result["returncode"] == 7
    assert result["receipt_valid"] is True
    assert (root / "attempts/attempt001/result.json").is_file()


def test_assembled_payload_keeps_canonical_suite_schema(monkeypatch, tmp_path):
    eval_id = "yaqa_wclip__YQ-Q32-W4"
    spec = fanout._spec(eval_id)
    monkeypatch.setattr(fanout.common, "OUTPUT_ROOT", tmp_path)
    quality_path = tmp_path / "evals" / eval_id / "quality_result.json"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text("{}", encoding="utf-8")
    payload = fanout._assemble_payload(
        spec=spec,
        artifact={"terminal_sha256": "a", "validation_sha256": "b"},
        reference_manifest={"fingerprint": fanout.EXPECTED_REFERENCE_FINGERPRINT},
        quality_receipt={"runtime": {"seed": 1234}},
        quality={"metrics": {"wikitext2": {}, "paper_qa": {}}},
        reasoning={task: {} for task in common.REASONING_TASKS},
        fanout_record={"kind": "test"},
    )
    assert payload["schema_version"] == 1
    assert payload["status"] == "generation_succeeded_official_humaneval_pending"
    assert payload["eval_id"] == eval_id
    assert payload["runtime"]["seed"] == 1234
    assert set(payload["reasoning"]) == set(common.REASONING_TASKS)
