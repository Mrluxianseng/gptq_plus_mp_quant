from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import formal_merged40 as formal
from experiments.realq_fullmodel_retune_20260817 import quality_merged40 as quality
from experiments.realq_fullmodel_retune_20260817 import reasoning_merged40 as reasoning
from experiments.realq_fullmodel_retune_20260817 import selection_v4_subset28 as s
from experiments.realq_fullmodel_retune_20260817 import selection_merged40 as merged


def test_v4_subset_has_disjoint_28_row_coverage() -> None:
    keys = s._expected_keys()
    assert len(keys) == 28
    assert all(not config.startswith("qwen3-32b_") for _, config in keys)
    assert all(config not in s.Q4_EXTERNAL_CONFIGS for _, config in keys)
    assert {branch for branch, _ in keys} == {"full_block", "single_linear"}


def test_merged_selection_partition_is_exactly_40_rows() -> None:
    assert sum(spec["rows"] for spec in merged.SOURCE_SPECS.values()) == 40
    assert len(merged._expected_keys()) == 40


def test_merged_command_parser_is_strict() -> None:
    command = ["python", "-m", "realq.ptq", "--seed", "1", "--rotate", "true"]
    assert merged._flags(command) == {"--seed": "1", "--rotate": "true"}


def _formal_source(*, branch: str, q32: bool) -> list[str]:
    module = (
        "experiments.realq_fullmodel_retune_20260817.q32_memory_entry"
        if q32
        else "realq.ptq"
    )
    flags = {
        **merged.FROZEN_FLAGS,
        "--a_loss_ratio": "1",
        "--full_block_refresh": c.BRANCH_VALUES[branch],
        "--hessian_accum_bsz": "32" if q32 else "64",
        "--grad_lr": "0",
        "--skip_eval": "false",
        "--skip_kl_ppl_eval": "false",
        "--lm_eval": "false",
        "--reasoning_eval": "false",
        "--require_static_cache_hit": "true",
        "--require_reference_cache_hit": "true",
        "--output_dir": "<RUNTIME_OUTPUT_DIR>",
        "--exp": "tuning",
        "--save_qmodel_path": "",
        "--unrelated_guard": "preserve-me",
    }
    command = ["python", "-m", module]
    for flag, value in flags.items():
        command.extend((flag, value))
    return command


@pytest.mark.parametrize(
    ("branch", "q32", "config", "hessian_bsz"),
    [
        ("full_block", False, "qwen3-4b_w4a16", "64"),
        ("single_linear", True, "qwen3-32b_w4a16", "32"),
    ],
)
def test_merged_formal_command_preserves_source_protocol(
    branch: str, q32: bool, config: str, hessian_bsz: str
) -> None:
    source = _formal_source(branch=branch, q32=q32)
    row = {
        "branch": branch,
        "config": config,
        "selected_lr": 7.498942093324559e-6,
        "source_command": source,
        "source_command_audit": {"command_sha256": c._canonical_sha256(source)},
    }
    command = formal._formal_command(row, Path("/tmp/formal-output"))
    source_flags, flags = formal._flags(source), formal._flags(command)
    assert command[2] == source[2]
    assert flags["--hessian_accum_bsz"] == hessian_bsz
    assert flags["--a_loss_ratio"] == "1"
    assert flags["--full_block_refresh"] == c.BRANCH_VALUES[branch]
    assert flags["--unrelated_guard"] == source_flags["--unrelated_guard"]
    assert flags["--grad_lr"] == "7.4989420933245589e-06"
    assert flags["--skip_eval"] == "true"
    assert flags["--skip_kl_ppl_eval"] == "true"
    assert flags["--require_reference_cache_hit"] == "false"
    assert flags["--save_qmodel_path"].endswith("/checkpoint/quantized.pt")


def test_merged_formal_rejects_source_hash_drift() -> None:
    source = _formal_source(branch="full_block", q32=False)
    row = {
        "branch": "full_block",
        "config": "qwen3-4b_w4a16",
        "selected_lr": 1e-5,
        "source_command": source,
        "source_command_audit": {"command_sha256": "0" * 64},
    }
    with pytest.raises(c.CampaignError, match="source command hash mismatch"):
        formal._formal_command(row, Path("/tmp/formal-output"))


def test_merged_quality_and_reasoning_are_checkpoint_only() -> None:
    source = _formal_source(branch="single_linear", q32=True)
    row = {
        "branch": "single_linear",
        "config": "qwen3-32b_w4a16",
        "selected_lr": 1e-5,
        "source_command": source,
        "source_command_audit": {"command_sha256": c._canonical_sha256(source)},
    }
    formal_command = formal._formal_command(row, Path("/tmp/formal-output"))
    plan = {
        "rows": [
            {
                "branch": "single_linear",
                "config": "qwen3-32b_w4a16",
                "command": formal_command,
            }
        ]
    }
    quality_command = quality._quality_command(
        plan,
        "single_linear",
        "qwen3-32b_w4a16",
        Path("/tmp/quality-output"),
    )
    quality_flags = formal._flags(quality_command)
    assert quality_flags["--load_qmodel_path"].endswith("/checkpoint/quantized.pt")
    assert quality_flags["--lm_eval"] == "true"
    assert quality_flags["--skip_kl_ppl_eval"] == "false"
    assert "--save_qmodel_path" not in quality_flags

    reasoning_command = reasoning._reasoning_command(
        plan,
        "single_linear",
        "qwen3-32b_w4a16",
        "math_500",
        Path("/tmp/reasoning-output"),
    )
    reasoning_flags = formal._flags(reasoning_command)
    assert reasoning_flags["--load_qmodel_path"].endswith("/checkpoint/quantized.pt")
    assert reasoning_flags["--reasoning_tasks"] == "math_500"
    assert reasoning_flags["--reasoning_do_sample"] == "false"
    assert reasoning_flags["--reasoning_seed"] == "1234"
    assert reasoning_flags["--hessian_accum_bsz"] == "32"
    assert "--save_qmodel_path" not in reasoning_flags


def test_reasoning_balanced_matrix_covers_three_tasks_for_all_rows() -> None:
    works = reasoning._balanced_works()
    assert len(works) == 120
    assert len(
        {
            reasoning._work_id(work["branch"], work["config"], work["task"])
            for work in works
        }
    ) == 120
    assert {work["task"] for work in works} == set(reasoning.TASKS)


def test_evalplus_parser_requires_and_counts_all_164_tasks(tmp_path: Path) -> None:
    task_ids = {f"HumanEval/{index}" for index in range(164)}
    evaluations = {
        task_id: [
            {
                "base_status": "pass" if index < 100 else "fail",
                "plus_status": "pass" if index < 80 else "fail",
            }
        ]
        for index, task_id in enumerate(sorted(task_ids))
    }
    path = tmp_path / "evalplus_samples_eval_results.json"
    path.write_text(json.dumps({"eval": evaluations}), encoding="utf-8")
    metrics = reasoning._evalplus_metrics(path, task_ids)
    assert metrics["base_pass"] == 100
    assert metrics["plus_pass"] == 80
    assert metrics["base_total"] == metrics["plus_total"] == 164
