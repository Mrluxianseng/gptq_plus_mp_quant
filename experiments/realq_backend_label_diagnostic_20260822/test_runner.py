from __future__ import annotations

from experiments.realq_backend_label_diagnostic_20260822 import runner
from experiments.realq_allopts_loss_ablation_20260821 import (
    cache_backend_diagnostic as source,
)


def test_paired_commands_preserve_the_source_scientific_contract():
    _receipt, source_command = source._source_command()
    source_flags = source._flags(source_command)
    scheduling = {
        "--static_cache_path",
        "--cache_dir",
        "--output_dir",
        "--exp",
    }
    for arm in runner.ARMS:
        command = runner._command(arm)
        assert command[1:3] == [
            "-m",
            "experiments.realq_backend_label_diagnostic_20260822.capture_driver",
        ]
        flags = source._flags(command)
        assert {
            key for key in flags if flags[key] != source_flags[key]
        } == scheduling | ({"--attention_backend"} if arm == "sdpa" else set())
        assert flags["--attention_backend"] == runner.ARMS[arm]["backend"]
        assert flags["--global_loss_bsz"] == "4"
        assert flags["--grad_hessian_topk"] == "-1"
        assert flags["--static_cache_path"] == ""


def test_label_comparison_counts_tokens_and_changed_samples():
    left = [0] * (256 * 2048)
    right = list(left)
    right[7] = 3
    right[2048 + 9] = 4
    result = runner._compare_values(left, right)
    assert result["unequal_labels"] == 2
    assert result["unequal_fraction"] == 2 / (256 * 2048)
    assert result["samples_with_any_change"] == 2
    assert result["per_sample_unequal_counts"][:3] == [1, 1, 0]


def test_plan_pins_pair_to_distinct_idle_physical_gpus():
    plan = runner._build_plan()
    assert plan["hostname"] == runner.HOST
    assert plan["arms"]["sdpa"]["physical_gpu"] == 0
    assert plan["arms"]["fa4"]["physical_gpu"] == 2
    assert plan["paired_contract"]["capture_changes_loss"] is False
