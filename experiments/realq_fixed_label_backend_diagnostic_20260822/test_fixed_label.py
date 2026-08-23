from __future__ import annotations

import torch

from experiments.realq_allopts_loss_ablation_20260821 import (
    cache_backend_diagnostic as source,
)
from experiments.realq_fixed_label_backend_diagnostic_20260822 import (
    fixed_label_driver,
    runner,
)


def test_paired_commands_only_change_backend_and_diagnostic_outputs():
    _receipt, source_command = source._source_command()
    source_flags = source._flags(source_command)
    scheduling = {
        "--static_cache_path",
        "--cache_dir",
        "--output_dir",
        "--exp",
        "--skip_eval",
        "--skip_kl_ppl_eval",
        "--lm_eval",
        "--reasoning_eval",
        "--require_static_cache_hit",
    }
    commands = {arm: runner._command(arm) for arm in runner.ARMS}
    for arm, command in commands.items():
        assert command[1:3] == [
            "-m",
            (
                "experiments.realq_fixed_label_backend_diagnostic_20260822."
                "fixed_label_driver"
            ),
        ]
        flags = source._flags(command)
        changed = {key for key in flags if flags[key] != source_flags.get(key)}
        expected = {
            key for key in scheduling if flags[key] != source_flags.get(key)
        }
        if runner.ARMS[arm]["backend"] != "flash_attention_4":
            expected.add("--attention_backend")
        assert changed == expected
        assert flags["--attention_backend"] == runner.ARMS[arm]["backend"]
        assert flags["--global_loss_bsz"] == "4"
        assert flags["--grad_hessian_topk"] == "-1"
    left = source._flags(commands["fixed_sdpa"])
    right = source._flags(commands["fixed_fa4"])
    scientific = {
        key for key in left if left[key] != right[key] and key not in scheduling
    }
    assert scientific == {"--attention_backend"}


def test_select_labels_returns_exact_rows_and_rejects_call_drift():
    fixed = torch.arange(8 * 2048, dtype=torch.long).reshape(8, 2048)
    record = {
        "global_sample_indices": [4, 5, 6, 7],
        "shape": [4, 2048],
        "base_seed": 0,
        "offset_labels": 4 * 2048,
    }
    logits = torch.empty((4, 2048, 3))
    selected = fixed_label_driver._select_labels(
        fixed,
        record,
        logits,
        [4, 5, 6, 7],
        base_seed=0,
    )
    assert torch.equal(selected, fixed[4:8])
    try:
        fixed_label_driver._select_labels(
            fixed,
            record,
            logits,
            [0, 1, 2, 3],
            base_seed=0,
        )
    except fixed_label_driver.FixedLabelError:
        pass
    else:
        raise AssertionError("changed global sample indices were accepted")


def test_plan_binds_native_sdpa_labels_and_disjoint_idle_gpus():
    plan = runner._build_plan()
    assert plan["fixed_labels"]["sha256"] == (
        "ae23e11d9096e7370ea7db3913a940f800510d8e06e2393a7841814358015741"
    )
    assert plan["paired_contract"]["production_label_sampler_called"] is False
    assert {
        definition["physical_gpu"] for definition in plan["arms"].values()
    } == {0, 2}
