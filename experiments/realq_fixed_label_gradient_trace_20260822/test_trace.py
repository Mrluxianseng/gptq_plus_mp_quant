from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.realq_allopts_loss_ablation_20260821 import (
    cache_backend_diagnostic as source,
)
from experiments.realq_fixed_label_gradient_trace_20260822 import (
    runner,
    trace_driver,
)


def test_signed_trace_commands_preserve_full_stage0_contract():
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
            "experiments.realq_fixed_label_gradient_trace_20260822.trace_driver",
        ]
        flags = source._flags(command)
        changed = {key for key in flags if flags[key] != source_flags.get(key)}
        expected = {
            key for key in scheduling if flags[key] != source_flags.get(key)
        }
        if runner.ARMS[arm]["backend"] != "flash_attention_4":
            expected.add("--attention_backend")
        assert changed == expected
        assert flags["--global_loss_bsz"] == "4"
        assert flags["--nsamples"] == "256"
        assert flags["--static_cache_path"] == ""
    left = source._flags(commands["trace_sdpa"])
    right = source._flags(commands["trace_fa4"])
    scientific = {
        key for key in left if left[key] != right[key] and key not in scheduling
    }
    assert scientific == {"--attention_backend"}


def test_tensor_sampler_uses_fixed_token_and_feature_coordinates():
    value = torch.arange(2 * 2048 * 65, dtype=torch.float32).reshape(2, 2048, 65)
    sample, metadata = trace_driver._sample_tensor(value)
    assert sample.shape == (2, len(trace_driver.TOKEN_INDICES), 64)
    assert metadata["token_indices"] == list(trace_driver.TOKEN_INDICES)
    assert metadata["feature_indices"][0] == 0
    assert metadata["feature_indices"][-1] == 64
    assert torch.equal(sample[:, 0, 0], value[:, 0, 0])
    with pytest.raises(trace_driver.GradientTraceError, match="sequence length"):
        trace_driver._sample_tensor(torch.empty(2, 32, 65))


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()

    def forward(self, value):
        attention = self.self_attn
        mixed = attention.q_proj(value) + attention.k_proj(value) + attention.v_proj(value)
        return attention.o_proj(mixed)


def test_trace_manager_captures_all_signed_forward_and_gradient_sites():
    layers = nn.ModuleList([_Layer() for _ in range(36)])
    analyzer = SimpleNamespace(get_layers=lambda: list(layers))
    manager = trace_driver.TraceManager(analyzer)
    manager.attach()
    value = torch.randn(1, 2048, 4, requires_grad=True)
    for layer in layers:
        value = layer(value)
    value.sum().backward()
    manager.remove()
    manager.validate()
    assert len(manager.samples) == 36 * 6 * 2
    assert manager.samples["layer00/q_proj_output/forward"].shape == (
        1,
        len(trace_driver.TOKEN_INDICES),
        4,
    )


def test_tensor_metrics_report_exact_relative_error_and_cosine():
    left = torch.tensor([1.0, 2.0, 3.0])
    right = torch.tensor([1.0, 4.0, 3.0])
    metrics = runner._tensor_metrics(left, right)
    assert metrics["sampled_values"] == 3
    assert metrics["unequal_values"] == 1
    assert metrics["diff_rms"] == pytest.approx((4.0 / 3.0) ** 0.5)
    assert 0 < metrics["cosine"] < 1


def test_plan_binds_fixed_label_diagnostic_and_two_idle_gpus():
    plan = runner._build_plan()
    assert plan["fixed_label_plan"]["fingerprint"] == (
        "3c68518f685b8e2d319e4de8873c102c06d7927c008605102f1adb39a54d65a2"
    )
    assert plan["paired_contract"]["numerical_path_changed"] is False
    assert {definition["physical_gpu"] for definition in plan["arms"].values()} == {
        0,
        2,
    }
