from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from realq.config import Config as DenseConfig  # noqa: E402
from realq_moe import model_adapter  # noqa: E402
from realq_moe.config import Config as MoeConfig  # noqa: E402
from realq_moe.precompute.static_e2e import (  # noqa: E402
    _validate_teacher_coverage,
)


_OPTIMIZED_DEFAULTS = {
    "quantizer_inner_fastpath": True,
    "prepared_clamp_bound_cache": True,
    "triton_column_block": True,
    "w_clip_search_impl": "symmetric_union_exact",
    "fisher_fp32_cache": True,
    "act_order_stitch_impl": "prefix_q_trailing_w_exact",
    "w_clip_update_impl": "where_out",
    "w_group_param_layout": "compact",
}


class _FakeSelfAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _FakeExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _FakeSparseMlp(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                _FakeExpert(hidden_size, intermediate_size)
                for _ in range(num_experts)
            ]
        )


class _FakeDenseMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _FakeSparseLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 4,
        intermediate_size: int = 3,
        num_experts: int = 2,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.self_attn = _FakeSelfAttention(hidden_size)
        self.mlp = _FakeSparseMlp(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )


class _FakeDenseLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 4,
        intermediate_size: int = 3,
    ) -> None:
        super().__init__()
        self.self_attn = _FakeSelfAttention(hidden_size)
        self.mlp = _FakeDenseMlp(hidden_size, intermediate_size)


def _fake_analyzer(layers: list[nn.Module]):
    return SimpleNamespace(
        model_arch="Qwen3MoeForCausalLM",
        config=SimpleNamespace(
            hidden_size=4,
            moe_intermediate_size=3,
        ),
        get_layers=lambda: layers,
    )


def test_inherited_performance_switches_default_to_optimized() -> None:
    config = MoeConfig()

    assert {
        name: getattr(config, name) for name in _OPTIMIZED_DEFAULTS
    } == _OPTIMIZED_DEFAULTS


@pytest.mark.parametrize(
    ("name", "candidate"),
    (
        ("w_clip_search_impl", "cartesian_legacy"),
        ("fisher_fp32_cache", False),
        ("act_order_stitch_impl", "full_weight_legacy"),
        ("w_clip_update_impl", "guarded"),
        ("w_group_param_layout", "expanded"),
        ("triton_column_block", False),
    ),
)
def test_legacy_performance_overrides_remain_selectable(
    name: str,
    candidate: object,
) -> None:
    assert getattr(MoeConfig(**{name: candidate}), name) == candidate


def test_p01_and_p10_must_be_disabled_together() -> None:
    config = MoeConfig(
        quantizer_inner_fastpath=False,
        prepared_clamp_bound_cache=False,
    )
    assert config.quantizer_inner_fastpath is False
    with pytest.raises(ValueError, match="requires"):
        MoeConfig(quantizer_inner_fastpath=False)


def test_moe_cold_joint_gpu_and_full_slide_defaults_are_locked() -> None:
    config = MoeConfig()

    assert config.moe_fail_on_teacher_cold is True
    assert config.moe_fail_on_student_cold is True
    assert config.moe_zero_route_fallback == "rtn"
    assert config.moe_gpu_resident is True
    assert config.moe_joint_column_block is True
    assert config.moe_expert_loss_slide_window is True
    assert config.moe_min_expert_assignments > 0
    assert config.moe_min_expert_unique_tokens > 0
    assert config.moe_min_expert_unique_samples > 0


@pytest.mark.parametrize(
    ("name", "disabled_value"),
    (
        ("moe_fail_on_teacher_cold", False),
        ("moe_fail_on_student_cold", False),
        ("moe_gpu_resident", False),
        ("moe_joint_column_block", False),
        ("moe_expert_loss_slide_window", False),
    ),
)
def test_moe_cold_joint_gpu_or_full_slide_policy_cannot_be_disabled(
    name: str,
    disabled_value: bool,
) -> None:
    with pytest.raises(ValueError, match=name):
        MoeConfig(**{name: disabled_value})


@pytest.mark.parametrize(
    "name",
    (
        "moe_min_expert_assignments",
        "moe_min_expert_unique_tokens",
        "moe_min_expert_unique_samples",
    ),
)
def test_moe_cold_threshold_cannot_be_disabled_with_zero(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        MoeConfig(**{name: 0})


def test_moe_zero_route_fallback_is_locked_to_rtn() -> None:
    with pytest.raises(ValueError, match="moe_zero_route_fallback"):
        MoeConfig(moe_zero_route_fallback="gptq")


def test_teacher_zero_route_uses_rtn_but_positive_undercoverage_fails() -> None:
    warm = {
        "assignment_count": torch.tensor([2, 3]),
        "unique_token_count": torch.tensor([2, 3]),
        "unique_sample_count": torch.tensor([1, 2]),
    }
    later_cold = {
        "assignment_count": torch.tensor([0, 4]),
        "unique_token_count": torch.tensor([0, 4]),
        "unique_sample_count": torch.tensor([0, 2]),
    }

    # Exact zero is the explicit RTN exception, including in a layer that is
    # actually quantized.
    _validate_teacher_coverage(
        MoeConfig(),
        [warm, later_cold],
        source="test",
    )

    positive_undercoverage = {
        "assignment_count": torch.tensor([1, 4]),
        "unique_token_count": torch.tensor([1, 4]),
        "unique_sample_count": torch.tensor([1, 2]),
    }
    with pytest.raises(RuntimeError, match=r"layer=1.*experts=\{0:"):
        _validate_teacher_coverage(
            MoeConfig(moe_min_expert_assignments=2),
            [warm, positive_undercoverage],
            source="test",
        )


def test_teacher_positive_undercoverage_after_stop_layer_is_ignored() -> None:
    warm = {
        "assignment_count": torch.tensor([2, 3]),
        "unique_token_count": torch.tensor([2, 3]),
        "unique_sample_count": torch.tensor([1, 2]),
    }
    later_undercovered = {
        "assignment_count": torch.tensor([1, 4]),
        "unique_token_count": torch.tensor([1, 4]),
        "unique_sample_count": torch.tensor([1, 2]),
    }
    _validate_teacher_coverage(
        MoeConfig(
            quant_stop_layer=0,
            moe_min_expert_assignments=2,
        ),
        [warm, later_undercovered],
        source="test",
    )


def test_dense_runtime_keeps_inherited_cpu_master_contract() -> None:
    config = MoeConfig(
        fsdp=True,
        cpu_master=True,
        loss_slide_window=False,
    )

    # Architecture-aware validation is intentionally a no-op for D0/D1 dense
    # analyzers; only sparse MoE formal runs reject the host-master lifecycle.
    config.validate_moe_runtime_contract(sparse_moe=False)


@pytest.mark.parametrize(
    "overrides",
    (
        {"fsdp": True},
        {"fsdp_cpu_offload": True},
        {"fsdp": True, "cpu_master": True},
    ),
)
def test_sparse_runtime_rejects_every_cpu_or_fsdp_reload_path(
    overrides: dict[str, object],
) -> None:
    config = MoeConfig(**overrides)

    with pytest.raises(ValueError, match="fully GPU-resident"):
        config.validate_moe_runtime_contract(sparse_moe=True)


def test_sparse_runtime_requires_the_inherited_full_slide_switch() -> None:
    config = MoeConfig(loss_slide_window=False)

    with pytest.raises(ValueError, match="loss_slide_window=True"):
        config.validate_moe_runtime_contract(sparse_moe=True)


def test_default_experiment_name_is_isolated_from_dense_realq() -> None:
    moe = MoeConfig()
    dense = DenseConfig()

    assert moe.exp == "realq_moe"
    assert dense.exp == "realq"
    assert moe.exp != dense.exp
    assert type(moe).__module__ == "realq_moe.config"
    assert type(dense).__module__ == "realq.config"


def test_all_sparse_qwen3_analyzer_is_accepted_and_described() -> None:
    layers = [_FakeSparseLayer(), _FakeSparseLayer()]
    analyzer = _fake_analyzer(layers)

    model_adapter.validate_analyzer(analyzer)

    for layer in layers:
        assert model_adapter.is_sparse_moe_layer(layer)
        assert model_adapter.describe_layer(layer) == model_adapter.LayerPlan(
            is_sparse=True,
            num_experts=2,
            top_k=1,
            router_path="mlp.gate",
        )
        assert model_adapter.group_order(layer) == ("attn_in", "attn_out")


def test_sparse_paths_exclude_router_and_keep_up_gate_down_order() -> None:
    layer = _FakeSparseLayer()
    expected_expert_entries = (
        (0, "up_proj", "mlp.experts.0.up_proj"),
        (0, "gate_proj", "mlp.experts.0.gate_proj"),
        (0, "down_proj", "mlp.experts.0.down_proj"),
        (1, "up_proj", "mlp.experts.1.up_proj"),
        (1, "gate_proj", "mlp.experts.1.gate_proj"),
        (1, "down_proj", "mlp.experts.1.down_proj"),
    )

    assert tuple(
        model_adapter.iter_expert_projection_paths(layer)
    ) == expected_expert_entries

    quantizable_paths = model_adapter.quantizable_linear_paths(layer)
    assert quantizable_paths[:4] == (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    assert quantizable_paths[4:] == tuple(
        path for _, _, path in expected_expert_entries
    )
    assert "mlp.gate" in dict(layer.named_modules())
    assert "mlp.gate" not in quantizable_paths
    assert (
        model_adapter.resolve_module(layer, "mlp.gate")
        is layer.mlp.gate
    )

    for expert_idx, projection, path in expected_expert_entries:
        assert (
            model_adapter.resolve_linear(layer, path)
            is getattr(layer.mlp.experts[expert_idx], projection)
        )

    with pytest.raises(KeyError, match="sparse"):
        model_adapter.group_paths(layer, "mlp_in")


def test_mixed_dense_sparse_qwen3_analyzer_fails_closed() -> None:
    analyzer = _fake_analyzer([_FakeSparseLayer(), _FakeDenseLayer()])

    with pytest.raises(NotImplementedError, match="Mixed dense/sparse") as exc:
        model_adapter.validate_analyzer(analyzer)

    assert "all-sparse Qwen3-30B-A3B" in str(exc.value)
    assert "dense layers=[1]" in str(exc.value)
