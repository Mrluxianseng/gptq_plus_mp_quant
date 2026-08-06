from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch.func import functional_call  # noqa: E402

from realq_moe.precompute.routed_stats import PackedLayerRoutes  # noqa: E402
from realq_moe.refresh.block_gd import RefreshContext  # noqa: E402
from realq_moe.refresh.fixed_route_moe import (  # noqa: E402
    build_fixed_route_moe_cache,
    evaluate_fixed_route_moe,
)
from realq_moe.refresh.joint_moe_block_gd import (  # noqa: E402
    make_joint_moe_block_gd_refresh_fn,
)
from realq_moe.runner.streams import LayerInputs  # noqa: E402
from utils import hadamard_utils  # noqa: E402
from utils.quant_utils import ActQuantWrapper  # noqa: E402


@pytest.fixture(autouse=True)
def _disable_tf32_for_exact_fp32_oracles():
    """Compare algorithms, not CUDA's shape-dependent TF32 kernel choices."""

    if not torch.cuda.is_available():
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class _Scheduler:
    def __init__(self, count: int) -> None:
        self.count = count
        self.calls = 0

    def next_indices(self) -> list[int]:
        self.calls += 1
        return list(range(self.count))


class _NativeExpert(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.gate_proj = torch.nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class _NativeSparseMlp(torch.nn.Module):
    """Tiny version of the native Qwen3 expert loop and accumulation order."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = True
        self.gate = torch.nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.experts = torch.nn.ModuleList(
            [
                _NativeExpert(
                    hidden_size,
                    intermediate_size,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_experts)
            ]
        )
        self.router_calls = 0
        self.last_selected: torch.Tensor | None = None
        self.last_weights: torch.Tensor | None = None
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.weight[0, 0] = 1.0
            self.gate.weight[1, 1] = 1.0
            self.gate.weight[2, 0] = -1.0
            self.gate.weight[2, 1] = -1.0
            # Every fixture token has hidden[2] == 1. Expert 3 is therefore
            # strictly below all other experts and intentionally receives no
            # assignment; routes among 0/1/2 remain highly imbalanced.
            self.gate.weight[3, 2] = -100.0

    def _route(
        self,
        flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.router_calls += 1
        logits = self.gate(flat)
        weights = F.softmax(logits, dim=1, dtype=torch.float)
        weights, selected = torch.topk(weights, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(flat.dtype)
        self.last_selected = selected.detach()
        self.last_weights = weights.detach()
        return logits, weights, selected

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        logits, routing_weights, selected_experts = self._route(flat)
        final = torch.zeros_like(flat)
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.num_experts,
        ).permute(2, 1, 0)
        for expert_idx, expert in enumerate(self.experts):
            topk_slot, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current = flat[None, token_idx].reshape(-1, shape[-1])
            contribution = (
                expert(current)
                * routing_weights[token_idx, topk_slot, None]
            )
            # This expert-ascending accumulation order is the numerical oracle.
            final.index_add_(
                0,
                token_idx,
                contribution.to(flat.dtype),
            )
        return final.reshape(shape), logits


class _NativeLayer(torch.nn.Module):
    def __init__(self, mlp: _NativeSparseMlp) -> None:
        super().__init__()
        self.mlp = mlp
        self.forward_calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor]:
        self.forward_calls += 1
        return (hidden_states + self.mlp(hidden_states)[0],)


def _inputs(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = torch.tensor(
        [
            [
                [3.0, 0.0, 1.0, 0.2],
                [2.0, 1.0, 1.0, -0.1],
                [1.0, 2.0, 1.0, 0.3],
                [-2.0, -2.0, 1.0, -0.2],
            ],
            [
                [4.0, 0.0, 1.0, 0.1],
                [3.0, 1.0, 1.0, -0.3],
                [2.0, 0.0, 1.0, 0.4],
                [-3.0, -2.0, 1.0, -0.4],
            ],
            [
                [0.0, 4.0, 1.0, 0.5],
                [1.0, 3.0, 1.0, -0.5],
                [0.0, 2.0, 1.0, 0.6],
                [-2.0, -3.0, 1.0, -0.6],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return F.pad(base, (0, 12))


def _pack_routes(
    selected: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_experts: int,
) -> PackedLayerRoutes:
    expert_mask = F.one_hot(
        selected,
        num_classes=num_experts,
    ).permute(2, 1, 0)
    flat_parts: list[torch.Tensor] = []
    slot_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    offsets = [0]
    for expert_idx in range(num_experts):
        topk_slot, token_idx = torch.where(expert_mask[expert_idx])
        flat_parts.append(token_idx.to(torch.int32))
        slot_parts.append(topk_slot.to(torch.uint8))
        weight_parts.append(weights[token_idx, topk_slot])
        offsets.append(offsets[-1] + int(token_idx.numel()))
    return PackedLayerRoutes(
        expert_offsets=torch.tensor(
            offsets,
            dtype=torch.int64,
            device=selected.device,
        ),
        flat_token_indices=torch.cat(flat_parts, dim=0),
        topk_slots=torch.cat(slot_parts, dim=0),
        route_weights=torch.cat(weight_parts, dim=0),
        _offset_values=tuple(offsets),
    )


def _make_fixture(
    dtype: torch.dtype,
    *,
    wrapped_down_hadamard: bool = False,
) -> tuple[_NativeSparseMlp, torch.Tensor, PackedLayerRoutes]:
    device = torch.device("cuda")
    torch.manual_seed(1907)
    mlp = _NativeSparseMlp(
        hidden_size=16,
        intermediate_size=24,
        num_experts=4,
        top_k=2,
        device=device,
        dtype=dtype,
    ).eval()
    mlp.requires_grad_(False)
    if wrapped_down_hadamard:
        for expert in mlp.experts:
            wrapper = ActQuantWrapper(expert.down_proj).to(
                device=device,
                dtype=dtype,
            )
            had_k, k = hadamard_utils.get_hadK(
                wrapper.module.in_features
            )
            wrapper.online_full_had = True
            wrapper.had_K = (
                None
                if had_k is None
                else had_k.to(device=device, dtype=dtype)
            )
            wrapper.K = k
            wrapper.fp32_had = False
            expert.down_proj = wrapper
    inputs = _inputs(device, dtype)
    with torch.no_grad():
        mlp(inputs)
    routes = _pack_routes(
        mlp.last_selected,
        mlp.last_weights,
        num_experts=mlp.num_experts,
    )
    assert routes[3]["flat_token_indices"].numel() == 0
    return mlp, inputs, routes


def _target_linears(
    mlp: _NativeSparseMlp,
    projection: str,
) -> tuple[torch.nn.Linear, ...]:
    return tuple(
        getattr(
            getattr(expert, projection),
            "module",
            getattr(expert, projection),
        )
        for expert in mlp.experts
    )


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    return (6e-2, 4e-2) if dtype == torch.bfloat16 else (5e-4, 2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("projection", ["up_proj", "gate_proj", "down_proj"])
@pytest.mark.parametrize("wrapped_down_hadamard", [False, True])
def test_packed_grouped_mm_output_and_all_expert_gradient_match_native_oracle(
    projection: str,
    wrapped_down_hadamard: bool,
) -> None:
    dtype = torch.bfloat16
    mlp, inputs, routes = _make_fixture(
        dtype,
        wrapped_down_hadamard=wrapped_down_hadamard,
    )
    target_linears = _target_linears(mlp, projection)
    router_calls_before_cache = mlp.router_calls
    cache = build_fixed_route_moe_cache(
        student_routes=routes,
        projection=projection,
        expert_modules=mlp.experts,
        target_linears=target_linears,
        mlp_inputs=inputs,
        seq_len=inputs.shape[1],
    )
    # Cache construction touches projections only, never the current router.
    assert mlp.router_calls == router_calls_before_cache

    selected_samples = torch.tensor([2, 0], device="cuda", dtype=torch.int64)
    candidate = torch.stack(
        [linear.weight.detach() for linear in target_linears],
        dim=0,
    ).clone().requires_grad_(True)
    fixed = evaluate_fixed_route_moe(
        cache,
        candidate,
        selected_samples,
    )
    # Fixed replay itself must remain router-free.
    assert mlp.router_calls == router_calls_before_cache

    oracle_candidate = candidate.detach().clone().requires_grad_(True)
    oracle_names = {
        (
            f"experts.{expert_idx}.{projection}.module.weight"
            if isinstance(
                getattr(mlp.experts[expert_idx], projection),
                ActQuantWrapper,
            )
            else f"experts.{expert_idx}.{projection}.weight"
        ): (
            oracle_candidate[expert_idx]
        )
        for expert_idx in range(mlp.num_experts)
    }
    oracle_output = functional_call(
        mlp,
        oracle_names,
        (inputs.index_select(0, selected_samples),),
        strict=False,
    )[0]
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(
        fixed.hidden_states,
        oracle_output,
        rtol=rtol,
        atol=atol,
    )

    target = torch.randn_like(oracle_output)
    fixed_loss = (
        fixed.hidden_states.float() - target.float()
    ).square().mean()
    oracle_loss = (
        oracle_output.float() - target.float()
    ).square().mean()
    fixed_grad = torch.autograd.grad(fixed_loss, candidate)[0]
    oracle_grad = torch.autograd.grad(
        oracle_loss,
        oracle_candidate,
    )[0]
    torch.testing.assert_close(
        fixed_grad,
        oracle_grad,
        rtol=rtol,
        atol=atol,
    )

    # The intentionally cold expert remains represented by the single batched
    # leaf and receives an exact zero gradient.
    torch.testing.assert_close(
        fixed_grad[3],
        torch.zeros_like(fixed_grad[3]),
        rtol=0,
        atol=0,
    )
    assert cache.padding.padding_efficiency == 1.0
    assert cache.padding.padded_rows == 0
    assert cache.padding.packed_rows == routes.assignment_count
    assert fixed.padding.active_assignments.item() == (
        selected_samples.numel() * inputs.shape[1] * mlp.top_k
    )
    assert fixed.padding.padded_rows == 0
    assert fixed.padding.bmm_rows_per_expert == 0
    assert fixed.padding.grouped_mm_rows == (
        selected_samples.numel() * inputs.shape[1] * mlp.top_k
    )
    torch.testing.assert_close(
        fixed.padding.padding_efficiency,
        torch.ones_like(fixed.padding.padding_efficiency),
    )
    for tensor in (
        cache.token_indices,
        cache.route_weights,
        cache.assignment_counts,
        cache.assignment_starts,
        cache.target_inputs,
        cache.fixed_values,
        fixed.padding.active_assignments,
        fixed.padding.padding_efficiency,
    ):
        assert tensor.is_cuda


def _psd_fisher(hidden_size: int, device: torch.device) -> torch.Tensor:
    factor = torch.randn(
        hidden_size,
        hidden_size,
        device=device,
    )
    return factor.transpose(0, 1) @ factor / hidden_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_joint_fixed_current_skips_router_but_slide_naturally_reroutes() -> None:
    dtype = torch.bfloat16
    current_mlp, inputs, routes = _make_fixture(dtype)
    next_mlp, _, _ = _make_fixture(dtype)
    current_layer = _NativeLayer(current_mlp).to("cuda").eval()
    next_layer = _NativeLayer(next_mlp).to("cuda").eval()
    current_layer.requires_grad_(False)
    next_layer.requires_grad_(False)
    state = LayerInputs(
        inps=inputs,
        fp_inps=inputs.clone(),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    with torch.no_grad():
        fp_current = (
            current_layer(inputs)[0] + 0.01 * torch.randn_like(inputs)
        )
        fp_next = (
            next_layer(fp_current)[0] + 0.01 * torch.randn_like(inputs)
        )
    current_mlp.router_calls = 0
    next_mlp.router_calls = 0
    current_layer.forward_calls = 0
    next_layer.forward_calls = 0

    projection = "down_proj"
    target_linears = _target_linears(current_mlp, projection)
    candidates = torch.stack(
        [
            linear.weight.detach().float()
            + 0.02 * torch.randn_like(linear.weight).float()
            for linear in target_linears
        ],
        dim=0,
    )
    scheduler = _Scheduler(inputs.shape[0])
    contexts = tuple(
        RefreshContext(
            module=linear,
            layer_lr=0.01,
            grad_clip=0.0,
            backward_bsz=inputs.shape[0],
            scheduler=scheduler,
        )
        for linear in target_linears
    )
    refresh = make_joint_moe_block_gd_refresh_fn(
        layer=current_layer,
        target_linears=target_linears,
        layer_state=state,
        fp_out_for_this_layer=fp_current,
        fisher=_psd_fisher(inputs.shape[-1], inputs.device),
        contexts=contexts,
        current_mlp_inputs=inputs,
        current_mlp_residuals=inputs,
        student_routes=routes,
        projection=projection,
        expert_modules=current_mlp.experts,
        next_layer=next_layer,
        next_fp_out=fp_next,
        next_fisher=_psd_fisher(inputs.shape[-1], inputs.device),
        slide_alpha_fn=lambda: 0.5,
    )
    assert current_mlp.router_calls == 0
    updates = refresh(
        candidates,
        (2,) * len(target_linears),
    )
    assert scheduler.calls == 1
    assert current_layer.forward_calls == 0
    assert current_mlp.router_calls == 0
    assert next_layer.forward_calls == 1
    assert next_mlp.router_calls == 1
    assert len(updates) == len(target_linears)
    assert hasattr(refresh, "fixed_route_padding_stats")
    assert hasattr(refresh, "fixed_route_last_batch_stats")
    torch.testing.assert_close(
        updates[3],
        torch.zeros_like(updates[3]),
        rtol=0,
        atol=0,
    )


def test_fixed_route_hot_evaluator_source_has_no_router_or_host_calls() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "refresh"
        / "fixed_route_moe.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    evaluator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "evaluate_fixed_route_moe"
    )
    forbidden_attributes = {
        "cpu",
        "item",
        "tolist",
        "softmax",
        "topk",
        "one_hot",
        "nonzero",
    }
    calls = {
        node.func.attr
        for node in ast.walk(evaluator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(forbidden_attributes)
