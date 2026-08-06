from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.func import functional_call

from realq_moe.refresh.block_gd import RefreshContext
from realq_moe.refresh.fisher_loss import fisher_mse_loss
from realq_moe.refresh.joint_moe_block_gd import (
    make_joint_moe_block_gd_refresh_fn,
)
from realq_moe.refresh.kl_loss import kl_topk_loss
from realq_moe.runner.streams import LayerInputs


class _CountingScheduler:
    def __init__(self, n_total: int) -> None:
        self.n_total = int(n_total)
        self.calls = 0

    def next_indices(self) -> list[int]:
        self.calls += 1
        return list(range(self.n_total))


class _TinyExpert(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.proj(hidden_states))


class _TinySparseMlp(torch.nn.Module):
    """Small hard-routed layer whose fourth expert receives no assignments."""

    def __init__(self, hidden_size: int = 4, num_experts: int = 4) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
        )
        self.experts = torch.nn.ModuleList(
            [_TinyExpert(hidden_size) for _ in range(num_experts)]
        )
        self.forward_calls = 0
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.weight[0, 0] = 1.0
            self.gate.weight[1, 1] = 1.0
            self.gate.weight[2, 0] = -1.0
            self.gate.weight[2, 1] = -1.0
            # Expert 3 has score zero.  The fixture below always gives one of
            # experts 0/1/2 a strictly positive score.

    def forward(
        self,
        hidden_states: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, ...]:
        self.forward_calls += 1
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        routes = self.gate(flat).argmax(dim=-1)
        mixed = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            token_indices = torch.nonzero(
                routes == expert_idx,
                as_tuple=False,
            ).flatten()
            if token_indices.numel() == 0:
                # Deliberately leave the unused expert disconnected from the
                # graph so joint refresh must materialize its None grad as 0.
                continue
            selected = flat.index_select(0, token_indices)
            contribution = expert(selected)
            mixed = torch.index_add(
                mixed,
                0,
                token_indices,
                contribution,
            )
        return (0.4 * mixed.reshape(shape),)


class _TinySparseLayer(torch.nn.Module):
    def __init__(self, hidden_size: int = 4, num_experts: int = 4) -> None:
        super().__init__()
        self.mlp = _TinySparseMlp(hidden_size, num_experts)
        self.forward_calls = 0

    def current_mlp_state(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Identity attention/norm fixture: these are respectively the
        # post-attention-layernorm MLP input and post-attention residual.
        return hidden_states, hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, ...]:
        self.forward_calls += 1
        mlp_inputs, residuals = self.current_mlp_state(hidden_states)
        return (residuals + self.mlp(mlp_inputs)[0],)


class _TinyAnalyzer:
    def __init__(self, hidden_size: int, device: torch.device) -> None:
        self.norm = torch.nn.LayerNorm(hidden_size, device=device)
        self.lm_head = torch.nn.Linear(
            hidden_size,
            hidden_size + 3,
            bias=False,
            device=device,
        )
        self.norm.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def get_layernorm_before_head(self) -> torch.nn.Module:
        return self.norm

    def get_lm_head(self) -> torch.nn.Module:
        return self.lm_head


def _psd_fisher(hidden_size: int, device: torch.device) -> torch.Tensor:
    factor = torch.randn(hidden_size, hidden_size, device=device)
    return factor.transpose(0, 1) @ factor / hidden_size


def _fixture_inputs(device: torch.device) -> torch.Tensor:
    # Every sample contains two tokens; experts 0, 1, and 2 each receive
    # assignments, while expert 3 receives none.
    return torch.tensor(
        [
            [[3.0, 0.0, 1.0, 0.2], [0.0, 3.0, 1.0, -0.1]],
            [[-3.0, -3.0, 1.0, 0.1], [2.0, 0.0, 1.0, -0.2]],
            [[0.0, 2.0, 1.0, 0.3], [-2.0, -2.0, 1.0, -0.3]],
        ],
        device=device,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("objective", ["fisher", "slide", "final_kl"])
@pytest.mark.parametrize("current_replay", ["fast_mlp", "full_layer"])
def test_joint_refresh_matches_joint_autograd_oracle_and_updates_all_experts(
    objective: str,
    current_replay: str,
) -> None:
    torch.manual_seed(
        {"fisher": 701, "slide": 702, "final_kl": 703}[objective]
        + int(current_replay == "full_layer") * 10
    )
    device = torch.device("cuda")
    hidden_size = 4
    use_slide = objective == "slide"
    use_final_kl = objective == "final_kl"
    layer = _TinySparseLayer(hidden_size).to(device).eval()
    next_layer = _TinySparseLayer(hidden_size).to(device).eval()
    analyzer = _TinyAnalyzer(hidden_size, device)
    layer.requires_grad_(False)
    next_layer.requires_grad_(False)
    inputs = _fixture_inputs(device)
    state = LayerInputs(
        inps=inputs,
        fp_inps=inputs.clone(),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )

    with torch.no_grad():
        fp_current = layer(inputs)[0] + 0.025 * torch.randn_like(inputs)
        fp_next = (
            next_layer(fp_current)[0]
            + 0.015 * torch.randn_like(inputs)
        )
    layer.forward_calls = 0
    layer.mlp.forward_calls = 0
    next_layer.forward_calls = 0
    next_layer.mlp.forward_calls = 0

    fisher = _psd_fisher(hidden_size, device)
    next_fisher = _psd_fisher(hidden_size, device)
    target_linears = tuple(expert.proj for expert in layer.mlp.experts)
    candidates = tuple(
        linear.weight.detach().float()
        + 0.04 * torch.randn_like(
            linear.weight,
            dtype=torch.float32,
        )
        for linear in target_linears
    )
    trailing_start = 2
    trailing_starts = (trailing_start,) * len(target_linears)
    permutations = tuple(
        torch.roll(
            torch.arange(hidden_size, device=device),
            shifts=expert_idx,
        )
        for expert_idx in range(len(target_linears))
    )

    scheduler = _CountingScheduler(n_total=inputs.shape[0])
    contexts = tuple(
        RefreshContext(
            module=linear,
            layer_lr=0.07 + 0.01 * expert_idx,
            grad_clip=0.0,
            backward_bsz=inputs.shape[0],
            scheduler=scheduler,
        )
        for expert_idx, linear in enumerate(target_linears)
    )
    alpha_calls = {"count": 0}
    slide_alpha = 0.35

    def alpha_fn() -> float:
        alpha_calls["count"] += 1
        return slide_alpha

    refresh = make_joint_moe_block_gd_refresh_fn(
        layer=layer,
        target_linears=target_linears,
        layer_state=state,
        fp_out_for_this_layer=fp_current,
        fisher=None if use_final_kl else fisher,
        contexts=contexts,
        current_mlp_inputs=(
            layer.current_mlp_state(inputs)[0]
            if current_replay == "fast_mlp"
            else None
        ),
        current_mlp_residuals=(
            layer.current_mlp_state(inputs)[1]
            if current_replay == "fast_mlp"
            else None
        ),
        next_layer=next_layer if use_slide else None,
        next_fp_out=fp_next if use_slide else None,
        next_fisher=next_fisher if use_slide else None,
        slide_alpha_fn=alpha_fn if use_slide else None,
        final_layer_analyzer=analyzer if use_final_kl else None,
        kl_topk=-1,
        a_loss_ratio=1.0,
    )
    # The primary batched analytic stepper supplies (E,R,C)/(E,C) tensors;
    # keep the full-layer oracle arm on the generic sequence interface.
    refresh_candidates = (
        torch.stack(candidates, dim=0)
        if current_replay == "fast_mlp"
        else candidates
    )
    refresh_permutations = (
        torch.stack(permutations, dim=0)
        if current_replay == "fast_mlp"
        else permutations
    )
    updates = refresh(
        refresh_candidates,
        trailing_starts,
        refresh_permutations,
    )

    # One shared block consumes one scheduler chunk and, because the protocol
    # batch fits backward_bsz, exactly one joint current forward/backward.
    assert scheduler.calls == 1
    assert layer.forward_calls == int(current_replay == "full_layer")
    assert layer.mlp.forward_calls == 1
    assert next_layer.forward_calls == int(use_slide)
    assert alpha_calls["count"] == int(use_slide)
    assert all(ctx.adam_step == 1 for ctx in contexts)

    oracle_weights = tuple(
        candidate.detach().clone().requires_grad_(True)
        for candidate in candidates
    )
    override_names = tuple(
        f"mlp.experts.{expert_idx}.proj.weight"
        for expert_idx in range(len(target_linears))
    )
    with torch.enable_grad():
        q_current = functional_call(
            layer,
            dict(zip(override_names, oracle_weights)),
            (inputs,),
            strict=False,
        )[0]
        if use_final_kl:
            loss_current = kl_topk_loss(
                q_current,
                fp_current,
                analyzer,
                -1,
            )
        else:
            loss_current = fisher_mse_loss(
                q_current,
                fp_current,
                fisher,
                a_loss_ratio=1.0,
            )
        if use_slide:
            q_next = next_layer(q_current)[0]
            loss_next = fisher_mse_loss(
                q_next,
                fp_next,
                next_fisher,
                a_loss_ratio=1.0,
            )
            oracle_loss = (
                slide_alpha * loss_current
                + (1.0 - slide_alpha) * loss_next
            )
        else:
            oracle_loss = loss_current
        oracle_grads = torch.autograd.grad(
            oracle_loss,
            oracle_weights,
            allow_unused=True,
        )

    for expert_idx, (
        ctx,
        update,
        oracle_grad,
        permutation,
    ) in enumerate(
        zip(contexts, updates, oracle_grads, permutations)
    ):
        natural_grad = (
            torch.zeros_like(candidates[expert_idx])
            if oracle_grad is None
            else oracle_grad.detach().float()
        )
        permuted_grad = natural_grad.index_select(1, permutation)
        expected_exp_avg = torch.zeros_like(permuted_grad)
        expected_exp_avg_sq = torch.zeros_like(permuted_grad)
        expected_exp_avg[:, trailing_start:] = (
            (1.0 - ctx.beta1)
            * permuted_grad[:, trailing_start:]
        )
        expected_exp_avg_sq[:, trailing_start:] = (
            (1.0 - ctx.beta2)
            * permuted_grad[:, trailing_start:].square()
        )
        torch.testing.assert_close(
            ctx.exp_avg,
            expected_exp_avg,
            rtol=2e-5,
            atol=2e-7,
        )
        torch.testing.assert_close(
            ctx.exp_avg_sq,
            expected_exp_avg_sq,
            rtol=3e-5,
            atol=2e-9,
        )

        bias_correction1 = 1.0 - ctx.beta1
        bias_correction2 = 1.0 - ctx.beta2
        expected_denominator = (
            expected_exp_avg_sq[:, trailing_start:].sqrt()
            / bias_correction2**0.5
        )
        expected_denominator.add_(ctx.eps)
        expected_update = (
            ctx.layer_lr
            / bias_correction1
            * (
                expected_exp_avg[:, trailing_start:]
                / expected_denominator
            )
        )
        torch.testing.assert_close(
            update,
            expected_update,
            rtol=3e-5,
            atol=2e-7,
        )

    # Expert 3 is not routed by the current layer.  It still advances Adam,
    # sees an exact zero gradient, and returns a zero trailing update.
    assert oracle_grads[-1] is None
    assert contexts[-1].adam_step == 1
    torch.testing.assert_close(
        updates[-1],
        torch.zeros_like(updates[-1]),
        rtol=0,
        atol=0,
    )
