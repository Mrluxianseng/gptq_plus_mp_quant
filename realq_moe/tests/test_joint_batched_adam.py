from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from realq_moe.refresh.block_gd import RefreshContext
from realq_moe.refresh.joint_batched_adam import JointBatchedAdamState


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="joint batched Adam is a CUDA-only MoE path",
)


class _UnusedScheduler:
    def next_indices(self) -> list[int]:
        raise AssertionError("Adam helper must not consume scheduler samples.")


def _make_contexts(
    *,
    expert_count: int,
    rows: int,
    columns: int,
    layer_lr: float,
    grad_clip: float,
) -> tuple[RefreshContext, ...]:
    scheduler = _UnusedScheduler()
    contexts = []
    for _ in range(expert_count):
        module = torch.nn.Linear(
            columns,
            rows,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        contexts.append(
            RefreshContext(
                module=module,
                layer_lr=layer_lr,
                grad_clip=grad_clip,
                backward_bsz=4,
                scheduler=scheduler,
            )
        )
    return tuple(contexts)


@torch.no_grad()
def _legacy_expert_loop_step(
    contexts: tuple[RefreshContext, ...],
    accumulated_grad_batch: torch.Tensor,
    trailing_start: int,
    perms: torch.Tensor | None,
) -> torch.Tensor:
    updates = []
    for expert_idx, (ctx, accum_grad) in enumerate(
        zip(contexts, accumulated_grad_batch)
    ):
        if perms is not None:
            accum_grad = accum_grad.index_select(
                1,
                perms[expert_idx].to(dtype=torch.int64),
            )
        grad_slice = accum_grad[:, trailing_start:]
        if ctx.grad_clip > 0:
            grad_slice = grad_slice.clamp(
                min=-ctx.grad_clip,
                max=ctx.grad_clip,
            )
        exp_avg = ctx.exp_avg[:, trailing_start:]
        exp_avg_sq = ctx.exp_avg_sq[:, trailing_start:]
        exp_avg.mul_(ctx.beta1).add_(
            grad_slice,
            alpha=1.0 - ctx.beta1,
        )
        exp_avg_sq.mul_(ctx.beta2).addcmul_(
            grad_slice,
            grad_slice,
            value=1.0 - ctx.beta2,
        )
        bias_correction1 = 1.0 - ctx.beta1**ctx.adam_step
        bias_correction2 = 1.0 - ctx.beta2**ctx.adam_step
        denominator = exp_avg_sq.sqrt() / math.sqrt(
            bias_correction2
        )
        denominator.add_(ctx.eps)
        step_size = ctx.layer_lr / bias_correction1
        updates.append(step_size * (exp_avg / denominator))
    return torch.stack(updates, dim=0)


@pytest.mark.parametrize("use_perms", [False, True], ids=["natural", "permuted"])
@pytest.mark.parametrize("grad_clip", [0.0, 0.17], ids=["no_clip", "clip"])
def test_joint_batched_adam_matches_legacy_expert_loop_across_steps(
    use_perms: bool,
    grad_clip: float,
) -> None:
    torch.manual_seed(9200 + int(use_perms) * 10 + int(grad_clip > 0))
    expert_count, rows, columns = 4, 5, 7
    layer_lr = 0.031
    joint_contexts = _make_contexts(
        expert_count=expert_count,
        rows=rows,
        columns=columns,
        layer_lr=layer_lr,
        grad_clip=grad_clip,
    )
    oracle_contexts = _make_contexts(
        expert_count=expert_count,
        rows=rows,
        columns=columns,
        layer_lr=layer_lr,
        grad_clip=grad_clip,
    )

    initial_avg = torch.randn(
        expert_count,
        rows,
        columns,
        device="cuda",
        dtype=torch.float32,
    ) * 0.03
    initial_avg_sq = (
        torch.rand(
            expert_count,
            rows,
            columns,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
        + 0.005
    )
    for expert_idx in range(expert_count):
        joint_contexts[expert_idx].exp_avg.copy_(initial_avg[expert_idx])
        joint_contexts[expert_idx].exp_avg_sq.copy_(
            initial_avg_sq[expert_idx]
        )
        oracle_contexts[expert_idx].exp_avg.copy_(initial_avg[expert_idx])
        oracle_contexts[expert_idx].exp_avg_sq.copy_(
            initial_avg_sq[expert_idx]
        )
        joint_contexts[expert_idx].adam_step = 2
        oracle_contexts[expert_idx].adam_step = 2

    state = JointBatchedAdamState(joint_contexts)
    for expert_idx, ctx in enumerate(joint_contexts):
        assert ctx.exp_avg is state._exp_avg_views[expert_idx]
        assert ctx.exp_avg_sq is state._exp_avg_sq_views[expert_idx]
        assert ctx.exp_avg.data_ptr() == state.exp_avg[expert_idx].data_ptr()
        assert (
            ctx.exp_avg_sq.data_ptr()
            == state.exp_avg_sq[expert_idx].data_ptr()
        )

    perms = None
    if use_perms:
        perms = torch.stack(
            [
                torch.roll(
                    torch.arange(columns, device="cuda"),
                    shifts=expert_idx + 1,
                )
                for expert_idx in range(expert_count)
            ],
            dim=0,
        )

    for iteration, trailing_start in enumerate((0, 2, 5), start=1):
        grad = torch.randn(
            expert_count,
            rows,
            columns,
            device="cuda",
            dtype=torch.float32,
        ) * (0.25 + 0.1 * iteration)
        for ctx in joint_contexts:
            ctx.adam_step += 1
        for ctx in oracle_contexts:
            ctx.adam_step += 1

        actual = state.step(grad, trailing_start, perms)
        expected = _legacy_expert_loop_step(
            oracle_contexts,
            grad,
            trailing_start,
            perms,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            state.exp_avg,
            torch.stack([ctx.exp_avg for ctx in oracle_contexts]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state.exp_avg_sq,
            torch.stack([ctx.exp_avg_sq for ctx in oracle_contexts]),
            rtol=0,
            atol=0,
        )


def test_joint_batched_adam_validates_synchronized_contract() -> None:
    contexts = list(
        _make_contexts(
            expert_count=2,
            rows=3,
            columns=4,
            layer_lr=0.02,
            grad_clip=0.0,
        )
    )
    contexts[1].layer_lr = 0.03
    with pytest.raises(ValueError, match="must share"):
        JointBatchedAdamState(contexts)

    contexts[1].layer_lr = contexts[0].layer_lr
    state = JointBatchedAdamState(contexts)
    for ctx in contexts:
        ctx.adam_step = 1

    with pytest.raises(ValueError, match="shape"):
        state.step(
            torch.zeros(2, 3, 3, device="cuda"),
            0,
            None,
        )
    with pytest.raises(ValueError, match="perms shape"):
        state.step(
            torch.zeros(2, 3, 4, device="cuda"),
            0,
            torch.zeros(2, 3, dtype=torch.int64, device="cuda"),
        )
