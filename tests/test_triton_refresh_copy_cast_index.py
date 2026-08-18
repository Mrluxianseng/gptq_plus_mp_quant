from __future__ import annotations

import pytest
import torch

from realq.quant import triton_refresh_stitch
from realq.refresh import triton_block_adam


CUDA_TRITON = bool(
    torch.cuda.is_available()
    and triton_refresh_stitch.triton is not None
    and triton_block_adam.is_available()
)


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA Triton")
@pytest.mark.parametrize("trailing_start", [1, 17, 63])
def test_one_pass_act_order_stitch_is_bit_exact(trailing_start: int) -> None:
    torch.manual_seed(2026080801 + trailing_start)
    device = torch.device("cuda")
    rows, columns = 19, 64
    q = torch.randn(rows, columns, device=device)
    w = torch.randn(rows, columns, device=device)
    perm = torch.randperm(columns, device=device)
    invperm = torch.argsort(perm)
    expected = w[:, invperm].clone()
    expected[:, perm[:trailing_start]] = q[:, invperm][
        :, perm[:trailing_start]
    ]

    actual = triton_refresh_stitch.stitch_natural_weight(
        q, w, invperm, trailing_start
    )
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def _adam_reference(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    columns: torch.Tensor | None,
    *,
    step: int,
    lr: float,
    grad_clip: float,
    grad_scale: float,
) -> torch.Tensor:
    selected_grad = (
        grad.float()
        if columns is None
        else grad.float().index_select(1, columns)
    )
    selected_grad = (selected_grad * grad_scale).clamp(
        -grad_clip, grad_clip
    )
    selected_avg = (
        exp_avg if columns is None else exp_avg.index_select(1, columns)
    )
    selected_sq = (
        exp_avg_sq
        if columns is None
        else exp_avg_sq.index_select(1, columns)
    )
    selected_avg.mul_(0.9).add_(selected_grad, alpha=0.1)
    selected_sq.mul_(0.999).addcmul_(
        selected_grad, selected_grad, value=0.001
    )
    if columns is not None:
        exp_avg.index_copy_(1, columns, selected_avg)
        exp_avg_sq.index_copy_(1, columns, selected_sq)
    bc1 = 1.0 - 0.9**step
    bc2 = 1.0 - 0.999**step
    return (lr / bc1) * (
        selected_avg / (selected_sq.sqrt() / bc2**0.5 + 1e-8)
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA Triton")
@pytest.mark.parametrize("current", [False, True])
def test_fused_block_adam_matches_fp32_oracle(current: bool) -> None:
    torch.manual_seed(2026080811 + int(current))
    device = torch.device("cuda")
    rows, columns = 37, 96
    grad = torch.randn(
        rows, columns, device=device, dtype=torch.bfloat16
    )
    exp_avg = torch.randn(rows, columns, device=device) * 0.01
    exp_avg_sq = torch.rand(rows, columns, device=device) * 0.002
    source = torch.randn(rows, columns, device=device)
    fused_avg = exp_avg.clone()
    fused_sq = exp_avg_sq.clone()
    fused_source = source.clone()
    active = (
        torch.randperm(columns, device=device)[23:].contiguous()
        if current
        else None
    )
    reference_update = _adam_reference(
        grad,
        exp_avg,
        exp_avg_sq,
        active,
        step=7,
        lr=8.5e-6,
        grad_clip=1.0,
        grad_scale=1.0,
    )
    if not current:
        source.sub_(reference_update)

    fused_update = triton_block_adam.fused_adam_step(
        grad,
        fused_avg,
        fused_sq,
        fused_source,
        active,
        step=7,
        lr=8.5e-6,
        grad_clip=1.0,
        grad_scale=1.0,
        update_source=not current,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fused_avg, exp_avg, rtol=2e-6, atol=2e-8)
    torch.testing.assert_close(fused_sq, exp_avg_sq, rtol=2e-6, atol=2e-9)
    if current:
        assert fused_update is not None
        assert fused_update.shape == reference_update.shape
        torch.testing.assert_close(
            fused_update, reference_update, rtol=3e-6, atol=2e-9
        )
        assert torch.equal(fused_source, source)
    else:
        assert fused_update is None
        torch.testing.assert_close(
            fused_source, source, rtol=2e-6, atol=2e-8
        )
