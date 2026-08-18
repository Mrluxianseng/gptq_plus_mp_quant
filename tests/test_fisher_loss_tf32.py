from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from realq.refresh.fisher_loss import fisher_mse_loss  # noqa: E402
from realq.refresh import triton_fisher  # noqa: E402


def test_cpu_fisher_loss_remains_exact_fp32_reference():
    torch.manual_seed(20260808)
    q = torch.randn(3, 5, 7, requires_grad=True)
    target = torch.randn_like(q)
    fisher = torch.randn(7, 7)
    delta = q - target
    expected = 0.5 * (
        (delta.reshape(-1, 7) @ fisher) * delta.reshape(-1, 7)
    ).sum(-1).mean()
    actual = fisher_mse_loss(q, target, fisher)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    actual_grad = q.grad.clone()
    q.grad = None
    expected.backward()
    torch.testing.assert_close(actual_grad, q.grad, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_fisher.is_available(),
    reason="requires CUDA Triton",
)
def test_cuda_fisher_tf32_forward_backward_and_policy_restoration():
    torch.manual_seed(20260808)
    q = torch.randn(
        4, 257, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.randn_like(q)
    fisher = torch.randn(128, 128, device="cuda", dtype=torch.float32)
    # Fisher is symmetric in production.  Use the same property here so the
    # comparison isolates TF32 input precision rather than conditioning.
    fisher = 0.5 * (fisher + fisher.T)
    q_ref = q.detach().clone().requires_grad_(True)
    delta_ref = (q_ref - target).float().reshape(-1, 128)
    expected = 0.5 * (
        (delta_ref @ fisher) * delta_ref
    ).sum(-1).mean()

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        actual = fisher_mse_loss(q, target, fisher)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        actual.backward()
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    expected.backward()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=4e-3, atol=3.125e-2)
