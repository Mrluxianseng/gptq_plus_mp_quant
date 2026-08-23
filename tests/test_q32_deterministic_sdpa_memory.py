from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from experiments.realq_fullmodel_retune_20260817.deterministic_sdpa_memory import (
    checkpointed_functional_call,
    tiled_scaled_dot_product_attention,
)


def test_tiled_sdpa_preserves_logical_batch_output_and_gradients():
    original = F.scaled_dot_product_attention
    torch.manual_seed(7)
    q = torch.randn(4, 2, 11, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(4, 2, 11, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(4, 2, 11, 8, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn_like(q)

    expected = original(q, k, v, is_causal=True)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)

    actual = tiled_scaled_dot_product_attention(
        q, k, v, is_causal=True, batch_tile=2, _force_tiling=True
    )
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)

    assert torch.equal(actual, expected)
    assert all(
        torch.equal(actual_grad, expected_grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads)
    )


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(5, 7, bias=False, dtype=torch.float64)
        self.right = nn.Linear(7, 5, bias=False, dtype=torch.float64)

    def forward(self, x: torch.Tensor, *, gain: torch.Tensor):
        hidden = torch.tanh(self.left(x))
        return (self.right(hidden) * gain,)


def test_checkpointed_functional_call_preserves_override_gradients():
    torch.manual_seed(11)
    block = _ToyBlock()
    x = torch.randn(3, 5, dtype=torch.float64)
    gain = torch.tensor(0.75, dtype=torch.float64)
    override = block.left.weight.detach().clone().requires_grad_(True)

    expected = functional_call(
        block,
        {"left.weight": override},
        (x,),
        {"gain": gain},
        strict=False,
    )[0]
    (expected_grad,) = torch.autograd.grad(expected.square().mean(), override)

    actual = checkpointed_functional_call(
        functional_call,
        block,
        {"left.weight": override},
        (x,),
        {"gain": gain},
        strict=False,
    )[0]
    (actual_grad,) = torch.autograd.grad(actual.square().mean(), override)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_grad, expected_grad)
