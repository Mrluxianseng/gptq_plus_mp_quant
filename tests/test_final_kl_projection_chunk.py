from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from realq.refresh import kl_loss


class _Analyzer:
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        for parameter in (*self.norm.parameters(), *self.head.parameters()):
            parameter.requires_grad_(False)

    def get_layernorm_before_head(self) -> nn.Module:
        return self.norm

    def get_lm_head(self) -> nn.Module:
        return self.head


@pytest.mark.parametrize("kl_topk", [-1, 5])
def test_projection_chunk_preserves_mean_kl_and_upstream_gradient(
    monkeypatch: pytest.MonkeyPatch,
    kl_topk: int,
) -> None:
    torch.manual_seed(11)
    analyzer = _Analyzer(hidden_size=7, vocab_size=19)
    inputs = torch.randn(4, 5, 7)
    teacher = torch.randn_like(inputs)

    one_shot_weight = torch.randn(7, 7, requires_grad=True)
    one_shot_hidden = inputs @ one_shot_weight
    one_shot_loss = kl_loss.kl_topk_loss(
        one_shot_hidden,
        teacher,
        analyzer,
        kl_topk,
    )
    (one_shot_grad,) = torch.autograd.grad(one_shot_loss, one_shot_weight)

    # Force several uneven token tiles in a tiny CPU test.  Production uses
    # the fixed audited constant declared by kl_loss.py.
    monkeypatch.setattr(kl_loss, "FINAL_KL_PROJECTION_TOKEN_CHUNK", 3)
    chunked_weight = one_shot_weight.detach().clone().requires_grad_(True)
    chunked_hidden = inputs @ chunked_weight
    chunked_loss = kl_loss.memory_bounded_kl_topk_loss(
        chunked_hidden,
        teacher,
        analyzer,
        kl_topk,
    )
    (chunked_grad,) = torch.autograd.grad(chunked_loss, chunked_weight)

    torch.testing.assert_close(chunked_loss, one_shot_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(chunked_grad, one_shot_grad, rtol=1e-6, atol=1e-7)


def test_projection_chunk_rejects_hidden_shape_mismatch() -> None:
    analyzer = _Analyzer(hidden_size=3, vocab_size=5)
    with pytest.raises(ValueError, match="shape mismatch"):
        kl_loss.memory_bounded_kl_topk_loss(
            torch.randn(2, 4, 3, requires_grad=True),
            torch.randn(2, 3, 3),
            analyzer,
            -1,
        )
