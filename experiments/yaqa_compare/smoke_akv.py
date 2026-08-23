#!/usr/bin/env python3
"""CPU smoke for YAQA A/K/V hooks, including weight-less projections."""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.yaqa_compare.akv_aware import (
    install_akv_quantization,
    quantize_post_rope_key,
)


def _manual_sym_per_token(
    value: torch.Tensor, *, bits: int = 4, clip_ratio: float = 0.9
) -> torch.Tensor:
    rows = value.reshape(-1, value.shape[-1])
    zeros = torch.zeros(rows.shape[0], dtype=rows.dtype, device=rows.device)
    xmin = torch.minimum(rows.min(dim=1).values, zeros) * clip_ratio
    xmax = torch.maximum(rows.max(dim=1).values, zeros) * clip_ratio
    absmax = torch.maximum(xmin.abs(), xmax)
    maxq = 2 ** (bits - 1) - 1
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / maxq)
    quantized = torch.clamp(
        torch.round(rows / scale.unsqueeze(1)), -(maxq + 1), maxq
    )
    return (quantized * scale.unsqueeze(1)).reshape_as(value)


class WeightlessProjection(nn.Module):
    """QTIP-like module deliberately exposing neither weight nor bias."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("matrix", torch.eye(width))
        self.last_mode = None

    def forward(self, value: torch.Tensor, mode=None) -> torch.Tensor:
        self.last_mode = mode
        return value @ self.matrix.T


class TinyAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = WeightlessProjection(width)
        self.k_proj = WeightlessProjection(width)
        self.v_proj = WeightlessProjection(width)
        self.o_proj = WeightlessProjection(width)

    def forward(self, value: torch.Tensor, mode=None):
        q = self.q_proj(value, mode)
        k = self.k_proj(value, mode)
        v = self.v_proj(value, mode)
        batch, seq, width = q.shape
        q = q.view(batch, seq, 1, width).transpose(1, 2)
        k = k.view(batch, seq, 1, width).transpose(1, 2)
        k = quantize_post_rope_key(self, k)
        v = v.view(batch, seq, 1, width).transpose(1, 2)
        return q, k, v


class TinyMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate_proj = WeightlessProjection(width)
        self.up_proj = WeightlessProjection(width)
        self.down_proj = WeightlessProjection(width)


class TinyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(width)
        self.mlp = TinyMLP(width)


class TinyBackbone(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyLayer(width) for _ in range(layers)]
        )


class TinyModel(nn.Module):
    def __init__(self, width: int = 5, layers: int = 2) -> None:
        super().__init__()
        self.model = TinyBackbone(width, layers)
        self.lm_head = nn.Linear(width, width, bias=False)


def main() -> None:
    model = TinyModel()
    state_keys_before = tuple(model.state_dict())
    summary = install_akv_quantization(
        model,
        a_bits=4,
        k_bits=4,
        v_bits=4,
        groupsize=-1,
        symmetric=True,
        clip_ratio=0.9,
        mode="aware",
    )
    assert summary.activation_input_sites == 14
    assert summary.value_output_sites == 2
    assert summary.post_rope_k_sites == 2
    assert tuple(model.state_dict()) == state_keys_before
    assert install_akv_quantization(
        model,
        a_bits=4,
        k_bits=4,
        v_bits=4,
        groupsize=-1,
        symmetric=True,
        clip_ratio=0.9,
        mode="aware",
    ) == summary

    value = torch.tensor(
        [[[-8.0, -1.0, 0.0, 2.0, 7.0],
          [0.0, 0.3, 1.1, -2.7, 4.2]]]
    )
    expected_a = _manual_sym_per_token(value)
    projection = model.model.layers[0].self_attn.q_proj
    actual_a = projection(value, "hessian-mode")
    torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
    assert projection.last_mode == "hessian-mode"

    value_projection = model.model.layers[0].self_attn.v_proj
    actual_v = value_projection(value, "hessian-mode")
    expected_v = _manual_sym_per_token(expected_a)
    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)

    q, actual_k, _ = model.model.layers[0].self_attn(value, "hessian-mode")
    expected_k = expected_a.view(1, 2, 1, 5).transpose(1, 2)
    expected_k = (
        _manual_sym_per_token(
            expected_k.transpose(1, 2).reshape(1, 2, 5)
        )
        .reshape(1, 2, 1, 5)
        .transpose(1, 2)
    )
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    # Q has A4 at its Linear input but no separate post-RoPE/cache QDQ.
    torch.testing.assert_close(
        q,
        expected_a.view(1, 2, 1, 5).transpose(1, 2),
        rtol=0,
        atol=0,
    )
    assert not hasattr(model.lm_head, "_yaqa_a_quant_hook")
    print(summary.as_dict())
    print("YAQA A/K/V smoke: PASS")


if __name__ == "__main__":
    main()
