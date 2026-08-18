"""Fused CUDA fake quantization for compact activation qparams.

The dynamic min/max reductions stay in :class:`utils.quant_utils.ActQuantizer`
so their established FP32-promotion semantics remain unchanged.  This module
fuses the elementwise divide, round-to-nearest-even, clamp, dequantize, and
final cast into one Triton kernel.  Qparams are compact ``(rows, groups)``
tensors; they are broadcast by address arithmetic rather than expanded to the
activation shape.

This is a streaming elementwise kernel.  Tensor-core instructions (WGMMA on
Hopper or TCGen05 on Blackwell) do not implement any operation in this
equation, and TMA would add a shared-memory/barrier round trip without tile
reuse.  Triton therefore emits coalesced vector loads/stores for this path.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - CPU-only environments.
    triton = None
    tl = None
    libdevice = None


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_COLS": 256}, num_warps=4),
            triton.Config({"BLOCK_COLS": 512}, num_warps=8),
            triton.Config({"BLOCK_COLS": 1024}, num_warps=8),
        ],
        key=["N_COLS", "GROUP_SIZE", "SYMMETRIC"],
    )
    @triton.jit
    def _fake_quant_compact_kernel(
        x_ptr,
        scale_ptr,
        zero_ptr,
        out_ptr,
        N_COLS: tl.constexpr,
        N_GROUPS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        MAXQ: tl.constexpr,
        SYMMETRIC: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask = cols < N_COLS
        offsets = row * N_COLS + cols
        group = cols // GROUP_SIZE
        param_offsets = row * N_GROUPS + group

        # Per-token qparams are FP32 in the production path.  Triton promotes
        # the activation load into the FP32 divide, exactly like the old mixed
        # BF16/FP32 PyTorch expression.  The output store performs the old
        # terminal ``.to(x.dtype)`` without a separate conversion kernel.
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(
            scale_ptr + param_offsets, mask=mask, other=1.0
        ).to(tl.float32)
        # Triton's ``/`` uses fast reciprocal division by default.  That can
        # cross a half-integer rounding boundary on large activation tensors
        # and change the quantized code.  PyTorch's established CUDA equation
        # uses IEEE round-to-nearest division, so request it explicitly.
        q = libdevice.rint(libdevice.div_rn(x, scale))
        if SYMMETRIC:
            q = tl.minimum(tl.maximum(q, -(MAXQ + 1)), MAXQ)
            out = scale * q
        else:
            zero = tl.load(
                zero_ptr + param_offsets, mask=mask, other=0.0
            ).to(tl.float32)
            q = tl.minimum(tl.maximum(q + zero, 0.0), MAXQ)
            out = scale * (q - zero)
        tl.store(out_ptr + offsets, out, mask=mask)


def is_available() -> bool:
    return triton is not None


class _FusedFakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        maxq: int,
        symmetric: bool,
        group_size: int,
    ) -> torch.Tensor:
        del ctx
        rows = x.numel() // x.shape[-1]
        columns = int(x.shape[-1])
        groups = (columns + group_size - 1) // group_size
        output = torch.empty_like(x)
        grid = lambda meta: (
            rows,
            triton.cdiv(columns, meta["BLOCK_COLS"]),
        )
        _fake_quant_compact_kernel[grid](
            x,
            scale,
            zero,
            output,
            N_COLS=columns,
            N_GROUPS=groups,
            GROUP_SIZE=group_size,
            MAXQ=maxq,
            SYMMETRIC=symmetric,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        del ctx
        # The qparams are dynamic constants.  Preserve the original STE:
        # identity for the activation and no gradients for qparams/config.
        return grad_output, None, None, None, None, None


def can_fuse(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor | None,
) -> bool:
    """Return whether the exact production arithmetic has a fused path."""

    return bool(
        triton is not None
        and x.is_cuda
        and x.is_contiguous()
        and x.ndim > 0
        and x.shape[-1] > 0
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        # Ordinary per-token quantization promotes its ranges to FP32.  A few
        # groupwise compatibility paths intentionally retain BF16 qparams;
        # keep those on eager ops instead of changing their rounding.
        and scale.dtype == torch.float32
        and scale.is_cuda
        and scale.is_contiguous()
        and (
            zero is None
            or (
                zero.dtype == torch.float32
                and zero.is_cuda
                and zero.is_contiguous()
            )
        )
    )


def fake_quant_compact(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor | None,
    *,
    maxq: int,
    symmetric: bool,
    group_size: int,
) -> torch.Tensor:
    """Apply fused QDQ using ``(flattened rows, groups)`` qparams."""

    if not can_fuse(x, scale, zero):
        raise ValueError("The fused activation fake-quant preconditions are not met.")
    zero_arg = scale if symmetric else zero
    return _FusedFakeQuantSTE.apply(
        x,
        scale,
        zero_arg,
        int(maxq),
        bool(symmetric),
        int(group_size),
    )
