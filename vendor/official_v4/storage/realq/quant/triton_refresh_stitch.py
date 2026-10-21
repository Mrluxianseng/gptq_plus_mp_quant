"""One-pass act-order weight stitching for Block-GD refresh.

The historical path materialised ``Q[:, invperm]`` and ``W[:, invperm]``,
cloned the latter, then scattered the quantized prefix into the clone.  This
kernel writes the same natural-order matrix directly from the permuted Q/W
sources in one pass.  It is an indexed streaming copy, so TMA and matrix
instructions cannot improve it: the column permutation prevents rectangular
TMA transfers and there is no multiply/accumulate work for Tensor Cores.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _stitch_natural_weight_kernel(
        q_ptr,
        w_ptr,
        invperm_ptr,
        output_ptr,
        Q_STRIDE_ROW: tl.constexpr,
        Q_STRIDE_COL: tl.constexpr,
        W_STRIDE_ROW: tl.constexpr,
        W_STRIDE_COL: tl.constexpr,
        OUT_STRIDE_ROW: tl.constexpr,
        OUT_STRIDE_COL: tl.constexpr,
        N_COLS: tl.constexpr,
        N_ELEMENTS,
        TRAILING_START,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N_ELEMENTS
        rows = offsets // N_COLS
        natural_cols = offsets - rows * N_COLS
        permuted_cols = tl.load(
            invperm_ptr + natural_cols, mask=mask, other=0
        ).to(tl.int64)
        q_offsets = rows * Q_STRIDE_ROW + permuted_cols * Q_STRIDE_COL
        w_offsets = rows * W_STRIDE_ROW + permuted_cols * W_STRIDE_COL
        source_ptrs = tl.where(
            permuted_cols < TRAILING_START,
            q_ptr + q_offsets,
            w_ptr + w_offsets,
        )
        values = tl.load(source_ptrs, mask=mask)
        out_offsets = rows * OUT_STRIDE_ROW + natural_cols * OUT_STRIDE_COL
        tl.store(output_ptr + out_offsets, values, mask=mask)


def can_stitch(
    q_permuted: torch.Tensor,
    w_permuted: torch.Tensor,
    invperm: torch.Tensor,
) -> bool:
    return bool(
        triton is not None
        and q_permuted.is_cuda
        and w_permuted.is_cuda
        and invperm.is_cuda
        and q_permuted.device == w_permuted.device == invperm.device
        and q_permuted.ndim == w_permuted.ndim == 2
        and q_permuted.shape == w_permuted.shape
        and q_permuted.dtype == w_permuted.dtype
        and invperm.dtype == torch.long
        and invperm.ndim == 1
        and invperm.numel() == q_permuted.shape[1]
        and invperm.is_contiguous()
    )


def stitch_natural_weight(
    q_permuted: torch.Tensor,
    w_permuted: torch.Tensor,
    invperm: torch.Tensor,
    trailing_start: int,
) -> torch.Tensor:
    """Return natural-order ``[Q prefix, W suffix]`` without intermediates."""

    if not can_stitch(q_permuted, w_permuted, invperm):
        raise ValueError("REAL-Q Triton refresh-stitch preconditions are not met")
    rows, columns = q_permuted.shape
    if not 0 <= trailing_start <= columns:
        raise ValueError(
            f"trailing_start must be in [0, {columns}], got {trailing_start}"
        )
    output = torch.empty_like(q_permuted)
    elements = rows * columns
    block = 256
    _stitch_natural_weight_kernel[(triton.cdiv(elements, block),)](
        q_permuted,
        w_permuted,
        invperm,
        output,
        Q_STRIDE_ROW=q_permuted.stride(0),
        Q_STRIDE_COL=q_permuted.stride(1),
        W_STRIDE_ROW=w_permuted.stride(0),
        W_STRIDE_COL=w_permuted.stride(1),
        OUT_STRIDE_ROW=output.stride(0),
        OUT_STRIDE_COL=output.stride(1),
        N_COLS=columns,
        N_ELEMENTS=elements,
        TRAILING_START=trailing_start,
        BLOCK=block,
        num_warps=4,
    )
    return output
