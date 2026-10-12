"""Small Triton kernels for opt-in GPTQ inner-loop experiments."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _fused_gptq_column_kernel(
    W_ptr,
    GH_ptr,
    Z_ptr,
    H_ptr,
    Scale_ptr,
    Q_ptr,
    WInt_ptr,
    Err_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    STRIDE_W: tl.constexpr,
    STRIDE_GH: tl.constexpr,
    STRIDE_Z: tl.constexpr,
    STRIDE_H0: tl.constexpr,
    STRIDE_H1: tl.constexpr,
    STRIDE_Q: tl.constexpr,
    STRIDE_WINT: tl.constexpr,
    STRIDE_ERR: tl.constexpr,
    COL,
    MAXQ: tl.constexpr,
    QLO: tl.constexpr,
    SECOND_ORDER_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # A CTA owns every trailing column for its row tile. This avoids a race:
    # W[:, COL] supplies q and is also updated as part of the tail.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    tail = tl.arange(0, BLOCK_N)
    cols = COL + tail
    row_mask = rows < ROWS
    col_mask = cols < COLS

    w_col = tl.load(W_ptr + rows * STRIDE_W + COL, row_mask, other=0).to(tl.float32)
    scale = tl.load(Scale_ptr + rows, row_mask, other=1).to(tl.float32)
    gh_col = tl.load(GH_ptr + rows * STRIDE_GH + COL, row_mask, other=0).to(tl.float32)
    d = tl.load(H_ptr + COL * STRIDE_H0 + COL * STRIDE_H1).to(tl.float32)

    # PyTorch torch.round and / use round-to-nearest-even FP32 semantics here.
    # Use explicit div_rn so iterative error propagation remains bit-identical.
    quantized = libdevice.nearbyint(libdevice.div_rn(w_col, scale))
    int_weight = tl.minimum(tl.maximum(quantized, QLO), MAXQ)
    q_col = (scale * int_weight).to(w_col.dtype)
    err = libdevice.div_rn((w_col - q_col) - gh_col, d)

    h_row = tl.load(
        H_ptr + COL * STRIDE_H0 + cols * STRIDE_H1,
        col_mask,
        other=0,
    ).to(tl.float32)
    gh_tail = tl.load(
        GH_ptr + rows[:, None] * STRIDE_GH + cols[None, :],
        row_mask[:, None] & col_mask[None, :],
        other=0,
    ).to(tl.float32)
    z_col = tl.load(Z_ptr + rows * STRIDE_Z + COL, row_mask, other=0).to(tl.float32)
    w_tail = tl.load(
        W_ptr + rows[:, None] * STRIDE_W + cols[None, :],
        row_mask[:, None] & col_mask[None, :],
        other=0,
    ).to(tl.float32)

    # Match the reference's sequence: outer product, add GHinv, scale, subtract.
    second_order = err[:, None] * h_row[None, :]
    w_delta = second_order + gh_tail
    w_delta = SECOND_ORDER_SCALE * w_delta
    w_new = w_tail - w_delta
    gh_new = gh_tail - z_col[:, None] * h_row[None, :]

    tl.store(
        W_ptr + rows[:, None] * STRIDE_W + cols[None, :],
        w_new,
        row_mask[:, None] & col_mask[None, :],
    )
    tl.store(
        GH_ptr + rows[:, None] * STRIDE_GH + cols[None, :],
        gh_new,
        row_mask[:, None] & col_mask[None, :],
    )
    tl.store(Q_ptr + rows * STRIDE_Q + COL, q_col, row_mask)
    tl.store(WInt_ptr + rows * STRIDE_WINT + COL, int_weight, row_mask)
    tl.store(Err_ptr + rows * STRIDE_ERR + COL, err, row_mask)


@torch.no_grad()
def fused_gptq_column_(
    W,
    GH,
    Z,
    H,
    scale,
    Q,
    W_int,
    Err,
    col,
    *,
    maxq,
    qlo,
    second_order_scale,
):
    """Apply one sequential GPTQ column update in-place with one kernel launch.

    This exact prototype supports FP32 tensors, a shared per-row symmetric
    quantization scale, and block widths up to 128.
    """
    rows, cols = W.shape
    if W.dtype != torch.float32 or GH.dtype != torch.float32:
        raise TypeError("fused GPTQ column kernel currently requires FP32 W and GH")
    if cols > 128:
        raise ValueError(f"fused GPTQ column kernel supports at most 128 columns; got {cols}")
    if not (W.is_cuda and GH.is_cuda and Z.is_cuda and H.is_cuda and scale.is_cuda):
        raise ValueError("fused GPTQ column kernel requires CUDA tensors")
    _fused_gptq_column_kernel[(triton.cdiv(rows, 8),)](
        W,
        GH,
        Z,
        H,
        scale,
        Q,
        W_int,
        Err,
        rows,
        cols,
        W.stride(0),
        GH.stride(0),
        Z.stride(0),
        H.stride(0),
        H.stride(1),
        Q.stride(0),
        W_int.stride(0),
        Err.stride(0),
        col,
        maxq,
        qlo,
        float(second_order_scale),
        8,
        triton.next_power_of_2(cols),
        num_warps=4,
        enable_fp_fusion=False,
    )
