"""Small Triton kernels for opt-in GPTQ inner-loop experiments."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import os


_COMPARE_ONCE_DONE = False


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
    STRIDE_HB: tl.constexpr,
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
    d = tl.load(
        H_ptr + rows * STRIDE_HB + COL * STRIDE_H0 + COL * STRIDE_H1,
        row_mask,
        other=1,
    ).to(tl.float32)

    # PyTorch torch.round and / use round-to-nearest-even FP32 semantics here.
    # Use explicit div_rn so iterative error propagation remains bit-identical.
    quantized = libdevice.nearbyint(libdevice.div_rn(w_col, scale))
    int_weight = tl.minimum(tl.maximum(quantized, QLO), MAXQ)
    q_col = (scale * int_weight).to(w_col.dtype)
    err = libdevice.div_rn((w_col - q_col) - gh_col, d)

    h_row = tl.load(
        H_ptr + rows[:, None] * STRIDE_HB + COL * STRIDE_H0 + cols[None, :] * STRIDE_H1,
        row_mask[:, None] & col_mask[None, :],
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
    # ``h_row`` is row-batched for both H layouts: with a shared 2-D H the
    # zero batch stride broadcasts the same row to every output row; with a
    # 3-D H each row loads its own Hessian row.
    second_order = err[:, None] * h_row
    w_delta = second_order + gh_tail
    w_delta = SECOND_ORDER_SCALE * w_delta
    w_new = w_tail - w_delta
    gh_new = gh_tail - z_col[:, None] * h_row

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

    This exact prototype supports FP32 tensors, per-row symmetric scales,
    shared or row-batched inverse-Hessian blocks, and widths up to 128.
    """
    rows, cols = W.shape
    if W.dtype != torch.float32 or GH.dtype != torch.float32:
        raise TypeError("fused GPTQ column kernel currently requires FP32 W and GH")
    if cols > 128:
        raise ValueError(f"fused GPTQ column kernel supports at most 128 columns; got {cols}")
    if not (W.is_cuda and GH.is_cuda and Z.is_cuda and H.is_cuda and scale.is_cuda):
        raise ValueError("fused GPTQ column kernel requires CUDA tensors")
    if H.ndim == 2:
        h_stride0, h_stride1, h_strideb = H.stride(0), H.stride(1), 0
    elif H.ndim == 3 and H.shape[0] == rows:
        h_stride0, h_stride1, h_strideb = H.stride(1), H.stride(2), H.stride(0)
    else:
        raise ValueError("H must have shape [cols, cols] or [rows, cols, cols]")
    global _COMPARE_ONCE_DONE
    compare_once = (
        os.environ.get("REALQ_TRITON_COMPARE_ONCE") == "1"
        and not _COMPARE_ONCE_DONE
    )
    if compare_once:
        before = {
            "W": W.clone(),
            "GH": GH.clone(),
            "Q": Q.clone(),
            "W_int": W_int.clone(),
            "Err": Err.clone(),
        }

    block_m = 8
    _fused_gptq_column_kernel[(triton.cdiv(rows, block_m),)](
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
        h_stride0,
        h_stride1,
        h_strideb,
        Q.stride(0),
        W_int.stride(0),
        Err.stride(0),
        col,
        maxq,
        qlo,
        float(second_order_scale),
        block_m,
        triton.next_power_of_2(cols),
        num_warps=4,
        enable_fp_fusion=False,
    )

    if compare_once:
        # Debug the first real invocation against the reference PyTorch
        # arithmetic using the exact tensors seen by the fused kernel.
        row_ids = torch.arange(rows, device=W.device)
        w_col = before["W"][:, col]
        scale_col = scale
        q_int = torch.clamp(torch.round(w_col / scale_col), qlo, maxq)
        q_col = (scale_col * q_int).to(w_col.dtype)
        if H.ndim == 2:
            d = H[col, col].expand(rows)
            h_row = H[col, col:]
        else:
            d = H[row_ids, col, col]
            h_row = H[row_ids, col, col:]
        err = (w_col - q_col - before["GH"][:, col]) / d
        expected_w = before["W"].clone()
        expected_w[:, col:] -= second_order_scale * (
            err.unsqueeze(1) * h_row + before["GH"][:, col:]
        )
        expected_gh = before["GH"].clone()
        z_col = Z[:, col]
        expected_gh[:, col:] -= z_col.unsqueeze(1) * h_row
        expected = {
            "W": expected_w,
            "GH": expected_gh,
            "Q": q_col,
            "W_int": q_int,
            "Err": err,
        }
        actual = {
            "W": W,
            "GH": GH,
            "Q": Q[:, col],
            "W_int": W_int[:, col],
            "Err": Err[:, col],
        }
        for name, ref in expected.items():
            got = actual[name]
            delta = (got - ref).abs() if got.is_floating_point() else (got != ref)
            count = int((delta != 0).sum().item())
            maximum = float(delta.max().item()) if delta.numel() else 0.0
            print(
                f"[triton-compare-once] col={col} tensor={name} "
                f"mismatches={count}/{delta.numel()} max_abs={maximum}",
                flush=True,
            )
        _COMPARE_ONCE_DONE = True
