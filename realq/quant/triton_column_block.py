"""Single-GPU Triton fusion for GPTQ's sequential column-block inner loop.

The dependency along the column axis is intentionally preserved: column ``i``
is quantized, its error is divided by the corresponding inverse-Hessian
diagonal, and that error compensates columns ``[i, block_end)`` before the
next iteration.  One Triton program owns a tile of output rows from one
Hessian group and keeps its weight/scale tile in on-chip SRAM for the entire
column loop.  ``BLOCK_ROWS`` is autotuned; the column extent is exactly the
logical GPTQ column-block length (with masked power-of-two storage).

This module only replaces the *in-block* Python/CUDA-launch loop.  Callers must
still apply the original one-shot cross-block compensation
``Err @ Hinv[block, trailing]`` after this function returns.

SM100 A/B rejected TMA and automatic warp specialization for this kernel: each
tile is transferred only once and the 128 loop iterations have a strict state
dependency, so descriptor/barrier/specialized-warp overhead exceeded any copy
overlap.  WGMMA/TCGen05 is likewise a poor fit for the per-iteration K=1 outer
product.  The retained path uses coalesced vector transactions and register
state; reproducible timings are recorded in the accompanying optimization doc.
"""
from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - exercised only on CPU-only hosts.
    triton = None
    tl = None
    libdevice = None


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=8, num_stages=2),
        ],
        key=["N_ROWS", "N_COLS", "ROWS_PER_GROUP"],
    )
    @triton.jit
    def _quantize_column_block_kernel(
        w_ptr,
        scale_ptr,
        hinv_ptr,
        maxq_ptr,
        q_ptr,
        err_ptr,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        ROWS_PER_GROUP: tl.constexpr,
        stride_wr: tl.constexpr,
        stride_wc: tl.constexpr,
        stride_sr: tl.constexpr,
        stride_sc: tl.constexpr,
        stride_hg: tl.constexpr,
        stride_hi: tl.constexpr,
        stride_hj: tl.constexpr,
        stride_qr: tl.constexpr,
        stride_qc: tl.constexpr,
        stride_er: tl.constexpr,
        stride_ec: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles_per_group = tl.cdiv(ROWS_PER_GROUP, BLOCK_ROWS)
        group_id = pid // tiles_per_group
        row_tile = pid - group_id * tiles_per_group

        row_in_group = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        rows = group_id * ROWS_PER_GROUP + row_in_group
        cols = tl.arange(0, BLOCK_COLS)
        row_mask = row_in_group < ROWS_PER_GROUP
        col_mask = cols < N_COLS
        tile_mask = row_mask[:, None] & col_mask[None, :]

        # One global read per W element. Triton keeps this recurrent tile in
        # registers/on-chip storage across the complete sequential loop.
        w = tl.load(
            w_ptr
            + rows[:, None] * stride_wr
            + cols[None, :] * stride_wc,
            mask=tile_mask,
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.float32)
        maxq = tl.load(maxq_ptr).to(tl.float32)
        minq = -(maxq + 1.0)
        if stride_sc == 0:
            q_tile = tl.zeros((BLOCK_ROWS, BLOCK_COLS), tl.float32)
            error_tile = tl.zeros((BLOCK_ROWS, BLOCK_COLS), tl.float32)

        # A broadcast per-row scale is invariant across the loop.  Grouped
        # per-column scales use a coalesced tile load below because column-wise
        # scalar loads would be badly strided in the row-major matrix.
        if stride_sc == 0:
            broadcast_scale = tl.load(
                scale_ptr + rows * stride_sr,
                mask=row_mask,
                other=1.0,
                cache_modifier=".ca",
            ).to(tl.float32)
        else:
            # Column-dependent qparams are row-major.  Loading their tile once
            # is coalesced; fetching one column per iteration directly from
            # global memory would use a 128-float stride between row lanes.
            scale_tile = tl.load(
                scale_ptr
                + rows[:, None] * stride_sr
                + cols[None, :] * stride_sc,
                mask=tile_mask,
                other=1.0,
                cache_modifier=".ca",
            ).to(tl.float32)

        # ``tl.range`` deliberately avoids unrolling a 128-column dependency
        # chain into a giant program. Each Hinv row is read exactly once and
        # retained in SRAM/registers for the current compensation step.
        for i in tl.range(
            0,
            N_COLS,
            num_stages=2,
            loop_unroll_factor=1,
        ):
            is_i = cols == i
            # Dynamic register gather replaces a BLOCK_COLS-wide masked
            # reduction for each row on every sequential iteration.
            gather_index = tl.full((BLOCK_ROWS, 1), i, tl.int32)
            # ``tl.gather`` keeps the gathered dimension.  Reshape it away;
            # dynamic tensor indexing (``[:, 0]``) is not legal Triton IR.
            w_i = tl.reshape(
                tl.gather(w, gather_index, axis=1),
                (BLOCK_ROWS,),
            )
            if stride_sc == 0:
                scale_i = broadcast_scale
            else:
                scale_i = tl.reshape(
                    tl.gather(scale_tile, gather_index, axis=1),
                    (BLOCK_ROWS,),
                )
            q_int = libdevice.rint(libdevice.div_rn(w_i, scale_i))
            q_int = tl.minimum(tl.maximum(q_int, minq), maxq)
            q_i = scale_i * q_int

            hinv_row = tl.load(
                hinv_ptr
                + group_id * stride_hg
                + i * stride_hi
                + cols * stride_hj,
                mask=col_mask,
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)
            # The diagonal is a scalar global load.  The old implementation
            # selected it from hinv_row with another BLOCK_COLS reduction.
            diagonal = tl.load(
                hinv_ptr
                + group_id * stride_hg
                + i * stride_hi
                + i * stride_hj,
                cache_modifier=".ca",
            ).to(tl.float32)
            error_i = libdevice.div_rn(w_i - q_i, diagonal)

            if stride_sc == 0:
                # With one broadcast scale the saved register tile leaves
                # room to accumulate outputs and issue two coalesced stores.
                q_tile = tl.where(is_i[None, :], q_i[:, None], q_tile)
                error_tile = tl.where(
                    is_i[None, :], error_i[:, None], error_tile
                )
            else:
                # A grouped-scale tile is already resident.  Retaining two
                # more output tiles spills registers, so store each completed
                # column immediately in this specialization.
                tl.store(
                    q_ptr + rows * stride_qr + i * stride_qc,
                    q_i,
                    mask=row_mask,
                )
                tl.store(
                    err_ptr + rows * stride_er + i * stride_ec,
                    error_i,
                    mask=row_mask,
                )

            active_suffix = (cols >= i)[None, :] & tile_mask
            compensated = w - error_i[:, None] * hinv_row[None, :]
            w = tl.where(active_suffix, compensated, w)

        if stride_sc == 0:
            tl.store(
                q_ptr
                + rows[:, None] * stride_qr
                + cols[None, :] * stride_qc,
                q_tile,
                mask=tile_mask,
            )
            tl.store(
                err_ptr
                + rows[:, None] * stride_er
                + cols[None, :] * stride_ec,
                error_tile,
                mask=tile_mask,
            )


def is_available() -> bool:
    """Return whether the Triton package is importable in this process."""

    return triton is not None


def prepared_scale_matrix(prepared: Any) -> torch.Tensor:
    """Expose P01's validated scale block as a contiguous ``(rows, cols)``."""

    if prepared is None:
        raise ValueError("prepared scale context is required.")
    scale = prepared.prepared_scale
    if prepared.grouped:
        # P01 stores grouped scales as (columns, rows, 1).
        return scale.squeeze(-1).transpose(0, 1).contiguous()
    if scale.dim() != 2 or scale.shape[1] != 1:
        raise ValueError(
            "per-row prepared scales must have shape (rows, 1), got "
            f"{tuple(scale.shape)}."
        )
    # A zero column stride broadcasts the single per-row value in the kernel
    # without allocating an expanded matrix.
    return scale


@torch.no_grad()
def quantize_column_block(
    weights: torch.Tensor,
    scales: torch.Tensor,
    hinv: torch.Tensor,
    maxq: torch.Tensor,
    *,
    rows_per_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse sequential WQ and in-block GPTQ compensation on one CUDA device.

    Args:
        weights: contiguous FP32 ``(rows, block_columns)`` working block.
        scales: FP32 ``(rows, block_columns)`` or broadcast ``(rows, 1)``.
        hinv: FP32 ``(groups, block_columns, block_columns)`` tensor/view.
        maxq: scalar symmetric quantizer maximum on the same CUDA device.
        rows_per_group: number of consecutive weight rows sharing one Hinv.
    """

    if triton is None:
        raise RuntimeError("Triton is unavailable in this environment.")
    if (
        not weights.is_cuda
        or weights.dtype != torch.float32
        or weights.dim() != 2
        or not weights.is_contiguous()
    ):
        raise ValueError(
            "Triton column-block weights must be contiguous CUDA float32 "
            f"(rows, columns), got {weights.device}/{weights.dtype}/"
            f"{tuple(weights.shape)} contiguous={weights.is_contiguous()}."
        )
    rows, columns = map(int, weights.shape)
    if columns <= 0 or columns > 256:
        raise ValueError(
            f"Triton column-block length must be in [1, 256], got {columns}."
        )
    if (
        scales.device != weights.device
        or scales.dtype != torch.float32
        or scales.dim() != 2
        or scales.shape[0] != rows
        or scales.shape[1] not in (1, columns)
    ):
        raise ValueError(
            "Triton scales must be CUDA float32 (rows, 1|columns), got "
            f"{scales.device}/{scales.dtype}/{tuple(scales.shape)}."
        )
    if (
        hinv.device != weights.device
        or hinv.dtype != torch.float32
        or hinv.dim() != 3
        or tuple(hinv.shape[1:]) != (columns, columns)
    ):
        raise ValueError(
            "Triton Hinv must be CUDA float32 "
            f"(groups, {columns}, {columns}), got "
            f"{hinv.device}/{hinv.dtype}/{tuple(hinv.shape)}."
        )
    if (
        not isinstance(rows_per_group, int)
        or isinstance(rows_per_group, bool)
        or rows_per_group <= 0
        or rows != int(hinv.shape[0]) * rows_per_group
    ):
        raise ValueError(
            "rows must equal groups * rows_per_group, got "
            f"rows={rows}, groups={hinv.shape[0]}, "
            f"rows_per_group={rows_per_group}."
        )
    if (
        maxq.device != weights.device
        or maxq.numel() < 1
    ):
        raise ValueError("maxq must contain a CUDA scalar on the input device.")

    q = torch.empty_like(weights)
    errors = torch.empty_like(weights)
    block_columns = triton.next_power_of_2(columns)
    grid = lambda meta: (
        int(hinv.shape[0])
        * triton.cdiv(rows_per_group, meta["BLOCK_ROWS"]),
    )
    # A (rows, 1) scale uses stride_sc=0 for exact broadcast.
    scale_col_stride = 0 if scales.shape[1] == 1 else scales.stride(1)
    _quantize_column_block_kernel[grid](
        weights,
        scales,
        hinv,
        maxq,
        q,
        errors,
        N_ROWS=rows,
        N_COLS=columns,
        ROWS_PER_GROUP=rows_per_group,
        stride_wr=weights.stride(0),
        stride_wc=weights.stride(1),
        stride_sr=scales.stride(0),
        stride_sc=scale_col_stride,
        stride_hg=hinv.stride(0),
        stride_hi=hinv.stride(1),
        stride_hj=hinv.stride(2),
        stride_qr=q.stride(0),
        stride_qc=q.stride(1),
        stride_er=errors.stride(0),
        stride_ec=errors.stride(1),
        BLOCK_COLS=block_columns,
        # The eager recurrence materializes ``error * Hinv`` before its
        # subtraction.  Contracting it into FFMA changes later rounding
        # boundaries after enough sequential columns.
        enable_fp_fusion=False,
    )
    return q, errors
