"""Single-pass FP32 Adam update for REAL-Q full-block refresh.

The unfused PyTorch path materialises a clipped gradient, two moment
intermediates, a denominator, a selected update, and finally a full-size zero
update for every active weight.  This kernel keeps the complete Adam chain in
registers and traverses gradient/moments/master once.  Future weights are
updated in-place; the current weight writes only its active suffix in GPTQ
quant-order, which is exactly the layout consumed by ``RealQLayer``.

This is a streaming elementwise operation with no tile reuse.  TMA/Tensor
Core instructions would add staging and synchronisation without useful reuse;
the relevant Hopper/Blackwell optimisation is one global-memory pass with
FP32 master and moment accumulation. Future full-weight access and current
output are coalesced; current gradient/moment loads necessarily follow the
act-order column permutation.
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
    def _block_adam_kernel(
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        source_ptr,
        active_columns_ptr,
        update_ptr,
        GRAD_STRIDE_ROW: tl.constexpr,
        GRAD_STRIDE_COL: tl.constexpr,
        MOMENT_STRIDE_ROW: tl.constexpr,
        MOMENT_STRIDE_COL: tl.constexpr,
        SOURCE_STRIDE_ROW: tl.constexpr,
        SOURCE_STRIDE_COL: tl.constexpr,
        N_ACTIVE_COLS,
        N_ELEMENTS,
        GRAD_SCALE,
        BETA1: tl.constexpr,
        BETA2: tl.constexpr,
        ONE_MINUS_BETA1: tl.constexpr,
        ONE_MINUS_BETA2: tl.constexpr,
        STEP_SIZE,
        INV_SQRT_BC2,
        EPS: tl.constexpr,
        GRAD_CLIP,
        APPLY_GRAD_CLIP: tl.constexpr,
        HAS_ACTIVE_COLUMNS: tl.constexpr,
        UPDATE_SOURCE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N_ELEMENTS
        rows = offsets // N_ACTIVE_COLS
        active_cols = offsets - rows * N_ACTIVE_COLS
        if HAS_ACTIVE_COLUMNS:
            cols = tl.load(
                active_columns_ptr + active_cols,
                mask=mask,
                other=0,
            ).to(tl.int64)
        else:
            cols = active_cols

        grad_offsets = rows * GRAD_STRIDE_ROW + cols * GRAD_STRIDE_COL
        moment_offsets = (
            rows * MOMENT_STRIDE_ROW + cols * MOMENT_STRIDE_COL
        )
        grad = tl.load(
            grad_ptr + grad_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        grad *= GRAD_SCALE
        if APPLY_GRAD_CLIP:
            grad = tl.minimum(tl.maximum(grad, -GRAD_CLIP), GRAD_CLIP)

        exp_avg = tl.load(
            exp_avg_ptr + moment_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        exp_avg_sq = tl.load(
            exp_avg_sq_ptr + moment_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        exp_avg = BETA1 * exp_avg + ONE_MINUS_BETA1 * grad
        exp_avg_sq = (
            BETA2 * exp_avg_sq + ONE_MINUS_BETA2 * grad * grad
        )
        denom = tl.sqrt(exp_avg_sq) * INV_SQRT_BC2 + EPS
        update = STEP_SIZE * exp_avg / denom

        tl.store(exp_avg_ptr + moment_offsets, exp_avg, mask=mask)
        tl.store(exp_avg_sq_ptr + moment_offsets, exp_avg_sq, mask=mask)
        if UPDATE_SOURCE:
            source_offsets = (
                rows * SOURCE_STRIDE_ROW + cols * SOURCE_STRIDE_COL
            )
            source = tl.load(
                source_ptr + source_offsets, mask=mask, other=0.0
            ).to(tl.float32)
            tl.store(
                source_ptr + source_offsets, source - update, mask=mask
            )
        else:
            # offsets enumerate [row, active quant-order column], so the
            # output is already the contiguous GPTQ suffix RealQLayer needs.
            tl.store(update_ptr + offsets, update, mask=mask)


def is_available() -> bool:
    return triton is not None


def can_fuse(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    source: torch.Tensor,
    active_columns: torch.Tensor | None,
) -> bool:
    if not (
        triton is not None
        and grad.is_cuda
        and exp_avg.is_cuda
        and exp_avg_sq.is_cuda
        and source.is_cuda
        and grad.device == exp_avg.device == exp_avg_sq.device == source.device
        and grad.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and exp_avg.dtype == exp_avg_sq.dtype == source.dtype == torch.float32
        and grad.ndim == exp_avg.ndim == exp_avg_sq.ndim == source.ndim == 2
        and grad.shape == exp_avg.shape == exp_avg_sq.shape == source.shape
        and grad.shape[0] > 0
        and grad.shape[1] > 0
    ):
        return False
    if active_columns is None:
        return True
    return bool(
        active_columns.is_cuda
        and active_columns.device == grad.device
        and active_columns.dtype == torch.long
        and active_columns.ndim == 1
        and active_columns.numel() > 0
        and active_columns.is_contiguous()
    )


def fused_adam_step(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    source: torch.Tensor,
    active_columns: torch.Tensor | None,
    *,
    step: int,
    lr: float,
    grad_clip: float,
    grad_scale: float,
    update_source: bool,
) -> torch.Tensor | None:
    """Advance moments and either update ``source`` or return active update.

    ``active_columns`` lists natural-coordinate columns in the desired output
    order.  Passing ``None`` means every natural column is active.
    """

    if not can_fuse(grad, exp_avg, exp_avg_sq, source, active_columns):
        raise ValueError("REAL-Q fused block Adam preconditions are not met")
    if step <= 0:
        raise ValueError(f"Adam step must be positive, got {step}")
    rows, columns = grad.shape
    active_count = (
        columns if active_columns is None else active_columns.numel()
    )
    output = (
        None
        if update_source
        else torch.empty(
            (rows, active_count),
            dtype=torch.float32,
            device=grad.device,
        )
    )
    output_arg = source if output is None else output
    columns_arg = grad if active_columns is None else active_columns
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    elements = rows * active_count
    block = 256
    _block_adam_kernel[(triton.cdiv(elements, block),)](
        grad,
        exp_avg,
        exp_avg_sq,
        source,
        columns_arg,
        output_arg,
        GRAD_STRIDE_ROW=grad.stride(0),
        GRAD_STRIDE_COL=grad.stride(1),
        MOMENT_STRIDE_ROW=exp_avg.stride(0),
        MOMENT_STRIDE_COL=exp_avg.stride(1),
        SOURCE_STRIDE_ROW=source.stride(0),
        SOURCE_STRIDE_COL=source.stride(1),
        N_ACTIVE_COLS=active_count,
        N_ELEMENTS=elements,
        GRAD_SCALE=float(grad_scale),
        BETA1=beta1,
        BETA2=beta2,
        ONE_MINUS_BETA1=1.0 - beta1,
        ONE_MINUS_BETA2=1.0 - beta2,
        STEP_SIZE=float(lr) / bc1,
        INV_SQRT_BC2=1.0 / bc2**0.5,
        EPS=eps,
        GRAD_CLIP=float(grad_clip),
        APPLY_GRAD_CLIP=grad_clip > 0.0,
        HAS_ACTIVE_COLUMNS=active_columns is not None,
        UPDATE_SOURCE=bool(update_source),
        BLOCK=block,
        num_warps=4,
    )
    return output
