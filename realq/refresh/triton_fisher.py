"""TF32-input, FP32-accumulate kernels for the Fisher quadratic.

TF32 is an arithmetic input format rather than a PyTorch storage dtype.
CUDA matmul selects it through the cuBLAS policy; the final row-wise inner
product is not a matmul and therefore needs an explicit mantissa conversion.
The kernels below round both multiplicands with PTX ``cvt.rna.tf32.f32`` and
perform the product/reduction in FP32, matching Tensor Core TF32 input
precision while retaining an FP32 accumulator.
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
    def _round_to_tf32(value):
        return tl.inline_asm_elementwise(
            "cvt.rna.tf32.f32 $0, $1;",
            "=r,r",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _tf32_row_inner_kernel(
        left_ptr,
        right_ptr,
        output_ptr,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_COLS)
        mask = cols < N_COLS
        offsets = row * N_COLS + cols
        left = tl.load(left_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        right = tl.load(right_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        left = _round_to_tf32(left)
        right = _round_to_tf32(right)
        # tl.sum on FP32 inputs uses an FP32 reduction accumulator.
        value = tl.sum(left * right, axis=0)
        tl.store(output_ptr + row, value * 0.5)


    @triton.jit
    def _tf32_quadratic_grad_kernel(
        left_ptr,
        right_ptr,
        grad_output_ptr,
        grad_delta_ptr,
        N_ELEMENTS: tl.constexpr,
        INV_ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N_ELEMENTS
        left = tl.load(left_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        right = tl.load(right_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_output = tl.load(grad_output_ptr).to(tl.float32)
        left = _round_to_tf32(left)
        right = _round_to_tf32(right)
        grad_output = _round_to_tf32(grad_output)
        # Both quadratic branches use TF32-rounded multiply inputs.  Their
        # sum and the output remain FP32.  Round the scalar factor and the
        # result of its first multiply as well, so every Fisher multiply sees
        # TF32-width inputs even for a non-power-of-two test batch.  The
        # production 0.5/N (N=32*2048) is exactly represented regardless.
        factor = tl.full((1,), 0.5 * INV_ROWS, tl.float32)
        factor = _round_to_tf32(factor)
        scale = _round_to_tf32(grad_output * factor)
        grad = left * scale + right * scale
        tl.store(grad_delta_ptr + offsets, grad, mask=mask)


def is_available() -> bool:
    return triton is not None


def can_fuse(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        triton is not None
        and left.is_cuda
        and right.is_cuda
        and left.device == right.device
        and left.dtype == right.dtype == torch.float32
        and left.ndim == right.ndim == 2
        and left.shape == right.shape
        and left.is_contiguous()
        and right.is_contiguous()
        and left.shape[0] > 0
        and 0 < left.shape[1] <= 8192
    )


def rowwise_inner_tf32(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Return ``0.5 * sum(left * right, -1)`` with FP32 accumulation."""

    if not can_fuse(left, right):
        raise ValueError("Fisher TF32 row-inner fusion preconditions are not met.")
    rows, columns = left.shape
    output = torch.empty(rows, dtype=torch.float32, device=left.device)
    block_columns = triton.next_power_of_2(columns)
    _tf32_row_inner_kernel[(rows,)](
        left,
        right,
        output,
        N_ROWS=rows,
        N_COLS=columns,
        BLOCK_COLS=block_columns,
        num_warps=8 if block_columns >= 1024 else 4,
    )
    return output


def quadratic_grad_tf32(
    left: torch.Tensor,
    right: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    rows: int,
) -> torch.Tensor:
    """Fuse the two quadratic gradient branches with FP32 output."""

    if not can_fuse(left, right):
        raise ValueError("Fisher TF32 gradient fusion preconditions are not met.")
    if grad_output.numel() != 1 or grad_output.device != left.device:
        raise ValueError("Fisher loss grad_output must be one CUDA scalar.")
    output = torch.empty_like(left)
    block = 1024
    _tf32_quadratic_grad_kernel[(triton.cdiv(left.numel(), block),)](
        left,
        right,
        grad_output,
        output,
        N_ELEMENTS=left.numel(),
        INV_ROWS=1.0 / float(rows),
        BLOCK=block,
        num_warps=8,
    )
    return output
