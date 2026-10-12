"""Exact compact storage helpers for symmetric empirical Fisher matrices."""

from __future__ import annotations

import torch


def lower_triangle_indices(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU indices for row-major lower-triangle packing.

    Keeping indices on CPU avoids spending GPU memory on two int64 index arrays.
    PyTorch accepts CPU advanced-index tensors when indexing CUDA tensors.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    indices = torch.tril_indices(dim, dim, offset=0, device="cpu")
    return indices[0].contiguous(), indices[1].contiguous()


def pack_symmetric_lower(
    matrix: torch.Tensor,
    rows: torch.Tensor | None = None,
    cols: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack the lower triangle of a symmetric square matrix into a vector."""
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected square matrix, got {tuple(matrix.shape)}")
    if rows is None or cols is None:
        rows, cols = lower_triangle_indices(matrix.shape[0])
    if rows.numel() != matrix.shape[0] * (matrix.shape[0] + 1) // 2:
        raise ValueError("triangle indices do not match matrix dimension")
    return matrix[rows, cols].contiguous()


def unpack_symmetric_lower(
    packed: torch.Tensor,
    dim: int,
    rows: torch.Tensor | None = None,
    cols: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct the exact symmetric matrix from its packed lower triangle."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    expected = dim * (dim + 1) // 2
    if packed.dim() != 1 or packed.numel() != expected:
        raise ValueError(
            f"expected packed vector of length {expected}, got {tuple(packed.shape)}"
        )
    if rows is None or cols is None:
        rows, cols = lower_triangle_indices(dim)
    if rows.numel() != expected:
        raise ValueError("triangle indices do not match packed vector length")
    matrix = torch.zeros((dim, dim), dtype=packed.dtype, device=packed.device)
    matrix[rows, cols] = packed
    matrix[cols, rows] = packed
    return matrix
