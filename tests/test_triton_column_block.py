from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from realq.quant import triton_column_block as fused  # noqa: E402


def _reference_inner(
    weights: torch.Tensor,
    scales: torch.Tensor,
    hinv: torch.Tensor,
    maxq: torch.Tensor,
    rows_per_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = weights.shape
    groups = hinv.shape[0]
    working = weights.clone()
    working_grouped = working.view(groups, rows_per_group, columns)
    q = torch.empty_like(working)
    errors = torch.empty_like(working)
    scale_matrix = scales.expand(rows, columns)
    for column in range(columns):
        q_column = torch.clamp(
            torch.round(
                working[:, column] / scale_matrix[:, column]
            ),
            -(maxq + 1),
            maxq,
        ) * scale_matrix[:, column]
        q[:, column] = q_column
        diagonal = hinv[:, column, column].repeat_interleave(
            rows_per_group
        )
        error = (working[:, column] - q_column) / diagonal
        errors[:, column] = error
        working_grouped[:, :, column:].sub_(
            error.view(groups, rows_per_group, 1)
            * hinv[:, column, column:].unsqueeze(1)
        )
    return q, errors


def _make_hinv(
    groups: int, columns: int, device: torch.device
) -> torch.Tensor:
    factor = torch.randn(
        groups, columns, columns, device=device
    )
    hessian = (
        factor @ factor.transpose(-1, -2)
        + torch.eye(columns, device=device) * 0.5
    )
    return torch.linalg.cholesky(hessian).inverse().transpose(
        -1, -2
    ).contiguous()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused.is_available(),
    reason="requires CUDA Triton",
)
@pytest.mark.parametrize("scale_columns", (1, 17))
def test_triton_inner_loop_matches_reference_quantized_columns(
    scale_columns: int,
) -> None:
    torch.manual_seed(20260728 + scale_columns)
    device = torch.device("cuda")
    groups, rows_per_group, columns = 4, 8, 17
    rows = groups * rows_per_group
    weights = torch.randn(rows, columns, device=device)
    scales = (
        torch.rand(rows, scale_columns, device=device) * 0.4 + 0.02
    )
    hinv = _make_hinv(groups, columns, device)
    maxq = torch.tensor(7, device=device)

    expected_q, expected_errors = _reference_inner(
        weights, scales, hinv, maxq, rows_per_group
    )
    actual_q, actual_errors = fused.quantize_column_block(
        weights,
        scales,
        hinv,
        maxq,
        rows_per_group=rows_per_group,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_q, expected_q)
    torch.testing.assert_close(
        actual_errors,
        expected_errors,
        rtol=2e-5,
        atol=5e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused.is_available(),
    reason="requires CUDA Triton",
)
def test_triton_keeps_separate_cross_block_compensation() -> None:
    """A second block sees the unchanged one-shot outer Err@Hinv update."""

    torch.manual_seed(20260730)
    device = torch.device("cuda")
    groups, rows_per_group = 4, 8
    rows, columns, blocksize = groups * rows_per_group, 31, 16
    initial = torch.randn(rows, columns, device=device)
    scales = torch.rand(rows, 1, device=device) * 0.4 + 0.02
    hinv = _make_hinv(groups, columns, device)
    maxq = torch.tensor(7, device=device)

    def run(use_triton: bool) -> torch.Tensor:
        working = initial.clone()
        quantized = torch.empty_like(working)
        for start in range(0, columns, blocksize):
            end = min(start + blocksize, columns)
            count = end - start
            block = working[:, start:end].clone()
            block_hinv = hinv[:, start:end, start:end]
            if use_triton:
                q, errors = fused.quantize_column_block(
                    block,
                    scales,
                    block_hinv,
                    maxq,
                    rows_per_group=rows_per_group,
                )
            else:
                q, errors = _reference_inner(
                    block,
                    scales,
                    block_hinv,
                    maxq,
                    rows_per_group,
                )
            quantized[:, start:end] = q
            if end < columns:
                # This is deliberately outside the fused inner kernel.
                working.view(
                    groups, rows_per_group, columns
                )[:, :, end:].sub_(
                    torch.bmm(
                        errors.view(
                            groups, rows_per_group, count
                        ),
                        hinv[:, start:end, end:],
                    )
                )
        return quantized

    actual = run(True)
    expected = run(False)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
