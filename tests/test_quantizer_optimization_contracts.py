"""Raw-byte contracts for quantizer hot-path optimizations.

The trusted inner fake-quant tests exercise the production private methods
directly. Later sections intentionally retain local prospective implementations
until their corresponding optimization switches are promoted.
"""

from __future__ import annotations

import itertools

import pytest
import torch

import utils.quant_utils as quant_utils
from utils.quant_utils import WeightQuantizer, sym_quant_dequant


def _quantizer(
    *,
    groupsize: int,
    sym: bool = True,
    mse: bool = False,
    grid: int = 8,
    maxshrink: float = 0.5,
    w_clip_search_impl: str = "cartesian_legacy",
) -> WeightQuantizer:
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=4,
        perchannel=True,
        sym=sym,
        mse=mse,
        norm=2.4,
        grid=grid,
        maxshrink=maxshrink,
        weight_groupsize=groupsize,
        w_clip_search_impl=w_clip_search_impl,
    )
    return quantizer


def _assert_raw_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare tensor metadata and payload, including signed zero/NaN bits."""

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    actual_bytes = actual.detach().cpu().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def _assert_inner_columns_match_public_raw_bytes(
    quantizer: WeightQuantizer,
    x: torch.Tensor,
    *,
    st_idx: int | None = None,
    end_idx: int | None = None,
    natural_columns: torch.Tensor | None = None,
):
    """Exercise the production private primitive one REAL-Q column at a time."""

    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=x.shape[0],
        column_count=x.shape[1],
        device=x.device,
        dtype=x.dtype,
        st_idx=st_idx,
        end_idx=end_idx,
        col_idx=natural_columns,
    )
    for column_offset in range(x.shape[1]):
        x_column = x[:, column_offset : column_offset + 1]
        natural_column = (
            int(natural_columns[column_offset])
            if natural_columns is not None
            else column_offset
        )
        public = quantizer.fake_quantize(
            x_column,
            st_idx=st_idx,
            end_idx=end_idx,
            col_idx=natural_column,
        )
        fast = quantizer._fake_quantize_prevalidated(
            x_column, prepared, column_offset
        )
        for actual, expected in zip(fast, public):
            _assert_raw_equal(actual, expected)
    return prepared


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_trusted_fake_quant_row_mode_is_raw_byte_identical_on_edge_values(dtype):
    quantizer = _quantizer(groupsize=-1)
    quantizer.scale = torch.tensor([[0.25], [0.5], [2.0]], dtype=dtype)
    multipliers = torch.tensor(
        [
            [-20.0, -7.5, -6.5, -0.5, -0.0, 0.0, 0.5, 6.5, 7.5, 20.0],
            [20.0, 7.5, 6.5, 0.5, 0.0, -0.0, -0.5, -6.5, -7.5, -20.0],
            [-8.0, -7.0, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 7.0, 8.0],
        ],
        dtype=dtype,
    )
    x = quantizer.scale * multipliers

    _assert_inner_columns_match_public_raw_bytes(quantizer, x)


def test_trusted_fake_quant_preserves_row_slice_contract_raw_bytes():
    quantizer = _quantizer(groupsize=-1)
    quantizer.scale = torch.tensor([[0.125], [0.25], [0.5], [1.0]])
    x = torch.tensor(
        [
            [-1.25, -0.125, -0.0, 0.0, 0.125, 1.25],
            [-5.0, -0.5, -0.0, 0.0, 0.5, 5.0],
        ]
    )

    _assert_inner_columns_match_public_raw_bytes(
        quantizer, x, st_idx=1, end_idx=3
    )


def test_trusted_fake_quant_group128_act_order_and_short_tail_raw_bytes():
    generator = torch.Generator().manual_seed(20260724)
    weight = torch.randn(4, 257, generator=generator)
    weight[:, :128].mul_(0.03125)
    weight[:, 128:256].mul_(8.0)
    weight[1, 256] = -0.0
    weight[2, 256] = 0.0

    quantizer = _quantizer(groupsize=128)
    quantizer.find_params(weight)
    # Non-monotonic natural coordinates emulate an act-order inner loop and
    # explicitly include the one-column final group.
    natural_columns = torch.tensor([256, 129, 0, 255, 127, 128, 1])
    row_start, row_end = 1, 3
    x = weight[row_start:row_end].index_select(-1, natural_columns)

    _assert_inner_columns_match_public_raw_bytes(
        quantizer,
        x,
        st_idx=row_start,
        end_idx=row_end,
        natural_columns=natural_columns,
    )

    # Counterexample for an unsafe hoist: treating the permuted loop position
    # as the natural column silently takes every scale from group 0.  The
    # deliberately different group ranges make that observably wrong.
    wrong_columns = torch.arange(natural_columns.numel())
    wrong_prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=x.shape[0],
        column_count=x.shape[1],
        device=x.device,
        dtype=x.dtype,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=wrong_columns,
    )
    wrong_quantized = torch.cat(
        [
            quantizer._fake_quantize_prevalidated(
                x[:, offset : offset + 1], wrong_prepared, offset
            )[0]
            for offset in range(x.shape[1])
        ],
        dim=1,
    )
    correct_quantized = torch.cat(
        [
            quantizer.fake_quantize(
                x[:, offset : offset + 1],
                st_idx=row_start,
                end_idx=row_end,
                col_idx=natural_columns[offset],
            )[0]
            for offset in range(x.shape[1])
        ],
        dim=1,
    )
    assert not torch.equal(
        wrong_quantized.contiguous().view(torch.uint8),
        correct_quantized.contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("groupsize,columns", [(-1, 37), (128, 257)])
@pytest.mark.parametrize("seed", [0, 1, 20260724])
def test_trusted_fake_quant_finite_fuzz_is_raw_byte_identical(
    groupsize, columns, seed
):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(5, columns, generator=generator)
    quantizer = _quantizer(groupsize=groupsize)
    quantizer.find_params(weight)

    natural_columns = (
        torch.arange(columns) if groupsize > 0 else None
    )
    _assert_inner_columns_match_public_raw_bytes(
        quantizer,
        weight,
        natural_columns=natural_columns,
    )


def test_trusted_fake_quant_rejects_replaced_or_mutated_scale_as_stale():
    quantizer = _quantizer(groupsize=-1)
    weight = torch.randn(3, 5, generator=torch.Generator().manual_seed(41))
    quantizer.find_params(weight)
    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=3,
        column_count=5,
        device=weight.device,
        dtype=weight.dtype,
    )
    quantizer.scale.add_(0.125)
    with pytest.raises(RuntimeError, match="Stale prevalidated"):
        quantizer._fake_quantize_prevalidated(weight[:, :1], prepared, 0)

    quantizer.find_params(weight)
    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=3,
        column_count=5,
        device=weight.device,
        dtype=weight.dtype,
    )
    quantizer.scale = quantizer.scale.clone()
    with pytest.raises(RuntimeError, match="Stale prevalidated"):
        quantizer._fake_quantize_prevalidated(weight[:, :1], prepared, 0)


def test_trusted_fake_quant_rejects_inference_tensors_with_actionable_error():
    quantizer = _quantizer(groupsize=-1)
    weight = torch.randn(3, 5, generator=torch.Generator().manual_seed(44))
    with torch.inference_mode():
        quantizer.find_params(weight)
        with pytest.raises(
            RuntimeError,
            match="cannot run under torch.inference_mode",
        ):
            quantizer._prepare_fake_quantize_inner(
                input_rows=3,
                column_count=5,
                device=weight.device,
                dtype=weight.dtype,
            )


def test_trusted_fake_quant_rejects_mutated_maxq_or_quantizer_metadata():
    quantizer = _quantizer(groupsize=-1)
    weight = torch.randn(3, 2, generator=torch.Generator().manual_seed(42))
    quantizer.find_params(weight)
    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=3,
        column_count=2,
        device=weight.device,
        dtype=weight.dtype,
    )
    quantizer.maxq.add_(1)
    with pytest.raises(RuntimeError, match="Stale prevalidated"):
        quantizer._fake_quantize_prevalidated(weight[:, :1], prepared, 0)

    quantizer.maxq.sub_(1)
    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=3,
        column_count=2,
        device=weight.device,
        dtype=weight.dtype,
    )
    quantizer.bits = 3
    with pytest.raises(RuntimeError, match="Stale prevalidated"):
        quantizer._fake_quantize_prevalidated(weight[:, :1], prepared, 0)


def test_trusted_fake_quant_rejects_invalid_natural_column_contracts():
    weight = torch.randn(
        3, 257, generator=torch.Generator().manual_seed(43)
    )
    quantizer = _quantizer(groupsize=128)
    quantizer.find_params(weight)
    common = dict(
        input_rows=3,
        column_count=3,
        device=weight.device,
        dtype=weight.dtype,
    )

    with pytest.raises(ValueError, match="requires one natural col_idx"):
        quantizer._prepare_fake_quantize_inner(**common)
    with pytest.raises(ValueError, match="one natural column index"):
        quantizer._prepare_fake_quantize_inner(
            **common, col_idx=torch.tensor([0, 1])
        )
    with pytest.raises(IndexError, match="outside"):
        quantizer._prepare_fake_quantize_inner(
            **common, col_idx=torch.tensor([0, 128, 257])
        )
    with pytest.raises(IndexError, match="outside"):
        quantizer._prepare_fake_quantize_inner(
            **common, col_idx=torch.tensor([-1, 0, 1])
        )
    with pytest.raises(TypeError, match="integer natural"):
        quantizer._prepare_fake_quantize_inner(
            **common, col_idx=torch.tensor([0.0, 1.0, 2.0])
        )


def _symmetric_union_candidates_and_first_keys(
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    *,
    grid: int,
    maxshrink: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the 2M endpoint candidates and their first legacy visit keys.

    For symmetric quantization, every legacy pair has range

    ``max(abs(alpha_i * xmin), alpha_j * xmax)``.

    A maximum is always one of its operands, so all ``M*M`` pairs are covered
    by the union of the ``M`` shrunken negative magnitudes and ``M`` shrunken
    positive endpoints.  ``first_keys`` preserves the original lexicographic
    ``(i, j)`` traversal, which is observable because updates use strict ``<``.
    A sentinel key marks an endpoint that can never be the maximum of a pair.
    """

    candidate_count = int(maxshrink * grid)
    if candidate_count <= 0:
        raise ValueError("the union candidate helper requires at least one candidate")

    negative_candidates = torch.stack(
        [
            (1 - i / grid) * torch.abs(xmin)
            for i in range(candidate_count)
        ]
    )
    positive_candidates = torch.stack(
        [(1 - j / grid) * xmax for j in range(candidate_count)]
    )
    sentinel = candidate_count * candidate_count
    key_shape = (candidate_count, *xmin.shape)
    endpoint_indices = torch.arange(
        candidate_count, device=xmin.device, dtype=torch.long
    ).reshape(candidate_count, *([1] * xmin.ndim))

    negative_keys = torch.full(
        key_shape, sentinel, device=xmin.device, dtype=torch.long
    )
    for i in range(candidate_count):
        # Earliest j for which max(A_i, B_j) is A_i.
        eligible_j = positive_candidates <= negative_candidates[i]
        first_j = torch.where(
            eligible_j,
            endpoint_indices,
            torch.full_like(endpoint_indices, sentinel),
        ).amin(dim=0)
        valid = first_j < sentinel
        negative_keys[i] = torch.where(
            valid, i * candidate_count + first_j, negative_keys[i]
        )

    positive_keys = torch.full(
        key_shape, sentinel, device=xmin.device, dtype=torch.long
    )
    for j in range(candidate_count):
        # Earliest i for which max(A_i, B_j) is B_j.
        eligible_i = negative_candidates <= positive_candidates[j]
        first_i = torch.where(
            eligible_i,
            endpoint_indices,
            torch.full_like(endpoint_indices, sentinel),
        ).amin(dim=0)
        valid = first_i < sentinel
        positive_keys[j] = torch.where(
            valid,
            first_i * candidate_count + j,
            positive_keys[j],
        )

    return (
        torch.cat([negative_candidates, positive_candidates], dim=0),
        torch.cat([negative_keys, positive_keys], dim=0),
    )


def _union_symmetric_scale_for_equal_width_groups(
    grouped_x: torch.Tensor,
    *,
    maxq: torch.Tensor,
    norm: float,
    grid: int,
    maxshrink: float,
    ordinary_row_error_order: bool,
) -> tuple[torch.Tensor, int]:
    """Evaluate at most 2M errors while preserving legacy strict ties."""

    xmin = torch.amin(grouped_x, dim=-1)
    xmax = torch.amax(grouped_x, dim=-1)
    candidates, first_keys = _symmetric_union_candidates_and_first_keys(
        xmin, xmax, grid=grid, maxshrink=maxshrink
    )
    candidate_count = int(maxshrink * grid)
    sentinel = candidate_count * candidate_count

    scales = candidates.clamp(min=1e-5) / maxq
    errors = []
    for scale in scales:
        expanded_scale = scale.unsqueeze(-1)
        quantized = sym_quant_dequant(grouped_x, expanded_scale, maxq)
        if ordinary_row_error_order:
            # Match WeightQuantizer.find_params's exact mutation/kernel order.
            quantized -= grouped_x
            quantized.abs_()
            quantized.pow_(norm)
            error = torch.sum(quantized, dim=-1)
        else:
            # Match find_params_weight_groupwise's exact expression.
            error = (quantized - grouped_x).abs().pow(norm).sum(dim=-1)
        errors.append(error)
    errors = torch.stack(errors)

    initial_scale = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5) / maxq
    result = initial_scale.clone()
    flat_result = result.reshape(-1)
    flat_scales = scales.reshape(scales.shape[0], -1)
    flat_errors = errors.reshape(errors.shape[0], -1)
    flat_keys = first_keys.reshape(first_keys.shape[0], -1)
    for lane in range(flat_result.numel()):
        best = torch.tensor(
            float("inf"), device=grouped_x.device, dtype=flat_errors.dtype
        )
        # Sorting by the exact first pair index reconstructs the nested loop.
        order = sorted(
            range(flat_keys.shape[0]),
            key=lambda candidate: int(flat_keys[candidate, lane]),
        )
        for candidate in order:
            if int(flat_keys[candidate, lane]) >= sentinel:
                continue
            error = flat_errors[candidate, lane]
            if bool(error < best):
                best = error
                flat_result[lane] = flat_scales[candidate, lane]
    return result, 2 * candidate_count


def _union_symmetric_find_params_candidate(
    quantizer: WeightQuantizer, x: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Candidate P02 scale with the same layout as ``quantizer.scale``."""

    if not quantizer.sym:
        raise ValueError("the endpoint-union reduction is symmetric-only")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("non-finite input must use the legacy search")

    if quantizer.weight_groupsize <= 0:
        flattened = x.flatten(1) if quantizer.perchannel else x.flatten().unsqueeze(0)
        return _union_symmetric_find_params_row_candidate(quantizer, flattened)

    rows, columns = x.shape
    full_columns = (
        columns // quantizer.weight_groupsize
    ) * quantizer.weight_groupsize
    scale_parts = []
    evaluations = 0
    if full_columns:
        grouped = x[:, :full_columns].reshape(
            rows,
            full_columns // quantizer.weight_groupsize,
            quantizer.weight_groupsize,
        )
        group_scale, group_evaluations = (
            _union_symmetric_scale_for_equal_width_groups(
                grouped,
                maxq=quantizer.maxq,
                norm=quantizer.norm,
                grid=quantizer.grid,
                maxshrink=quantizer.maxshrink,
                ordinary_row_error_order=False,
            )
        )
        scale_parts.append(
            group_scale.unsqueeze(-1)
            .expand_as(grouped)
            .reshape(rows, full_columns)
        )
        evaluations = max(evaluations, group_evaluations)
    if full_columns < columns:
        tail = x[:, full_columns:].unsqueeze(1)
        tail_scale, tail_evaluations = (
            _union_symmetric_scale_for_equal_width_groups(
                tail,
                maxq=quantizer.maxq,
                norm=quantizer.norm,
                grid=quantizer.grid,
                maxshrink=quantizer.maxshrink,
                ordinary_row_error_order=False,
            )
        )
        scale_parts.append(
            tail_scale.unsqueeze(-1).expand_as(tail).reshape(rows, -1)
        )
        evaluations = max(evaluations, tail_evaluations)
    return torch.cat(scale_parts, dim=1), evaluations


def _union_symmetric_find_params_row_candidate(
    quantizer: WeightQuantizer, flattened: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Ordinary per-row P02 candidate with the legacy error expression."""

    zero = torch.zeros(
        flattened.shape[0], device=flattened.device, dtype=flattened.dtype
    )
    xmin = torch.minimum(flattened.min(1)[0], zero)
    xmax = torch.maximum(flattened.max(1)[0], zero)
    candidates, first_keys = _symmetric_union_candidates_and_first_keys(
        xmin,
        xmax,
        grid=quantizer.grid,
        maxshrink=quantizer.maxshrink,
    )
    candidate_count = int(quantizer.maxshrink * quantizer.grid)
    sentinel = candidate_count * candidate_count
    scales = candidates.clamp(min=1e-5) / quantizer.maxq
    errors = []
    for scale in scales:
        quantized = sym_quant_dequant(
            flattened, scale.unsqueeze(1), quantizer.maxq
        )
        quantized -= flattened
        quantized.abs_()
        quantized.pow_(quantizer.norm)
        errors.append(torch.sum(quantized, dim=1))
    errors = torch.stack(errors)

    result = (
        torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
        / quantizer.maxq
    )
    for row in range(flattened.shape[0]):
        best = torch.tensor(
            float("inf"), device=flattened.device, dtype=errors.dtype
        )
        order = sorted(
            range(first_keys.shape[0]),
            key=lambda candidate: int(first_keys[candidate, row]),
        )
        for candidate in order:
            if int(first_keys[candidate, row]) >= sentinel:
                continue
            error = errors[candidate, row]
            if bool(error < best):
                best = error
                result[row] = scales[candidate, row]
    if not quantizer.perchannel:
        result = result.repeat(flattened.shape[0])
    return result.reshape(-1, 1), 2 * candidate_count


def _finite_cases(columns: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(20260724)
    fuzz = torch.randn(5, columns, generator=generator)
    if columns >= 257:
        fuzz[:, :128].mul_(0.03125)
        fuzz[:, 128:256].mul_(9.0)
        fuzz[:, 256:].mul_(0.5)

    edge = torch.zeros(5, columns)
    edge[0, 0] = -0.0
    edge[1, 0] = -1.0
    edge[1, 1] = 1.0  # equal endpoint candidates / duplicate ties
    edge[2].fill_(3.0)  # one-sided positive
    edge[3].fill_(-2.0)  # one-sided negative
    edge[4, 0] = -10.0
    edge[4, 1] = 12.0

    # Finite endpoints can still overflow the |error|**2.4 accumulator.  The
    # legacy strict comparison then leaves the initial scale in place when no
    # candidate has error < +inf; the reduced search must retain that behavior.
    extreme = torch.zeros(5, columns)
    finite_max = torch.finfo(extreme.dtype).max
    extreme[0, 0] = -finite_max
    extreme[0, 1] = finite_max / 2
    extreme[1, 0] = finite_max / 4
    extreme[2, 0] = -finite_max / 4
    extreme[3, 0] = torch.finfo(extreme.dtype).tiny
    extreme[4, 0] = -torch.finfo(extreme.dtype).tiny
    return [fuzz, edge, extreme]


@pytest.mark.parametrize("groupsize,columns", [(-1, 37), (128, 257)])
def test_symmetric_union_clip_search_matches_legacy_raw_bytes(
    groupsize, columns
):
    for weight in _finite_cases(columns):
        legacy = _quantizer(groupsize=groupsize, mse=True)
        candidate = _quantizer(
            groupsize=groupsize,
            mse=True,
            w_clip_search_impl="symmetric_union_exact",
        )
        legacy.find_params(weight)
        candidate.find_params(weight)

        _assert_raw_equal(candidate.scale, legacy.scale)
        _assert_raw_equal(candidate.zero, legacy.zero)
        public_quantized, public_int, public_scale = legacy.fake_quantize(weight)
        candidate_quantized, candidate_int, returned_scale = (
            candidate.fake_quantize(weight)
        )
        _assert_raw_equal(candidate_quantized, public_quantized)
        _assert_raw_equal(candidate_int, public_int)
        _assert_raw_equal(returned_scale, public_scale)


@pytest.mark.parametrize("groupsize,columns", [(-1, 29), (128, 257)])
@pytest.mark.parametrize("seed", [0, 1, 11, 20260724])
def test_symmetric_union_clip_search_finite_fuzz_raw_bytes(
    groupsize, columns, seed
):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(4, columns, generator=generator)
    legacy = _quantizer(groupsize=groupsize, mse=True)
    candidate = _quantizer(
        groupsize=groupsize,
        mse=True,
        w_clip_search_impl="symmetric_union_exact",
    )
    legacy.find_params(weight)
    candidate.find_params(weight)
    _assert_raw_equal(candidate.scale, legacy.scale)
    _assert_raw_equal(candidate.zero, legacy.zero)


def test_symmetric_union_covers_exact_pair_set_and_first_tie_order():
    # Include equal endpoints, a zero/signed-zero lane, and both dominant sides.
    xmin = torch.tensor([-10.0, -12.0, -1.0, -0.0])
    xmax = torch.tensor([12.0, 10.0, 1.0, 0.0])
    grid, maxshrink = 8, 0.5
    count = int(grid * maxshrink)
    candidates, keys = _symmetric_union_candidates_and_first_keys(
        xmin, xmax, grid=grid, maxshrink=maxshrink
    )
    assert candidates.shape[0] == 2 * count

    negative = candidates[:count]
    positive = candidates[count:]
    for lane in range(xmin.numel()):
        exhaustive: dict[float, int] = {}
        for i, j in itertools.product(range(count), repeat=2):
            value = float(torch.maximum(negative[i, lane], positive[j, lane]))
            exhaustive.setdefault(value, i * count + j)

        represented: dict[float, int] = {}
        for candidate in range(candidates.shape[0]):
            key = int(keys[candidate, lane])
            if key < count * count:
                value = float(candidates[candidate, lane])
                represented[value] = min(represented.get(value, key), key)
        assert represented == exhaustive


@pytest.mark.parametrize("groupsize,columns", [(-1, 9), (128, 129)])
@pytest.mark.parametrize(
    "nonfinite",
    [float("nan"), float("inf"), -float("inf")],
)
def test_symmetric_union_nonfinite_input_falls_back_to_legacy_raw_bytes(
    groupsize, columns, nonfinite
):
    weight = torch.linspace(-2.0, 3.0, steps=2 * columns).reshape(2, columns)
    weight[0, -1] = nonfinite
    expected = _quantizer(groupsize=groupsize, sym=True, mse=True)
    actual = _quantizer(
        groupsize=groupsize,
        sym=True,
        mse=True,
        w_clip_search_impl="symmetric_union_exact",
    )
    expected.find_params(weight)
    actual.find_params(weight)

    _assert_raw_equal(actual.scale, expected.scale)
    _assert_raw_equal(actual.zero, expected.zero)


@pytest.mark.parametrize("groupsize,columns", [(-1, 9), (128, 129)])
def test_symmetric_union_asymmetric_mode_falls_back_to_legacy_raw_bytes(
    groupsize, columns
):
    weight = torch.linspace(-2.0, 3.0, steps=3 * columns).reshape(3, columns)
    expected = _quantizer(groupsize=groupsize, sym=False, mse=True)
    actual = _quantizer(
        groupsize=groupsize,
        sym=False,
        mse=True,
        w_clip_search_impl="symmetric_union_exact",
    )
    expected.find_params(weight)
    actual.find_params(weight)

    _assert_raw_equal(actual.scale, expected.scale)
    _assert_raw_equal(actual.zero, expected.zero)


def test_symmetric_union_nonfinite_dispatch_executes_cartesian_loop(monkeypatch):
    calls = 0
    original = quant_utils.sym_quant_dequant

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(quant_utils, "sym_quant_dequant", counted)
    weight = torch.tensor([[float("nan"), -1.0, 2.0]])
    quantizer = _quantizer(
        groupsize=-1,
        mse=True,
        grid=8,
        maxshrink=0.5,
        w_clip_search_impl="symmetric_union_exact",
    )
    quantizer.find_params(weight)
    assert calls == int(quantizer.grid * quantizer.maxshrink) ** 2


def test_symmetric_union_unsupported_search_shape_falls_back_raw_bytes(
    monkeypatch,
):
    calls = 0
    original = quant_utils.sym_quant_dequant

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(quant_utils, "sym_quant_dequant", counted)
    weight = torch.linspace(-3.0, 2.0, steps=34).reshape(2, 17)
    legacy = _quantizer(
        groupsize=-1, mse=True, grid=8, maxshrink=1.25
    )
    candidate = _quantizer(
        groupsize=-1,
        mse=True,
        grid=8,
        maxshrink=1.25,
        w_clip_search_impl="symmetric_union_exact",
    )
    legacy.find_params(weight)
    legacy_calls = calls
    calls = 0
    candidate.find_params(weight)
    candidate_calls = calls
    expected_calls = int(8 * 1.25) ** 2
    assert (legacy_calls, candidate_calls) == (expected_calls, expected_calls)
    _assert_raw_equal(candidate.scale, legacy.scale)
    _assert_raw_equal(candidate.zero, legacy.zero)


def test_symmetric_union_unsupported_row_dtype_preserves_legacy_error():
    weight = torch.randn(2, 9, dtype=torch.float64)
    for implementation in (
        "cartesian_legacy",
        "symmetric_union_exact",
    ):
        quantizer = _quantizer(
            groupsize=-1,
            mse=True,
            w_clip_search_impl=implementation,
        )
        with pytest.raises(
            RuntimeError,
            match="source and destination dtypes match",
        ):
            quantizer.find_params(weight)


def test_symmetric_union_reduces_actual_qdq_error_evaluations_625_to_50(
    monkeypatch,
):
    calls = 0
    original = quant_utils.sym_quant_dequant

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(quant_utils, "sym_quant_dequant", counted)
    weight = torch.randn(4, 257, generator=torch.Generator().manual_seed(17))

    default = _quantizer(
        groupsize=-1, mse=True, grid=50, maxshrink=0.5
    )
    assert default.w_clip_search_impl == "cartesian_legacy"
    default.find_params(weight)
    legacy_calls = calls

    calls = 0
    union = _quantizer(
        groupsize=-1,
        mse=True,
        grid=50,
        maxshrink=0.5,
        w_clip_search_impl="symmetric_union_exact",
    )
    union.find_params(weight)
    union_calls = calls

    assert (legacy_calls, union_calls) == (625, 50)
    _assert_raw_equal(union.scale, default.scale)
    _assert_raw_equal(union.zero, default.zero)


def test_symmetric_union_group128_short_tail_is_two_exact_50_eval_batches(
    monkeypatch,
):
    calls = 0
    original = quant_utils.sym_quant_dequant

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(quant_utils, "sym_quant_dequant", counted)
    weight = torch.randn(3, 257, generator=torch.Generator().manual_seed(19))
    union = _quantizer(
        groupsize=128,
        mse=True,
        grid=50,
        maxshrink=0.5,
        w_clip_search_impl="symmetric_union_exact",
    )
    union.find_params(weight)
    # Full groups are vectorized together; the unequal one-column tail is a
    # second batch.  Each batch is 50 QDQ/error evaluations instead of 625.
    assert calls == 100
