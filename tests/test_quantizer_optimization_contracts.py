"""Raw-byte contracts for prospective quantizer hot-path optimizations.

These tests intentionally keep the candidate implementations local to the test
module.  They let us prove the algebra and the observable tie-breaking contract
before changing :mod:`utils.quant_utils` while the formal baseline source is
frozen.  When the production fast paths are added, the local candidate calls
should be replaced by the corresponding private methods/configured backends so
the same cases become direct implementation regressions.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from utils.quant_utils import WeightQuantizer, sym_quant_dequant


def _quantizer(
    *,
    groupsize: int,
    sym: bool = True,
    mse: bool = False,
    grid: int = 8,
    maxshrink: float = 0.5,
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
    )
    return quantizer


def _assert_raw_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare tensor metadata and payload, including signed zero/NaN bits."""

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    actual_bytes = actual.detach().cpu().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def _trusted_fake_quant_candidate(
    quantizer: WeightQuantizer,
    x: torch.Tensor,
    preselected_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Candidate for a *prevalidated* inner-loop fake-quant primitive.

    The caller remains responsible for all public API work: checking
    ``ready()/bits``, moving scale to the input device, slicing output rows,
    and mapping act-order columns back to natural group coordinates.  Keeping
    the exact division -> round -> clamp -> multiply order (and the tensor
    ``maxq`` operand) is part of the raw-byte contract.
    """

    quantized_int = torch.clamp(
        torch.round(x / preselected_scale),
        -(quantizer.maxq + 1),
        quantizer.maxq,
    )
    return (
        (preselected_scale * quantized_int).to(x.dtype),
        quantized_int,
        preselected_scale,
    )


def _public_preselected_scale(
    quantizer: WeightQuantizer,
    x: torch.Tensor,
    *,
    st_idx: int | None = None,
    end_idx: int | None = None,
    col_idx: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Repeat only the public scale-selection prelude used by the call site."""

    scale = quantizer.scale.to(x.device)
    if st_idx is not None and end_idx is not None:
        scale = scale[st_idx:end_idx]
    if quantizer.weight_groupsize > 0 and col_idx is not None:
        natural_columns = torch.as_tensor(
            col_idx, dtype=torch.long, device=scale.device
        ).reshape(-1)
        scale = scale.index_select(-1, natural_columns)
    return scale


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

    public = quantizer.fake_quantize(x)
    candidate = _trusted_fake_quant_candidate(
        quantizer,
        x,
        _public_preselected_scale(quantizer, x),
    )
    for actual, expected in zip(candidate, public):
        _assert_raw_equal(actual, expected)


def test_trusted_fake_quant_preserves_row_slice_contract_raw_bytes():
    quantizer = _quantizer(groupsize=-1)
    quantizer.scale = torch.tensor([[0.125], [0.25], [0.5], [1.0]])
    x = torch.tensor(
        [
            [-1.25, -0.125, -0.0, 0.0, 0.125, 1.25],
            [-5.0, -0.5, -0.0, 0.0, 0.5, 5.0],
        ]
    )

    public = quantizer.fake_quantize(x, st_idx=1, end_idx=3)
    candidate = _trusted_fake_quant_candidate(
        quantizer,
        x,
        _public_preselected_scale(quantizer, x, st_idx=1, end_idx=3),
    )
    for actual, expected in zip(candidate, public):
        _assert_raw_equal(actual, expected)


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

    public = quantizer.fake_quantize(
        x,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=natural_columns,
    )
    candidate = _trusted_fake_quant_candidate(
        quantizer,
        x,
        _public_preselected_scale(
            quantizer,
            x,
            st_idx=row_start,
            end_idx=row_end,
            col_idx=natural_columns,
        ),
    )
    for actual, expected in zip(candidate, public):
        _assert_raw_equal(actual, expected)

    # Counterexample for an unsafe hoist: treating the permuted loop position
    # as the natural column silently takes every scale from group 0.  The
    # deliberately different group ranges make that observably wrong.
    wrong_scale = _public_preselected_scale(
        quantizer,
        x,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=torch.arange(natural_columns.numel()),
    )
    wrong_quantized, _, _ = _trusted_fake_quant_candidate(
        quantizer, x, wrong_scale
    )
    assert not torch.equal(
        wrong_quantized.contiguous().view(torch.uint8),
        public[0].contiguous().view(torch.uint8),
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

    public = quantizer.fake_quantize(weight)
    candidate = _trusted_fake_quant_candidate(
        quantizer,
        weight,
        _public_preselected_scale(quantizer, weight),
    )
    for actual, expected in zip(candidate, public):
        _assert_raw_equal(actual, expected)


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
        legacy.find_params(weight)
        candidate_scale, evaluations = _union_symmetric_find_params_candidate(
            legacy, weight
        )

        _assert_raw_equal(candidate_scale, legacy.scale)
        assert evaluations == 2 * int(legacy.maxshrink * legacy.grid)
        assert evaluations < int(legacy.maxshrink * legacy.grid) ** 2

        public_quantized, public_int, public_scale = legacy.fake_quantize(weight)
        candidate_quantized, candidate_int, returned_scale = (
            _trusted_fake_quant_candidate(legacy, weight, candidate_scale)
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
    legacy.find_params(weight)
    candidate_scale, _ = _union_symmetric_find_params_candidate(legacy, weight)
    _assert_raw_equal(candidate_scale, legacy.scale)


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


def _fallback_dispatch(
    quantizer: WeightQuantizer, x: torch.Tensor
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Safety dispatch required by P02; legacy owns unsupported domains."""

    if not quantizer.sym or not bool(torch.isfinite(x).all()):
        quantizer.find_params(x)
        return "legacy", quantizer.scale, quantizer.zero
    scale, _ = _union_symmetric_find_params_candidate(quantizer, x)
    return "union", scale, torch.zeros_like(scale)


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
    actual = _quantizer(groupsize=groupsize, sym=True, mse=True)
    expected.find_params(weight)

    backend, scale, zero = _fallback_dispatch(actual, weight)
    assert backend == "legacy"
    _assert_raw_equal(scale, expected.scale)
    _assert_raw_equal(zero, expected.zero)


@pytest.mark.parametrize("groupsize,columns", [(-1, 9), (128, 129)])
def test_symmetric_union_asymmetric_mode_falls_back_to_legacy_raw_bytes(
    groupsize, columns
):
    weight = torch.linspace(-2.0, 3.0, steps=3 * columns).reshape(3, columns)
    expected = _quantizer(groupsize=groupsize, sym=False, mse=True)
    actual = _quantizer(groupsize=groupsize, sym=False, mse=True)
    expected.find_params(weight)

    backend, scale, zero = _fallback_dispatch(actual, weight)
    assert backend == "legacy"
    _assert_raw_equal(scale, expected.scale)
    _assert_raw_equal(zero, expected.zero)


def test_default_symmetric_union_reduces_error_evaluations_625_to_50():
    grid, maxshrink = 50, 0.5
    legacy_evaluations = int(grid * maxshrink) ** 2
    union_evaluations = 2 * int(grid * maxshrink)
    assert (legacy_evaluations, union_evaluations) == (625, 50)
