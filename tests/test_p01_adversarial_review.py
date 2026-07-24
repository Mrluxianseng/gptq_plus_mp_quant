"""Independent CPU-only adversarial review for P01's inner quantizer path."""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn as nn

from realq.quant.realq_layer import RealQLayer
from utils.quant_utils import WeightQuantizer


def _raw_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8),
        expected.detach().cpu().contiguous().view(torch.uint8),
    )


def _quantizer(
    *,
    bits: int,
    groupsize: int,
    mse: bool = False,
) -> WeightQuantizer:
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=bits,
        perchannel=True,
        sym=True,
        mse=mse,
        grid=4,
        maxshrink=0.5,
        weight_groupsize=groupsize,
    )
    return quantizer


@pytest.mark.parametrize(
    "bits,dtype,grouped",
    itertools.product(
        (2, 3, 4, 8, 15),
        (torch.float16, torch.bfloat16, torch.float32, torch.float64),
        (False, True),
    ),
)
def test_inner_primitive_raw_equal_across_bits_dtypes_and_nonfinite_inputs(
    bits,
    dtype,
    grouped,
):
    rows = 4
    quantizer = _quantizer(
        bits=bits,
        groupsize=3 if grouped else -1,
    )
    if grouped:
        full_columns = 11
        row_factors = torch.tensor(
            [0.125, 0.5, 1.25, 3.0], dtype=dtype
        ).unsqueeze(1)
        column_factors = torch.linspace(
            0.75, 2.25, full_columns, dtype=dtype
        ).unsqueeze(0)
        quantizer.scale = row_factors * column_factors
        natural_columns = torch.tensor([10, 0, 6, 3, 9, 1, 7])
        selected_scale = quantizer.scale.index_select(-1, natural_columns)
    else:
        quantizer.scale = torch.tensor(
            [[0.125], [0.5], [1.25], [3.0]], dtype=dtype
        )
        natural_columns = None
        selected_scale = quantizer.scale.expand(rows, 7)

    multipliers = torch.tensor(
        [
            [-100.0, -7.5, -0.5, -0.0, 0.5, 7.5, 100.0],
            [100.0, 6.5, 1.5, 0.0, -1.5, -6.5, -100.0],
            [-8.0, -7.0, -0.0, 0.0, 7.0, 8.0, 0.5],
            [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        ],
        dtype=dtype,
    )
    x = selected_scale * multipliers
    x[0, 1] = float("inf")
    x[1, 2] = float("-inf")
    x[2, 3] = float("nan")

    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=rows,
        column_count=x.shape[1],
        device=x.device,
        dtype=x.dtype,
        col_idx=natural_columns,
    )
    for offset in range(x.shape[1]):
        natural_column = (
            int(natural_columns[offset]) if grouped else offset
        )
        public = quantizer.fake_quantize(
            x[:, offset : offset + 1],
            col_idx=natural_column,
        )
        fast = quantizer._fake_quantize_prevalidated(
            x[:, offset : offset + 1], prepared, offset
        )
        for actual, expected in zip(fast, public):
            _raw_equal(actual, expected)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_inner_primitive_raw_equal_with_nonfinite_scale_payloads(dtype):
    quantizer = _quantizer(bits=4, groupsize=2)
    quantizer.scale = torch.tensor(
        [
            [0.25, float("nan"), float("inf"), 0.5],
            [0.75, 1.0, float("nan"), float("inf")],
        ],
        dtype=dtype,
    )
    natural_columns = torch.tensor([3, 1, 2, 0])
    x = torch.tensor(
        [[-1.25, -0.5, 2.0, float("inf")], [3.5, 0.5, -2.0, float("nan")]],
        dtype=dtype,
    )
    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=2,
        column_count=4,
        device=x.device,
        dtype=x.dtype,
        col_idx=natural_columns,
    )
    for offset, natural_column in enumerate(natural_columns.tolist()):
        public = quantizer.fake_quantize(
            x[:, offset : offset + 1],
            col_idx=natural_column,
        )
        fast = quantizer._fake_quantize_prevalidated(
            x[:, offset : offset + 1], prepared, offset
        )
        for actual, expected in zip(fast, public):
            _raw_equal(actual, expected)


def _run_realq(
    *,
    weight: torch.Tensor,
    hessians: torch.Tensor,
    act_square: torch.Tensor,
    groupsize: int,
    num_groups: int,
    group_parallel_quant: str,
    act_order: bool,
    blocksize: int,
    w_clip: bool,
    fast: bool,
) -> torch.Tensor:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight.data.copy_(weight)
    quantizer = _quantizer(bits=4, groupsize=groupsize, mse=w_clip)
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, num_groups),
        quantizer=quantizer,
        num_groups=num_groups,
        dev=torch.device("cpu"),
        group_parallel_quant=group_parallel_quant,
    )
    realq.H = hessians.clone()
    realq.act_square = act_square.clone()
    realq._finalized = True
    realq.quantize(
        blocksize=blocksize,
        percdamp=0.01,
        act_order=act_order,
        w_clip=w_clip,
        group_parallel_quant=group_parallel_quant,
        quantizer_inner_fastpath=fast,
    )
    return linear.weight.detach().clone()


@pytest.mark.parametrize(
    (
        "groupsize",
        "num_groups",
        "group_parallel_quant",
        "act_order",
        "columns",
        "blocksize",
        "w_clip",
    ),
    [
        (-1, 1, "none", False, 9, 4, False),
        (-1, 1, "rank", True, 9, 4, True),
        (-1, 2, "none", True, 7, 3, False),
        (-1, 2, "rank", False, 7, 3, True),
        (4, 1, "rank", False, 10, 4, False),
        (4, 2, "rank", False, 10, 4, True),
        (3, 1, "rank", True, 10, 4, False),
        (3, 2, "rank", True, 10, 4, True),
        (3, 2, "none", True, 10, 4, False),
    ],
)
def test_realq_fastpath_raw_equal_with_nondiagonal_hessian_and_tail(
    groupsize,
    num_groups,
    group_parallel_quant,
    act_order,
    columns,
    blocksize,
    w_clip,
):
    generator = torch.Generator().manual_seed(
        1000
        + columns
        + 10 * num_groups
        + 100 * int(act_order)
        + 1000 * int(w_clip)
    )
    weight = torch.randn(4, columns, generator=generator)
    weight[:, ::2].mul_(0.125)
    weight[:, 1::3].mul_(4.0)
    hessians = []
    for group in range(num_groups):
        factor = torch.randn(columns, columns, generator=generator)
        hessian = factor.matmul(factor.T)
        hessian.add_(torch.eye(columns) * (0.25 + group))
        hessians.append(hessian)
    hessians = torch.stack(hessians)
    act_square = torch.rand(columns, generator=generator)

    baseline = _run_realq(
        weight=weight,
        hessians=hessians,
        act_square=act_square,
        groupsize=groupsize,
        num_groups=num_groups,
        group_parallel_quant=group_parallel_quant,
        act_order=act_order,
        blocksize=blocksize,
        w_clip=w_clip,
        fast=False,
    )
    candidate = _run_realq(
        weight=weight,
        hessians=hessians,
        act_square=act_square,
        groupsize=groupsize,
        num_groups=num_groups,
        group_parallel_quant=group_parallel_quant,
        act_order=act_order,
        blocksize=blocksize,
        w_clip=w_clip,
        fast=True,
    )
    _raw_equal(candidate, baseline)


def test_stale_context_rejects_replaced_maxq_and_prepared_scale_mutation():
    quantizer = _quantizer(bits=4, groupsize=2)
    quantizer.scale = torch.full((2, 4), 0.25)
    x = torch.ones(2, 2)
    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=2,
        column_count=2,
        device=x.device,
        dtype=x.dtype,
        col_idx=torch.tensor([0, 3]),
    )
    quantizer.maxq = quantizer.maxq.clone()
    with pytest.raises(RuntimeError, match="Stale prevalidated"):
        quantizer._fake_quantize_prevalidated(x[:, :1], prepared, 0)

    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=2,
        column_count=2,
        device=x.device,
        dtype=x.dtype,
        col_idx=torch.tensor([0, 3]),
    )
    prepared.prepared_scale.mul_(2)
    with pytest.raises(RuntimeError, match="Stale prevalidated"):
        quantizer._fake_quantize_prevalidated(x[:, :1], prepared, 0)


def test_input_contract_rejects_dtype_shape_and_foreign_owner():
    quantizer = _quantizer(bits=4, groupsize=-1)
    quantizer.scale = torch.full((2, 1), 0.25)
    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=2,
        column_count=2,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="Input no longer matches"):
        quantizer._fake_quantize_prevalidated(
            torch.ones(2, 1, dtype=torch.float64), prepared, 0
        )
    with pytest.raises(ValueError, match="Input no longer matches"):
        quantizer._fake_quantize_prevalidated(torch.ones(1, 2), prepared, 0)

    other = _quantizer(bits=4, groupsize=-1)
    other.scale = torch.full((2, 1), 0.25)
    with pytest.raises(RuntimeError, match="another quantizer"):
        other._fake_quantize_prevalidated(torch.ones(2, 1), prepared, 0)


def test_probe_unsafe_data_mutation_bypasses_version_and_stales_grouped_copy():
    """Document the PyTorch ``.data`` escape-hatch counterexample."""

    quantizer = _quantizer(bits=4, groupsize=2)
    quantizer.scale = torch.ones(2, 4)
    x = torch.full((2, 1), 1.4)
    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=2,
        column_count=1,
        device=x.device,
        dtype=x.dtype,
        col_idx=torch.tensor([0]),
    )
    version = quantizer.scale._version
    quantizer.scale.data.mul_(2)
    assert quantizer.scale._version == version

    fast = quantizer._fake_quantize_prevalidated(x, prepared, 0)[0]
    public = quantizer.fake_quantize(x, col_idx=0)[0]
    assert not torch.equal(
        fast.contiguous().view(torch.uint8),
        public.contiguous().view(torch.uint8),
    )


def test_probe_inference_tensor_version_behavior():
    quantizer = _quantizer(bits=4, groupsize=2)
    weight = torch.randn(2, 4)
    with torch.inference_mode():
        quantizer.find_params(weight)
        with pytest.raises(
            RuntimeError, match="cannot run under torch.inference_mode"
        ):
            quantizer._prepare_fake_quantize_inner(
                input_rows=2,
                column_count=1,
                device=weight.device,
                dtype=weight.dtype,
                col_idx=torch.tensor([0]),
            )


def test_probe_private_primitive_torch_compile_fullgraph_capability():
    """Validate compiled bytes when supported; record an explicit capability gap."""

    quantizer = _quantizer(bits=4, groupsize=2)
    quantizer.scale = torch.full((2, 4), 0.25)
    x = torch.tensor([[0.375], [-0.625]])
    prepared = quantizer._prepare_fake_quantize_inner(
        input_rows=2,
        column_count=1,
        device=x.device,
        dtype=x.dtype,
        col_idx=torch.tensor([3]),
    )

    def apply_inner(value):
        return quantizer._fake_quantize_prevalidated(value, prepared, 0)[0]

    if not hasattr(torch, "compile"):
        pytest.xfail("this PyTorch build does not expose torch.compile")

    compiled = torch.compile(apply_inner, backend="eager", fullgraph=True)
    dynamo_errors = getattr(getattr(torch, "_dynamo", None), "exc", None)
    capability_errors = tuple(
        error
        for error in (
            getattr(dynamo_errors, "Unsupported", None),
            getattr(dynamo_errors, "UserError", None),
        )
        if isinstance(error, type)
    )
    try:
        actual = compiled(x)
    except Exception as exc:
        if capability_errors and isinstance(exc, capability_errors):
            pytest.xfail(
                "the private version-checked primitive is not supported by "
                f"torch.compile(fullgraph=True): {type(exc).__name__}: {exc}"
            )
        raise

    eager = apply_inner(x)
    _raw_equal(actual, eager)
