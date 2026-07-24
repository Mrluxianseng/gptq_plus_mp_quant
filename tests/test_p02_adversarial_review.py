from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn as nn

from realq.quant.realq_layer import RealQLayer
from utils.quant_utils import WeightQuantizer


def _quantizer(
    implementation: str,
    *,
    bits: int = 4,
    groupsize: int = -1,
    perchannel: bool = True,
    grid: int = 8,
    maxshrink: float = 0.5,
    norm: float = 2.4,
) -> WeightQuantizer:
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=bits,
        perchannel=perchannel,
        sym=True,
        mse=True,
        norm=norm,
        grid=grid,
        maxshrink=maxshrink,
        weight_groupsize=groupsize,
        w_clip_search_impl=implementation,
    )
    return quantizer


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.detach().cpu().contiguous().view(torch.uint8),
        right.detach().cpu().contiguous().view(torch.uint8),
    )


def _assert_exact(weight: torch.Tensor, **kwargs) -> None:
    legacy = _quantizer("cartesian_legacy", **kwargs)
    union = _quantizer("symmetric_union_exact", **kwargs)
    legacy.find_params(weight.clone())
    union.find_params(weight.clone())
    assert _raw_equal(union.scale, legacy.scale)
    assert _raw_equal(union.zero, legacy.zero)
    legacy_q, legacy_i, legacy_s = legacy.fake_quantize(weight)
    union_q, union_i, union_s = union.fake_quantize(weight)
    assert _raw_equal(union_q, legacy_q)
    assert _raw_equal(union_i, legacy_i)
    assert _raw_equal(union_s, legacy_s)


_EDGE_BITS = [
    0x00000000,  # +0
    0x80000000,  # -0
    0x00000001,  # minimum positive subnormal
    0x80000001,  # minimum negative subnormal
    0x007FFFFF,  # maximum positive subnormal
    0x807FFFFF,  # maximum negative subnormal
    0x00800000,  # minimum positive normal
    0x80800000,  # minimum negative normal
    0x3727C5AB,  # immediately below float32(1e-5)
    0x3727C5AC,  # float32(1e-5)
    0x3727C5AD,  # immediately above float32(1e-5)
    0x3EFFFFFF,  # immediately below 0.5
    0x3F000000,  # 0.5
    0x3F000001,  # immediately above 0.5
    0x3F7FFFFF,  # immediately below 1
    0x3F800000,  # 1
    0x3F800001,  # immediately above 1
    0x7F7FFFFF,  # maximum finite
    0xFF7FFFFF,  # minimum finite
]


@pytest.mark.parametrize("bits", [2, 3, 4, 8, 15])
@pytest.mark.parametrize(
    ("grid", "maxshrink"),
    [(1, 1.0), (2, 1.0), (3, 0.67), (7, 1.0), (8, 0.5), (50, 0.5)],
)
def test_all_float32_edge_pairs(bits, grid, maxshrink):
    values = torch.tensor(_EDGE_BITS, dtype=torch.uint32).view(torch.float32)
    # Each row is a separate production per-channel lane.  Cover every ordered
    # pair plus a third value to make the reduction less degenerate.
    rows = torch.tensor(
        [(float(a), float(b), 0.25) for a, b in itertools.product(values, repeat=2)],
        dtype=torch.float32,
    )
    _assert_exact(
        rows,
        bits=bits,
        grid=grid,
        maxshrink=maxshrink,
    )


@pytest.mark.parametrize("groupsize", [-1, 1, 2, 3, 4, 7, 8])
@pytest.mark.parametrize("columns", [1, 2, 3, 7, 8, 9, 15, 17])
def test_group_and_tail_random_bitpattern_fuzz(groupsize, columns):
    generator = torch.Generator().manual_seed(0xC0FFEE + columns + groupsize)
    words = torch.randint(
        0,
        2**32,
        (19, columns),
        dtype=torch.int64,
        generator=generator,
    ).to(torch.uint32)
    weight = words.view(torch.float32)
    finite = torch.isfinite(weight)
    weight = torch.where(finite, weight, torch.zeros_like(weight))
    _assert_exact(weight, groupsize=groupsize, grid=7, maxshrink=1.0)


def test_large_random_finite_bitpattern_fuzz_at_production_grid():
    generator = torch.Generator().manual_seed(0x5EED)
    words = torch.randint(
        0,
        2**32,
        (4096, 4),
        dtype=torch.int64,
        generator=generator,
    ).to(torch.uint32)
    weight = words.view(torch.float32)
    weight = torch.where(
        torch.isfinite(weight), weight, torch.zeros_like(weight)
    )
    _assert_exact(weight, grid=50, maxshrink=0.5)


@pytest.mark.parametrize("perchannel", [False, True])
def test_tie_heavy_rows_and_global_mode(perchannel):
    threshold = torch.tensor(1e-5, dtype=torch.float32)
    below = torch.nextafter(threshold, torch.tensor(0.0))
    above = torch.nextafter(threshold, torch.tensor(float("inf")))
    weight = torch.tensor(
        [
            [+0.0, -0.0, float(below), -float(below)],
            [+0.0, -0.0, float(threshold), -float(threshold)],
            [+0.0, -0.0, float(above), -float(above)],
            [+1.0, -1.0, +1.0, -1.0],
            [+3.5, +3.5, +3.5, +3.5],
            [-2.5, -2.5, -2.5, -2.5],
        ],
        dtype=torch.float32,
    )
    _assert_exact(
        weight,
        perchannel=perchannel,
        grid=50,
        maxshrink=0.5,
    )


def test_default_backend_does_not_call_union(monkeypatch):
    import utils.quant_utils as quant_utils

    def forbidden(*args, **kwargs):
        raise AssertionError("union backend was entered")

    monkeypatch.setattr(
        quant_utils, "_select_symmetric_union_scale", forbidden
    )
    quantizer = _quantizer("cartesian_legacy")
    quantizer.find_params(torch.randn(4, 17))


def _run_realq(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    implementation: str,
    groupsize: int,
    act_order: bool,
    group_parallel_quant: str,
) -> torch.Tensor:
    linear = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device=weight.device,
    )
    linear.weight.data.copy_(weight)
    quantizer = _quantizer(
        implementation,
        groupsize=groupsize,
        grid=8,
        maxshrink=0.5,
    )
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, 1, device=weight.device),
        quantizer=quantizer,
        num_groups=1,
        dev=weight.device,
        group_parallel_quant=group_parallel_quant,
    )
    realq.H = hessian.unsqueeze(0)
    realq.act_square = torch.linspace(
        -3.0, 5.0, steps=weight.shape[1], device=weight.device
    ).roll(5)
    realq._finalized = True
    realq.quantize(
        blocksize=8,
        percdamp=0.01,
        act_order=act_order,
        w_clip=True,
        group_parallel_quant=group_parallel_quant,
    )
    return linear.weight.detach().clone()


@pytest.mark.parametrize("groupsize", [-1, 8])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("group_parallel_quant", ["none", "rank"])
def test_full_realq_static_dynamic_act_order_and_tail_exact(
    groupsize, act_order, group_parallel_quant
):
    generator = torch.Generator().manual_seed(314159)
    weight = torch.randn(6, 17, generator=generator)
    basis = torch.randn(17, 17, generator=generator)
    hessian = basis @ basis.T + 0.5 * torch.eye(17)
    legacy = _run_realq(
        weight,
        hessian,
        implementation="cartesian_legacy",
        groupsize=groupsize,
        act_order=act_order,
        group_parallel_quant=group_parallel_quant,
    )
    union = _run_realq(
        weight,
        hessian,
        implementation="symmetric_union_exact",
        groupsize=groupsize,
        act_order=act_order,
        group_parallel_quant=group_parallel_quant,
    )
    assert _raw_equal(union, legacy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_edge_group_tail_and_full_realq_exact():
    device = torch.device("cuda")
    values = torch.tensor(
        _EDGE_BITS, dtype=torch.uint32, device=device
    ).view(torch.float32)
    rows = torch.stack(
        [
            torch.stack([a, b, values[4], values[5]])
            for a, b in itertools.product(values, repeat=2)
        ]
    )
    _assert_exact(rows, bits=4, grid=50, maxshrink=0.5)

    generator = torch.Generator(device=device).manual_seed(271828)
    grouped = torch.randn(64, 257, generator=generator, device=device)
    grouped[:, :128].mul_(0.03125)
    grouped[:, 128:256].mul_(9.0)
    _assert_exact(
        grouped,
        bits=4,
        groupsize=128,
        grid=50,
        maxshrink=0.5,
    )

    weight = torch.randn(6, 17, generator=generator, device=device)
    basis = torch.randn(17, 17, generator=generator, device=device)
    hessian = basis @ basis.T + 0.5 * torch.eye(17, device=device)
    for groupsize, act_order, group_parallel_quant in itertools.product(
        [-1, 8], [False, True], ["none", "rank"]
    ):
        legacy = _run_realq(
            weight=weight,
            hessian=hessian,
            implementation="cartesian_legacy",
            groupsize=groupsize,
            act_order=act_order,
            group_parallel_quant=group_parallel_quant,
        )
        union = _run_realq(
            weight=weight,
            hessian=hessian,
            implementation="symmetric_union_exact",
            groupsize=groupsize,
            act_order=act_order,
            group_parallel_quant=group_parallel_quant,
        )
        assert _raw_equal(union, legacy)
