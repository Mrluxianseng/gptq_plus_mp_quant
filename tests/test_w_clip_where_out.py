"""Exactness and fallback contracts for P03 clip-winner updates."""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

import utils.quant_utils as quant_utils
from realq.quant.realq_layer import RealQLayer
from utils.quant_utils import WeightQuantizer


def _quantizer(
    *,
    update_impl: str,
    groupsize: int = -1,
    perchannel: bool = True,
    sym: bool = True,
    search_impl: str = "cartesian_legacy",
    grid: int = 8,
) -> WeightQuantizer:
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=4,
        perchannel=perchannel,
        sym=sym,
        mse=True,
        norm=2.4,
        grid=grid,
        maxshrink=0.5,
        weight_groupsize=groupsize,
        w_clip_search_impl=search_impl,
        w_clip_update_impl=update_impl,
    )
    return quantizer


def _assert_raw_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8),
        expected.detach().cpu().contiguous().view(torch.uint8),
    )


def _assert_observers_raw_equal(
    weight: torch.Tensor,
    *,
    groupsize: int,
    perchannel: bool,
    sym: bool,
    search_impl: str = "cartesian_legacy",
) -> None:
    guarded = _quantizer(
        update_impl="guarded",
        groupsize=groupsize,
        perchannel=perchannel,
        sym=sym,
        search_impl=search_impl,
    )
    where_out = _quantizer(
        update_impl="where_out",
        groupsize=groupsize,
        perchannel=perchannel,
        sym=sym,
        search_impl=search_impl,
    )
    guarded.find_params(weight.clone())
    where_out.find_params(weight.clone())

    _assert_raw_equal(where_out.scale, guarded.scale)
    _assert_raw_equal(where_out.zero, guarded.zero)
    guarded_q, guarded_i, guarded_scale = guarded.fake_quantize(weight)
    actual_q, actual_i, actual_scale = where_out.fake_quantize(weight)
    _assert_raw_equal(actual_q, guarded_q)
    _assert_raw_equal(actual_i, guarded_i)
    _assert_raw_equal(actual_scale, guarded_scale)


@pytest.mark.parametrize("groupsize", [-1, 4])
def test_implicit_default_update_matches_explicit_guarded_raw(groupsize):
    """Keep the repository-local half of the cross-revision default gate.

    The actual parent/candidate comparison is intentionally performed with
    ``tools/p01_cross_revision_probe.py`` in two checkouts: a fixed hash in a
    unit test would be unnecessarily tied to the PyTorch build and CPU math
    libraries.  This test makes any local change from the public implicit
    default to the explicit historical implementation immediately visible.
    """

    weight = torch.randn(
        4, 10, generator=torch.Generator().manual_seed(20260727)
    )
    implicit = WeightQuantizer()
    implicit.configure(
        bits=4,
        perchannel=True,
        sym=True,
        mse=True,
        norm=2.4,
        grid=8,
        maxshrink=0.5,
        weight_groupsize=groupsize,
    )
    explicit = _quantizer(
        update_impl="guarded",
        groupsize=groupsize,
        perchannel=True,
        sym=True,
    )

    assert implicit.w_clip_update_impl == "guarded"
    implicit.find_params(weight.clone())
    explicit.find_params(weight.clone())
    _assert_raw_equal(implicit.scale, explicit.scale)
    _assert_raw_equal(implicit.zero, explicit.zero)
    for actual, expected in zip(
        implicit.fake_quantize(weight), explicit.fake_quantize(weight)
    ):
        _assert_raw_equal(actual, expected)


@pytest.mark.parametrize(
    "groupsize,perchannel,sym,columns",
    [
        (-1, True, True, 17),
        (-1, True, False, 17),
        (-1, False, True, 17),
        (-1, False, False, 17),
        (4, True, True, 10),
        (4, True, False, 10),
    ],
)
@pytest.mark.parametrize("seed", [0, 11, 20260724])
def test_where_out_matches_guarded_raw_for_ordinary_and_grouped_fuzz(
    groupsize, perchannel, sym, columns, seed
):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(5, columns, generator=generator)
    weight[0].mul_(0.03125)
    weight[1].mul_(7.0)
    _assert_observers_raw_equal(
        weight,
        groupsize=groupsize,
        perchannel=perchannel,
        sym=sym,
    )


@pytest.mark.parametrize(
    "groupsize,perchannel,sym",
    [
        (-1, True, True),
        (-1, False, False),
        (4, True, True),
        (4, True, False),
    ],
)
def test_where_out_preserves_edge_nonfinite_and_tie_payloads_raw(
    groupsize, perchannel, sym
):
    # Ten columns exercise two complete groups plus a short grouped tail.
    weight = torch.tensor(
        [
            [+0.0, -0.0, +1.0, -1.0, +1.0, -1.0, +0.0, -0.0, 2.0, -2.0],
            [float("nan"), -1.0, 2.0, 0.0, 3.0, -3.0, 1.0, 1.0, 0.0, 0.0],
            [float("inf"), 1.0, -2.0, 0.0, 4.0, -4.0, 2.0, 2.0, 0.0, 0.0],
            [-float("inf"), 1.0, -2.0, 0.0, 5.0, -5.0, 3.0, 3.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0, -2.0, -2.0, -2.0, -2.0, 0.0, -0.0],
        ],
        dtype=torch.float32,
    )
    _assert_observers_raw_equal(
        weight,
        groupsize=groupsize,
        perchannel=perchannel,
        sym=sym,
    )


def _observer_outcome(
    weight: torch.Tensor,
    *,
    update_impl: str,
    groupsize: int,
):
    quantizer = _quantizer(
        update_impl=update_impl,
        groupsize=groupsize,
        perchannel=True,
        sym=True,
    )
    try:
        quantizer.find_params(weight)
    except Exception as exc:  # noqa: BLE001 - historical behavior is the oracle
        return ("error", type(exc), str(exc))
    return (
        "success",
        quantizer.scale.detach().clone(),
        quantizer.zero.detach().clone(),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("groupsize", [-1, 4])
def test_where_out_non_fp32_inputs_follow_strict_legacy_path(dtype, groupsize):
    weight = torch.linspace(-2.0, 3.0, steps=20).reshape(2, 10).to(dtype)
    guarded = _observer_outcome(
        weight.clone(), update_impl="guarded", groupsize=groupsize
    )
    actual = _observer_outcome(
        weight.clone(), update_impl="where_out", groupsize=groupsize
    )

    assert actual[0] == guarded[0]
    if guarded[0] == "error":
        assert actual[1] is guarded[1]
        assert actual[2] == guarded[2]
    else:
        _assert_raw_equal(actual[1], guarded[1])
        _assert_raw_equal(actual[2], guarded[2])


@pytest.mark.parametrize("groupsize", [-1, 4])
def test_where_out_requires_grad_input_follows_strict_legacy_path(groupsize):
    weight = torch.linspace(-2.0, 3.0, steps=20).reshape(2, 10)
    weight.requires_grad_(True)
    guarded = _observer_outcome(
        weight.clone(), update_impl="guarded", groupsize=groupsize
    )
    actual = _observer_outcome(
        weight.clone(), update_impl="where_out", groupsize=groupsize
    )

    assert actual[0] == guarded[0]
    if guarded[0] == "error":
        assert actual[1] is guarded[1]
        assert actual[2] == guarded[2]
    else:
        _assert_raw_equal(actual[1], guarded[1])
        _assert_raw_equal(actual[2], guarded[2])


def test_weight_quantizer_rejects_unknown_update_implementation():
    quantizer = WeightQuantizer()
    with pytest.raises(ValueError, match=re.escape("guarded' or 'where_out")):
        quantizer.configure(bits=4, w_clip_update_impl="masked")


@pytest.mark.parametrize("groupsize", [-1, 4])
def test_where_out_keeps_fixed_updates_for_strict_tie_masks(
    groupsize, monkeypatch
):
    conditions = []
    original_where = torch.where

    def recording_where(*args, **kwargs):
        if kwargs.get("out") is not None:
            conditions.append(args[0].detach().clone())
        return original_where(*args, **kwargs)

    monkeypatch.setattr(torch, "where", recording_where)
    quantizer = _quantizer(
        update_impl="where_out",
        groupsize=groupsize,
        perchannel=True,
        sym=True,
    )
    quantizer.find_params(torch.zeros(2, 8))

    candidate_count = int(quantizer.grid * quantizer.maxshrink)
    assert len(conditions) == 3 * candidate_count**2
    # The first candidate wins every lane. All later candidates tie exactly,
    # so strict ``<`` makes every later update mask false; unlike the legacy
    # Python guard, the optimized path remains a fixed-shape device operation.
    assert all(bool(condition.all()) for condition in conditions[:3])
    assert not any(bool(condition.any()) for condition in conditions[3:])


def test_symmetric_union_composes_without_cartesian_out_updates(monkeypatch):
    weight = torch.randn(
        4, 17, generator=torch.Generator().manual_seed(20260724)
    )
    out_calls = 0
    original_where = torch.where

    def counted_where(*args, **kwargs):
        nonlocal out_calls
        if kwargs.get("out") is not None:
            out_calls += 1
        return original_where(*args, **kwargs)

    monkeypatch.setattr(torch, "where", counted_where)
    _assert_observers_raw_equal(
        weight,
        groupsize=-1,
        perchannel=True,
        sym=True,
        search_impl="symmetric_union_exact",
    )
    # P02's finite symmetric union owns selection entirely, so P03 has no
    # incremental update work in this combination.
    assert out_calls == 0


def test_symmetric_union_asymmetric_fallback_composes_with_where_out(
    monkeypatch,
):
    weight = torch.randn(
        4, 17, generator=torch.Generator().manual_seed(20260725)
    )
    out_calls = 0
    original_where = torch.where

    def counted_where(*args, **kwargs):
        nonlocal out_calls
        if kwargs.get("out") is not None:
            out_calls += 1
        return original_where(*args, **kwargs)

    monkeypatch.setattr(torch, "where", counted_where)
    _assert_observers_raw_equal(
        weight,
        groupsize=-1,
        perchannel=True,
        sym=False,
        search_impl="symmetric_union_exact",
    )
    assert out_calls > 0


@pytest.mark.parametrize(
    "groupsize,cartesian_batches",
    [(-1, 1), (4, 2)],
)
def test_symmetric_union_nonfinite_fallback_uses_where_out_raw(
    groupsize,
    cartesian_batches,
    monkeypatch,
):
    # Group size four gives two vectorized batches: two complete groups and a
    # short two-column tail.  NaN/Inf make P02 reject its finite-only union and
    # must hand the exact Cartesian selection back to P03.
    weight = torch.tensor(
        [
            [float("nan"), -0.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.0, 2.0],
            [float("inf"), 0.0, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -0.0, -2.0],
        ],
        dtype=torch.float32,
    )
    out_calls = 0
    original_where = torch.where

    def counted_where(*args, **kwargs):
        nonlocal out_calls
        if kwargs.get("out") is not None:
            out_calls += 1
        return original_where(*args, **kwargs)

    monkeypatch.setattr(torch, "where", counted_where)
    _assert_observers_raw_equal(
        weight,
        groupsize=groupsize,
        perchannel=True,
        sym=True,
        search_impl="symmetric_union_exact",
    )

    candidate_count = int(8 * 0.5)
    assert out_calls == 3 * candidate_count**2 * cartesian_batches


def _realq_result(
    weight: torch.Tensor,
    *,
    update_impl: str,
    quantizer_inner_fastpath: bool,
) -> torch.Tensor:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight.data.copy_(weight)
    quantizer = _quantizer(
        update_impl=update_impl,
        groupsize=-1,
        perchannel=True,
        sym=True,
    )
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, 1),
        quantizer=quantizer,
        num_groups=1,
        dev=torch.device("cpu"),
        group_parallel_quant="none",
    )
    realq.H = torch.eye(weight.shape[1]).unsqueeze(0)
    realq.act_square = torch.arange(weight.shape[1], dtype=torch.float32)
    realq._finalized = True
    realq.quantize(
        blocksize=4,
        percdamp=0.01,
        act_order=True,
        w_clip=True,
        group_parallel_quant="none",
        quantizer_inner_fastpath=quantizer_inner_fastpath,
    )
    return linear.weight.detach().clone()


def test_where_out_composes_with_p01_inner_fastpath_end_to_end_raw():
    weight = torch.randn(
        3, 9, generator=torch.Generator().manual_seed(20260724)
    )
    guarded = _realq_result(
        weight,
        update_impl="guarded",
        quantizer_inner_fastpath=False,
    )
    optimized = _realq_result(
        weight,
        update_impl="where_out",
        quantizer_inner_fastpath=True,
    )
    _assert_raw_equal(optimized, guarded)


def test_dynamic_group_quantizer_propagates_where_out(monkeypatch):
    weight = torch.randn(
        3, 9, generator=torch.Generator().manual_seed(20260726)
    )
    linear = nn.Linear(9, 3, bias=False)
    linear.weight.data.copy_(weight)
    quantizer = _quantizer(
        update_impl="where_out",
        groupsize=4,
        perchannel=True,
        sym=True,
    )
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, 1),
        quantizer=quantizer,
        num_groups=1,
        dev=torch.device("cpu"),
        group_parallel_quant="none",
    )
    realq.H = torch.eye(weight.shape[1]).unsqueeze(0)
    realq.act_square = torch.arange(weight.shape[1], dtype=torch.float32)
    realq._finalized = True

    configured_updates = []
    original_configure = quant_utils.WeightQuantizer.configure

    def recording_configure(self, *args, **kwargs):
        configured_updates.append(kwargs.get("w_clip_update_impl"))
        return original_configure(self, *args, **kwargs)

    monkeypatch.setattr(
        quant_utils.WeightQuantizer, "configure", recording_configure
    )
    realq.quantize(
        blocksize=4,
        percdamp=0.01,
        act_order=False,
        w_clip=True,
        group_parallel_quant="none",
    )
    assert configured_updates
    assert configured_updates == ["where_out"] * len(configured_updates)


def _dynamic_realq_result(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    update_impl: str,
    quantizer_inner_fastpath: bool,
    group_parallel_quant: str,
) -> torch.Tensor:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight.data.copy_(weight)
    quantizer = _quantizer(
        update_impl=update_impl,
        groupsize=4,
        perchannel=True,
        sym=True,
    )
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, 1),
        quantizer=quantizer,
        num_groups=1,
        dev=torch.device("cpu"),
        group_parallel_quant=group_parallel_quant,
    )
    realq.H = hessian.unsqueeze(0)
    realq.act_square = torch.arange(weight.shape[1], dtype=torch.float32)
    realq._finalized = True
    realq.quantize(
        blocksize=4,
        percdamp=0.01,
        act_order=False,
        w_clip=True,
        group_parallel_quant=group_parallel_quant,
        quantizer_inner_fastpath=quantizer_inner_fastpath,
    )
    return linear.weight.detach().clone()


@pytest.mark.parametrize("group_parallel_quant", ["none", "rank"])
@pytest.mark.parametrize("quantizer_inner_fastpath", [False, True])
def test_dynamic_group_where_out_matches_guarded_p01_matrix_raw(
    group_parallel_quant,
    quantizer_inner_fastpath,
):
    generator = torch.Generator().manual_seed(20260728)
    weight = torch.randn(4, 9, generator=generator)
    weight[:, ::2].mul_(0.125)
    weight[:, 1::3].mul_(4.0)
    factor = torch.randn(9, 9, generator=generator)
    hessian = factor.matmul(factor.T).add_(torch.eye(9))

    guarded = _dynamic_realq_result(
        weight,
        hessian,
        update_impl="guarded",
        quantizer_inner_fastpath=quantizer_inner_fastpath,
        group_parallel_quant=group_parallel_quant,
    )
    where_out = _dynamic_realq_result(
        weight,
        hessian,
        update_impl="where_out",
        quantizer_inner_fastpath=quantizer_inner_fastpath,
        group_parallel_quant=group_parallel_quant,
    )
    _assert_raw_equal(where_out, guarded)
