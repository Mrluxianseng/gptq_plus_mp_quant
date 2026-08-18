from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from realq.quant.realq_layer import RealQLayer
from utils.quant_utils import WeightQuantizer


def _quantize(
    module_weight: torch.Tensor,
    *,
    act_order: bool,
    refresh_fn=None,
    initial_weight_fp32: torch.Tensor | None = None,
    refresh_layout: str = "full_natural",
) -> torch.Tensor:
    if refresh_fn is not None:
        # Hand-written callbacks here intentionally exercise the new complete
        # natural-coordinate update contract. Production factory closures
        # carry the same marker themselves.
        refresh_fn._realq_update_layout = refresh_layout
    rows, columns = module_weight.shape
    linear = nn.Linear(columns, rows, bias=False)
    linear.weight.data.copy_(module_weight)
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=3,
        perchannel=True,
        sym=True,
        mse=False,
        weight_groupsize=-1,
    )
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, 1),
        quantizer=quantizer,
        num_groups=1,
        dev=torch.device("cpu"),
        group_parallel_quant="none",
    )
    realq.H = torch.eye(columns).unsqueeze(0)
    # Deliberately non-monotonic so act_order exercises scattered natural
    # locked/active columns rather than a trivial prefix.
    realq.act_square = torch.tensor(
        [0.5, 4.0, 1.0, 6.0, 2.0, 3.0],
        dtype=torch.float32,
    )[:columns]
    realq._finalized = True
    realq.quantize(
        blocksize=2,
        percdamp=0.01,
        act_order=act_order,
        grad_refresh_fn=refresh_fn,
        initial_weight_fp32=initial_weight_fp32,
        group_parallel_quant="none",
    )
    return linear.weight.detach().clone()


def _base_weight() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260729)
    return torch.randn(4, 6, generator=generator) * 0.37


@pytest.mark.parametrize("act_order", [False, True])
def test_realq_layer_ignores_locked_columns_in_full_natural_update(
    act_order: bool,
) -> None:
    weight = _base_weight()

    def zero_refresh(stitched, _trailing_start, perm=None):
        return torch.zeros_like(stitched)

    def locked_only_refresh(stitched, trailing_start, perm=None):
        update = torch.zeros_like(stitched)
        if perm is None:
            update[:, :trailing_start] = 10_000.0
        else:
            update[:, perm[:trailing_start]] = 10_000.0
        return update

    expected = _quantize(
        weight,
        act_order=act_order,
        refresh_fn=zero_refresh,
    )
    actual = _quantize(
        weight,
        act_order=act_order,
        refresh_fn=locked_only_refresh,
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("act_order", [False, True])
def test_realq_layer_applies_active_columns_from_full_natural_update(
    act_order: bool,
) -> None:
    weight = _base_weight()

    def zero_refresh(stitched, _trailing_start, perm=None):
        return torch.zeros_like(stitched)

    def active_refresh(stitched, trailing_start, perm=None):
        update = torch.zeros_like(stitched)
        if perm is None:
            update[:, trailing_start:] = 0.5
        else:
            update[:, perm[trailing_start:]] = 0.5
        return update

    baseline = _quantize(
        weight,
        act_order=act_order,
        refresh_fn=zero_refresh,
    )
    changed = _quantize(
        weight,
        act_order=act_order,
        refresh_fn=active_refresh,
    )
    assert not torch.equal(changed, baseline)


@pytest.mark.parametrize("act_order", [False, True])
def test_transferred_fp32_master_is_the_gptq_starting_weight(
    act_order: bool,
) -> None:
    module_weight = _base_weight()
    transferred = module_weight + torch.linspace(
        -0.4,
        0.6,
        module_weight.numel(),
    ).reshape_as(module_weight)

    expected = _quantize(
        transferred,
        act_order=act_order,
    )
    actual = _quantize(
        module_weight,
        act_order=act_order,
        initial_weight_fp32=transferred.clone(),
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("act_order", [False, True])
def test_compact_full_block_update_matches_full_natural_contract(
    act_order: bool,
) -> None:
    weight = _base_weight()

    def full_refresh(stitched, trailing_start, perm=None):
        update = torch.zeros_like(stitched)
        active = (
            slice(trailing_start, None)
            if perm is None
            else perm[trailing_start:]
        )
        update[:, active] = 0.03125
        return update

    def compact_refresh(stitched, trailing_start, perm=None):
        del perm
        return torch.full(
            (stitched.shape[0], stitched.shape[1] - trailing_start),
            0.03125,
            dtype=stitched.dtype,
            device=stitched.device,
        )

    expected = _quantize(
        weight,
        act_order=act_order,
        refresh_fn=full_refresh,
    )
    actual = _quantize(
        weight,
        act_order=act_order,
        refresh_fn=compact_refresh,
        refresh_layout="trailing_quant_order_full_block",
    )
    assert torch.equal(actual, expected)
