from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from realq_moe.quant.joint_column_quant import (
    JointColumnQuantState,
    JointMoeProjectionStepper,
    _find_groupwise_params_batched_,
)
from realq_moe.quant.realq_layer import RealQLayer
from realq_moe.quant.routed_realq_layer import RoutedRealQLayer
from utils.quant_utils import WeightQuantizer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="joint MoE quantization is a CUDA-only execution path",
)


def _make_quantizer(
    *,
    weight_groupsize: int,
    mse: bool,
    optimized: bool = False,
) -> WeightQuantizer:
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=4,
        perchannel=True,
        sym=True,
        mse=mse,
        weight_groupsize=weight_groupsize,
        w_clip_search_impl=(
            "symmetric_union_exact"
            if optimized
            else "cartesian_legacy"
        ),
        w_clip_update_impl="where_out" if optimized else "guarded",
        w_group_param_layout="compact" if optimized else "expanded",
    )
    return quantizer


def _make_realq(
    *,
    weight: torch.Tensor,
    inputs: torch.Tensor,
    saliency: torch.Tensor,
    weight_groupsize: int,
    mse: bool,
    optimized: bool = False,
) -> RealQLayer:
    dev = weight.device
    linear = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device=dev,
        dtype=weight.dtype,
    )
    linear.weight.data.copy_(weight)
    quantizer = _make_quantizer(
        weight_groupsize=weight_groupsize,
        mse=mse,
        optimized=optimized,
    )
    realq = RealQLayer(
        linear=linear,
        saliency=saliency,
        quantizer=quantizer,
        num_groups=4,
        dev=dev,
        group_parallel_quant="rank",
    )
    realq.add_batch(inputs)
    realq.finalize_hessian()
    return realq


@pytest.mark.parametrize("mse", [False, True], ids=["symmetric", "w_clip"])
@pytest.mark.parametrize(
    "columns",
    [384, 273],
    ids=["three_full_groups", "two_groups_plus_short_tail"],
)
def test_batched_groupwise_params_match_per_expert_find_params(
    mse: bool,
    columns: int,
) -> None:
    torch.manual_seed(8110 + columns + int(mse))
    device = torch.device("cuda")
    expert_count = 3
    rows = 5
    group_width = 128
    weights = (
        torch.randn(
            expert_count,
            rows,
            columns,
            device=device,
            dtype=torch.float32,
        )
        * 0.17
    )
    # Exercise different positive/negative extrema and the clamp floor in
    # distinct expert/group lanes.
    weights[0, 0, 0] = -1.75
    weights[1, 1, min(group_width, columns - 1)] = 2.25
    last_group_start = ((columns - 1) // group_width) * group_width
    weights[2, 2, last_group_start:] = 0.0
    weights[2, 3, -min(7, columns)] = 1.0e-7

    references = [
        _make_quantizer(
            weight_groupsize=group_width,
            mse=mse,
        )
        for _ in range(expert_count)
    ]
    candidates = [
        _make_quantizer(
            weight_groupsize=group_width,
            mse=mse,
        )
        for _ in range(expert_count)
    ]
    for expert_idx, quantizer in enumerate(references):
        quantizer.find_params(weights[expert_idx])

    batched_scale, batched_zero = _find_groupwise_params_batched_(
        candidates,
        weights,
    )
    assert batched_scale.shape == (expert_count, rows, columns)
    assert batched_zero.shape == (expert_count, rows, columns)
    assert batched_scale.is_cuda
    assert batched_zero.is_cuda

    for expert_idx, (reference, candidate) in enumerate(
        zip(references, candidates)
    ):
        assert candidate.sym
        assert candidate.weight_ncolumns == columns
        torch.testing.assert_close(
            candidate.scale,
            reference.scale,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate.zero,
            reference.zero,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate.maxq,
            reference.maxq,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate.quantize(weights[expert_idx]),
            reference.quantize(weights[expert_idx]),
            rtol=0,
            atol=0,
        )


def test_batched_groupwise_params_reject_nonuniform_quantizers() -> None:
    device = torch.device("cuda")
    weights = torch.randn(2, 3, 257, device=device)
    quantizers = [
        _make_quantizer(weight_groupsize=128, mse=True),
        _make_quantizer(weight_groupsize=64, mse=True),
    ]
    with pytest.raises(ValueError, match="identical quantizer settings"):
        _find_groupwise_params_batched_(quantizers, weights)


@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("with_refresh", [False, True])
def test_joint_stepper_matches_monolithic_target_branch(
    act_order: bool,
    with_refresh: bool,
) -> None:
    torch.manual_seed(9207)
    dev = torch.device("cuda")
    # Non-square shape also exercises output-row Hessian grouping.
    weight = torch.randn(
        12, 10, device=dev, dtype=torch.float32
    ) * 0.05
    inputs = torch.randn(
        3, 5, 10, device=dev, dtype=torch.float32
    )
    saliency = (
        torch.rand(3, 5, 4, device=dev, dtype=torch.float32)
        + 0.1
    )
    blocksize = 4
    weight_groupsize = blocksize

    reference = _make_realq(
        weight=weight,
        inputs=inputs,
        saliency=saliency,
        weight_groupsize=weight_groupsize,
        mse=False,
    )
    candidate = _make_realq(
        weight=weight,
        inputs=inputs,
        saliency=saliency,
        weight_groupsize=weight_groupsize,
        mse=False,
    )

    reference_calls: list[tuple[int, torch.Tensor | None]] = []

    def deterministic_refresh(
        stitched_weight_fp32: torch.Tensor,
        trailing_col_start: int,
        perm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reference_calls.append(
            (
                trailing_col_start,
                None if perm is None else perm.detach().clone(),
            )
        )
        value = 1.0e-4 * len(reference_calls)
        return torch.full(
            (
                stitched_weight_fp32.shape[0],
                stitched_weight_fp32.shape[1] - trailing_col_start,
            ),
            value,
            dtype=stitched_weight_fp32.dtype,
            device=stitched_weight_fp32.device,
        )

    reference.quantize(
        blocksize=blocksize,
        percdamp=0.01,
        act_order=act_order,
        w_clip=False,
        grad_refresh_fn=(
            deterministic_refresh if with_refresh else None
        ),
        group_parallel_quant="rank",
        quantizer_inner_fastpath=False,
        act_order_stitch_impl="full_weight_legacy",
    )

    state = JointColumnQuantState(
        candidate,
        blocksize=blocksize,
        percdamp=0.01,
        act_order=act_order,
        w_clip=False,
    )
    joint_calls = 0
    while not state.finished:
        boundary = state.advance_block()
        if boundary is None:
            continue
        if with_refresh:
            joint_calls += 1
            update = torch.full(
                (
                    boundary.weight_fp32.shape[0],
                    boundary.weight_fp32.shape[1]
                    - boundary.trailing_col_start,
                ),
                1.0e-4 * joint_calls,
                dtype=boundary.weight_fp32.dtype,
                device=boundary.weight_fp32.device,
            )
            state.apply_update(update)
    state.writeback()

    assert joint_calls == len(reference_calls)
    assert torch.equal(
        candidate.linear.weight, reference.linear.weight
    )
    assert state.Hinv.is_cuda
    assert state.W.is_cuda
    assert state.Q.is_cuda


@pytest.mark.parametrize("with_refresh", [False, True])
def test_expert_batched_stepper_matches_independent_experts(
    with_refresh: bool,
) -> None:
    torch.manual_seed(11209)
    dev = torch.device("cuda")
    expert_count = 3
    blocksize = 4
    inputs = torch.randn(
        2, 4, 10, device=dev, dtype=torch.float32
    )
    saliencies = [
        torch.rand(2, 4, 4, device=dev) + 0.2
        for _ in range(expert_count)
    ]
    weights = [
        torch.randn(12, 10, device=dev) * 0.04
        for _ in range(expert_count)
    ]

    references = [
        _make_realq(
            weight=weights[e],
            inputs=inputs,
            saliency=saliencies[e],
            weight_groupsize=blocksize,
            mse=False,
        )
        for e in range(expert_count)
    ]
    candidates = [
        _make_realq(
            weight=weights[e],
            inputs=inputs,
            saliency=saliencies[e],
            weight_groupsize=blocksize,
            mse=False,
        )
        for e in range(expert_count)
    ]

    def make_reference_refresh(expert_idx: int):
        call = {"n": 0}

        def refresh(
            stitched_weight_fp32: torch.Tensor,
            trailing_col_start: int,
            perm: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del perm
            call["n"] += 1
            return torch.full(
                (
                    stitched_weight_fp32.shape[0],
                    stitched_weight_fp32.shape[1]
                    - trailing_col_start,
                ),
                (expert_idx + 1) * call["n"] * 1.0e-4,
                device=stitched_weight_fp32.device,
                dtype=stitched_weight_fp32.dtype,
            )

        return refresh

    for expert_idx, reference in enumerate(references):
        reference.quantize(
            blocksize=blocksize,
            percdamp=0.01,
            act_order=True,
            w_clip=False,
            grad_refresh_fn=(
                make_reference_refresh(expert_idx)
                if with_refresh
                else None
            ),
            group_parallel_quant="rank",
            quantizer_inner_fastpath=False,
            act_order_stitch_impl="full_weight_legacy",
        )

    stepper = JointMoeProjectionStepper(
        candidates,
        blocksize=blocksize,
        percdamp=0.01,
        act_order=True,
        w_clip=False,
    )
    # The batched stepper has copied the expert Hessians and activation
    # ordering; retaining the source statistics would double the projection's
    # Hessian residency on real 128-expert layers.
    assert all(candidate.H is None for candidate in candidates)
    assert all(candidate.act_square is None for candidate in candidates)
    for reference, candidate in zip(references, candidates):
        torch.testing.assert_close(
            candidate.quantizer.scale,
            reference.quantizer.scale,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate.quantizer.zero,
            reference.quantizer.zero,
            rtol=0,
            atol=0,
        )
    refresh_step = 0
    while not stepper.finished:
        boundary = stepper.advance_block()
        if boundary is None:
            continue
        if with_refresh:
            refresh_step += 1
            updates = torch.stack(
                [
                    torch.full(
                        (
                            stepper.rows,
                            stepper.columns
                            - boundary.trailing_col_start,
                        ),
                        (expert_idx + 1)
                        * refresh_step
                        * 1.0e-4,
                        device=dev,
                        dtype=torch.float32,
                    )
                    for expert_idx in range(expert_count)
                ],
                dim=0,
            )
            stepper.apply_updates(updates)
    stepper.writeback()

    for reference, candidate in zip(references, candidates):
        torch.testing.assert_close(
            candidate.linear.weight,
            reference.linear.weight,
            rtol=0,
            atol=2e-7,
        )
    assert stepper.Hinv.is_cuda
    assert stepper.W.is_cuda
    assert stepper.Q.is_cuda


def test_zero_route_lane_is_exact_rtn_and_ignores_refresh() -> None:
    torch.manual_seed(11210)
    dev = torch.device("cuda")
    rows, columns, blocksize = 12, 10, 4
    hot_weight = torch.randn(rows, columns, device=dev) * 0.04
    cold_weight = torch.randn(rows, columns, device=dev) * 0.04
    hot = _make_realq(
        weight=hot_weight,
        inputs=torch.randn(2, 4, columns, device=dev),
        saliency=torch.rand(2, 4, 4, device=dev) + 0.2,
        weight_groupsize=blocksize,
        mse=False,
    )

    cold_linear = nn.Linear(
        columns,
        rows,
        bias=False,
        device=dev,
        dtype=cold_weight.dtype,
    )
    cold_linear.weight.data.copy_(cold_weight)
    cold_quantizer = _make_quantizer(
        weight_groupsize=blocksize,
        mse=False,
    )
    cold = RoutedRealQLayer(
        linear=cold_linear,
        saliency=torch.empty(0, 1, 4, device=dev),
        quantizer=cold_quantizer,
        num_groups=4,
        dev=dev,
        normalization_token_count=8,
        group_parallel_quant="rank",
    )
    cold.finalize_rtn_fallback("teacher_zero_route")

    reference_quantizer = _make_quantizer(
        weight_groupsize=blocksize,
        mse=False,
    )
    reference_quantizer.find_params(cold_weight)
    expected_rtn = reference_quantizer.quantize(cold_weight)

    stepper = JointMoeProjectionStepper(
        [hot, cold],
        blocksize=blocksize,
        percdamp=0.01,
        act_order=True,
        w_clip=False,
    )
    while not stepper.finished:
        boundary = stepper.advance_block()
        if boundary is None:
            continue
        stepper.apply_updates(
            torch.full(
                (
                    2,
                    rows,
                    columns - boundary.trailing_col_start,
                ),
                0.25,
                device=dev,
            )
        )
    stepper.writeback()

    torch.testing.assert_close(
        cold_linear.weight,
        expected_rtn,
        rtol=0,
        atol=0,
    )


def test_optimized_moe_stepper_triton_matches_reference() -> None:
    dev = torch.device("cuda")
    torch.manual_seed(20260728)
    expert_count, rows, columns, blocksize = 2, 12, 18, 8
    inputs = torch.randn(2, 5, columns, device=dev)
    saliencies = [
        torch.rand(2, 5, 4, device=dev)
        for _ in range(expert_count)
    ]
    weights = [
        torch.randn(rows, columns, device=dev) * 0.04
        for _ in range(expert_count)
    ]

    def build():
        return [
            _make_realq(
                weight=weights[expert_idx],
                inputs=inputs,
                saliency=saliencies[expert_idx],
                weight_groupsize=blocksize,
                mse=True,
                optimized=True,
            )
            for expert_idx in range(expert_count)
        ]

    reference = JointMoeProjectionStepper(
        build(),
        blocksize=blocksize,
        percdamp=0.01,
        act_order=True,
        w_clip=True,
        triton_column_block=False,
    )
    candidate = JointMoeProjectionStepper(
        build(),
        blocksize=blocksize,
        percdamp=0.01,
        act_order=True,
        w_clip=True,
        triton_column_block=True,
    )
    assert reference.scales.shape[-1] == (
        columns + blocksize - 1
    ) // blocksize
    while not reference.finished:
        reference.advance_block()
    while not candidate.finished:
        candidate.advance_block()
    reference.writeback()
    candidate.writeback()
    torch.cuda.synchronize()

    for expected, actual in zip(
        reference.linears, candidate.linears
    ):
        assert torch.equal(actual.weight, expected.weight)
