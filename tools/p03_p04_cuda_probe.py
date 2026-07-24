"""Strict CUDA/NCCL exactness gate for the P03/P04 weight optimizations.

This probe intentionally has no CPU fallback.  The caller selects the physical
cards with ``CUDA_VISIBLE_DEVICES`` and launches one, two, or four ranks:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 REALQ_P03_P04_PROBE_DEVICE=cuda \
      CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \
      torchrun --standalone --nproc_per_node=1 \
        tools/p03_p04_cuda_probe.py

    CUBLAS_WORKSPACE_CONFIG=:4096:8 REALQ_P03_P04_PROBE_DEVICE=cuda \
      CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. \
      torchrun --standalone --nproc_per_node=4 \
        tools/p03_p04_cuda_probe.py

World size one covers both ``group_parallel_quant=none`` and ``rank``.  Larger
world sizes exercise the real NCCL row-sharded path with Fisher-MSE backward,
gradient all-reduce, Adam updates, and exact P06 weight stitching.  The gate
compares raw bytes, not tolerances.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

from realq.quant.realq_layer import RealQLayer
from realq.refresh.block_gd import (
    RefreshContext,
    _SharedSampleScheduler,
    make_grad_refresh_fn,
)
from utils.quant_utils import WeightQuantizer


_CARTESIAN = "cartesian_legacy"
_UNION = "symmetric_union_exact"
_GUARDED = "guarded"
_WHERE_OUT = "where_out"
_EXPANDED = "expanded"
_COMPACT = "compact"
_EXACT_STITCH = "prefix_q_trailing_w_exact"


def _assert_raw_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
) -> None:
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype {actual.dtype} != {expected.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    actual_bytes = actual.detach().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes), (
        f"{label}: raw bytes differ; "
        f"elements={(actual != expected).sum().item()}"
    )


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _quantizer(
    *,
    groupsize: int,
    sym: bool = True,
    mse: bool = True,
    search_impl: str = _CARTESIAN,
    update_impl: str = _GUARDED,
    layout: str = _EXPANDED,
) -> WeightQuantizer:
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=4,
        perchannel=True,
        sym=sym,
        mse=mse,
        norm=2.4,
        grid=4,
        maxshrink=0.5,
        weight_groupsize=groupsize,
        w_clip_search_impl=search_impl,
        w_clip_update_impl=update_impl,
        w_group_param_layout=layout,
    )
    return quantizer


def _observer_outputs(
    weight: torch.Tensor,
    *,
    groupsize: int,
    sym: bool,
    search_impl: str,
    update_impl: str,
    layout: str,
) -> tuple[WeightQuantizer, tuple[torch.Tensor, ...]]:
    quantizer = _quantizer(
        groupsize=groupsize,
        sym=sym,
        search_impl=search_impl,
        update_impl=update_impl,
        layout=layout,
    )
    quantizer.find_params(weight.clone())
    fake, integer, returned_scale = quantizer.fake_quantize(weight)
    return quantizer, (
        quantizer.scale.detach().clone(),
        quantizer.zero.detach().clone(),
        fake.detach().clone(),
        integer.detach().clone(),
        returned_scale.detach().clone(),
    )


def _structured_weight(
    rows: int,
    columns: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, columns, generator=generator)
    weight[:, ::3].mul_(0.0625)
    weight[:, 1::4].mul_(6.0)
    weight[0].zero_()
    weight[0, ::2] = -0.0
    return weight.to(device)


def _run_p03_observer_gate(device: torch.device) -> list[dict[str, object]]:
    """Guarded and where-out must match in every P02 dispatch domain."""

    cases = [
        ("row_cartesian", -1, True, _CARTESIAN, False),
        ("row_asymmetric_fallback", -1, False, _UNION, False),
        ("row_union_bypass", -1, True, _UNION, False),
        ("row_nonfinite_fallback", -1, True, _UNION, True),
        ("group_short_cartesian", 4, True, _CARTESIAN, False),
        ("group_short_asymmetric_fallback", 4, False, _UNION, False),
        ("group_short_union_bypass", 4, True, _UNION, False),
        ("group_short_nonfinite_fallback", 4, True, _UNION, True),
    ]
    summaries = []
    for offset, (name, groupsize, sym, search, nonfinite) in enumerate(cases):
        weight = _structured_weight(
            5, 10 if groupsize > 0 else 17,
            seed=3100 + offset,
            device=device,
        )
        if nonfinite:
            weight[1, 1] = float("nan")
            weight[2, -2] = float("inf")
            weight[3, -1] = -float("inf")
        guarded, guarded_outputs = _observer_outputs(
            weight,
            groupsize=groupsize,
            sym=sym,
            search_impl=search,
            update_impl=_GUARDED,
            layout=_EXPANDED,
        )
        where_out, where_outputs = _observer_outputs(
            weight,
            groupsize=groupsize,
            sym=sym,
            search_impl=search,
            update_impl=_WHERE_OUT,
            layout=_EXPANDED,
        )
        for field, actual, expected in zip(
            ("scale", "zero", "fake", "integer_q", "returned_scale"),
            where_outputs,
            guarded_outputs,
        ):
            _assert_raw_equal(actual, expected, label=f"P03/{name}/{field}")
        expected_dispatch = (
            "union"
            if search == _UNION and sym and not nonfinite
            else "cartesian"
        )
        assert guarded._can_use_symmetric_union(weight) == (
            expected_dispatch == "union"
        )
        assert where_out._can_use_symmetric_union(weight) == (
            expected_dispatch == "union"
        )
        summaries.append(
            {
                "name": name,
                "groupsize": groupsize,
                "sym": sym,
                "p02_dispatch": expected_dispatch,
                "scale_shape": tuple(where_out.scale.shape),
            }
        )
    return summaries


def _expand_qparam(
    quantizer: WeightQuantizer,
    parameter: torch.Tensor,
    columns: int,
) -> torch.Tensor:
    if quantizer.weight_groupsize <= 0:
        return parameter
    group_ids = torch.arange(
        columns, device=parameter.device, dtype=torch.long
    ) // quantizer.weight_groupsize
    return parameter.index_select(-1, group_ids)


def _run_p04_observer_gate(device: torch.device) -> list[dict[str, object]]:
    """Expanded and compact qparams must expose identical public arithmetic."""

    cases = [
        ("row_noop_symmetric", -1, 17, True, _CARTESIAN),
        ("row_noop_asymmetric", -1, 17, False, _UNION),
        ("group_exact_symmetric", 4, 8, True, _CARTESIAN),
        ("group_short_symmetric", 4, 10, True, _UNION),
        ("group_short_asymmetric", 4, 10, False, _UNION),
        ("group_larger_than_width_symmetric", 32, 10, True, _UNION),
        ("group_larger_than_width_asymmetric", 32, 10, False, _CARTESIAN),
    ]
    summaries = []
    for offset, (name, groupsize, columns, sym, search) in enumerate(cases):
        weight = _structured_weight(
            6, columns, seed=4100 + offset, device=device
        )
        expanded, expanded_outputs = _observer_outputs(
            weight,
            groupsize=groupsize,
            sym=sym,
            search_impl=search,
            update_impl=_WHERE_OUT,
            layout=_EXPANDED,
        )
        compact, compact_outputs = _observer_outputs(
            weight,
            groupsize=groupsize,
            sym=sym,
            search_impl=search,
            update_impl=_WHERE_OUT,
            layout=_COMPACT,
        )
        if groupsize > 0:
            _assert_raw_equal(
                _expand_qparam(compact, compact_outputs[0], columns),
                expanded_outputs[0],
                label=f"P04/{name}/expanded_scale",
            )
            _assert_raw_equal(
                _expand_qparam(compact, compact_outputs[1], columns),
                expanded_outputs[1],
                label=f"P04/{name}/expanded_zero",
            )
            fields = ("fake", "integer_q", "returned_scale")
            compact_public = compact_outputs[2:]
            expanded_public = expanded_outputs[2:]
        else:
            # P04 is deliberately inactive for ordinary per-row qparams.
            fields = (
                "scale",
                "zero",
                "fake",
                "integer_q",
                "returned_scale",
            )
            compact_public = compact_outputs
            expanded_public = expanded_outputs
        for field, actual, expected in zip(
            fields,
            compact_public,
            expanded_public,
        ):
            _assert_raw_equal(actual, expected, label=f"P04/{name}/{field}")
        _assert_raw_equal(
            compact.quantize(weight),
            expanded.quantize(weight),
            label=f"P04/{name}/generic_quantize",
        )
        summaries.append(
            {
                "name": name,
                "groupsize": groupsize,
                "columns": columns,
                "sym": sym,
                "expanded_shape": tuple(expanded.scale.shape),
                "compact_shape": tuple(compact.scale.shape),
            }
        )

    # Act-order uses natural (pre-permutation) column coordinates.  Check both
    # the public API and P01's prevalidated private path on a row slice.
    columns = 10
    groupsize = 4
    weight = _structured_weight(8, columns, seed=4199, device=device)
    expanded, _ = _observer_outputs(
        weight,
        groupsize=groupsize,
        sym=True,
        search_impl=_UNION,
        update_impl=_WHERE_OUT,
        layout=_EXPANDED,
    )
    compact, _ = _observer_outputs(
        weight,
        groupsize=groupsize,
        sym=True,
        search_impl=_UNION,
        update_impl=_WHERE_OUT,
        layout=_COMPACT,
    )
    permutation = torch.tensor(
        [9, 0, 7, 1, 8, 2, 6, 3, 5, 4],
        device=device,
        dtype=torch.long,
    )
    row_start, row_end = 2, 7
    selected = weight[row_start:row_end].index_select(1, permutation)
    expanded_public = expanded.fake_quantize(
        selected,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=permutation,
    )
    compact_public = compact.fake_quantize(
        selected,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=permutation,
    )
    for field, actual, expected in zip(
        ("fake", "integer_q", "returned_scale"),
        compact_public,
        expanded_public,
    ):
        _assert_raw_equal(
            actual, expected, label=f"P04/act_order/public/{field}"
        )
    expanded_prepared = expanded._prepare_fake_quantize_inner(
        input_rows=row_end - row_start,
        column_count=columns,
        device=device,
        dtype=selected.dtype,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=permutation,
    )
    compact_prepared = compact._prepare_fake_quantize_inner(
        input_rows=row_end - row_start,
        column_count=columns,
        device=device,
        dtype=selected.dtype,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=permutation,
    )
    _assert_raw_equal(
        compact_prepared.prepared_scale,
        expanded_prepared.prepared_scale,
        label="P04/act_order/P01/prepared_scale",
    )
    for column_offset in range(columns):
        expanded_column = expanded._fake_quantize_prevalidated(
            selected[:, column_offset : column_offset + 1],
            expanded_prepared,
            column_offset,
        )
        compact_column = compact._fake_quantize_prevalidated(
            selected[:, column_offset : column_offset + 1],
            compact_prepared,
            column_offset,
        )
        for field, actual, expected in zip(
            ("fake", "integer_q", "returned_scale"),
            compact_column,
            expanded_column,
        ):
            _assert_raw_equal(
                actual,
                expected,
                label=(
                    f"P04/act_order/P01/column_{column_offset}/{field}"
                ),
            )
    summaries.append(
        {
            "name": "act_order_natural_mapping",
            "groupsize": groupsize,
            "columns": columns,
            "row_slice": (row_start, row_end),
            "permutation": permutation.tolist(),
        }
    )
    return summaries


@dataclass(frozen=True)
class _Implementation:
    name: str
    p01: bool
    p02: str
    p03: str
    p04: str


_LEGACY = _Implementation(
    "legacy", False, _CARTESIAN, _GUARDED, _EXPANDED
)


@dataclass(frozen=True)
class _QuantCase:
    name: str
    rows: int
    columns: int
    blocksize: int
    groupsize: int
    seed: int


@dataclass
class _QuantResult:
    weight: torch.Tensor
    expanded_scale: torch.Tensor
    expanded_zero: torch.Tensor
    stitched_inputs: list[torch.Tensor]
    updates: list[torch.Tensor]
    exp_avg: torch.Tensor | None
    exp_avg_sq: torch.Tensor | None
    adam_step: int
    collective_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]]
    peak_memory_bytes: int


class _ToyLayer(nn.Module):
    def __init__(self, columns: int, rows: int) -> None:
        super().__init__()
        self.proj = nn.Linear(columns, rows, bias=False)

    def forward(self, hidden_states, **_kwargs):
        return (self.proj(hidden_states),)


def _case_tensors(
    case: _QuantCase,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(case.seed)
    weight = torch.randn(case.rows, case.columns, generator=generator)
    weight[:, ::3].mul_(0.125)
    weight[:, 1::4].mul_(5.0)
    factor = torch.randn(
        case.columns, case.columns, generator=generator
    )
    hessian = factor.matmul(factor.T).add_(torch.eye(case.columns))
    act_square = torch.rand(case.columns, generator=generator)
    return weight.to(device), hessian.to(device), act_square.to(device)


def _run_quant_case(
    case: _QuantCase,
    implementation: _Implementation,
    *,
    device: torch.device,
    group_parallel_quant: str,
    with_refresh: bool,
) -> _QuantResult:
    rank = dist.get_rank()
    world = dist.get_world_size()
    weight, hessian, act_square = _case_tensors(case, device)
    layer = _ToyLayer(case.columns, case.rows).to(device)
    layer.proj.weight.data.copy_(weight)
    quantizer = _quantizer(
        groupsize=case.groupsize,
        sym=True,
        mse=True,
        search_impl=implementation.p02,
        update_impl=implementation.p03,
        layout=implementation.p04,
    )
    realq = RealQLayer(
        linear=layer.proj,
        saliency=torch.ones(1, 1, 1),
        quantizer=quantizer,
        num_groups=1,
        dev=device,
        group_parallel_quant=group_parallel_quant,
    )
    realq.H = hessian.unsqueeze(0)
    realq.act_square = act_square
    realq._finalized = True

    stitched_inputs: list[torch.Tensor] = []
    updates: list[torch.Tensor] = []
    context = None
    refresh = None
    if with_refresh:
        samples_per_rank = 2
        global_samples = world * samples_per_rank
        generator = torch.Generator().manual_seed(case.seed + 91)
        global_inputs = torch.randn(
            global_samples, 3, case.columns, generator=generator
        ).to(device)
        global_targets = torch.randn(
            global_samples, 3, case.rows, generator=generator
        ).to(device)
        local_slice = slice(
            rank * samples_per_rank,
            (rank + 1) * samples_per_rank,
        )
        layer_state = SimpleNamespace(
            inps=global_inputs[local_slice].clone(),
            attention_mask=None,
            position_ids=None,
            position_embeddings=None,
        )
        context = RefreshContext(
            module=layer.proj,
            layer_lr=3e-4,
            grad_clip=1.0,
            backward_bsz=1,
            scheduler=_SharedSampleScheduler(
                global_samples,
                global_samples,
                seed=77,
            ),
        )
        real_refresh = make_grad_refresh_fn(
            layer=layer,
            module=layer.proj,
            layer_state=layer_state,
            fp_out_for_this_layer=global_targets[local_slice].clone(),
            fisher=torch.eye(case.rows, device=device),
            ctx=context,
        )

        def observed_refresh(stitched, trailing_col_start, perm=None):
            stitched_inputs.append(stitched.detach().clone())
            update = real_refresh(
                stitched, trailing_col_start, perm=perm
            )
            updates.append(update.detach().clone())
            return update

        refresh = observed_refresh

    collective_shapes: list[
        tuple[tuple[int, ...], tuple[int, ...]]
    ] = []
    original_all_gather = dist.all_gather_into_tensor

    def observed_all_gather(output, local, *args, **kwargs):
        collective_shapes.append((tuple(output.shape), tuple(local.shape)))
        return original_all_gather(output, local, *args, **kwargs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    dist.all_gather_into_tensor = observed_all_gather
    try:
        realq.quantize(
            blocksize=case.blocksize,
            percdamp=0.01,
            act_order=True,
            w_clip=True,
            grad_refresh_fn=refresh,
            group_parallel_quant=group_parallel_quant,
            quantizer_inner_fastpath=implementation.p01,
            act_order_stitch_impl=_EXACT_STITCH,
        )
    finally:
        dist.all_gather_into_tensor = original_all_gather
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory = 0

    scale = quantizer.scale.detach().clone()
    zero = quantizer.zero.detach().clone()
    if case.groupsize > 0 and implementation.p04 == _COMPACT:
        scale = _expand_qparam(quantizer, scale, case.columns)
        zero = _expand_qparam(quantizer, zero, case.columns)
    return _QuantResult(
        weight=layer.proj.weight.detach().clone(),
        expanded_scale=scale,
        expanded_zero=zero,
        stitched_inputs=stitched_inputs,
        updates=updates,
        exp_avg=(
            context.exp_avg.detach().clone() if context is not None else None
        ),
        exp_avg_sq=(
            context.exp_avg_sq.detach().clone()
            if context is not None
            else None
        ),
        adam_step=context.adam_step if context is not None else 0,
        collective_shapes=collective_shapes,
        peak_memory_bytes=peak_memory,
    )


def _assert_quant_results_equal(
    actual: _QuantResult,
    expected: _QuantResult,
    *,
    label: str,
) -> None:
    _assert_raw_equal(actual.weight, expected.weight, label=f"{label}/weight")
    _assert_raw_equal(
        actual.expanded_scale,
        expected.expanded_scale,
        label=f"{label}/expanded_scale",
    )
    _assert_raw_equal(
        actual.expanded_zero,
        expected.expanded_zero,
        label=f"{label}/expanded_zero",
    )
    assert actual.adam_step == expected.adam_step, (
        f"{label}/adam_step: {actual.adam_step} != {expected.adam_step}"
    )
    assert len(actual.stitched_inputs) == len(expected.stitched_inputs)
    assert len(actual.updates) == len(expected.updates)
    for index, (actual_tensor, expected_tensor) in enumerate(
        zip(actual.stitched_inputs, expected.stitched_inputs)
    ):
        _assert_raw_equal(
            actual_tensor,
            expected_tensor,
            label=f"{label}/stitched_{index}",
        )
    for index, (actual_tensor, expected_tensor) in enumerate(
        zip(actual.updates, expected.updates)
    ):
        _assert_raw_equal(
            actual_tensor,
            expected_tensor,
            label=f"{label}/update_{index}",
        )
    if expected.exp_avg is None:
        assert actual.exp_avg is None and actual.exp_avg_sq is None
    else:
        assert actual.exp_avg is not None and actual.exp_avg_sq is not None
        _assert_raw_equal(
            actual.exp_avg,
            expected.exp_avg,
            label=f"{label}/exp_avg",
        )
        _assert_raw_equal(
            actual.exp_avg_sq,
            expected.exp_avg_sq,
            label=f"{label}/exp_avg_sq",
        )


def _expected_collectives(
    case: _QuantCase,
    implementation: _Implementation,
    world: int,
    *,
    with_refresh: bool,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if world == 1:
        return []
    local_rows = case.rows // world
    if case.groupsize <= 0:
        qparam_columns = 1
    elif implementation.p04 == _COMPACT:
        qparam_columns = math.ceil(case.columns / case.groupsize)
    else:
        qparam_columns = case.columns
    expected = [
        ((case.rows, qparam_columns), (local_rows, qparam_columns)),
        ((case.rows, qparam_columns), (local_rows, qparam_columns)),
    ]
    if with_refresh:
        for start in range(0, case.columns, case.blocksize):
            end = min(start + case.blocksize, case.columns)
            if end >= case.columns:
                break
            count = end - start
            expected.extend(
                [
                    ((case.rows, count), (local_rows, count)),
                    (
                        (case.rows, case.columns - end),
                        (local_rows, case.columns - end),
                    ),
                ]
            )
    expected.append(
        ((case.rows, case.columns), (local_rows, case.columns))
    )
    return expected


def _candidate_matrix(
    case: _QuantCase,
) -> list[_Implementation]:
    candidates = []
    layouts = (_EXPANDED,) if case.groupsize <= 0 else (
        _EXPANDED,
        _COMPACT,
    )
    for p01, p02, p03, p04 in product(
        (False, True),
        (_CARTESIAN, _UNION),
        (_GUARDED, _WHERE_OUT),
        layouts,
    ):
        name = (
            f"p01_{int(p01)}-"
            f"p02_{'cart' if p02 == _CARTESIAN else 'union'}-"
            f"p03_{'guard' if p03 == _GUARDED else 'where'}-"
            f"p04_{p04}"
        )
        candidates.append(_Implementation(name, p01, p02, p03, p04))
    return candidates


def _assert_replicated(result: _QuantResult, world: int, label: str) -> str:
    digest = _sha256(result.weight)
    gathered = [None] * world
    dist.all_gather_object(gathered, digest)
    assert all(item == digest for item in gathered), (
        f"{label}: ranks produced different final weights: {gathered}"
    )
    return digest


def _run_world_one_gate(
    device: torch.device,
) -> list[dict[str, object]]:
    assert dist.get_world_size() == 1
    cases = [
        _QuantCase("row", 8, 10, 4, -1, 5101),
        _QuantCase("group_short", 8, 9, 4, 4, 5102),
        _QuantCase("group_gt_width", 8, 9, 4, 32, 5103),
    ]
    summaries = []
    for mode in ("none", "rank"):
        for case in cases:
            baseline = _run_quant_case(
                case,
                _LEGACY,
                device=device,
                group_parallel_quant=mode,
                with_refresh=False,
            )
            max_peak = baseline.peak_memory_bytes
            for implementation in _candidate_matrix(case):
                result = _run_quant_case(
                    case,
                    implementation,
                    device=device,
                    group_parallel_quant=mode,
                    with_refresh=False,
                )
                _assert_quant_results_equal(
                    result,
                    baseline,
                    label=f"world1/{mode}/{case.name}/{implementation.name}",
                )
                assert result.collective_shapes == []
                max_peak = max(max_peak, result.peak_memory_bytes)
            # Detect order/allocator-history sensitivity after every candidate.
            repeat = _run_quant_case(
                case,
                _LEGACY,
                device=device,
                group_parallel_quant=mode,
                with_refresh=False,
            )
            _assert_quant_results_equal(
                repeat,
                baseline,
                label=f"world1/{mode}/{case.name}/legacy_repeat",
            )
            summaries.append(
                {
                    "mode": mode,
                    "case": case.name,
                    "candidate_count": len(_candidate_matrix(case)),
                    "sha256": _sha256(baseline.weight),
                    "peak_memory_bytes": max_peak,
                }
            )
    return summaries


def _run_distributed_gate(
    device: torch.device,
) -> list[dict[str, object]]:
    world = dist.get_world_size()
    assert world in (2, 4)
    cases = [
        _QuantCase("row", 8, 10, 4, -1, 6100 + world),
        _QuantCase("group_short", 8, 9, 4, 4, 6200 + world),
    ]
    summaries = []
    for case in cases:
        baseline = _run_quant_case(
            case,
            _LEGACY,
            device=device,
            group_parallel_quant="rank",
            with_refresh=True,
        )
        assert baseline.collective_shapes == _expected_collectives(
            case, _LEGACY, world, with_refresh=True
        )
        assert baseline.adam_step == math.ceil(
            case.columns / case.blocksize
        ) - 1
        max_peak = baseline.peak_memory_bytes
        candidate_count = 0
        collective_variants = {}
        for implementation in _candidate_matrix(case):
            result = _run_quant_case(
                case,
                implementation,
                device=device,
                group_parallel_quant="rank",
                with_refresh=True,
            )
            _assert_quant_results_equal(
                result,
                baseline,
                label=(
                    f"world{world}/{case.name}/{implementation.name}"
                ),
            )
            expected_collectives = _expected_collectives(
                case, implementation, world, with_refresh=True
            )
            assert result.collective_shapes == expected_collectives, (
                f"{case.name}/{implementation.name}: collectives "
                f"{result.collective_shapes} != {expected_collectives}"
            )
            key = f"{implementation.p04}"
            collective_variants[key] = result.collective_shapes
            _assert_replicated(
                result,
                world,
                f"world{world}/{case.name}/{implementation.name}",
            )
            candidate_count += 1
            max_peak = max(max_peak, result.peak_memory_bytes)
            dist.barrier()
        repeat = _run_quant_case(
            case,
            _LEGACY,
            device=device,
            group_parallel_quant="rank",
            with_refresh=True,
        )
        _assert_quant_results_equal(
            repeat,
            baseline,
            label=f"world{world}/{case.name}/legacy_repeat",
        )
        digest = _assert_replicated(
            repeat, world, f"world{world}/{case.name}/legacy_repeat"
        )
        summaries.append(
            {
                "case": case.name,
                "candidate_count": candidate_count,
                "refreshes": baseline.adam_step,
                "collective_count": len(baseline.collective_shapes),
                "collective_shapes_by_layout": collective_variants,
                "sha256": digest,
                "peak_memory_bytes": max_peak,
            }
        )
        dist.barrier()
    return summaries


def _resolve_device() -> tuple[torch.device, int]:
    assert os.environ.get("REALQ_P03_P04_PROBE_DEVICE") == "cuda", (
        "Set REALQ_P03_P04_PROBE_DEVICE=cuda explicitly."
    )
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    assert workspace in (":4096:8", ":16:8"), (
        "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts."
    )
    assert torch.cuda.is_available(), "CUDA is unavailable."
    local_rank = int(os.environ["LOCAL_RANK"])
    assert 0 <= local_rank < torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda", local_rank), local_rank


def main() -> None:
    device, local_rank = _resolve_device()
    torch.set_num_threads(1)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world not in (1, 2, 4):
        raise ValueError(
            f"P03/P04 CUDA probe requires world size 1, 2, or 4; got {world}."
        )

    # Observer-level gates are independent on each selected CUDA device.  This
    # catches device-specific where/index and compact/index_select behavior
    # rather than checking rank zero only.
    p03_observers = _run_p03_observer_gate(device)
    p04_observers = _run_p04_observer_gate(device)
    dist.barrier()

    if world == 1:
        quant_summaries = _run_world_one_gate(device)
    else:
        quant_summaries = _run_distributed_gate(device)

    properties = torch.cuda.get_device_properties(device)
    local_device_info = {
        "rank": dist.get_rank(),
        "local_rank": local_rank,
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "unavailable")),
        "total_memory_bytes": properties.total_memory,
        "peak_memory_bytes": max(
            (item["peak_memory_bytes"] for item in quant_summaries),
            default=torch.cuda.max_memory_allocated(device),
        ),
    }
    device_infos = [None] * world
    dist.all_gather_object(device_infos, local_device_info)
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "world_size": world,
                    "backend": dist.get_backend(),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ),
                    "cublas_workspace_config": os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    ),
                    "devices": device_infos,
                    "p03_observer_cases": p03_observers,
                    "p04_observer_cases": p04_observers,
                    "quantization_cases": quant_summaries,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
