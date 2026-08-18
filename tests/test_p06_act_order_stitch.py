from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from realq.config import Config, parse_cli
from realq.quant.realq_layer import (
    RealQLayer,
    _rebuild_permuted_weight_from_prefix_and_trailing_,
)
from utils import checkpoint_utils, dist_utils
from utils.quant_utils import WeightQuantizer


LEGACY = "full_weight_legacy"
EXACT = "prefix_q_trailing_w_exact"


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _float32_from_bits(patterns: list[int], shape: tuple[int, ...]) -> torch.Tensor:
    signed = [
        pattern if pattern < 2**31 else pattern - 2**32
        for pattern in patterns
    ]
    return torch.tensor(signed, dtype=torch.int32).view(torch.float32).reshape(
        shape
    )


@pytest.mark.parametrize("trailing_col_start", [0, 1, 4, 7])
def test_rebuild_helper_preserves_arbitrary_float32_raw_bits(
    trailing_col_start: int,
):
    # Includes positive/negative zero, infinities, distinct NaN payloads,
    # subnormals, and ordinary values. The helper is a copy-only primitive and
    # must preserve all payload bits rather than merely compare numerically.
    q_patterns = [
        0x00000000,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0x7FC01234,
        0x00000001,
        0x80000001,
        0x3F800000,
        0xBF800000,
        0x40490FDB,
        0xC0490FDB,
        0x00800000,
        0x80800000,
    ]
    w_patterns = [
        0x7FC0ABCD,
        0xFFC00002,
        0x00000000,
        0x80000000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x3EAAAAAB,
        0xBEAAAAAB,
        0x00000002,
        0x80000002,
        0x3F000000,
        0xBF000000,
        0x41200000,
        0xC1200000,
    ]
    full_q = _float32_from_bits(q_patterns, (2, 7))
    full_w = _float32_from_bits(w_patterns, (2, 7))
    trailing_w = full_w[:, trailing_col_start:].clone()
    destination = torch.empty_like(full_q)
    expected = full_q.clone()
    expected[:, trailing_col_start:] = trailing_w

    pointer = destination.data_ptr()
    returned = _rebuild_permuted_weight_from_prefix_and_trailing_(
        destination,
        full_q,
        trailing_w,
        trailing_col_start,
    )

    assert returned is destination
    assert destination.data_ptr() == pointer
    assert torch.equal(_raw_bytes(destination), _raw_bytes(expected))


def test_config_default_cli_validation_and_checkpoint_provenance():
    assert Config().act_order_stitch_impl == EXACT
    assert parse_cli([]).act_order_stitch_impl == EXACT
    assert (
        parse_cli(["--act_order_stitch_impl", EXACT]).act_order_stitch_impl
        == EXACT
    )
    with pytest.raises(ValueError, match="act_order_stitch_impl"):
        Config(act_order_stitch_impl="unordered")

    source = Config(act_order_stitch_impl=EXACT)
    manifest = checkpoint_utils.build_runtime_manifest(source)
    assert manifest["weight_quantization"]["act_order_stitch_impl"] == EXACT

    restored = Config(act_order_stitch_impl=LEGACY)
    assert checkpoint_utils.apply_runtime_manifest(restored, manifest)
    assert restored.act_order_stitch_impl == EXACT

    # Version-1 artifacts created before P06 must restore the historical
    # implementation, not inherit a candidate setting from the evaluation CLI.
    historical = copy.deepcopy(manifest)
    historical["weight_quantization"].pop("act_order_stitch_impl")
    restored = Config(act_order_stitch_impl=EXACT)
    assert checkpoint_utils.apply_runtime_manifest(restored, historical)
    assert restored.act_order_stitch_impl == LEGACY

    malformed = copy.deepcopy(manifest)
    malformed["weight_quantization"]["act_order_stitch_impl"] = "unordered"
    target = Config(a_bits=16, act_order_stitch_impl=LEGACY)
    with pytest.raises(ValueError, match="act_order_stitch_impl"):
        checkpoint_utils.apply_runtime_manifest(target, malformed)
    # Validation precedes all manifest mutation.
    assert target.a_bits == 16
    assert target.act_order_stitch_impl == LEGACY


def _make_ready_quantizer(weight: torch.Tensor) -> WeightQuantizer:
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=4,
        perchannel=True,
        sym=True,
        mse=False,
        weight_groupsize=-1,
    )
    # Pre-observe the complete tensor so the mocked distributed test isolates
    # refresh collectives rather than scale/zero parameter collectives.
    quantizer.find_params(weight)
    return quantizer


def _run_mock_two_rank_case(
    *,
    num_groups: int,
    act_order: bool,
    implementation: str,
) -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[tuple, tuple]]]:
    rows, columns, blocksize = 4, 10, 4
    generator = torch.Generator().manual_seed(2026072406 + num_groups)
    base_rows = torch.randn(2, columns, generator=generator)
    base_rows[:, ::2].mul_(0.25)
    # Identical synthetic rank shards let the fake all-gather reconstruct
    # exactly what a real two-rank collective would return.
    weight = torch.cat([base_rows, base_rows], dim=0)
    linear = nn.Linear(columns, rows, bias=False)
    linear.weight.data.copy_(weight)
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, num_groups),
        quantizer=_make_ready_quantizer(weight),
        num_groups=num_groups,
        dev=torch.device("cpu"),
        group_parallel_quant="rank",
    )
    # Identical group Hessians keep the duplicated synthetic shards coherent.
    factor = torch.randn(columns, columns, generator=generator)
    hessian = factor.matmul(factor.T).add_(torch.eye(columns))
    full_hessians = hessian.unsqueeze(0).repeat(num_groups, 1, 1)
    realq.H = full_hessians.index_select(0, realq.hessian_group_ids)
    realq.act_square = torch.rand(columns, generator=generator)
    realq._finalized = True

    stitched_inputs: list[torch.Tensor] = []

    def refresh(stitched, trailing_start, perm=None):
        stitched_inputs.append(stitched.clone())
        step = len(stitched_inputs)
        update_natural = torch.sin(
            stitched * 0.375 + step * 0.125
        ) * (step * 1e-3)
        update_quant_order = (
            update_natural[:, perm] if perm is not None else update_natural
        )
        return update_quant_order[:, trailing_start:].clone()

    calls: list[tuple[tuple, tuple]] = []

    def fake_all_gather_into_tensor(output, local):
        calls.append((tuple(output.shape), tuple(local.shape)))
        output.copy_(torch.cat([local, local], dim=0))

    original = torch.distributed.all_gather_into_tensor
    torch.distributed.all_gather_into_tensor = fake_all_gather_into_tensor
    try:
        realq.quantize(
            blocksize=blocksize,
            percdamp=0.01,
            act_order=act_order,
            grad_refresh_fn=refresh,
            group_parallel_quant="rank",
            act_order_stitch_impl=implementation,
        )
    finally:
        torch.distributed.all_gather_into_tensor = original

    return linear.weight.detach().clone(), stitched_inputs, calls


@pytest.mark.parametrize("num_groups", [1, 2])
def test_exact_stitch_matches_legacy_and_removes_only_full_weight_collectives(
    monkeypatch,
    num_groups: int,
):
    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist_utils, "get_rank", lambda: 0)

    legacy_weight, legacy_stitched, legacy_calls = _run_mock_two_rank_case(
        num_groups=num_groups,
        act_order=True,
        implementation=LEGACY,
    )
    exact_weight, exact_stitched, exact_calls = _run_mock_two_rank_case(
        num_groups=num_groups,
        act_order=True,
        implementation=EXACT,
    )

    assert torch.equal(_raw_bytes(exact_weight), _raw_bytes(legacy_weight))
    assert len(exact_stitched) == len(legacy_stitched) == 2
    assert all(
        torch.equal(_raw_bytes(exact), _raw_bytes(legacy))
        for exact, legacy in zip(exact_stitched, legacy_stitched)
    )

    # C=10, B=4 has refresh boundaries i2=4,8 and a final short tail of 2.
    # Legacy: [Q block, W suffix, full W] per refresh, then final Q.
    assert legacy_calls == [
        ((4, 4), (2, 4)),
        ((4, 6), (2, 6)),
        ((4, 10), (2, 10)),
        ((4, 4), (2, 4)),
        ((4, 2), (2, 2)),
        ((4, 10), (2, 10)),
        ((4, 10), (2, 10)),
    ]
    # Candidate removes exactly the two redundant full-W collectives.
    assert exact_calls == [
        ((4, 4), (2, 4)),
        ((4, 6), (2, 6)),
        ((4, 4), (2, 4)),
        ((4, 2), (2, 2)),
        ((4, 10), (2, 10)),
    ]


@pytest.mark.parametrize("num_groups", [1, 2])
def test_stitch_switch_is_inactive_without_act_order(
    monkeypatch,
    num_groups: int,
):
    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist_utils, "get_rank", lambda: 0)

    legacy_weight, legacy_stitched, legacy_calls = _run_mock_two_rank_case(
        num_groups=num_groups,
        act_order=False,
        implementation=LEGACY,
    )
    exact_weight, exact_stitched, exact_calls = _run_mock_two_rank_case(
        num_groups=num_groups,
        act_order=False,
        implementation=EXACT,
    )
    assert torch.equal(_raw_bytes(exact_weight), _raw_bytes(legacy_weight))
    assert all(
        torch.equal(_raw_bytes(exact), _raw_bytes(legacy))
        for exact, legacy in zip(exact_stitched, legacy_stitched)
    )
    assert exact_calls == legacy_calls == [
        ((4, 4), (2, 4)),
        ((4, 6), (2, 6)),
        ((4, 4), (2, 4)),
        ((4, 2), (2, 2)),
        ((4, 10), (2, 10)),
    ]


def test_realq_rejects_invalid_stitch_implementation():
    weight = torch.randn(4, 5, generator=torch.Generator().manual_seed(6))
    linear = nn.Linear(5, 4, bias=False)
    linear.weight.data.copy_(weight)
    realq = RealQLayer(
        linear,
        torch.ones(1, 1, 1),
        _make_ready_quantizer(weight),
        1,
        torch.device("cpu"),
        group_parallel_quant="rank",
    )
    realq.H = torch.eye(5).unsqueeze(0)
    realq.act_square = torch.arange(5, dtype=torch.float32)
    realq._finalized = True

    with pytest.raises(ValueError, match="act_order_stitch_impl"):
        realq.quantize(
            blocksize=2,
            act_order=True,
            group_parallel_quant="rank",
            act_order_stitch_impl="unordered",
        )


def test_uneven_rank_rows_remain_rejected(monkeypatch):
    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist_utils, "get_rank", lambda: 0)
    weight = torch.randn(5, 5, generator=torch.Generator().manual_seed(7))
    linear = nn.Linear(5, 5, bias=False)
    linear.weight.data.copy_(weight)
    realq = RealQLayer(
        linear,
        torch.ones(1, 1, 1),
        _make_ready_quantizer(weight),
        1,
        torch.device("cpu"),
        group_parallel_quant="rank",
    )
    realq.H = torch.eye(5).unsqueeze(0)
    realq.act_square = torch.arange(5, dtype=torch.float32)
    realq._finalized = True

    with pytest.raises(
        ValueError,
        match=r"out_features \(5\) divisible by world_size \(2\)",
    ):
        realq.quantize(
            blocksize=2,
            act_order=True,
            grad_refresh_fn=lambda *args, **kwargs: None,
            group_parallel_quant="rank",
            act_order_stitch_impl=EXACT,
        )
