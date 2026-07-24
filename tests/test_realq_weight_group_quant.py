from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from realq.quant.hessian import cholesky_inverse_with_damp
from realq.quant.realq_layer import RealQLayer
from realq.runner.layer_loop import _make_quantizer
from utils import dist_utils
from utils.quant_utils import WeightQuantizer


def _quantizer(*, groupsize: int, bits: int = 4, mse: bool = False):
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


def _structured_weight(rows: int, columns: int) -> torch.Tensor:
    """Give adjacent 128-column groups deliberately different ranges."""
    generator = torch.Generator().manual_seed(20260724)
    weight = torch.randn(rows, columns, generator=generator)
    if columns > 128:
        weight[:, :128].mul_(0.07)
        weight[:, 128:256].mul_(3.5)
    if columns > 256:
        weight[:, 256:].mul_(0.4)
    return weight


def _old_static_group_reference(
    weight: torch.Tensor, groupsize: int, *, mse: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the legacy static-group construction.

    Old GPTQ+ deep-copies one ordinary per-row quantizer for each natural
    column group and calls find_params on that group's weight slice.
    """
    quantized_parts = []
    scale_parts = []
    for start in range(0, weight.shape[1], groupsize):
        end = min(start + groupsize, weight.shape[1])
        group_quantizer = _quantizer(groupsize=-1, mse=mse)
        group_quantizer.find_params(weight[:, start:end])
        quantized, _, scale = group_quantizer.fake_quantize(weight[:, start:end])
        quantized_parts.append(quantized)
        scale_parts.append(scale.expand(-1, end - start))
    return torch.cat(quantized_parts, dim=1), torch.cat(scale_parts, dim=1)


@pytest.mark.parametrize("mse", [False, True])
def test_group128_matches_legacy_static_groups_including_short_tail(mse):
    weight = _structured_weight(rows=5, columns=257)
    expected_q, expected_scale = _old_static_group_reference(
        weight, groupsize=128, mse=mse
    )

    quantizer = _quantizer(groupsize=128, mse=mse)
    quantizer.find_params(weight)
    actual_q, _, actual_scale = quantizer.fake_quantize(weight)

    assert actual_scale.shape == weight.shape
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_group128_mse_matches_legacy_for_one_sided_groups(sign):
    # One-sided groups exercise the legacy min/max inclusion of zero. Symmetric
    # clipping must remain exactly equivalent for all-positive/all-negative
    # inputs, not merely for the usual mixed-sign Gaussian weights.
    generator = torch.Generator().manual_seed(19)
    weight = sign * (torch.rand(3, 257, generator=generator) + 0.05)
    weight[:, 128:256].mul_(5.0)
    expected_q, expected_scale = _old_static_group_reference(
        weight, groupsize=128, mse=True
    )
    quantizer = _quantizer(groupsize=128, mse=True)
    quantizer.find_params(weight)
    actual_q, _, actual_scale = quantizer.fake_quantize(weight)

    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)


def test_group_partial_fake_quant_requires_and_uses_natural_column_index():
    weight = _structured_weight(rows=4, columns=256)
    quantizer = _quantizer(groupsize=128)
    quantizer.find_params(weight)
    full_q, _, _ = quantizer.fake_quantize(weight)

    with pytest.raises(ValueError, match="requires col_idx"):
        quantizer.fake_quantize(weight[:, :1])

    # Reverse order emulates act-order. Each partial call still names its
    # natural (pre-permutation) column, so it must reproduce the full RTN
    # result after concatenating in that same permuted order.
    permutation = torch.arange(weight.shape[1] - 1, -1, -1)
    partial_q = torch.cat(
        [
            quantizer.fake_quantize(
                weight[:, natural_col : natural_col + 1],
                col_idx=natural_col,
            )[0]
            for natural_col in permutation.tolist()
        ],
        dim=1,
    )
    torch.testing.assert_close(partial_q, full_q[:, permutation], rtol=0, atol=0)


def _run_realq(
    weight: torch.Tensor,
    *,
    groupsize: int,
    num_groups: int,
    group_parallel_quant: str,
    act_order: bool,
    hessian: torch.Tensor | None = None,
    blocksize: int = 128,
    quantizer_inner_fastpath: bool = False,
) -> torch.Tensor:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight.data.copy_(weight)
    quantizer = _quantizer(groupsize=groupsize)
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, num_groups),
        quantizer=quantizer,
        num_groups=num_groups,
        dev=torch.device("cpu"),
        group_parallel_quant=group_parallel_quant,
    )
    # A diagonal Hessian removes cross-column compensation, isolating the
    # quantizer/group-coordinate contract in all RealQLayer execution paths.
    if hessian is None:
        hessian = torch.eye(weight.shape[1])
    realq.H = hessian.unsqueeze(0).repeat(num_groups, 1, 1)
    realq.act_square = torch.arange(weight.shape[1], dtype=torch.float32)
    realq._finalized = True
    realq.quantize(
        blocksize=blocksize,
        percdamp=0.01,
        act_order=act_order,
        w_clip=False,
        group_parallel_quant=group_parallel_quant,
        quantizer_inner_fastpath=quantizer_inner_fastpath,
    )
    return linear.weight.detach().clone()


@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize("group_parallel_quant", ["none", "rank"])
@pytest.mark.parametrize("act_order", [False, True])
def test_group128_realq_all_row_group_paths_match_natural_group_rtn(
    num_groups, group_parallel_quant, act_order
):
    weight = _structured_weight(rows=4, columns=256)
    reference_quantizer = _quantizer(groupsize=128)
    reference_quantizer.find_params(weight)
    expected, _, _ = reference_quantizer.fake_quantize(weight)

    actual = _run_realq(
        weight,
        groupsize=128,
        num_groups=num_groups,
        group_parallel_quant=group_parallel_quant,
        act_order=act_order,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "groupsize,num_groups,group_parallel_quant,act_order,columns",
    [
        (-1, 1, "rank", False, 17),
        (128, 1, "rank", False, 129),
        (128, 1, "rank", True, 129),
        (128, 2, "rank", True, 129),
        (128, 2, "none", True, 129),
    ],
)
def test_quantizer_inner_fastpath_is_raw_byte_identical_end_to_end(
    groupsize,
    num_groups,
    group_parallel_quant,
    act_order,
    columns,
):
    weight = _structured_weight(rows=4, columns=columns)
    baseline = _run_realq(
        weight,
        groupsize=groupsize,
        num_groups=num_groups,
        group_parallel_quant=group_parallel_quant,
        act_order=act_order,
        blocksize=min(128, columns),
        quantizer_inner_fastpath=False,
    )
    fast = _run_realq(
        weight,
        groupsize=groupsize,
        num_groups=num_groups,
        group_parallel_quant=group_parallel_quant,
        act_order=act_order,
        blocksize=min(128, columns),
        quantizer_inner_fastpath=True,
    )
    assert torch.equal(
        fast.contiguous().view(torch.uint8),
        baseline.contiguous().view(torch.uint8),
    )


def _legacy_dynamic_group_reference(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    blocksize: int,
) -> torch.Tensor:
    working = weight.float().clone()
    output = torch.zeros_like(working)
    hinv = cholesky_inverse_with_damp(hessian, percdamp=0.01)
    for start in range(0, working.shape[1], blocksize):
        end = min(start + blocksize, working.shape[1])
        block = working[:, start:end].clone()
        group_quantizer = _quantizer(groupsize=-1)
        group_quantizer.find_params(block)
        errors = torch.zeros_like(block)
        hinv_block = hinv[start:end, start:end]
        for local_column in range(end - start):
            column = block[:, local_column]
            diagonal = hinv_block[local_column, local_column]
            quantized, _, _ = group_quantizer.fake_quantize(column.unsqueeze(1))
            quantized = quantized.flatten()
            output[:, start + local_column] = quantized
            error = (column - quantized) / diagonal
            block[:, local_column:] -= error.unsqueeze(1).matmul(
                hinv_block[local_column, local_column:].unsqueeze(0)
            )
            errors[:, local_column] = error
        working[:, end:] -= errors.matmul(hinv[start:end, end:])
    return output


def test_non_act_order_group128_uses_legacy_dynamic_block_observer():
    generator = torch.Generator().manual_seed(20260724)
    weight = torch.randn(4, 256, generator=generator)
    weight[:, :128].mul_(4.0)
    weight[:, 128:].mul_(0.01)
    hessian = torch.eye(256)
    paired = torch.arange(128)
    # SPD block matrix [[I, rho*I], [rho*I, I]] creates deterministic
    # cross-group compensation, so observing group 2 before vs after group 1
    # is numerically distinguishable.
    hessian[paired, paired + 128] = 0.75
    hessian[paired + 128, paired] = 0.75
    expected = _legacy_dynamic_group_reference(
        weight, hessian, blocksize=128
    )
    actual = _run_realq(
        weight,
        groupsize=128,
        num_groups=1,
        group_parallel_quant="none",
        act_order=False,
        hessian=hessian,
        blocksize=128,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    static_quantizer = _quantizer(groupsize=128)
    static_quantizer.find_params(weight)
    static_q, _, _ = static_quantizer.fake_quantize(weight)
    assert not torch.equal(expected, static_q)


def test_dynamic_group_size_must_match_blocksize():
    weight = _structured_weight(rows=4, columns=256)
    with pytest.raises(ValueError, match="weight_groupsize == blocksize"):
        _run_realq(
            weight,
            groupsize=128,
            num_groups=1,
            group_parallel_quant="none",
            act_order=False,
            blocksize=64,
        )


@pytest.mark.parametrize("groupsize, expected_param_columns", [(-1, 1), (128, 256)])
def test_rank_gather_preserves_per_row_and_group_parameter_shapes(
    monkeypatch, groupsize, expected_param_columns
):
    # Simulate rank 0 of a two-rank job. Both row shards are intentionally
    # identical, allowing a deterministic fake all-gather without launching
    # a process group while still exercising the exact distributed branch.
    base_rows = _structured_weight(rows=2, columns=256)
    weight = torch.cat([base_rows, base_rows], dim=0)
    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist_utils, "get_rank", lambda: 0)

    gather_shapes = []

    def fake_all_gather_into_tensor(output, local):
        gather_shapes.append((tuple(output.shape), tuple(local.shape)))
        output.copy_(torch.cat([local, local], dim=0))

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )

    actual = _run_realq(
        weight,
        groupsize=groupsize,
        num_groups=1,
        group_parallel_quant="rank",
        act_order=True,
    )
    reference_quantizer = _quantizer(groupsize=groupsize)
    reference_quantizer.find_params(weight)
    expected, _, _ = reference_quantizer.fake_quantize(weight)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert gather_shapes[:2] == [
        ((4, expected_param_columns), (2, expected_param_columns)),
        ((4, expected_param_columns), (2, expected_param_columns)),
    ]
    assert gather_shapes[-1] == ((4, 256), (2, 256))


def test_rank_dynamic_group_gathers_per_block_row_scales(monkeypatch):
    base_rows = _structured_weight(rows=2, columns=256)
    weight = torch.cat([base_rows, base_rows], dim=0)
    monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist_utils, "get_rank", lambda: 0)
    gather_shapes = []

    def fake_all_gather_into_tensor(output, local):
        gather_shapes.append((tuple(output.shape), tuple(local.shape)))
        output.copy_(torch.cat([local, local], dim=0))

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    actual = _run_realq(
        weight,
        groupsize=128,
        num_groups=1,
        group_parallel_quant="rank",
        act_order=False,
        blocksize=128,
    )
    reference_quantizer = _quantizer(groupsize=128)
    reference_quantizer.find_params(weight)
    expected, _, _ = reference_quantizer.fake_quantize(weight)

    # Identity H means dynamic and static group values coincide here; this test
    # isolates collective shapes. Two blocks each gather scale+zero, then Q.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert gather_shapes == [
        ((4, 1), (2, 1)),
        ((4, 1), (2, 1)),
        ((4, 1), (2, 1)),
        ((4, 1), (2, 1)),
        ((4, 256), (2, 256)),
    ]


def test_per_row_quantization_ignores_column_selector_and_is_unchanged():
    weight = _structured_weight(rows=3, columns=11)
    quantizer = _quantizer(groupsize=-1)
    quantizer.find_params(weight)
    full_q, _, _ = quantizer.fake_quantize(weight)
    column_q = torch.cat(
        [
            quantizer.fake_quantize(
                weight[:, column : column + 1],
                # RealQLayer now passes this uniformly; per-row mode must
                # deliberately ignore it.
                col_idx=weight.shape[1] - column - 1,
            )[0]
            for column in range(weight.shape[1])
        ],
        dim=1,
    )
    torch.testing.assert_close(column_q, full_q, rtol=0, atol=0)


def test_w16_realq_is_exact_noop_before_hessian_work():
    weight = _structured_weight(rows=4, columns=17)
    linear = nn.Linear(17, 4, bias=False)
    linear.weight.data.copy_(weight)
    quantizer = _quantizer(groupsize=128, bits=16)
    realq = RealQLayer(
        linear=linear,
        saliency=torch.ones(1, 1, 1),
        quantizer=quantizer,
        num_groups=1,
        dev=torch.device("cpu"),
    )
    # Deliberately leave H invalid and unfinalized: W16 must return before
    # Hessian validation/Cholesky and before using the uninitialised scale.
    realq.H.fill_(float("nan"))
    realq.quantize(
        blocksize=8,
        act_order=True,
        # Also must not care that the disabled quantizer was configured without
        # MSE clipping.
        w_clip=True,
        group_parallel_quant="rank",
    )
    torch.testing.assert_close(linear.weight, weight, rtol=0, atol=0)


def test_new_realq_rejects_asymmetric_weight_mode_explicitly():
    cfg = SimpleNamespace(
        w_asym=True,
        w_bits=4,
        w_clip=False,
        w_groupsize=128,
    )
    with pytest.raises(ValueError, match="only supports symmetric weights"):
        _make_quantizer(cfg)


def test_w16_allows_nominal_asymmetry_because_it_is_still_a_strict_noop():
    cfg = SimpleNamespace(
        w_asym=True,
        w_bits=16,
        w_clip=True,
        w_groupsize=128,
    )
    quantizer = _make_quantizer(cfg)
    assert quantizer.bits == 16
    assert not quantizer.sym


def test_make_quantizer_plumbs_opt_in_clip_search_backend():
    cfg = SimpleNamespace(
        w_asym=False,
        w_bits=4,
        w_clip=True,
        w_groupsize=128,
        w_clip_search_impl="symmetric_union_exact",
    )
    quantizer = _make_quantizer(cfg)
    assert quantizer.w_clip_search_impl == "symmetric_union_exact"
