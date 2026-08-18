from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptq_utils import gptq_utils
from realq.config import Config, parse_cli
from utils.quant_utils import WeightQuantizer


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert left.stride() == right.stride()
    assert torch.equal(
        left.detach().contiguous().view(torch.uint8),
        right.detach().contiguous().view(torch.uint8),
    )


def _quantizer(
    groupsize: int,
    *,
    clip_impl: str = "cartesian_legacy",
    mse: bool = False,
    sym: bool = True,
    layout: str = "expanded",
) -> WeightQuantizer:
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits=4,
        perchannel=True,
        sym=sym,
        mse=mse,
        grid=8,
        maxshrink=0.5,
        weight_groupsize=groupsize,
        w_clip_search_impl=clip_impl,
        w_group_param_layout=layout,
    )
    return quantizer


def _expanded_and_compact(
    weight: torch.Tensor,
    groupsize: int,
    *,
    clip_impl: str = "cartesian_legacy",
    mse: bool = False,
    sym: bool = True,
) -> tuple[WeightQuantizer, WeightQuantizer]:
    expanded = _quantizer(
        groupsize,
        clip_impl=clip_impl,
        mse=mse,
        sym=sym,
    )
    compact = _quantizer(
        groupsize,
        clip_impl=clip_impl,
        mse=mse,
        sym=sym,
        layout="compact",
    )
    expanded.find_params(weight)
    compact.find_params(weight)
    return expanded, compact


@pytest.mark.parametrize(
    ("groupsize", "columns"),
    [
        (1, 1),
        (1, 17),
        (2, 17),
        (3, 17),
        (8, 16),
        (8, 17),
        (128, 17),
        (128, 256),
        (128, 257),
    ],
)
@pytest.mark.parametrize("mse", [False, True])
@pytest.mark.parametrize(
    "clip_impl", ["cartesian_legacy", "symmetric_union_exact"]
)
def test_compact_observer_and_full_fake_quant_are_raw_byte_identical(
    groupsize, columns, mse, clip_impl
):
    generator = torch.Generator().manual_seed(
        1701 + groupsize + columns
    )
    weight = torch.randn(7, columns, generator=generator)
    weight[:, ::3].mul_(11.0)
    weight[0].zero_()
    weight[0, ::2] = -0.0
    expanded, compact = _expanded_and_compact(
        weight,
        groupsize,
        clip_impl=clip_impl,
        mse=mse,
    )

    natural_columns = torch.arange(columns)
    group_ids = natural_columns // groupsize
    reconstructed_scale = compact.scale.index_select(1, group_ids)
    reconstructed_zero = compact.zero.index_select(1, group_ids)
    expected_groups = (columns + groupsize - 1) // groupsize
    assert compact.scale.shape == (weight.shape[0], expected_groups)
    assert compact.zero.shape == compact.scale.shape
    assert compact.weight_ncolumns == columns
    _raw_equal(reconstructed_scale, expanded.scale)
    _raw_equal(reconstructed_zero, expanded.zero)

    for actual, expected in zip(
        compact.fake_quantize(weight),
        expanded.fake_quantize(weight),
    ):
        _raw_equal(actual, expected)


@pytest.mark.parametrize(
    ("groupsize", "columns"),
    [(3, 17), (8, 16), (8, 17), (128, 17), (128, 257)],
)
def test_compact_act_order_row_slice_and_p01_are_raw_byte_identical(
    groupsize, columns
):
    generator = torch.Generator().manual_seed(20260724 + columns)
    weight = torch.randn(8, columns, generator=generator)
    expanded, compact = _expanded_and_compact(
        weight,
        groupsize,
        clip_impl="symmetric_union_exact",
        mse=True,
    )

    permutation = torch.randperm(columns, generator=generator)
    natural_columns = permutation[: min(11, columns)]
    row_start, row_end = 2, 6
    x = weight[row_start:row_end].index_select(1, natural_columns)
    expanded_public = expanded.fake_quantize(
        x,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=natural_columns,
    )
    compact_public = compact.fake_quantize(
        x,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=natural_columns,
    )
    for actual, expected in zip(compact_public, expanded_public):
        _raw_equal(actual, expected)

    expanded_prepared = expanded._prepare_fake_quantize_inner(
        input_rows=row_end - row_start,
        column_count=natural_columns.numel(),
        device=x.device,
        dtype=x.dtype,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=natural_columns,
    )
    compact_prepared = compact._prepare_fake_quantize_inner(
        input_rows=row_end - row_start,
        column_count=natural_columns.numel(),
        device=x.device,
        dtype=x.dtype,
        st_idx=row_start,
        end_idx=row_end,
        col_idx=natural_columns,
    )
    _raw_equal(
        compact_prepared.prepared_scale,
        expanded_prepared.prepared_scale,
    )
    for offset in range(x.shape[1]):
        expanded_result = expanded._fake_quantize_prevalidated(
            x[:, offset : offset + 1],
            expanded_prepared,
            offset,
        )
        compact_result = compact._fake_quantize_prevalidated(
            x[:, offset : offset + 1],
            compact_prepared,
            offset,
        )
        for actual, expected in zip(compact_result, expanded_result):
            _raw_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize(
    ("groupsize", "columns"),
    [(3, 17), (8, 16), (8, 17), (128, 17), (128, 257)],
)
def test_compact_preserves_generic_quantize_and_fake_quantize_behavior(
    dtype, sym, groupsize, columns
):
    generator = torch.Generator().manual_seed(99 + columns)
    weight = torch.randn(5, columns, generator=generator).to(dtype)
    expanded, compact = _expanded_and_compact(
        weight,
        groupsize,
        sym=sym,
    )

    _raw_equal(compact.quantize(weight), expanded.quantize(weight))
    # fake_quantize intentionally retains its historical symmetric arithmetic
    # even for a standalone asymmetric quantizer. P04 preserves that public
    # behavior instead of coupling an unrelated algorithm fix to the layout.
    for actual, expected in zip(
        compact.fake_quantize(weight),
        expanded.fake_quantize(weight),
    ):
        _raw_equal(actual, expected)


def test_compact_preserves_nonfinite_union_fallback_raw_bytes():
    weight = torch.randn(
        3, 17, generator=torch.Generator().manual_seed(1001)
    )
    weight[0, 1] = float("nan")
    weight[1, 9] = float("inf")
    weight[2, 16] = -float("inf")
    expanded, compact = _expanded_and_compact(
        weight,
        8,
        clip_impl="symmetric_union_exact",
        mse=True,
    )
    group_ids = torch.arange(weight.shape[1]) // 8
    _raw_equal(compact.scale.index_select(1, group_ids), expanded.scale)
    _raw_equal(compact.zero.index_select(1, group_ids), expanded.zero)
    for actual, expected in zip(
        compact.fake_quantize(weight),
        expanded.fake_quantize(weight),
    ):
        _raw_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float64]
)
def test_compact_preserves_union_unsupported_dtype_fallback_raw_bytes(dtype):
    weight = torch.randn(
        3, 17, generator=torch.Generator().manual_seed(1002)
    ).to(dtype)
    expanded, compact = _expanded_and_compact(
        weight,
        8,
        clip_impl="symmetric_union_exact",
        mse=True,
    )
    group_ids = torch.arange(weight.shape[1]) // 8
    _raw_equal(compact.scale.index_select(1, group_ids), expanded.scale)
    _raw_equal(compact.zero.index_select(1, group_ids), expanded.zero)
    for actual, expected in zip(
        compact.fake_quantize(weight),
        expanded.fake_quantize(weight),
    ):
        _raw_equal(actual, expected)


@pytest.mark.parametrize(
    ("groupsize", "columns"),
    [(3, 17), (8, 16), (8, 17), (128, 17), (128, 256), (128, 257)],
)
def test_row_shard_compact_gather_reconstructs_full_observer(
    groupsize, columns
):
    generator = torch.Generator().manual_seed(4242 + columns)
    weight = torch.randn(8, columns, generator=generator)
    full = _quantizer(
        groupsize,
        clip_impl="symmetric_union_exact",
        mse=True,
    )
    full.find_params(weight)

    compact_scale_shards = []
    compact_zero_shards = []
    for shard in weight.chunk(4, dim=0):
        local = _quantizer(
            groupsize,
            clip_impl="symmetric_union_exact",
            mse=True,
            layout="compact",
        )
        local.find_params(shard)
        compact_scale_shards.append(local.scale)
        compact_zero_shards.append(local.zero)
    gathered_scale = torch.cat(compact_scale_shards, dim=0)
    gathered_zero = torch.cat(compact_zero_shards, dim=0)
    group_ids = torch.arange(columns) // groupsize
    reconstructed_scale = gathered_scale.index_select(1, group_ids)
    reconstructed_zero = gathered_zero.index_select(1, group_ids)
    _raw_equal(reconstructed_scale, full.scale)
    _raw_equal(reconstructed_zero, full.zero)
    assert (
        gathered_scale.nbytes + gathered_zero.nbytes
        == 2
        * weight.shape[0]
        * ((columns + groupsize - 1) // groupsize)
        * weight.element_size()
    )


def test_compact_partial_without_col_idx_never_confuses_groups_for_columns():
    weight = torch.randn(3, 257, generator=torch.Generator().manual_seed(12))
    _, compact = _expanded_and_compact(weight, 128)
    # Three compact groups happen to equal this partial input width. Natural
    # column metadata, rather than scale.shape[-1], must still reject it.
    with pytest.raises(ValueError, match="requires col_idx"):
        compact.fake_quantize(weight[:, :3])
    with pytest.raises(ValueError, match="requires col_idx"):
        compact.quantize(weight[:, :3])


def test_compact_public_col_idx_keeps_legacy_long_coercion():
    weight = torch.randn(3, 17, generator=torch.Generator().manual_seed(13))
    expanded, compact = _expanded_and_compact(weight, 3)
    x = weight[:, [1, 16]]
    floating_columns = torch.tensor([1.9, 16.8])
    for actual, expected in zip(
        compact.fake_quantize(x, col_idx=floating_columns),
        expanded.fake_quantize(x, col_idx=floating_columns),
    ):
        _raw_equal(actual, expected)
    with pytest.raises(TypeError, match="integer natural"):
        compact._prepare_fake_quantize_inner(
            input_rows=weight.shape[0],
            column_count=x.shape[1],
            device=x.device,
            dtype=x.dtype,
            col_idx=floating_columns,
        )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("w_group_param_layout", "expanded"),
        ("weight_ncolumns", 18),
    ],
)
def test_compact_p01_context_rejects_stale_layout_metadata(attribute, value):
    weight = torch.randn(3, 17, generator=torch.Generator().manual_seed(14))
    _, compact = _expanded_and_compact(weight, 3)
    columns = torch.tensor([0, 4, 16])
    x = weight.index_select(1, columns)
    prepared = compact._prepare_fake_quantize_inner(
        input_rows=weight.shape[0],
        column_count=columns.numel(),
        device=x.device,
        dtype=x.dtype,
        col_idx=columns,
    )
    setattr(compact, attribute, value)
    with pytest.raises(RuntimeError, match="Stale prevalidated"):
        compact._fake_quantize_prevalidated(x[:, :1], prepared, 0)


@pytest.mark.parametrize(
    ("metadata", "scale_columns", "error"),
    [
        (None, 6, "natural-column count"),
        (17, 5, "invalid shape"),
    ],
)
def test_compact_fails_closed_on_missing_or_inconsistent_metadata(
    metadata, scale_columns, error
):
    quantizer = _quantizer(3, layout="compact")
    quantizer.scale = torch.ones(2, scale_columns)
    quantizer.zero = torch.zeros_like(quantizer.scale)
    quantizer.weight_ncolumns = metadata
    with pytest.raises(RuntimeError, match=error):
        quantizer.fake_quantize(torch.ones(2, 17))


@pytest.mark.parametrize("delete_before_observer", [False, True])
def test_legacy_grouped_quantizer_without_layout_metadata_falls_back_expanded(
    delete_before_observer,
):
    weight = torch.randn(3, 17, generator=torch.Generator().manual_seed(15))
    reference = _quantizer(3)
    reference.find_params(weight)
    expected = reference.fake_quantize(weight)

    legacy = _quantizer(3)
    if delete_before_observer:
        del legacy.w_group_param_layout
        del legacy.weight_ncolumns
    legacy.find_params(weight)
    if not delete_before_observer:
        del legacy.w_group_param_layout
        del legacy.weight_ncolumns
    assert legacy.scale.shape == weight.shape
    actual = legacy.fake_quantize(weight)
    for left, right in zip(actual, expected):
        _raw_equal(left, right)


def test_invalid_layout_is_rejected_before_configure_mutates_quantizer():
    quantizer = _quantizer(3)
    before = (
        quantizer.bits,
        quantizer.weight_groupsize,
        quantizer.w_group_param_layout,
    )
    with pytest.raises(ValueError, match="w_group_param_layout"):
        quantizer.configure(
            bits=2,
            perchannel=False,
            weight_groupsize=7,
            w_group_param_layout="packed",
        )
    assert (
        quantizer.bits,
        quantizer.weight_groupsize,
        quantizer.w_group_param_layout,
    ) == before


def test_compact_layout_config_and_cli_default_to_optimized():
    assert Config().w_group_param_layout == "compact"
    assert (
        parse_cli(["--w_group_param_layout", "compact"])
        .w_group_param_layout
        == "compact"
    )
    with pytest.raises(ValueError, match="w_group_param_layout"):
        Config(w_group_param_layout="packed")


class _TinyRtnLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(17, 4, bias=False)


class _TinyRtnModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyRtnLayer()])


class _TinyRtnAnalyzer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def get_layers(self):
        return self.model.layers

    def get_quantizable_modules(self, layer):
        return {"proj": layer.proj}


def test_rtn_plumbs_compact_layout_and_preserves_quantized_weight(
    monkeypatch,
):
    model = _TinyRtnModel()
    generator = torch.Generator().manual_seed(91)
    model.layers[0].proj.weight.data.copy_(
        torch.randn(4, 17, generator=generator)
    )
    original = model.layers[0].proj.weight.detach().clone()
    expanded = _quantizer(3)
    expanded.find_params(original)
    expected, _, _ = expanded.fake_quantize(original)

    monkeypatch.setattr(
        gptq_utils.memory_utils, "cleanup_memory", lambda **_kwargs: None
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    quantizers = gptq_utils.rtn_fwrd(
        SimpleNamespace(
            w_bits=4,
            w_groupsize=3,
            w_asym=False,
            w_clip=False,
            w_group_param_layout="compact",
        ),
        _TinyRtnAnalyzer(model),
        torch.device("cpu"),
    )

    actual = model.layers[0].proj.weight.detach()
    _raw_equal(actual, expected)
    quantizer = quantizers["model.layers.0.proj"]
    assert quantizer.w_group_param_layout == "compact"
    assert quantizer.weight_ncolumns == 17
    assert quantizer.scale.shape == (4, 6)
