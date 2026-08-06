from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from realq.alignment import RefreshTraceWriter
from realq.refresh.block_gd import (
    BlockRefreshState,
    RefreshContext,
    _SharedSampleScheduler,
    make_grad_refresh_fn,
)
from realq.refresh.kl_loss import make_kl_refresh_fn


class _ToyBlock(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.first = nn.Linear(4, 4, bias=False)
        self.second = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.first.weight.copy_(
                torch.randn(4, 4, generator=generator) * 0.2
            )
            self.second.weight.copy_(
                torch.randn(4, 4, generator=generator) * 0.2
            )

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        hidden_states = torch.tanh(self.first(hidden_states))
        return (self.second(hidden_states),)


class _AliasWrapper(nn.Module):
    """Minimal ActQuantWrapper-style tied ``weight`` alias."""

    def __init__(self, module: nn.Linear) -> None:
        super().__init__()
        self.module = module
        self.weight = module.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.module(hidden_states)


class _WrappedToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = _AliasWrapper(nn.Linear(4, 4, bias=False))
        self.second = _AliasWrapper(nn.Linear(4, 4, bias=False))

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        hidden_states = torch.tanh(self.first(hidden_states))
        return (self.second(hidden_states),)


class _ToyAnalyzer:
    def __init__(self) -> None:
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 7, bias=False)

    def get_layernorm_before_head(self) -> nn.Module:
        return self.norm

    def get_lm_head(self) -> nn.Module:
        return self.head


def _named_linears(block: _ToyBlock) -> list[tuple[str, nn.Module]]:
    return [("first", block.first), ("second", block.second)]


def _layer_inputs(x: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        inps=x,
        fp_inps=x.clone(),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )


def _context(module: nn.Module) -> RefreshContext:
    return RefreshContext(
        module=module,
        layer_lr=1e-2,
        grad_clip=-1.0,
        backward_bsz=2,
        scheduler=_SharedSampleScheduler(
            n_total=2,
            chunk_size=2,
            seed=0,
        ),
    )


def test_full_block_trace_records_actual_backward_and_master_updates(
    tmp_path,
) -> None:
    torch.manual_seed(5)
    block = _ToyBlock(seed=7)
    x = torch.randn(4, 3, 4)
    with torch.no_grad():
        fp_out = block(x)[0].clone()
    state = BlockRefreshState(block, _named_linears(block))
    first_master = state.begin_quantization("first")
    second_storage = (
        state._states["second"].master.untyped_storage().data_ptr()
    )
    trace_path = tmp_path / "refresh.jsonl"
    writer = RefreshTraceWriter(
        str(trace_path),
        implementation="realq-plus",
        run_id="runtime-proof",
        config={},
    )
    ctx = RefreshContext(
        module=block.first,
        layer_lr=1e-2,
        grad_clip=-1.0,
        backward_bsz=2,
        scheduler=_SharedSampleScheduler(
            n_total=4,
            chunk_size=4,
            seed=0,
        ),
        trace_writer=writer,
        trace_layer=0,
        trace_module="first",
        blocksize=2,
    )
    refresh = make_grad_refresh_fn(
        layer=block,
        module=block.first,
        module_name="first",
        block_state=state,
        layer_state=_layer_inputs(x),
        fp_out_for_this_layer=fp_out,
        fisher=torch.eye(4),
        ctx=ctx,
    )
    refresh(first_master + 0.1, trailing_col_start=2)
    writer.close()

    records = [
        json.loads(line)
        for line in trace_path.read_text().splitlines()
    ]
    step = records[1]
    assert step["schema_version"] == 4
    assert step["objective"] == "fisher_mse_full_block"
    assert step["backward_invocation_id"] == 1
    assert step["backward_chunk_sizes"] == [2, 2]
    assert step["global_count"] == 4
    assert [
        (item["scope"], item["name"], item["used"])
        for item in step["active_weights"]
    ] == [
        ("current_block", "first", True),
        ("current_block", "second", True),
    ]
    future = step["active_weights"][1]
    assert str(second_storage) in future["source_storage_id"]
    assert future["storage_id_after"] == future["source_storage_id"]
    assert future["update_applied"] is True
    assert future["update_l2"] > 0
    assert future["optimizer_step_after"] == (
        future["optimizer_step_before"] + 1
    )


def test_full_block_refresh_locks_prefix_and_updates_future_weight() -> None:
    torch.manual_seed(7)
    block = _ToyBlock(seed=11)
    x = torch.randn(2, 3, 4)
    with torch.no_grad():
        fp_out = block(x)[0].clone()

    first_actual_before = block.first.weight.detach().clone()
    second_actual_before = block.second.weight.detach().clone()
    state = BlockRefreshState(block, _named_linears(block))
    first_master = state.begin_quantization("first")
    second_master_before = state._states["second"].master.clone()

    refresh = make_grad_refresh_fn(
        layer=block,
        module=block.first,
        module_name="first",
        block_state=state,
        layer_state=_layer_inputs(x),
        fp_out_for_this_layer=fp_out,
        fisher=torch.eye(4),
        ctx=_context(block.first),
    )
    stitched = first_master.clone()
    stitched.add_(0.15)
    update = refresh(stitched, trailing_col_start=2)

    assert torch.count_nonzero(update[:, :2]) == 0
    assert torch.count_nonzero(update[:, 2:]) > 0
    assert not torch.equal(
        state._states["second"].master,
        second_master_before,
    )
    assert state._states["first"].step == 1
    assert state._states["second"].step == 1
    # Functional overrides and FP32 masters must not contaminate the teacher
    # module. RealQLayer alone commits the final quantized weight.
    assert torch.equal(block.first.weight, first_actual_before)
    assert torch.equal(block.second.weight, second_actual_before)


def test_unquantized_master_mapping_preserves_live_storage_handoff() -> None:
    block = _ToyBlock(seed=13)
    state = BlockRefreshState(block, _named_linears(block))

    masters = state.unquantized_master_mapping()
    assert tuple(masters) == ("first", "second")
    assert (
        masters["first"].data_ptr()
        == state._states["first"].master.data_ptr()
    )
    assert (
        masters["second"].data_ptr()
        == state._states["second"].master.data_ptr()
    )

    masters["second"].add_(0.125)
    torch.testing.assert_close(
        state._states["second"].master,
        masters["second"],
    )
    transferred = state.begin_quantization("first")
    assert transferred.data_ptr() == masters["first"].data_ptr()
    # The view may only be acquired at a pristine block boundary; its
    # existing tensor references remain live through the handoff.
    with pytest.raises(RuntimeError, match="pristine block boundary"):
        state.unquantized_master_mapping()


def test_sliding_refresh_updates_both_blocks_and_preserves_state_handoff() -> None:
    torch.manual_seed(17)
    block = _ToyBlock(seed=19)
    next_block = _ToyBlock(seed=23)
    x = torch.randn(2, 3, 4)
    with torch.no_grad():
        fp_out = block(x)[0].clone()
        next_fp_out = next_block(fp_out)[0].clone()

    state = BlockRefreshState(block, _named_linears(block))
    next_state = BlockRefreshState(
        next_block,
        _named_linears(next_block),
    )
    first_master = state.begin_quantization("first")
    second_master_before = state._states["second"].master.clone()
    next_masters_before = {
        name: item.master.clone()
        for name, item in next_state._states.items()
    }

    refresh = make_grad_refresh_fn(
        layer=block,
        module=block.first,
        module_name="first",
        block_state=state,
        layer_state=_layer_inputs(x),
        fp_out_for_this_layer=fp_out,
        fisher=torch.eye(4),
        ctx=_context(block.first),
        next_layer=next_block,
        next_block_state=next_state,
        next_fp_out=next_fp_out,
        next_fisher=torch.eye(4),
        slide_alpha_fn=lambda: 0.5,
    )
    stitched = first_master.clone()
    stitched.add_(0.2)
    perm = torch.tensor([2, 0, 3, 1])
    update = refresh(stitched, trailing_col_start=2, perm=perm)

    locked_natural_columns = perm[:2]
    active_natural_columns = perm[2:]
    assert torch.count_nonzero(
        update.index_select(1, locked_natural_columns)
    ) == 0
    assert torch.count_nonzero(
        update.index_select(1, active_natural_columns)
    ) > 0
    assert not torch.equal(
        state._states["second"].master,
        second_master_before,
    )
    for name, before in next_masters_before.items():
        assert not torch.equal(next_state._states[name].master, before)
        assert next_state._states[name].step == 1

    # The future current-block linear carries its already accumulated Adam
    # state and updated FP32 master into its own subsequent GPTQ sweep.
    state.finish_quantization("first")
    transferred_second = state.begin_quantization("second")
    assert state._states["second"].master is None
    assert not torch.equal(transferred_second, second_master_before)
    assert state._states["second"].step == 1


def test_final_kl_refresh_also_updates_the_full_unquantized_block() -> None:
    torch.manual_seed(29)
    block = _ToyBlock(seed=31)
    analyzer = _ToyAnalyzer()
    x = torch.randn(2, 3, 4)
    with torch.no_grad():
        fp_out = block(x)[0].clone()

    state = BlockRefreshState(block, _named_linears(block))
    first_master = state.begin_quantization("first")
    second_master_before = state._states["second"].master.clone()
    refresh = make_kl_refresh_fn(
        layer=block,
        module=block.first,
        module_name="first",
        block_state=state,
        layer_state=_layer_inputs(x),
        fp_out_for_this_layer=fp_out,
        analyzer=analyzer,
        kl_topk=-1,
        ctx=_context(block.first),
    )
    stitched = first_master.clone()
    stitched.add_(0.1)
    update = refresh(stitched, trailing_col_start=2)

    assert torch.count_nonzero(update[:, :2]) == 0
    assert torch.count_nonzero(update[:, 2:]) > 0
    assert not torch.equal(
        state._states["second"].master,
        second_master_before,
    )


def test_functional_overrides_follow_act_quant_wrapper_weight_aliases() -> None:
    torch.manual_seed(37)
    block = _WrappedToyBlock()
    named_linears = [
        ("first", block.first.module),
        ("second", block.second.module),
    ]
    state = BlockRefreshState(block, named_linears)
    first_master = state.begin_quantization("first")
    overrides, active = state.make_overrides(
        current_name="first",
        current_weight_fp32=first_master + 0.2,
        trailing_col_start=2,
    )
    x = torch.randn(2, 3, 4)
    original = block(x)[0]
    changed = functional_call(
        block,
        overrides,
        (x,),
        strict=False,
    )[0]
    assert not torch.equal(changed, original)
    loss = changed.square().mean()
    grads = torch.autograd.grad(loss, [entry.leaf for entry in active])
    assert all(torch.count_nonzero(grad) > 0 for grad in grads)
