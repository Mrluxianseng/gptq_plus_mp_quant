from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from realq.config import Config, parse_cli
from realq.refresh.block_gd import (
    RefreshContext,
    _SharedSampleScheduler,
    make_grad_refresh_fn,
)
from realq.refresh.fisher_loss import fisher_mse_loss
from realq.runner.layer_loop import (
    _should_stage_fisher_for_refresh,
    _stage_fisher_for_refresh,
)


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 2, bias=False)

    def forward(self, hidden_states, **_kwargs):
        return (self.proj(hidden_states),)


class _ToyNextLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)

    def forward(self, hidden_states, **_kwargs):
        return (self.proj(hidden_states),)


def test_fisher_fp32_cache_is_default_on_and_validated():
    assert Config().fisher_fp32_cache is True
    assert parse_cli(
        ["--fisher_fp32_cache", "false"]
    ).fisher_fp32_cache is False
    with pytest.raises(ValueError, match="fisher_fp32_cache"):
        Config(fisher_fp32_cache=1)


def test_default_staging_retains_historical_dtype_and_direct_tensor():
    persisted = torch.tensor(
        [[1.0, -0.25], [-0.25, 2.0]],
        dtype=torch.bfloat16,
    )
    staged = _stage_fisher_for_refresh(
        persisted,
        torch.device("cpu"),
        fp32_cache=False,
    )
    assert staged is persisted
    assert staged.dtype == torch.bfloat16


def test_opt_in_staging_matches_per_call_conversion_and_is_reusable():
    persisted = torch.tensor(
        [[1.0, -0.25], [-0.25, 2.0]],
        dtype=torch.bfloat16,
    )
    staged = _stage_fisher_for_refresh(
        persisted,
        torch.device("cpu"),
        fp32_cache=True,
    )
    expected = persisted.to(dtype=torch.float32)
    assert staged.dtype == torch.float32
    assert torch.equal(_raw_bytes(staged), _raw_bytes(expected))

    # This is the conversion expression inside fisher_mse_loss. Once staged
    # as FP32 on the target device it returns the same tensor instead of
    # allocating and expanding BF16 again.
    reused = staged.to(device=staged.device, dtype=torch.float32)
    assert reused is staged


def test_opt_in_skips_unused_final_kl_fisher_without_changing_default_path():
    assert not _should_stage_fisher_for_refresh(
        block_gd_enabled=False,
        use_kl_refresh=False,
        fp32_cache=False,
    )
    assert _should_stage_fisher_for_refresh(
        block_gd_enabled=True,
        use_kl_refresh=False,
        fp32_cache=True,
    )
    # Default-off deliberately retains the historical unused BF16 allocation.
    assert _should_stage_fisher_for_refresh(
        block_gd_enabled=True,
        use_kl_refresh=True,
        fp32_cache=False,
    )
    assert not _should_stage_fisher_for_refresh(
        block_gd_enabled=True,
        use_kl_refresh=True,
        fp32_cache=True,
    )


def test_opt_in_staging_preserves_fisher_loss_and_gradient_raw_bytes():
    persisted = torch.tensor(
        [
            [1.0, -0.125, 0.25],
            [-0.125, 0.75, 0.0625],
            [0.25, 0.0625, 1.5],
        ],
        dtype=torch.bfloat16,
    )
    fp_target = torch.tensor(
        [[[0.5, -0.25, 0.75], [0.0, 1.0, -1.0]]],
        dtype=torch.bfloat16,
    )
    q_seed = torch.tensor(
        [[[0.75, -0.5, 1.25], [-0.25, 0.5, -0.75]]],
        dtype=torch.bfloat16,
    )

    q_legacy = q_seed.clone().requires_grad_(True)
    q_cached = q_seed.clone().requires_grad_(True)
    legacy_fisher = _stage_fisher_for_refresh(
        persisted,
        torch.device("cpu"),
        fp32_cache=False,
    )
    cached_fisher = _stage_fisher_for_refresh(
        persisted,
        torch.device("cpu"),
        fp32_cache=True,
    )

    legacy_loss = fisher_mse_loss(q_legacy, fp_target, legacy_fisher)
    cached_loss = fisher_mse_loss(q_cached, fp_target, cached_fisher)
    legacy_grad = torch.autograd.grad(legacy_loss, q_legacy)[0]
    cached_grad = torch.autograd.grad(cached_loss, q_cached)[0]

    assert torch.equal(_raw_bytes(cached_loss), _raw_bytes(legacy_loss))
    assert torch.equal(_raw_bytes(cached_grad), _raw_bytes(legacy_grad))


def test_fp32_cache_preserves_slide_refresh_adam_state_and_updates_raw_bytes():
    torch.manual_seed(20260724)
    base_layer = _ToyLayer()
    base_next = _ToyNextLayer()
    legacy_layer = copy.deepcopy(base_layer)
    cached_layer = copy.deepcopy(base_layer)
    legacy_next = copy.deepcopy(base_next)
    cached_next = copy.deepcopy(base_next)
    state = SimpleNamespace(
        inps=torch.randn(4, 2, 4),
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    )
    current_target = torch.randn(4, 2, 2)
    next_target = torch.randn(4, 2, 2)
    current_fisher = torch.tensor(
        [[1.0, -0.125], [-0.125, 0.75]],
        dtype=torch.bfloat16,
    )
    next_fisher = torch.tensor(
        [[0.625, 0.0625], [0.0625, 1.25]],
        dtype=torch.bfloat16,
    )

    def build(layer, next_layer, *, cached):
        context = RefreshContext(
            module=layer.proj,
            layer_lr=3e-4,
            grad_clip=1.0,
            backward_bsz=2,
            scheduler=_SharedSampleScheduler(4, 4, seed=19),
        )
        refresh = make_grad_refresh_fn(
            layer=layer,
            module=layer.proj,
            layer_state=state,
            fp_out_for_this_layer=current_target,
            fisher=current_fisher.float() if cached else current_fisher,
            ctx=context,
            next_layer=next_layer,
            next_fp_out=next_target,
            next_fisher=next_fisher.float() if cached else next_fisher,
            slide_alpha_fn=lambda: 0.375,
        )
        return context, refresh

    legacy_ctx, legacy_refresh = build(
        legacy_layer, legacy_next, cached=False
    )
    cached_ctx, cached_refresh = build(
        cached_layer, cached_next, cached=True
    )
    legacy_weight = legacy_layer.proj.weight.detach().float().clone()
    cached_weight = cached_layer.proj.weight.detach().float().clone()
    for trailing_start in (2, 3):
        legacy_update = legacy_refresh(legacy_weight, trailing_start)
        cached_update = cached_refresh(cached_weight, trailing_start)
        assert torch.equal(
            _raw_bytes(cached_update), _raw_bytes(legacy_update)
        )
        with torch.no_grad():
            legacy_weight[:, trailing_start:].sub_(legacy_update)
            cached_weight[:, trailing_start:].sub_(cached_update)
        assert torch.equal(
            _raw_bytes(cached_weight), _raw_bytes(legacy_weight)
        )
        assert torch.equal(
            _raw_bytes(cached_ctx.exp_avg),
            _raw_bytes(legacy_ctx.exp_avg),
        )
        assert torch.equal(
            _raw_bytes(cached_ctx.exp_avg_sq),
            _raw_bytes(legacy_ctx.exp_avg_sq),
        )
        assert cached_ctx.adam_step == legacy_ctx.adam_step
