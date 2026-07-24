from __future__ import annotations

import pytest
import torch

from realq.config import Config, parse_cli
from realq.refresh.fisher_loss import fisher_mse_loss
from realq.runner.layer_loop import (
    _should_stage_fisher_for_refresh,
    _stage_fisher_for_refresh,
)


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def test_fisher_fp32_cache_is_explicit_default_off_and_validated():
    assert Config().fisher_fp32_cache is False
    assert parse_cli(
        ["--fisher_fp32_cache", "true"]
    ).fisher_fp32_cache is True
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
