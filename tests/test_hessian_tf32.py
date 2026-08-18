from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from realq.config import Config, parse_cli
from realq.quant.realq_layer import RealQLayer, _hessian_matmul


def test_hessian_tf32_config_defaults_on_and_validates() -> None:
    assert Config().hessian_tf32 is True
    assert parse_cli(["--hessian_tf32", "false"]).hessian_tf32 is False
    with pytest.raises(ValueError, match="hessian_tf32"):
        Config(hessian_tf32=1)


def test_cpu_hessian_stays_fp32_and_does_not_change_cuda_policy() -> None:
    previous = torch.backends.cuda.matmul.allow_tf32
    left = torch.randn(2, 4, 8, dtype=torch.float32)
    right = torch.randn(2, 8, 4, dtype=torch.float32)
    actual = _hessian_matmul(left, right, use_tf32=True)
    expected = torch.bmm(left, right)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert torch.backends.cuda.matmul.allow_tf32 is previous

    layer = RealQLayer(
        nn.Linear(4, 4, bias=False),
        torch.ones(1, 2, 1),
        SimpleNamespace(),
        num_groups=1,
        dev=torch.device("cpu"),
        hessian_tf32=True,
    )
    layer.add_batch(torch.randn(1, 2, 4, dtype=torch.bfloat16))
    assert layer.H.dtype == torch.float32
    assert layer.act_square.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_tf32_hessian_matches_fp32_with_fp32_accumulation() -> None:
    torch.manual_seed(20260807)
    device = torch.device("cuda")
    groups, tokens, columns = 4, 8192, 384
    # RealQ inputs originate in BF16, then become FP32 before weighting.
    inp = torch.randn(
        tokens, columns, device=device, dtype=torch.bfloat16
    ).float()
    saliency = torch.rand(
        tokens, groups, device=device, dtype=torch.float32
    )
    left = inp.transpose(0, 1).unsqueeze(0).expand(groups, -1, -1)
    right = inp.unsqueeze(0) * saliency.transpose(0, 1).unsqueeze(-1)

    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        exact = _hessian_matmul(left, right, use_tf32=False)
        candidate = _hessian_matmul(left, right, use_tf32=True)
        torch.cuda.synchronize()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

    assert exact.dtype == candidate.dtype == torch.float32
    relative_l2 = (
        (candidate - exact).norm() / exact.norm().clamp_min(1e-30)
    ).item()
    assert relative_l2 < 1e-3

    accumulated = torch.zeros_like(candidate, dtype=torch.float32)
    accumulated.add_(candidate)
    accumulated.add_(candidate)
    assert accumulated.dtype == torch.float32
    torch.testing.assert_close(
        accumulated, candidate * 2.0, atol=0.0, rtol=0.0
    )
    assert torch.backends.cuda.matmul.allow_tf32 is previous
