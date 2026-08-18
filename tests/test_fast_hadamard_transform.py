from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from utils import hadamard_utils  # noqa: E402


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or hadamard_utils.fast_hadamard_transform is None,
    reason="requires the CUDA fast-hadamard-transform extension",
)
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    (
        (torch.float32, 2e-6, 3e-6),
        (torch.bfloat16, 3e-2, 6.25e-2),
    ),
)
def test_fht_forward_and_backward_match_torch_fallback(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    torch.manual_seed(20260807)
    x = torch.randn(
        257, 128, device="cuda", dtype=dtype, requires_grad=True
    )
    scale = 1.0 / math.sqrt(x.shape[-1])
    grad = torch.randn_like(x)

    actual = hadamard_utils.scaled_hadamard_transform(x, scale=scale)
    expected = hadamard_utils._hadamard_transform_torch(x) * scale
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    actual.backward(grad)
    expected_grad = (
        hadamard_utils._hadamard_transform_torch(grad) * scale
    )
    torch.testing.assert_close(x.grad, expected_grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or hadamard_utils.fast_hadamard_transform is None,
    reason="requires the CUDA fast-hadamard-transform extension",
)
def test_fht_accepts_partial_rotation_noncontiguous_layout() -> None:
    torch.manual_seed(20260808)
    base = torch.randn(8, 16, 128, device="cuda", dtype=torch.bfloat16)
    x = base.transpose(1, 2)
    assert not x.is_contiguous()
    scale = 1.0 / math.sqrt(x.shape[-1])
    actual = hadamard_utils.scaled_hadamard_transform(x, scale=scale)
    expected = hadamard_utils._hadamard_transform_torch(x) * scale
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=6.25e-2)
