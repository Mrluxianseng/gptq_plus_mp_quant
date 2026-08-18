from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from utils import quant_utils, triton_activation_quant  # noqa: E402


def _expanded_reference(
    x: torch.Tensor, *, bits: int, symmetric: bool, clip_ratio: float
) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    # Preserve the production path's intentional FP32 promotion exactly.
    zero_ref = torch.zeros(rows.shape[0], device=x.device)
    xmin = torch.minimum(rows.min(1).values, zero_ref) * clip_ratio
    xmax = torch.maximum(rows.max(1).values, zero_ref) * clip_ratio
    if symmetric:
        maxq = 2 ** (bits - 1) - 1
        absmax = torch.maximum(xmin.abs(), xmax)
        scale = torch.where(
            absmax == 0, torch.ones_like(absmax), absmax / maxq
        ).unsqueeze(1)
        q = torch.clamp(
            torch.round(rows / scale), -(maxq + 1), maxq
        )
        return (scale * q).to(x.dtype).reshape_as(x)

    maxq = 2**bits - 1
    all_zero = (xmin == 0) & (xmax == 0)
    xmin = xmin.clone()
    xmax = xmax.clone()
    xmin[all_zero] = -1
    xmax[all_zero] = 1
    scale = ((xmax - xmin) / maxq).unsqueeze(1)
    zero = torch.round(-xmin.unsqueeze(1) / scale)
    q = torch.clamp(torch.round(rows / scale) + zero, 0, maxq)
    return (scale * (q - zero)).to(x.dtype).reshape_as(x)


def _reference_from_quantizer(
    x: torch.Tensor, quantizer: quant_utils.ActQuantizer
) -> torch.Tensor:
    """Run the pre-fusion equation with the production qparams.

    ``ActQuantizer.find_params`` divides by its device scalar ``maxq``.
    Recomputing scales with a Python integer differs by one FP32 ULP on CUDA,
    so that is not an exact regression oracle for the fused QDQ kernel.
    """

    rows = x.reshape(-1, x.shape[-1])
    if quantizer.sym:
        q = torch.clamp(
            torch.round(rows / quantizer.scale),
            -(quantizer.maxq + 1),
            quantizer.maxq,
        )
        return (quantizer.scale * q).to(x.dtype).reshape_as(x)
    q = torch.clamp(
        torch.round(rows / quantizer.scale) + quantizer.zero,
        0,
        quantizer.maxq,
    )
    return (
        quantizer.scale * (q - quantizer.zero)
    ).to(x.dtype).reshape_as(x)


@pytest.mark.parametrize("symmetric", (True, False))
@pytest.mark.parametrize("clip_ratio", (1.0, 0.9))
def test_compact_activation_qparams_match_expanded_reference_cpu(
    symmetric: bool, clip_ratio: float
) -> None:
    torch.manual_seed(20260807)
    x = torch.randn(2, 3, 17)
    x[0, 0].zero_()
    quantizer = quant_utils.ActQuantizer()
    quantizer.configure(
        bits=4,
        groupsize=-1,
        sym=symmetric,
        clip_ratio=clip_ratio,
    )
    quantizer.find_params(x)

    actual = quantizer(x)
    expected = _expanded_reference(
        x, bits=4, symmetric=symmetric, clip_ratio=clip_ratio
    )
    assert torch.equal(actual, expected)
    assert quantizer.scale.shape == (6, 1)
    assert quantizer.zero.shape == (6, 1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_activation_quant.is_available(),
    reason="requires CUDA Triton",
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("symmetric", (True, False))
def test_fused_activation_fake_quant_is_exact_and_ste(
    dtype: torch.dtype, symmetric: bool
) -> None:
    torch.manual_seed(20260808)
    x = torch.randn(4, 7, 513, device="cuda", dtype=dtype)
    x[0, 0].zero_()
    x.requires_grad_(True)
    quantizer = quant_utils.ActQuantizer()
    quantizer.configure(bits=4, groupsize=-1, sym=symmetric, clip_ratio=0.9)
    quantizer.find_params(x)

    actual = quantizer(x)
    expected = _reference_from_quantizer(x.detach(), quantizer)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert quantizer.scale.shape == (28, 1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_activation_quant.is_available(),
    reason="requires CUDA Triton",
)
def test_noncontiguous_activation_uses_exact_eager_fallback() -> None:
    torch.manual_seed(20260809)
    x = torch.randn(3, 11, 5, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    assert not x.is_contiguous()
    quantizer = quant_utils.ActQuantizer()
    quantizer.configure(bits=4, groupsize=-1, sym=True, clip_ratio=0.9)
    quantizer.find_params(x)
    actual = quantizer(x)
    expected = _expanded_reference(
        x, bits=4, symmetric=True, clip_ratio=0.9
    )
    assert torch.equal(actual, expected)
