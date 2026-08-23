from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
EFFICIENTQAT_ROOT = REPO_ROOT / "EfficientQAT"
# Bind the repository's namespace package before exposing EfficientQAT's
# standalone ``utils.py`` module, matching the production bootstrap.
importlib.import_module("utils")
if str(EFFICIENTQAT_ROOT) not in sys.path:
    # Keep the repository root ahead of EfficientQAT so its standalone
    # ``utils.py`` cannot shadow the workspace ``utils`` package.
    sys.path.append(str(EFFICIENTQAT_ROOT))

from quantize.int_linear_fake import QuantLinear as FakeQuantLinear
from quantize.int_linear_real import QuantLinear as PackedQuantLinear
from quantize.quantizer import UniformAffineQuantizer

from experiments.efficientqat_compare.materialize import (
    SYMMETRIC_SOURCE_QUANTIZER,
    decode_efficientqat_weight_cpu,
    materialize_efficientqat_model,
)


def test_w3_symmetric_uses_realq_signed_grid_and_fixed_zero() -> None:
    weight = torch.tensor(
        [[-6.0, -4.0, -2.0, 0.0, 1.0, 2.0, 4.0, 6.0]],
        dtype=torch.float32,
    )
    quantizer = UniformAffineQuantizer(
        n_bits=3,
        group_size=4,
        weight=weight,
        symmetric=True,
    )

    assert quantizer.qmin == 0
    assert quantizer.qmax == 7
    assert quantizer.symmetric is True
    assert "zero_point" not in dict(quantizer.named_parameters())
    assert "zero_point" in dict(quantizer.named_buffers())
    assert torch.equal(quantizer.zero_point, torch.full((2, 1), 4.0))
    assert torch.equal(quantizer.scale.detach(), torch.full((2, 1), 2.0))

    expected_codes = torch.clamp(
        torch.round(weight.reshape(-1, 4) / 2.0), -4, 3
    )
    expected = (expected_codes * 2.0).reshape_as(weight)
    assert torch.equal(quantizer.fake_quant(weight), expected)


def test_asymmetric_default_keeps_learnable_zero_point() -> None:
    weight = torch.tensor([[-2.0, -1.0, 1.0, 5.0]], dtype=torch.float32)
    quantizer = UniformAffineQuantizer(3, 4, weight=weight)
    assert quantizer.symmetric is False
    assert "zero_point" in dict(quantizer.named_parameters())
    assert "zero_point" not in dict(quantizer.named_buffers())


def test_symmetric_fake_pack_decode_roundtrip() -> None:
    linear = nn.Linear(8, 3, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [
                    [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                    [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
                    [-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0],
                ]
            )
        )
    fake = FakeQuantLinear(linear, wbits=3, group_size=4, symmetric=True)
    expected = fake.weight_quantizer.fake_quant(fake.weight).detach()
    fake.weight.data.copy_(expected)

    scales = (
        fake.weight_quantizer.scale.detach()
        .view(linear.out_features, -1)
        .transpose(0, 1)
        .contiguous()
    )
    zeros = (
        fake.weight_quantizer.zero_point.detach()
        .view(linear.out_features, -1)
        .transpose(0, 1)
        .contiguous()
    )
    packed = PackedQuantLinear(3, 4, 8, 3, False)
    packed.pack(fake.cpu(), scales.float(), zeros.float())
    decoded = decode_efficientqat_weight_cpu(
        packed,
        expected_bits=3,
        expected_group_size=4,
        source_quantizer=SYMMETRIC_SOURCE_QUANTIZER,
    )
    assert torch.equal(decoded, expected.to(torch.bfloat16))


def test_symmetric_decoder_rejects_nonfixed_zero_point() -> None:
    linear = nn.Linear(4, 2, bias=False, dtype=torch.float32)
    fake = FakeQuantLinear(linear, wbits=3, group_size=4, symmetric=True)
    fake.weight.data.copy_(fake.weight_quantizer.fake_quant(fake.weight).detach())
    scales = fake.weight_quantizer.scale.detach().view(2, 1).transpose(0, 1)
    bad_zeros = torch.tensor([[3.0, 4.0]])
    packed = PackedQuantLinear(3, 4, 4, 2, False)
    packed.pack(fake.cpu(), scales.float(), bad_zeros)

    with pytest.raises(ValueError, match="fixed packed zero-point 4"):
        decode_efficientqat_weight_cpu(
            packed,
            expected_bits=3,
            expected_group_size=4,
            source_quantizer=SYMMETRIC_SOURCE_QUANTIZER,
        )


def test_symmetric_materialization_manifest_preserves_quantizer_identity() -> None:
    linear = nn.Linear(4, 2, bias=False, dtype=torch.float32)
    fake = FakeQuantLinear(linear, wbits=3, group_size=4, symmetric=True)
    fake.weight.data.copy_(fake.weight_quantizer.fake_quant(fake.weight).detach())
    scales = fake.weight_quantizer.scale.detach().view(2, 1).transpose(0, 1)
    zeros = fake.weight_quantizer.zero_point.detach().view(2, 1).transpose(0, 1)
    packed = PackedQuantLinear(3, 4, 4, 2, False)
    packed.pack(fake.cpu(), scales.float(), zeros.float())
    model = nn.Module()
    model.proj = packed

    manifest = materialize_efficientqat_model(
        model,
        expected_count=1,
        expected_bits=3,
        expected_group_size=4,
        expected_module_names=["proj"],
        provenance={"source_quantizer": SYMMETRIC_SOURCE_QUANTIZER},
        source_quantizer=SYMMETRIC_SOURCE_QUANTIZER,
    )
    assert manifest["source_quantizer"] == SYMMETRIC_SOURCE_QUANTIZER
    assert manifest["provenance"]["source_quantizer"] == SYMMETRIC_SOURCE_QUANTIZER
    assert isinstance(model.proj, nn.Linear)
