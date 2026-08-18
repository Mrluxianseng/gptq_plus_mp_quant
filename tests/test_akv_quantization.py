from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gptq_utils.quant_aware_utils import (  # noqa: E402
    configure_activation_quantizers_for_gptq,
    disable_fp_path_quant,
)
from realq import akv  # noqa: E402
from realq.config import Config  # noqa: E402
from utils import hadamard_utils, quant_utils, rotation_utils  # noqa: E402


def _manual_sym_per_token(x: torch.Tensor, bits: int, clip: float) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    zero = torch.zeros(rows.shape[0], device=x.device, dtype=x.dtype)
    xmin = torch.minimum(rows.min(dim=1).values, zero) * clip
    xmax = torch.maximum(rows.max(dim=1).values, zero) * clip
    absmax = torch.maximum(xmin.abs(), xmax)
    maxq = 2 ** (bits - 1) - 1
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / maxq)
    q = torch.clamp(
        torch.round(rows / scale.unsqueeze(1)),
        -(maxq + 1),
        maxq,
    )
    return (q * scale.unsqueeze(1)).reshape_as(x)


@pytest.mark.parametrize("clip", [1.0, 0.9])
def test_a4_symmetric_per_token_matches_shared_implementation_equation(clip):
    x = torch.tensor(
        [[[-8.0, -1.0, 0.0, 2.0, 7.0], [0.0, 0.3, 1.1, -2.7, 4.2]]]
    )
    quantizer = quant_utils.ActQuantizer()
    quantizer.configure(bits=4, groupsize=-1, sym=True, clip_ratio=clip)
    quantizer.find_params(x)

    actual = quantizer(x)
    expected = _manual_sym_per_token(x, bits=4, clip=clip)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Dynamic per-token quantization stores one compact scale per flattened
    # token row.  The QDQ kernel broadcasts it without a full-shape tensor.
    assert quantizer.scale.shape == (2, 1)


def test_a16_is_an_exact_noop_and_groupwise_supports_a_short_tail():
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0]])

    fp = quant_utils.ActQuantizer()
    fp.configure(bits=16, groupsize=-1, sym=True, clip_ratio=1.0)
    assert fp(x) is x

    grouped = quant_utils.ActQuantizer()
    grouped.configure(bits=4, groupsize=2, sym=True, clip_ratio=0.9)
    grouped.find_params(x)
    actual = grouped(x)
    expected = torch.cat(
        [
            _manual_sym_per_token(x[:, start : start + 2], 4, 0.9)
            for start in range(0, x.shape[-1], 2)
        ],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert grouped.scale.shape == (1, 3)

    affine = quant_utils.ActQuantizer()
    affine.configure(bits=4, groupsize=2, sym=False, clip_ratio=1.0)
    affine.find_params(x.abs())
    # Affine ranges must include real zero even for an all-positive group.
    assert torch.all(affine.zero >= 0)


@pytest.mark.parametrize("bits,clip", [(16, 1.0), (4, 1.0), (4, 0.9)])
def test_v_output_quantizer_uses_the_same_per_token_equation(bits, clip):
    linear = torch.nn.Linear(5, 5, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(5))
    wrapper = quant_utils.ActQuantWrapper(linear)
    wrapper.out_quantizer.configure(
        bits=bits, groupsize=-1, sym=True, clip_ratio=clip
    )
    x = torch.tensor([[[-8.0, -1.0, 0.0, 2.0, 7.0]]])
    expected = (
        x if bits == 16 else _manual_sym_per_token(x, bits=bits, clip=clip)
    )
    torch.testing.assert_close(wrapper(x), expected, rtol=0, atol=0)


class _WrappedSites(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = quant_utils.ActQuantWrapper(
            torch.nn.Linear(5, 4, bias=False)
        )
        self.v_proj = quant_utils.ActQuantWrapper(
            torch.nn.Linear(5, 4, bias=False)
        )
        # add_actquant intentionally does not install this in production; the
        # synthetic site verifies the defensive lm_head exception itself.
        self.lm_head = quant_utils.ActQuantWrapper(
            torch.nn.Linear(5, 4, bias=False)
        )


def _paper_akv_config() -> Config:
    return Config(
        model="unused",
        rotate=False,
        a_bits=4,
        a_groupsize=-1,
        a_asym=False,
        a_clip_ratio=0.9,
        k_bits=4,
        k_groupsize=-1,
        k_asym=False,
        k_clip_ratio=0.9,
        v_bits=4,
        v_groupsize=-1,
        v_asym=False,
        v_clip_ratio=0.9,
    )


def test_legacy_and_realq_share_exact_a_v_sites_and_lm_head_exception():
    torch.manual_seed(0)
    legacy = _WrappedSites()
    refactored = _WrappedSites()
    refactored.load_state_dict(legacy.state_dict())
    cfg = _paper_akv_config()

    legacy_counts = configure_activation_quantizers_for_gptq(cfg, legacy)
    akv.configure_a_v_quantizers(SimpleNamespace(model=refactored), cfg)

    assert legacy_counts == (2, 1)
    for old_site, new_site in (
        (legacy.q_proj, refactored.q_proj),
        (legacy.v_proj, refactored.v_proj),
        (legacy.lm_head, refactored.lm_head),
    ):
        assert old_site.quantizer.bits == new_site.quantizer.bits
        assert old_site.quantizer.sym == new_site.quantizer.sym
        assert old_site.quantizer.clip_ratio == new_site.quantizer.clip_ratio
        assert old_site.out_quantizer.bits == new_site.out_quantizer.bits
        assert old_site.out_quantizer.sym == new_site.out_quantizer.sym
        assert (
            old_site.out_quantizer.clip_ratio
            == new_site.out_quantizer.clip_ratio
        )

    # A4 enters q/v projection inputs, but Q has no independent output
    # quantizer. V4 is exclusively attached to v_proj's output.
    assert legacy.q_proj.quantizer.bits == 4
    assert legacy.q_proj.out_quantizer.bits == 16
    assert legacy.v_proj.quantizer.bits == 4
    assert legacy.v_proj.out_quantizer.bits == 4
    assert legacy.lm_head.quantizer.bits == 16
    assert legacy.lm_head.out_quantizer.bits == 16

    x = torch.tensor([[[-8.0, -1.0, 0.0, 2.0, 7.0]]])
    for old_site, new_site in (
        (legacy.q_proj, refactored.q_proj),
        (legacy.v_proj, refactored.v_proj),
        (legacy.lm_head, refactored.lm_head),
    ):
        torch.testing.assert_close(
            old_site(x), new_site(x), rtol=0, atol=0
        )
    torch.testing.assert_close(
        legacy.lm_head(x), legacy.lm_head.module(x), rtol=0, atol=0
    )


def test_disable_fp_path_quant_restores_a_and_v_even_after_exception():
    model = _WrappedSites()
    cfg = _paper_akv_config()
    configure_activation_quantizers_for_gptq(cfg, model)

    with pytest.raises(RuntimeError, match="teacher failure"):
        with disable_fp_path_quant(model):
            assert model.q_proj.quantizer.bits == 16
            assert model.v_proj.quantizer.bits == 16
            assert model.v_proj.out_quantizer.bits == 16
            raise RuntimeError("teacher failure")

    assert model.q_proj.quantizer.bits == 4
    assert model.v_proj.quantizer.bits == 4
    assert model.v_proj.out_quantizer.bits == 4


def _rope_site(q, k):
    # Non-identity values make it observable that the wrapper consumes the
    # post-RoPE tensors rather than the inputs to this function.
    return q + 0.25, k - 0.5


class _FakeAttention(torch.nn.Module):
    def forward(self, q, k):
        return _rope_site(q, k)


def _hadamard_last_dim(x):
    return hadamard_utils.scaled_hadamard_transform(
        x.float(), scale=1.0 / math.sqrt(x.shape[-1])
    ).to(x.dtype)


@pytest.mark.parametrize("clip", [1.0, 0.9])
def test_k4_is_post_rope_per_token_q_is_rotation_only_and_patch_is_idempotent(
    clip,
):
    attention = _FakeAttention()
    first = rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
        attention,
        "_rope_site",
        head_dim=4,
        k_bits=4,
        k_groupsize=-1,
        k_sym=True,
        k_clip_ratio=clip,
    )
    second = rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
        attention,
        "_rope_site",
        head_dim=4,
        k_bits=4,
        k_groupsize=-1,
        k_sym=True,
        k_clip_ratio=clip,
    )
    assert first is second
    assert sum(
        isinstance(module, rotation_utils.QKRotationWrapper)
        for module in attention.modules()
    ) == 1

    q = torch.tensor(
        [[[[1.0, 2.0, -3.0, 4.0], [0.5, -1.0, 2.0, -4.0]]]]
    )
    k = torch.tensor(
        [[[[8.0, -1.0, 0.0, 2.0], [1.5, -3.0, 4.0, 0.25]]]]
    )
    actual_q, actual_k = attention(q, k)
    rope_q, rope_k = _rope_site(q, k)
    rotated_q = _hadamard_last_dim(rope_q)
    rotated_k = _hadamard_last_dim(rope_k)
    expected_k = _manual_sym_per_token(rotated_k, bits=4, clip=clip)

    # Q participates in the orthogonal Q/K basis change, preserving QK^T,
    # but W*x*A4KV4 does not fake-quantize Q as a separate cache tensor.
    torch.testing.assert_close(actual_q, rotated_q, rtol=0, atol=0)
    assert not torch.equal(
        actual_q, _manual_sym_per_token(rotated_q, bits=4, clip=clip)
    )
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)


def test_k16_rotates_but_does_not_quantize_and_disable_restore_is_exact():
    wrapper = rotation_utils.QKRotationWrapper(
        _rope_site,
        head_dim=4,
        k_bits=16,
        k_groupsize=-1,
        k_sym=True,
        k_clip_ratio=1.0,
    )
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    k = torch.tensor([[[[-4.0, 3.0, -2.0, 1.0]]]])
    expected_q, expected_k = (_hadamard_last_dim(x) for x in _rope_site(q, k))
    actual_q, actual_k = wrapper(q, k)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)

    wrapper.configure_k_quant(k_bits=4, k_sym=True, k_clip_ratio=0.9)
    holder = torch.nn.Module()
    holder.wrapper = wrapper
    with disable_fp_path_quant(holder):
        assert wrapper.k_quant_enabled is False
        disabled_q, disabled_k = wrapper(q, k)
        torch.testing.assert_close(disabled_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(disabled_k, expected_k, rtol=0, atol=0)
    assert wrapper.k_quant_enabled is True
    assert not torch.equal(wrapper(q, k)[1], expected_k)


def test_aware_and_unaware_setup_are_disjoint_in_time(monkeypatch):
    events = []
    analyzer = SimpleNamespace(model=SimpleNamespace())

    monkeypatch.setattr(
        akv,
        "install_actquant_wrappers",
        lambda _analyzer: events.append("install"),
    )
    monkeypatch.setattr(
        akv,
        "configure_a_v_quantizers",
        lambda _analyzer, _cfg: events.append("av"),
    )
    monkeypatch.setattr(
        akv,
        "install_k_cache_wrappers",
        lambda _analyzer, _cfg: events.append("k"),
    )

    aware = _paper_akv_config()
    aware.act_quant_aware_gptq = True
    aware.k_cache_quant_aware_gptq = True
    akv.setup_aware_pre_quant(analyzer, aware)
    akv.setup_unaware_post_quant(analyzer, aware)
    assert events == ["install", "av", "k", "install"]

    events.clear()
    unaware = _paper_akv_config()
    akv.setup_aware_pre_quant(analyzer, unaware)
    akv.setup_unaware_post_quant(analyzer, unaware)
    assert events == ["install", "install", "av", "k"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("a_bits", 1),
        ("k_bits", 17),
        ("v_groupsize", 0),
        ("a_clip_ratio", 0.0),
        ("k_clip_ratio", float("nan")),
    ],
)
def test_config_rejects_invalid_akv_domains(field, value):
    with pytest.raises(ValueError):
        Config(**{field: value})


def test_k_groupsize_domain_is_explicit():
    wrapper = rotation_utils.QKRotationWrapper(_rope_site, head_dim=4)
    with pytest.raises(ValueError, match="groupsize"):
        wrapper.configure_k_quant(
            k_bits=4,
            k_groupsize=2,
            k_sym=True,
            k_clip_ratio=0.9,
        )
