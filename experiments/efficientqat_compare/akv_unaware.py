"""Post-training A/V/K fake quantisation for EfficientQAT low-activation runs.

The controlled EfficientQAT experiment is explicitly *unaware*: weight QAT
finishes before any activation or cache quantiser is installed.  A and V use
REAL-Q's existing ``ActQuantWrapper`` implementation.  K uses the same
``ActQuantizer`` equation at the same post-RoPE site, but intentionally does
not apply the Q/K Hadamard basis change embedded in REAL-Q's
``QKRotationWrapper``.

Keeping the pure-K hook in this comparison-only package makes the
``rotation=False`` claim mechanically testable and avoids changing REAL-Q's
runtime semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from gptq_utils.quant_aware_utils import (
    configure_activation_quantizers_for_gptq,
)
from realq import akv
from utils import monkeypatch, quant_utils, rotation_utils


_WRAPPER_ATTRIBUTE = "apply_rotary_pos_emb_efficientqat_k_quant_wrapper"
_EXPECTED_BITS = 4
_EXPECTED_GROUPSIZE = -1
_EXPECTED_CLIP = 0.9


def _require_controlled_config(cfg: Any) -> None:
    fields = {
        "rotate": False,
        "act_quant_aware_gptq": False,
        "k_cache_quant_aware_gptq": False,
        "a_bits": _EXPECTED_BITS,
        "k_bits": _EXPECTED_BITS,
        "v_bits": _EXPECTED_BITS,
        "a_groupsize": _EXPECTED_GROUPSIZE,
        "k_groupsize": _EXPECTED_GROUPSIZE,
        "v_groupsize": _EXPECTED_GROUPSIZE,
        "a_asym": False,
        "k_asym": False,
        "v_asym": False,
    }
    for name, expected in fields.items():
        if not hasattr(cfg, name):
            raise ValueError(
                f"Unaware A4/K4/V4 config is missing required field {name!r}."
            )
        actual = getattr(cfg, name)
        if type(expected) is bool:
            matches = type(actual) is bool and actual is expected
        else:
            matches = type(actual) is type(expected) and actual == expected
        if not matches:
            raise ValueError(
                f"A4/K4/V4-unaware-no-rotation requires {name}={expected!r}; "
                f"got {actual!r}."
            )
    for name in ("a_clip_ratio", "k_clip_ratio", "v_clip_ratio"):
        if not hasattr(cfg, name):
            raise ValueError(
                f"Unaware A4/K4/V4 config is missing required field {name!r}."
            )
        value = getattr(cfg, name)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            numeric_value = float("nan")
        if (
            isinstance(value, bool)
            or not math.isfinite(numeric_value)
            or numeric_value != _EXPECTED_CLIP
        ):
            raise ValueError(
                f"A4/K4/V4-unaware-no-rotation requires {name}=0.9; "
                f"got {value!r}."
            )


class PostRoPEKQuantWrapper(nn.Module):
    """Wrap ``apply_rotary_pos_emb`` and fake-quantise only its K output."""

    def __init__(
        self,
        func,
        *,
        bits: int,
        groupsize: int,
        sym: bool,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        if not callable(func):
            raise TypeError("Post-RoPE K wrapper requires a callable RoPE function.")
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.bits = 16
        self.groupsize = -1
        self.sym = False
        self.clip_ratio = 1.0
        self.configure(
            bits=bits,
            groupsize=groupsize,
            sym=sym,
            clip_ratio=clip_ratio,
        )

    def configure(
        self,
        *,
        bits: int,
        groupsize: int,
        sym: bool,
        clip_ratio: float,
    ) -> None:
        if type(bits) is not int or not 2 <= bits < 16:
            raise ValueError(
                f"Post-RoPE K bits must be an integer in [2, 15]; got {bits!r}."
            )
        if type(groupsize) is not int or groupsize == 0 or groupsize < -1:
            raise ValueError(
                "Post-RoPE K groupsize must be -1 or a positive integer; "
                f"got {groupsize!r}."
            )
        if type(sym) is not bool:
            raise ValueError(f"Post-RoPE K sym must be bool; got {sym!r}.")
        if isinstance(clip_ratio, bool) or not 0.0 < float(clip_ratio) <= 1.0:
            raise ValueError(
                "Post-RoPE K clip_ratio must be in (0, 1]; "
                f"got {clip_ratio!r}."
            )
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = float(clip_ratio)
        # Per-head grouping, when requested, is handled explicitly in forward.
        self.k_quantizer.configure(
            bits=bits,
            groupsize=-1,
            sym=sym,
            clip_ratio=self.clip_ratio,
        )

    def forward(self, *args, **kwargs):
        output = self.func(*args, **kwargs)
        if (
            not isinstance(output, (tuple, list))
            or len(output) != 2
            or not all(isinstance(item, torch.Tensor) for item in output)
        ):
            raise TypeError(
                "apply_rotary_pos_emb must return exactly two tensors (Q, K)."
            )
        q, k = output
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(
                "Post-RoPE Q/K tensors must have shape "
                f"[batch, heads, sequence, head_dim]; got "
                f"Q={tuple(q.shape)}, K={tuple(k.shape)}."
            )
        if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
            raise ValueError(
                "Post-RoPE Q/K batch, sequence and head dimensions disagree: "
                f"Q={tuple(q.shape)}, K={tuple(k.shape)}."
            )

        bsz, num_kv_heads, seq_len, head_dim = k.shape
        if self.groupsize not in (-1, head_dim):
            raise ValueError(
                "Post-RoPE K groupsize must be -1 (all KV heads per token) "
                f"or head_dim={head_dim}; got {self.groupsize}."
            )

        k_dtype = k.dtype
        if self.groupsize == -1:
            rows = k.transpose(1, 2).reshape(
                -1, num_kv_heads * head_dim
            )
            self.k_quantizer.find_params(rows)
            try:
                quantized = self.k_quantizer(rows)
            finally:
                self.k_quantizer.free()
            k = (
                quantized.reshape(
                    bsz, seq_len, num_kv_heads, head_dim
                )
                .transpose(1, 2)
                .to(k_dtype)
            )
        else:
            rows = k.reshape(-1, head_dim)
            self.k_quantizer.find_params(rows)
            try:
                quantized = self.k_quantizer(rows)
            finally:
                self.k_quantizer.free()
            k = quantized.reshape_as(k).to(k_dtype)

        # Q is returned verbatim: no fake quantisation, dtype conversion,
        # clone, or Hadamard transformation is permitted.
        return q, k


def _existing_rotation_wrappers(module: nn.Module) -> list[str]:
    return [
        name
        for name, child in module.named_modules()
        if isinstance(child, rotation_utils.QKRotationWrapper)
    ]


def add_post_rope_k_quantizer(
    attention: nn.Module,
    *,
    bits: int = _EXPECTED_BITS,
    groupsize: int = _EXPECTED_GROUPSIZE,
    sym: bool = True,
    clip_ratio: float = _EXPECTED_CLIP,
) -> PostRoPEKQuantWrapper:
    """Install or idempotently validate one pure post-RoPE K quantiser."""

    rotations = _existing_rotation_wrappers(attention)
    legacy_attribute = "apply_rotary_pos_emb_qk_rotation_wrapper"
    if rotations or hasattr(attention, legacy_attribute):
        raise RuntimeError(
            "Refusing to install pure post-RoPE K quantisation on an "
            f"attention module that already contains Q/K rotation: {rotations!r}."
        )

    if hasattr(attention, _WRAPPER_ATTRIBUTE):
        wrapper = getattr(attention, _WRAPPER_ATTRIBUTE)
        if not isinstance(wrapper, PostRoPEKQuantWrapper):
            raise TypeError(
                f"{type(attention).__qualname__}.{_WRAPPER_ATTRIBUTE} exists "
                "but is not PostRoPEKQuantWrapper."
            )
        requested = (bits, groupsize, sym, float(clip_ratio))
        existing = (
            wrapper.bits,
            wrapper.groupsize,
            wrapper.sym,
            wrapper.clip_ratio,
        )
        if existing != requested:
            raise RuntimeError(
                "Pure post-RoPE K wrapper was already configured differently: "
                f"existing={existing!r}, requested={requested!r}."
            )
        return wrapper

    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        attention,
        "forward",
        "apply_rotary_pos_emb",
        lambda original: PostRoPEKQuantWrapper(
            original,
            bits=bits,
            groupsize=groupsize,
            sym=sym,
            clip_ratio=clip_ratio,
        ),
    )
    setattr(attention, _WRAPPER_ATTRIBUTE, wrapper)
    return wrapper


def install_post_rope_k_quantizers(analyzer: Any, cfg: Any) -> int:
    """Install one pure K wrapper in every decoder attention block."""

    count = 0
    for index, layer in enumerate(analyzer.get_layers()):
        attention = getattr(layer, "self_attn", None)
        if not isinstance(attention, nn.Module):
            raise TypeError(
                f"Decoder layer {index} does not expose an nn.Module self_attn."
            )
        add_post_rope_k_quantizer(
            attention,
            bits=cfg.k_bits,
            groupsize=cfg.k_groupsize,
            sym=not cfg.k_asym,
            clip_ratio=cfg.k_clip_ratio,
        )
        count += 1
    return count


@dataclass(frozen=True)
class UnawareAKVSummary:
    decoder_layers: int
    activation_input_sites: int
    value_output_sites: int
    post_rope_k_sites: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "unaware",
            "global_rotation": False,
            "local_qk_hadamard": False,
            "q_cache_quantized": False,
            "decoder_layers": self.decoder_layers,
            "activation_input_sites": self.activation_input_sites,
            "value_output_sites": self.value_output_sites,
            "post_rope_k_sites": self.post_rope_k_sites,
        }


def _assert_no_rotation_state(model: nn.Module) -> None:
    if bool(getattr(model, "_gptqplus_rotation_wrappers_installed", False)):
        raise RuntimeError("Model is marked as containing QuaRot wrappers.")
    if bool(getattr(model, "_gptqplus_checkpoint_is_rotated", False)):
        raise RuntimeError("Model is marked as a rotated checkpoint.")
    rotations = _existing_rotation_wrappers(model)
    if rotations:
        raise RuntimeError(
            f"Model contains forbidden Q/K Hadamard wrappers: {rotations!r}."
        )


def setup_unaware_post_quant_no_rotation(
    analyzer: Any,
    cfg: Any,
) -> UnawareAKVSummary:
    """Install controlled W2A4KV4 runtime quantisation after QAT.

    The live model must already be in evaluation mode.  This is a fail-closed
    temporal gate: callers cannot accidentally install the supposedly-unaware
    path on a model that is still training.
    """

    _require_controlled_config(cfg)
    model = analyzer.model
    if not isinstance(model, nn.Module):
        raise TypeError("Analyzer must expose an nn.Module as `.model`.")
    if model.training:
        raise RuntimeError(
            "Unaware A/V/K quantisation may only be installed on model.eval() "
            "after Block-AP, E2E-QP and materialisation have completed."
        )
    _assert_no_rotation_state(model)

    layers = list(analyzer.get_layers())
    if not layers:
        raise RuntimeError("Analyzer returned no decoder layers.")

    # This reuses REAL-Q's exact wrapper topology and exact A/V configuration
    # function, while intentionally bypassing its Hadamard-bearing K helper.
    akv.install_actquant_wrappers(analyzer)
    activation_count, value_count = (
        configure_activation_quantizers_for_gptq(cfg, model)
    )
    k_count = install_post_rope_k_quantizers(analyzer, cfg)

    expected_activation_count = 7 * len(layers)
    if activation_count != expected_activation_count:
        raise RuntimeError(
            "A4 site count mismatch: "
            f"{activation_count} != 7*{len(layers)}="
            f"{expected_activation_count}."
        )
    if value_count != len(layers):
        raise RuntimeError(
            f"V4 site count mismatch: {value_count} != {len(layers)}."
        )
    if k_count != len(layers):
        raise RuntimeError(
            f"K4 site count mismatch: {k_count} != {len(layers)}."
        )

    summary = UnawareAKVSummary(
        decoder_layers=len(layers),
        activation_input_sites=activation_count,
        value_output_sites=value_count,
        post_rope_k_sites=k_count,
    )
    assert_unaware_post_quant_no_rotation(analyzer, cfg, summary=summary)
    model._efficientqat_unaware_akv_summary = summary.as_dict()
    return summary


def install_unaware_av_and_pure_post_rope_k(
    analyzer: Any,
    *,
    bits: int = _EXPECTED_BITS,
    groupsize: int = _EXPECTED_GROUPSIZE,
    symmetric: bool = True,
    clip_ratio: float = _EXPECTED_CLIP,
) -> UnawareAKVSummary:
    """Runner-facing entry point for the frozen W2A4KV4-unaware protocol."""

    cfg = SimpleNamespace(
        rotate=False,
        act_quant_aware_gptq=False,
        k_cache_quant_aware_gptq=False,
        a_bits=bits,
        a_groupsize=groupsize,
        a_asym=not symmetric,
        a_clip_ratio=clip_ratio,
        k_bits=bits,
        k_groupsize=groupsize,
        k_asym=not symmetric,
        k_clip_ratio=clip_ratio,
        v_bits=bits,
        v_groupsize=groupsize,
        v_asym=not symmetric,
        v_clip_ratio=clip_ratio,
    )
    return setup_unaware_post_quant_no_rotation(analyzer, cfg)


def assert_unaware_post_quant_no_rotation(
    analyzer: Any,
    cfg: Any,
    *,
    summary: UnawareAKVSummary | None = None,
) -> UnawareAKVSummary:
    """Validate the full runtime topology without mutating it."""

    _require_controlled_config(cfg)
    model = analyzer.model
    _assert_no_rotation_state(model)
    layers = list(analyzer.get_layers())
    expected_a = 7 * len(layers)
    expected_v = len(layers)
    expected_k = len(layers)

    wrappers = quant_utils.find_qlayers(
        model, layers=[quant_utils.ActQuantWrapper]
    )
    activation_count = 0
    value_count = 0
    for name, wrapper in wrappers.items():
        if "lm_head" in name:
            raise RuntimeError("lm_head must not be wrapped for A/V quantisation.")
        if wrapper.online_full_had or wrapper.online_partial_had:
            raise RuntimeError(
                f"ActQuantWrapper {name!r} contains forbidden Hadamard state."
            )
        if (
            wrapper.quantizer.bits != _EXPECTED_BITS
            or wrapper.quantizer.groupsize != _EXPECTED_GROUPSIZE
            or wrapper.quantizer.sym is not True
            or wrapper.quantizer.clip_ratio != _EXPECTED_CLIP
        ):
            raise RuntimeError(
                f"ActQuantWrapper {name!r} does not match symmetric A4/.9."
            )
        activation_count += 1
        is_v = "v_proj" in name
        expected_output_bits = _EXPECTED_BITS if is_v else 16
        if wrapper.out_quantizer.bits != expected_output_bits:
            raise RuntimeError(
                f"ActQuantWrapper {name!r} has output bits "
                f"{wrapper.out_quantizer.bits}, expected {expected_output_bits}."
            )
        if is_v:
            if (
                wrapper.out_quantizer.groupsize != _EXPECTED_GROUPSIZE
                or wrapper.out_quantizer.sym is not True
                or wrapper.out_quantizer.clip_ratio != _EXPECTED_CLIP
            ):
                raise RuntimeError(
                    f"V wrapper {name!r} does not match symmetric V4/.9."
                )
            value_count += 1

    k_count = 0
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        wrapper = getattr(attention, _WRAPPER_ATTRIBUTE, None)
        if not isinstance(wrapper, PostRoPEKQuantWrapper):
            raise RuntimeError(
                f"Decoder layer {index} is missing its pure post-RoPE K wrapper."
            )
        if (
            wrapper.bits != _EXPECTED_BITS
            or wrapper.groupsize != _EXPECTED_GROUPSIZE
            or wrapper.sym is not True
            or wrapper.clip_ratio != _EXPECTED_CLIP
        ):
            raise RuntimeError(
                f"Decoder layer {index} K wrapper does not match symmetric K4/.9."
            )
        k_count += 1

    actual = UnawareAKVSummary(
        decoder_layers=len(layers),
        activation_input_sites=activation_count,
        value_output_sites=value_count,
        post_rope_k_sites=k_count,
    )
    if (
        activation_count != expected_a
        or value_count != expected_v
        or k_count != expected_k
    ):
        raise RuntimeError(
            "Unaware A/V/K topology mismatch: "
            f"actual={actual.as_dict()!r}, "
            f"expected A/V/K=({expected_a}, {expected_v}, {expected_k})."
        )
    if summary is not None and actual != summary:
        raise RuntimeError(
            f"Unaware A/V/K summary changed: {actual!r} != {summary!r}."
        )
    return actual


__all__ = [
    "PostRoPEKQuantWrapper",
    "UnawareAKVSummary",
    "add_post_rope_k_quantizer",
    "assert_unaware_post_quant_no_rotation",
    "install_unaware_av_and_pure_post_rope_k",
    "install_post_rope_k_quantizers",
    "setup_unaware_post_quant_no_rotation",
]
