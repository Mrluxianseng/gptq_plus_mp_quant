"""A/K/V fake-quant adapters shared by YAQA Hessian collection and evaluation.

YAQA's Hessian collector replaces ordinary ``nn.Linear`` modules with
``CustomLinear(input, mode)`` while an hfized QTIP model uses
``QuantizedLinear`` modules without a public ``weight`` attribute.  REAL-Q's
``ActQuantWrapper`` therefore cannot wrap either representation safely.

This module reuses REAL-Q's exact dynamic ``ActQuantizer`` arithmetic through
forward hooks.  The hook callables retain their quantizers as ordinary Python
objects rather than registered child modules, so YAQA's broad activation
checkpoint policy cannot wrap them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
from typing import Any, Iterable

import torch
import torch.nn as nn

from utils import monkeypatch
from utils.quant_utils import ActQuantizer


_A_MARKER = "_yaqa_a_quant_hook"
_V_MARKER = "_yaqa_v_quant_hook"
_K_MARKER = "_yaqa_post_rope_k_quantizer"
_K_WRAPPER_MARKER = "_yaqa_post_rope_k_forward_wrapper"
_MODEL_MARKER = "_yaqa_akv_quantization_summary"


def _validate_quant_config(
    *, bits: int, groupsize: int, symmetric: bool, clip_ratio: float
) -> None:
    if type(bits) is not int or not 2 <= bits <= 16:
        raise ValueError(f"bits must be an integer in [2, 16]; got {bits!r}")
    if groupsize != -1:
        raise ValueError(
            "The controlled YAQA comparison requires per-token groupsize=-1; "
            f"got {groupsize!r}"
        )
    if type(symmetric) is not bool or not symmetric:
        raise ValueError("The controlled YAQA comparison requires symmetric QDQ")
    if isinstance(clip_ratio, bool) or not 0.0 < float(clip_ratio) <= 1.0:
        raise ValueError(
            f"clip_ratio must be finite and in (0, 1]; got {clip_ratio!r}"
        )


class _TensorQuantizer:
    """Dynamic QDQ callable backed by REAL-Q's exact ``ActQuantizer``."""

    def __init__(
        self,
        *,
        bits: int,
        groupsize: int,
        symmetric: bool,
        clip_ratio: float,
    ) -> None:
        _validate_quant_config(
            bits=bits,
            groupsize=groupsize,
            symmetric=symmetric,
            clip_ratio=clip_ratio,
        )
        self.bits = bits
        self.groupsize = groupsize
        self.symmetric = symmetric
        self.clip_ratio = float(clip_ratio)
        self.quantizer = ActQuantizer()
        self.quantizer.configure(
            bits=bits,
            groupsize=groupsize,
            sym=symmetric,
            clip_ratio=self.clip_ratio,
        )

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "YAQA activation quantization expects a Tensor, got "
                f"{type(value).__qualname__}"
            )
        if self.bits >= 16:
            return value
        self.quantizer.find_params(value)
        try:
            return self.quantizer(value)
        finally:
            self.quantizer.free()

    def config_tuple(self) -> tuple[int, int, bool, float]:
        return (
            self.bits,
            self.groupsize,
            self.symmetric,
            self.clip_ratio,
        )


class _InputQuantHook:
    def __init__(self, quantizer: _TensorQuantizer) -> None:
        self.quantizer = quantizer

    def __call__(
        self, _module: nn.Module, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        if not args:
            raise RuntimeError("YAQA projection forward received no positional input")
        return (self.quantizer(args[0]), *args[1:])


class _OutputQuantHook:
    def __init__(self, quantizer: _TensorQuantizer) -> None:
        self.quantizer = quantizer

    def __call__(
        self, _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> torch.Tensor:
        return self.quantizer(output)


class PostRoPEKQuantizer:
    """Quantize K after RoPE with one scale per token across all KV heads."""

    def __init__(
        self,
        *,
        bits: int,
        groupsize: int,
        symmetric: bool,
        clip_ratio: float,
    ) -> None:
        self.tensor_quantizer = _TensorQuantizer(
            bits=bits,
            groupsize=groupsize,
            symmetric=symmetric,
            clip_ratio=clip_ratio,
        )

    @property
    def bits(self) -> int:
        return self.tensor_quantizer.bits

    @property
    def groupsize(self) -> int:
        return self.tensor_quantizer.groupsize

    @property
    def symmetric(self) -> bool:
        return self.tensor_quantizer.symmetric

    @property
    def clip_ratio(self) -> float:
        return self.tensor_quantizer.clip_ratio

    def __call__(self, key_states: torch.Tensor) -> torch.Tensor:
        if key_states.ndim != 4:
            raise ValueError(
                "Post-RoPE K must have shape [batch, kv_heads, sequence, "
                f"head_dim]; got {tuple(key_states.shape)}"
            )
        bsz, kv_heads, seq_len, head_dim = key_states.shape
        original_dtype = key_states.dtype
        token_rows = (
            key_states.transpose(1, 2)
            .reshape(bsz, seq_len, kv_heads * head_dim)
        )
        quantized = self.tensor_quantizer(token_rows)
        return (
            quantized.reshape(bsz, seq_len, kv_heads, head_dim)
            .transpose(1, 2)
            .to(original_dtype)
        )

    def config_tuple(self) -> tuple[int, int, bool, float]:
        return self.tensor_quantizer.config_tuple()


@dataclass(frozen=True)
class AKVQuantizationSummary:
    mode: str
    decoder_layers: int
    activation_input_sites: int
    value_output_sites: int
    post_rope_k_sites: int
    a_bits: int
    k_bits: int
    v_bits: int
    groupsize: int
    symmetric: bool
    clip_ratio: float
    query_quantized: bool = False
    extra_qk_hadamard: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decoder_layers(model: nn.Module) -> list[nn.Module]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError(
            "YAQA A/K/V adapter requires model.model.layers on a Llama model"
        )
    result = list(layers)
    if not result:
        raise RuntimeError("YAQA A/K/V adapter found no decoder layers")
    return result


def _layer_projections(layer: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    attention = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    if not isinstance(attention, nn.Module) or not isinstance(mlp, nn.Module):
        raise TypeError("Each YAQA decoder layer must expose self_attn and mlp")
    for name, owner in (
        ("q_proj", attention),
        ("k_proj", attention),
        ("v_proj", attention),
        ("o_proj", attention),
        ("gate_proj", mlp),
        ("up_proj", mlp),
        ("down_proj", mlp),
    ):
        projection = getattr(owner, name, None)
        if not isinstance(projection, nn.Module):
            raise TypeError(
                f"YAQA decoder projection {name!r} is not an nn.Module"
            )
        yield name, projection


def _install_input_hook(
    module: nn.Module,
    *,
    bits: int,
    groupsize: int,
    symmetric: bool,
    clip_ratio: float,
) -> None:
    requested = (bits, groupsize, symmetric, float(clip_ratio))
    existing = getattr(module, _A_MARKER, None)
    if existing is not None:
        if existing["config"] != requested:
            raise RuntimeError(
                "YAQA A quantizer was already installed with a different "
                f"configuration: {existing['config']!r} != {requested!r}"
            )
        return
    hook = _InputQuantHook(
        _TensorQuantizer(
            bits=bits,
            groupsize=groupsize,
            symmetric=symmetric,
            clip_ratio=clip_ratio,
        )
    )
    handle = module.register_forward_pre_hook(hook)
    setattr(module, _A_MARKER, {"config": requested, "hook": hook, "handle": handle})


def _install_output_hook(
    module: nn.Module,
    *,
    bits: int,
    groupsize: int,
    symmetric: bool,
    clip_ratio: float,
) -> None:
    requested = (bits, groupsize, symmetric, float(clip_ratio))
    existing = getattr(module, _V_MARKER, None)
    if existing is not None:
        if existing["config"] != requested:
            raise RuntimeError(
                "YAQA V quantizer was already installed with a different "
                f"configuration: {existing['config']!r} != {requested!r}"
            )
        return
    hook = _OutputQuantHook(
        _TensorQuantizer(
            bits=bits,
            groupsize=groupsize,
            symmetric=symmetric,
            clip_ratio=clip_ratio,
        )
    )
    handle = module.register_forward_hook(hook)
    setattr(module, _V_MARKER, {"config": requested, "hook": hook, "handle": handle})


def install_akv_quantization(
    model: nn.Module,
    *,
    a_bits: int,
    k_bits: int,
    v_bits: int,
    groupsize: int = -1,
    symmetric: bool = True,
    clip_ratio: float = 0.9,
    mode: str = "aware",
) -> AKVQuantizationSummary:
    """Install the controlled YAQA A/V hooks and post-RoPE K callable.

    The caller controls *when* the adapter is installed.  Installing it before
    Sketch-B Hessian collection makes the weight quantizer A/K/V-aware;
    reinstalling the same topology after loading the hfized QTIP model gives
    evaluation the identical deployed QDQ semantics.
    """

    for bits in (a_bits, k_bits, v_bits):
        _validate_quant_config(
            bits=bits,
            groupsize=groupsize,
            symmetric=symmetric,
            clip_ratio=clip_ratio,
        )
    requested = {
        "mode": mode,
        "a_bits": a_bits,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "groupsize": groupsize,
        "symmetric": symmetric,
        "clip_ratio": float(clip_ratio),
    }
    previous = getattr(model, _MODEL_MARKER, None)
    if previous is not None:
        previous_request = {
            key: previous[key] for key in requested
        }
        if previous_request != requested:
            raise RuntimeError(
                "YAQA A/K/V quantization was already installed differently: "
                f"{previous_request!r} != {requested!r}"
            )
        return AKVQuantizationSummary(**{
            key: value
            for key, value in previous.items()
            if key in AKVQuantizationSummary.__dataclass_fields__
        })

    layers = _decoder_layers(model)
    a_count = 0
    v_count = 0
    k_count = 0
    for layer in layers:
        attention = layer.self_attn
        for name, projection in _layer_projections(layer):
            if a_bits < 16:
                _install_input_hook(
                    projection,
                    bits=a_bits,
                    groupsize=groupsize,
                    symmetric=symmetric,
                    clip_ratio=clip_ratio,
                )
                a_count += 1
            if name == "v_proj" and v_bits < 16:
                _install_output_hook(
                    projection,
                    bits=v_bits,
                    groupsize=groupsize,
                    symmetric=symmetric,
                    clip_ratio=clip_ratio,
                )
                v_count += 1
        if k_bits < 16:
            if hasattr(attention, _K_MARKER):
                raise RuntimeError(
                    "YAQA post-RoPE K quantizer unexpectedly existed before "
                    "model-level installation"
                )
            setattr(
                attention,
                _K_MARKER,
                PostRoPEKQuantizer(
                    bits=k_bits,
                    groupsize=groupsize,
                    symmetric=symmetric,
                    clip_ratio=clip_ratio,
                ),
            )
            _install_post_rope_k_forward_wrapper(attention)
            k_count += 1

    summary = AKVQuantizationSummary(
        mode=mode,
        decoder_layers=len(layers),
        activation_input_sites=a_count,
        value_output_sites=v_count,
        post_rope_k_sites=k_count,
        a_bits=a_bits,
        k_bits=k_bits,
        v_bits=v_bits,
        groupsize=groupsize,
        symmetric=symmetric,
        clip_ratio=float(clip_ratio),
    )
    expected_a = 0 if a_bits >= 16 else 7 * len(layers)
    expected_v = 0 if v_bits >= 16 else len(layers)
    expected_k = 0 if k_bits >= 16 else len(layers)
    if (a_count, v_count, k_count) != (expected_a, expected_v, expected_k):
        raise RuntimeError(
            "YAQA A/V/K topology mismatch: "
            f"{(a_count, v_count, k_count)!r} != "
            f"{(expected_a, expected_v, expected_k)!r}"
        )
    stored = {**requested, **summary.as_dict()}
    setattr(model, _MODEL_MARKER, stored)
    return summary


def _forward_has_native_post_rope_k(attention: nn.Module) -> bool:
    """Whether the copied YAQA model already calls our K hook directly."""

    method = attention.forward
    function = getattr(method, "__func__", method)
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    globals_dict = getattr(function, "__globals__", {})
    return globals_dict.get("quantize_post_rope_key") is quantize_post_rope_key


def _install_post_rope_k_forward_wrapper(attention: nn.Module) -> None:
    """Patch stock HF attention after RoPE while preserving copied Llama."""

    if _forward_has_native_post_rope_k(attention):
        return
    if getattr(attention, _K_WRAPPER_MARKER, None) is not None:
        return

    def wrapper_factory(original_apply_rotary):
        @functools.wraps(original_apply_rotary)
        def wrapped_apply_rotary(*args, **kwargs):
            query_states, key_states = original_apply_rotary(*args, **kwargs)
            return query_states, quantize_post_rope_key(
                attention, key_states
            )

        return wrapped_apply_rotary

    try:
        wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
            attention,
            "forward",
            "apply_rotary_pos_emb",
            wrapper_factory,
        )
    except (AttributeError, KeyError) as exc:
        raise TypeError(
            "YAQA post-RoPE K-aware support requires attention.forward to "
            "call apply_rotary_pos_emb directly"
        ) from exc
    setattr(attention, _K_WRAPPER_MARKER, wrapper)


def quantize_post_rope_key(attention: nn.Module, key_states: torch.Tensor) -> torch.Tensor:
    """Apply an installed K quantizer; otherwise return K exactly unchanged."""

    quantizer = getattr(attention, _K_MARKER, None)
    if quantizer is None:
        return key_states
    if not isinstance(quantizer, PostRoPEKQuantizer):
        raise TypeError(
            f"{_K_MARKER} must be PostRoPEKQuantizer, got "
            f"{type(quantizer).__qualname__}"
        )
    return quantizer(key_states)


__all__ = [
    "AKVQuantizationSummary",
    "PostRoPEKQuantizer",
    "install_akv_quantization",
    "quantize_post_rope_key",
]
