"""Optional attention backends used by the RealQ pipeline.

The FlashAttention-4 integration is intentionally narrow: RealQ calibration
and block refresh use dense, unpadded, causal sequences and do not use a KV
cache.  Keeping that contract explicit avoids silently dropping a padding or
custom mask when Transformers dispatches to a custom attention backend.
"""
from __future__ import annotations

import logging
from importlib import metadata
from typing import Any

import torch


FA4_BACKEND = "flash_attention_4"
_PINNED_FA4_VERSION = "4.0.0b25"
_SUPPORTED_BACKENDS = ("sdpa", FA4_BACKEND)


def _load_fa4():
    """Import the pinned CuTe DSL implementation only when it is selected."""

    try:
        installed = metadata.version("flash-attn-4")
        if installed != _PINNED_FA4_VERSION:
            raise RuntimeError(
                "RealQ's FlashAttention-4 adapter is pinned to "
                f"flash-attn-4=={_PINNED_FA4_VERSION}, but found {installed}. "
                "Install the versions from requirements.txt."
            )
        from flash_attn.cute import flash_attn_func
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "attention_backend='flash_attention_4' requires the optional "
            "FA4 dependencies pinned in requirements.txt."
        ) from exc
    return flash_attn_func


def _reject_unsupported_model_inputs(
    _module: torch.nn.Module,
    _args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Fail before Transformers can discard masks for a custom backend."""

    if kwargs.get("attention_mask") is not None:
        raise ValueError(
            "RealQ's flash_attention_4 backend supports only dense, unpadded "
            "causal inputs; attention_mask must be None."
        )
    if kwargs.get("past_key_values") is not None:
        raise ValueError(
            "RealQ's flash_attention_4 backend is for calibration/refresh and "
            "does not support KV-cache decoding."
        )


def flash_attention_4_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Transformers attention adapter for FA4's CuTe DSL API.

    Transformers supplies Q/K/V as ``[batch, heads, sequence, head_dim]``;
    FA4 consumes ``[batch, sequence, heads, head_dim]`` and natively handles
    grouped-query attention, so K/V are never repeated to the Q-head count.
    """

    if attention_mask is not None:
        raise ValueError(
            "flash_attention_4 received an attention mask. Only dense, "
            "unpadded causal attention is supported by this RealQ adapter."
        )
    if (
        kwargs.get("output_attentions", False)
        or kwargs.get("head_mask") is not None
    ):
        raise ValueError("flash_attention_4 does not return attention weights.")
    if dropout != 0.0:
        raise ValueError(
            "flash_attention_4 in RealQ does not implement attention dropout; "
            "run the model in eval mode."
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("flash_attention_4 expects rank-4 Q/K/V tensors.")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("flash_attention_4 requires CUDA Q/K/V tensors.")
    if not (query.device == key.device == value.device):
        raise ValueError("flash_attention_4 requires Q/K/V on the same device.")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("flash_attention_4 requires matching Q/K/V dtypes.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "RealQ's flash_attention_4 backward path supports FP16/BF16 Q/K/V."
        )
    if query.shape[0] != key.shape[0] or key.shape != value.shape:
        raise ValueError(
            "flash_attention_4 requires matching Q/K/V batch and K/V shapes."
        )
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("flash_attention_4 requires matching Q/K head dimensions.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("Q head count must be divisible by KV head count for GQA.")

    capability = torch.cuda.get_device_capability(query.device)
    if capability not in ((9, 0), (10, 0), (12, 0)):
        raise RuntimeError(
            "RealQ's flash_attention_4 backend targets Hopper/Blackwell "
            f"(SM90/SM100/SM120); got compute capability {capability}."
        )

    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = bool(getattr(module, "is_causal", True))
    if not is_causal:
        raise ValueError("RealQ's flash_attention_4 backend requires causal attention.")

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    window_size = (
        (int(sliding_window) - 1, 0)
        if sliding_window is not None and key.shape[-2] > sliding_window
        else (None, None)
    )

    flash_attn_func = _load_fa4()
    out, _lse = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=scaling,
        causal=True,
        window_size=window_size,
        softcap=0.0 if softcap is None else float(softcap),
        pack_gqa=query.shape[1] != key.shape[1],
        deterministic=torch.are_deterministic_algorithms_enabled(),
        return_lse=False,
    )
    return out, None


def configure_attention_backend(model: torch.nn.Module, backend: str) -> None:
    """Configure a loaded Transformers model for the selected backend."""

    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"attention_backend must be one of {_SUPPORTED_BACKENDS}; got {backend!r}."
        )
    if backend == "sdpa":
        model.config._attn_implementation = "sdpa"
        return

    # Validate the dependency before changing model state. AttentionInterface
    # is process-global, while the config key is local to this model.
    _load_fa4()
    from transformers import AttentionInterface

    AttentionInterface.register(FA4_BACKEND, flash_attention_4_forward)
    model.config._attn_implementation = FA4_BACKEND

    if not hasattr(model, "_realq_fa4_input_guard_handle"):
        model._realq_fa4_input_guard_handle = model.register_forward_pre_hook(
            _reject_unsupported_model_inputs,
            with_kwargs=True,
        )
    logging.info(
        "[realq.attention] backend=%s version=%s dense_causal_only=true",
        FA4_BACKEND,
        _PINNED_FA4_VERSION,
    )
