"""Memory-only execution helpers for deterministic Qwen3-32B refreshes.

The frozen V4 runs keep ``backward_bsz=32`` and force CUDA SDPA's math
backend so attention backward is deterministic.  For Qwen3-32B, one
un-tiled ``32 x 2048`` math-SDPA call materialises a 32 GiB attention
workspace.  A sliding refresh needs two Transformer blocks in one graph and
therefore does not fit on one 192-GB L20C.

This module changes neither the logical sample batch nor the loss/optimizer
cadence:

* math SDPA is tiled only along its independent batch axis and concatenated;
* Transformer-block functional forwards are non-reentrant-checkpointed so the
  two sliding blocks are recomputed, rather than retained simultaneously;
* the legacy single-linear sliding arm receives the same checkpoint wrapper.

The helpers are installed only by the dedicated Q32 entry point.  The frozen
V4 source files and their protocol fingerprint remain untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


DEFAULT_SDPA_BATCH_TILE = 4

_ORIGINAL_SDPA = F.scaled_dot_product_attention
_INSTALLED = False


def _slice_batch_operand(
    tensor: torch.Tensor,
    *,
    start: int,
    stop: int,
    full_batch: int,
) -> torch.Tensor:
    """Slice a batch-shaped SDPA operand while retaining broadcast operands."""

    if tensor.ndim > 0 and tensor.shape[0] == full_batch:
        return tensor[start:stop]
    return tensor


def tiled_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
    batch_tile: int = DEFAULT_SDPA_BATCH_TILE,
    _force_tiling: bool = False,
) -> torch.Tensor:
    """Run SDPA in independent batch tiles without changing its logical batch.

    Tiling is limited to CUDA grad-enabled calls: those are the only calls
    that hit the deterministic math-SDPA 32-GiB workspace.  Calibration,
    Hessian accumulation, reference forwards, and evaluation retain their
    frozen execution path.
    """

    batch = int(query.shape[0]) if query.ndim else 0
    if (
        batch_tile <= 0
        or batch <= batch_tile
        or (
            not _force_tiling
            and (not query.is_cuda or not torch.is_grad_enabled())
        )
    ):
        return _ORIGINAL_SDPA(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    outputs: list[torch.Tensor] = []
    for start in range(0, batch, batch_tile):
        stop = min(start + batch_tile, batch)
        key_tile = _slice_batch_operand(
            key, start=start, stop=stop, full_batch=batch
        )
        value_tile = _slice_batch_operand(
            value, start=start, stop=stop, full_batch=batch
        )
        mask_tile = (
            _slice_batch_operand(
                attn_mask, start=start, stop=stop, full_batch=batch
            )
            if attn_mask is not None
            else None
        )
        outputs.append(
            _ORIGINAL_SDPA(
                query[start:stop],
                key_tile,
                value_tile,
                attn_mask=mask_tile,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        )
    return torch.cat(outputs, dim=0)


def checkpointed_functional_call(
    original_functional_call: Any,
    module: nn.Module,
    parameter_and_buffer_dicts: (
        Mapping[str, torch.Tensor]
        | Sequence[Mapping[str, torch.Tensor]]
    ),
    args: Any = None,
    kwargs: dict[str, Any] | None = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
) -> Any:
    """Checkpoint a grad-enabled ``torch.func.functional_call``.

    Override tensors are explicit checkpoint inputs, so gradients continue to
    target the exact leaves created by ``BlockRefreshState.make_overrides``.
    """

    if args is None:
        positional: tuple[Any, ...] = ()
    elif isinstance(args, tuple):
        positional = args
    else:
        positional = (args,)
    call_kwargs = {} if kwargs is None else kwargs

    if not torch.is_grad_enabled():
        return original_functional_call(
            module,
            parameter_and_buffer_dicts,
            positional,
            call_kwargs,
            tie_weights=tie_weights,
            strict=strict,
        )
    if not all(isinstance(item, torch.Tensor) for item in positional):
        raise TypeError(
            "checkpointed functional_call requires tensor positional inputs"
        )

    is_sequence = not isinstance(parameter_and_buffer_dicts, Mapping)
    dictionaries = (
        tuple(parameter_and_buffer_dicts)
        if is_sequence
        else (parameter_and_buffer_dicts,)
    )
    keys_by_dict = tuple(tuple(mapping.keys()) for mapping in dictionaries)
    flat_overrides = tuple(
        value for mapping in dictionaries for value in mapping.values()
    )
    positional_count = len(positional)

    def run(*flat_inputs: torch.Tensor) -> Any:
        call_args = flat_inputs[:positional_count]
        flat_values = iter(flat_inputs[positional_count:])
        rebuilt = tuple(
            {key: next(flat_values) for key in keys}
            for keys in keys_by_dict
        )
        call_state: Any = rebuilt if is_sequence else rebuilt[0]
        return original_functional_call(
            module,
            call_state,
            call_args,
            call_kwargs,
            tie_weights=tie_weights,
            strict=strict,
        )

    return checkpoint(
        run,
        *positional,
        *flat_overrides,
        use_reentrant=False,
        preserve_rng_state=True,
    )


class _CheckpointedModule(nn.Module):
    """Checkpoint an ordinary module call while preserving its public output."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> Any:
        if not torch.is_grad_enabled():
            return self.module(*args, **kwargs)

        def run(*inputs: torch.Tensor) -> Any:
            return self.module(*inputs, **kwargs)

        return checkpoint(
            run,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
        )


def install() -> None:
    """Install the Q32 deterministic-memory wrappers once per process."""

    global _INSTALLED
    if _INSTALLED:
        return

    from realq.refresh import block_gd, kl_loss
    from realq.runner import layer_loop

    original_block_functional_call = block_gd.functional_call
    original_kl_functional_call = kl_loss.functional_call
    original_make_grad_refresh_fn = layer_loop.make_grad_refresh_fn

    def block_functional_call(
        module: nn.Module,
        parameter_and_buffer_dicts: Any,
        args: Any = None,
        kwargs: dict[str, Any] | None = None,
        *,
        tie_weights: bool = True,
        strict: bool = False,
    ) -> Any:
        return checkpointed_functional_call(
            original_block_functional_call,
            module,
            parameter_and_buffer_dicts,
            args,
            kwargs,
            tie_weights=tie_weights,
            strict=strict,
        )

    def kl_functional_call(
        module: nn.Module,
        parameter_and_buffer_dicts: Any,
        args: Any = None,
        kwargs: dict[str, Any] | None = None,
        *,
        tie_weights: bool = True,
        strict: bool = False,
    ) -> Any:
        return checkpointed_functional_call(
            original_kl_functional_call,
            module,
            parameter_and_buffer_dicts,
            args,
            kwargs,
            tie_weights=tie_weights,
            strict=strict,
        )

    def make_grad_refresh_fn(**kwargs: Any) -> Any:
        # Full-block sliding uses functional_call and is already covered above.
        # The legacy single-linear path calls next_layer directly, so wrap just
        # that arm without changing its module or parameters.
        if kwargs.get("block_state") is None and kwargs.get("next_layer") is not None:
            kwargs = dict(kwargs)
            kwargs["next_layer"] = _CheckpointedModule(kwargs["next_layer"])
        return original_make_grad_refresh_fn(**kwargs)

    block_gd.functional_call = block_functional_call
    kl_loss.functional_call = kl_functional_call
    layer_loop.make_grad_refresh_fn = make_grad_refresh_fn
    F.scaled_dot_product_attention = tiled_scaled_dot_product_attention
    _INSTALLED = True
    logging.info(
        "[realq.q32_memory] installed checkpointed block forwards and "
        "deterministic math-SDPA batch tiling; logical backward_bsz unchanged, "
        "sdpa_batch_tile=%d",
        DEFAULT_SDPA_BATCH_TILE,
    )


def is_installed() -> bool:
    return _INSTALLED
