"""Llama-3.2 / Transformers 4.56 compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


_BLOCK_CONTEXT_KEYS = (
    "attention_mask",
    "position_ids",
    "cache_position",
    "position_embeddings",
)


def ensure_fast_eos_tokenizer(
    tokenizer: Any,
    *,
    expected_vocab_size: int | None = None,
) -> Any:
    """Require a fast tokenizer and assign EOS as PAD without adding a token."""

    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError(
            f"EfficientQAT comparison requires a fast tokenizer; "
            f"got {type(tokenizer).__qualname__}."
        )
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token is None or eos_token_id is None:
        raise ValueError("Tokenizer must define both eos_token and eos_token_id.")

    vocab_before = len(tokenizer)
    if expected_vocab_size is not None and vocab_before != expected_vocab_size:
        raise ValueError(
            "Tokenizer vocabulary does not match the frozen model vocabulary: "
            f"{vocab_before} != {expected_vocab_size}."
        )

    tokenizer.pad_token = eos_token

    vocab_after = len(tokenizer)
    if vocab_after != vocab_before:
        raise RuntimeError(
            "Assigning EOS as PAD changed tokenizer vocabulary size "
            f"from {vocab_before} to {vocab_after}; vocabulary resize is forbidden."
        )
    if tokenizer.pad_token_id != eos_token_id:
        raise RuntimeError(
            "Tokenizer PAD/EOS ids differ after configuration: "
            f"{tokenizer.pad_token_id} != {eos_token_id}."
        )
    return tokenizer


def load_fast_eos_tokenizer(
    model_path: str | Path,
    *,
    expected_vocab_size: int | None = None,
    trust_remote_code: bool = False,
):
    """Load the frozen model tokenizer without invoking a slow-tokenizer path."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    return ensure_fast_eos_tokenizer(
        tokenizer,
        expected_vocab_size=expected_vocab_size,
    )


def assert_model_vocab_unchanged(
    model: Any,
    tokenizer: Any,
    *,
    expected_vocab_size: int,
    expected_tokenizer_vocab_size: int | None = None,
) -> None:
    """Fail closed if any caller resized embeddings or the LM head."""

    if expected_tokenizer_vocab_size is None:
        expected_tokenizer_vocab_size = expected_vocab_size
    tokenizer_size = len(tokenizer)
    if tokenizer_size != expected_tokenizer_vocab_size:
        raise RuntimeError(
            "Tokenizer vocabulary changed during EfficientQAT setup: "
            f"{tokenizer_size} != {expected_tokenizer_vocab_size}."
        )

    config_size = int(getattr(model.config, "vocab_size"))
    if config_size != expected_vocab_size:
        raise RuntimeError(
            "Model config vocabulary changed during EfficientQAT setup: "
            f"{config_size} != {expected_vocab_size}."
        )

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise RuntimeError("Model does not expose input embeddings.")
    input_size = int(input_embeddings.weight.shape[0])
    if input_size != expected_vocab_size:
        raise RuntimeError(
            "Input embedding vocabulary changed during EfficientQAT setup: "
            f"{input_size} != {expected_vocab_size}."
        )

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Model does not expose output embeddings.")
    output_size = int(output_embeddings.weight.shape[0])
    if output_size != expected_vocab_size:
        raise RuntimeError(
            "LM-head vocabulary changed during EfficientQAT setup: "
            f"{output_size} != {expected_vocab_size}."
        )
    if bool(getattr(model.config, "tie_word_embeddings", False)):
        if (
            input_embeddings.weight.data_ptr()
            != output_embeddings.weight.data_ptr()
        ):
            raise RuntimeError(
                "The checkpoint declares tied word embeddings, but the "
                "reloaded LM head no longer shares embedding storage."
            )

    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise RuntimeError(
            "Tokenizer PAD must remain aliased to EOS for the controlled run."
        )
    model.config.pad_token_id = tokenizer.eos_token_id


def _detach_tree(value: Any, *, to_cpu: bool) -> Any:
    if isinstance(value, torch.Tensor):
        result = value.detach()
        return result.cpu() if to_cpu else result
    if isinstance(value, tuple):
        return tuple(_detach_tree(item, to_cpu=to_cpu) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item, to_cpu=to_cpu) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _detach_tree(item, to_cpu=to_cpu)
            for key, item in value.items()
        }
    return value


def _move_tree(value: Any, *, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device=device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device=device) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _move_tree(item, device=device)
            for key, item in value.items()
        }
    return value


def _align_attention_mask_batch(
    attention_mask: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Expand only an explicitly singleton mask batch dimension."""

    if attention_mask.ndim == 0:
        return attention_mask
    mask_batch = int(attention_mask.shape[0])
    if mask_batch == batch_size:
        return attention_mask
    if mask_batch == 1:
        return attention_mask.expand(batch_size, *attention_mask.shape[1:])
    raise ValueError(
        "Captured attention-mask batch dimension is incompatible with the "
        f"current block batch: {mask_batch} not in (1, {batch_size})."
    )


@dataclass(frozen=True)
class BlockForwardContext:
    """Captured kwargs needed for a standalone Transformers 4.56 decoder block."""

    values: dict[str, Any]

    @classmethod
    def capture(
        cls,
        kwargs: Mapping[str, Any],
        *,
        to_cpu: bool = False,
        require_position_embeddings: bool = True,
    ) -> "BlockForwardContext":
        values = {
            key: _detach_tree(kwargs[key], to_cpu=to_cpu)
            for key in _BLOCK_CONTEXT_KEYS
            if key in kwargs and kwargs[key] is not None
        }
        if require_position_embeddings and "position_embeddings" not in values:
            raise KeyError(
                "First-block capture did not receive `position_embeddings`; "
                "Transformers 4.56 Llama decoder blocks cannot be replayed exactly."
            )
        # Block-AP explicitly disables the model cache.  Passing this value
        # prevents a block default from allocating or mutating a cache.
        values["use_cache"] = False
        return cls(values=values)

    def for_hidden_states(self, hidden_states: torch.Tensor) -> dict[str, Any]:
        if hidden_states.ndim < 1:
            raise ValueError(
                "Decoder hidden_states must expose a batch dimension; "
                f"got shape {tuple(hidden_states.shape)}."
            )
        result = _move_tree(self.values, device=hidden_states.device)
        attention_mask = result.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            result["attention_mask"] = _align_attention_mask_batch(
                attention_mask,
                batch_size=int(hidden_states.shape[0]),
            )
        return result


def extract_hidden_states(output: Any) -> torch.Tensor:
    """Normalize modern Tensor and legacy tuple/model-output block returns."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        if output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(
            "Decoder block returned a tuple/list without a leading Tensor."
        )
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state
    raise TypeError(
        "Unsupported decoder-block output type: "
        f"{type(output).__qualname__}."
    )


def call_decoder_block(
    block: Any,
    hidden_states: torch.Tensor,
    context: BlockForwardContext,
) -> torch.Tensor:
    """Replay one decoder block and return its full hidden-state tensor."""

    output = block(
        hidden_states,
        **context.for_hidden_states(hidden_states),
    )
    result = extract_hidden_states(output)
    if result.ndim != hidden_states.ndim or result.shape[:2] != hidden_states.shape[:2]:
        raise RuntimeError(
            "Decoder block changed batch/sequence dimensions unexpectedly: "
            f"{tuple(hidden_states.shape)} -> {tuple(result.shape)}."
        )
    return result


__all__ = [
    "BlockForwardContext",
    "assert_model_vocab_unchanged",
    "call_decoder_block",
    "ensure_fast_eos_tokenizer",
    "extract_hidden_states",
    "load_fast_eos_tokenizer",
]
