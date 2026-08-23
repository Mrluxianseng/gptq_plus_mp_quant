"""Exact calibration-data adapters for the controlled EfficientQAT runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset


EXPECTED_SAMPLES = 2048
EXPECTED_SEQ_LEN = 2048


def validate_exact_tokens(
    value: Any,
    *,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_seq_len: int = EXPECTED_SEQ_LEN,
) -> list[torch.Tensor]:
    """Validate the byte-bearing token-cache contract without normalizing it.

    The comparison protocol deliberately requires the serialized artifact to
    be a Python ``list`` of independent one-dimensional ``torch.int64``
    tensors.  Accepting a stacked tensor, silently casting an integer dtype, or
    truncating/padding a sample would make an invalid cache look legitimate.
    """

    if isinstance(expected_samples, bool) or expected_samples <= 0:
        raise ValueError(
            f"expected_samples must be a positive integer; got {expected_samples!r}."
        )
    if isinstance(expected_seq_len, bool) or expected_seq_len <= 0:
        raise ValueError(
            f"expected_seq_len must be a positive integer; got {expected_seq_len!r}."
        )
    if type(value) is not list:
        raise TypeError(
            "Exact token cache must be a Python list, "
            f"not {type(value).__qualname__}."
        )
    if len(value) != expected_samples:
        raise ValueError(
            "Exact token cache has the wrong sample count: "
            f"{len(value)} != {expected_samples}."
        )

    for index, sample in enumerate(value):
        if not isinstance(sample, torch.Tensor):
            raise TypeError(
                f"Token sample {index} must be a torch.Tensor; "
                f"got {type(sample).__qualname__}."
            )
        if sample.dtype != torch.int64:
            raise TypeError(
                f"Token sample {index} must have dtype torch.int64; "
                f"got {sample.dtype}."
            )
        if sample.ndim != 1 or tuple(sample.shape) != (expected_seq_len,):
            raise ValueError(
                f"Token sample {index} has shape {tuple(sample.shape)}; "
                f"expected ({expected_seq_len},)."
            )
        if sample.device.type != "cpu":
            raise ValueError(
                f"Token sample {index} is on {sample.device}; "
                "formal token caches must be CPU artifacts."
            )

    return value


def load_exact_token_cache(
    path: str | Path,
    *,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_seq_len: int = EXPECTED_SEQ_LEN,
) -> list[torch.Tensor]:
    """Load and strictly validate an existing token cache.

    This function never creates or resamples data.  ``weights_only=True`` is
    used because the formal cache contains only a list of tensors.
    """

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Exact token cache does not exist: {cache_path}")
    try:
        value = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions that predate the keyword.
        value = torch.load(cache_path, map_location="cpu")
    return validate_exact_tokens(
        value,
        expected_samples=expected_samples,
        expected_seq_len=expected_seq_len,
    )


class BlockAPTokenDataset(Sequence[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy list-compatible view expected by upstream Block-AP.

    Each item mirrors ``EfficientQAT.datautils_block.get_wikitext2``:
    ``input_ids`` and labels both have shape ``[1, sequence_length]`` and every
    label except the final token is masked with ``-100``.  Block-AP itself only
    consumes item ``[0]``, but preserving the original pair prevents accidental
    incompatibility with diagnostic callers.
    """

    def __init__(self, tokens: list[torch.Tensor]) -> None:
        if not tokens:
            raise ValueError("Block-AP tokens must not be empty.")
        seq_len = int(tokens[0].numel())
        self._tokens = validate_exact_tokens(
            tokens,
            expected_samples=len(tokens),
            expected_seq_len=seq_len,
        )

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(
        self, index: int | slice
    ) -> tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        sample = self._tokens[index]
        input_ids = sample.unsqueeze(0)
        labels = input_ids.clone()
        labels[:, :-1] = -100
        return input_ids, labels


def build_block_ap_trainloader(
    tokens: list[torch.Tensor],
) -> BlockAPTokenDataset:
    """Return the list-compatible exact-token view consumed by Block-AP."""

    return BlockAPTokenDataset(tokens)


def build_e2e_fixed_dataset(tokens: list[torch.Tensor]):
    """Build a fixed-length Hugging Face Dataset from the exact token list.

    No tokenizer is called, no samples are shuffled, and no padding token is
    inserted.  The returned dataset emits torch tensors and contains only
    ``input_ids`` and causal-LM ``labels``; all rows already have identical
    length, so the ordinary default data collator can stack them directly.
    """

    if not tokens:
        raise ValueError("E2E-QP tokens must not be empty.")
    seq_len = int(tokens[0].numel())
    checked = validate_exact_tokens(
        tokens,
        expected_samples=len(tokens),
        expected_seq_len=seq_len,
    )

    try:
        from datasets import Dataset, Features, Sequence as HFSequence, Value
    except ImportError as exc:
        raise RuntimeError(
            "build_e2e_fixed_dataset requires the Hugging Face `datasets` package."
        ) from exc

    # A list-of-lists is intentionally used here.  It maps unambiguously to a
    # fixed-size Arrow list on every supported datasets/pyarrow release,
    # avoiding version-dependent inference from a rank-2 torch tensor.
    rows = [sample.tolist() for sample in checked]
    features = Features(
        {
            "input_ids": HFSequence(Value("int64"), length=seq_len),
            "labels": HFSequence(Value("int64"), length=seq_len),
        }
    )
    dataset = Dataset.from_dict(
        {"input_ids": rows, "labels": rows},
        features=features,
    )
    return dataset.with_format(
        "torch",
        columns=["input_ids", "labels"],
        output_all_columns=False,
    )


__all__ = [
    "BlockAPTokenDataset",
    "EXPECTED_SAMPLES",
    "EXPECTED_SEQ_LEN",
    "build_block_ap_trainloader",
    "build_e2e_fixed_dataset",
    "load_exact_token_cache",
    "validate_exact_tokens",
]
