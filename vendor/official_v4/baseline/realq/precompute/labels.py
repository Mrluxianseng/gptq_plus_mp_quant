"""Deterministic Categorical sampling for unbiased Fisher estimation.

Each sample's labels depend ONLY on its global sample index — not on batch
shape or DP world size. This guarantees that the saliency / Fisher cache
produced under different DP topologies (1 GPU vs 4 GPU) describes the same
statistics over the same calibration data.
"""
from __future__ import annotations

from typing import Sequence

import torch


def deterministic_categorical_labels(
    logits: torch.Tensor,
    global_sample_indices: Sequence[int],
    base_seed: int = 0,
) -> torch.Tensor:
    """Per-sample `Categorical(logits).sample()` with a deterministic seed.

    logits: (B, ..., V)
    global_sample_indices: length B; each entry is the calibration sample's
        global id.
    base_seed: namespacing tag, lets different precompute call sites avoid
        seed collisions.

    Returns labels of shape `logits.shape[:-1]`, dtype long.
    """
    bsz = logits.shape[0]
    if len(global_sample_indices) != bsz:
        raise ValueError(
            f"deterministic_categorical_labels: expected {bsz} global indices, "
            f"got {len(global_sample_indices)}"
        )
    trailing_shape = logits.shape[1:-1]
    vocab = logits.shape[-1]
    out = torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)
    for i in range(bsz):
        # Deliberately collision-resistant: the multiplier is prime and the
        # additive constant is non-zero so seed collisions across (base_seed,
        # global_idx) pairs only happen at >2^31 sample ids.
        seed = int(1000003 * (base_seed * 100000 + int(global_sample_indices[i])) + 17)
        gen = torch.Generator(device=logits.device).manual_seed(seed)
        probs = torch.softmax(logits[i].reshape(-1, vocab).float(), dim=-1)
        sample = torch.multinomial(probs, 1, generator=gen).reshape(trailing_shape)
        out[i] = sample
    return out
