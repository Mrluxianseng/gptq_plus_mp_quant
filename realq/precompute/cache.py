"""Per-rank disk cache for static precompute results.

Sweeps over learning rate / refresh loss schedule reuse the same precompute,
so the wall-time cost (a full end-to-end backward over ``nsamples`` samples)
should be paid once per model+dataset+seed combination.

Cache layout:

    <static_cache_path>/
        <key>_world{W}_rank{R}.pt

Each file holds a dict ``{"saliency": [...], "fisher": [...]}`` produced by
``static_e2e.run`` — see that module for the exact tensor layouts.

The cache key is built from every config field that could change the
numerical contents of the cache. Anything not in the key is assumed to leave
the cache invariant; if you add a precompute knob that changes the maths,
add it to ``build_cache_key`` too.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import torch


def build_cache_key(cfg, world_size: int) -> str:
    """Stable string key for the precompute output.

    Inputs that DO change the cache contents (and so are part of the key):
        model_name, dataset, nsamples, seq_len, rotate, num_groups,
        saliency_clip_percentile, grad_hessian_topk, global_loss_bsz,
        seed, world_size.

    Excluded on purpose: anything that only affects the quantisation phase
    (grad_lr, blocksize, w_bits, ...).
    """
    parts = [
        f"model={cfg.model_name}",
        f"ds={cfg.dataset}",
        f"n={cfg.nsamples}",
        f"sl={cfg.seq_len}",
        f"rot={int(bool(cfg.rotate))}",
        f"ng={cfg.num_groups}",
        f"salclip={cfg.saliency_clip_percentile:g}",
        f"topk={cfg.grad_hessian_topk}",
        f"glbsz={cfg.global_loss_bsz}",
        f"seed={cfg.seed}",
        f"world={world_size}",
    ]
    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"{cfg.model_name}_{cfg.dataset}_n{cfg.nsamples}_sl{cfg.seq_len}_{digest}"


def cache_path(cache_dir: str, key: str, world_size: int, rank: int) -> str:
    return os.path.join(cache_dir, f"{key}_world{world_size}_rank{rank}.pt")


def try_load(cache_dir: str, key: str, world_size: int, rank: int) -> dict | None:
    path = cache_path(cache_dir, key, world_size, rank)
    if not os.path.isfile(path):
        return None
    return torch.load(path, weights_only=False)


def save(cache_dir: str, key: str, world_size: int, rank: int, payload: dict[str, Any]) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, key, world_size, rank)
    torch.save(payload, path)
    return path
