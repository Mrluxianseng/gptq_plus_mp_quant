"""Per-rank disk cache for static precompute results.

Sweeps over learning rate / refresh loss schedule reuse the same precompute,
so the wall-time cost (a full end-to-end backward over ``nsamples`` samples)
should be paid once per model+dataset+seed combination.

Cache layout:

    <static_cache_path>/
        <key>_world{W}_rank{R}.pt

Each file holds the dense/routed saliency, Fisher matrices, teacher routes,
and globally reduced per-expert coverage produced by ``static_e2e.run``.
See that module for the exact tensor layouts.

The cache key is built from every config field that could change the
numerical contents of the cache. Anything not in the key is assumed to leave
the cache invariant; if you add a precompute knob that changes the maths,
add it to ``build_cache_key`` too.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import Any

import torch

from realq_moe.precompute.routed_stats import PackedLayerRoutes
from utils.cache_identity import artifact_identity


_CACHE_SCHEMA_VERSION = 8


def _valid_routed_payload_structure(payload: dict[str, Any]) -> bool:
    """Cheap cache-hit validation without scanning large tensor contents."""

    list_fields = (
        "saliency",
        "fisher",
        "routes",
        "expert_global_assignment_counts",
        "expert_global_coverage",
    )
    if any(not isinstance(payload[name], list) for name in list_fields):
        return False
    num_layers = len(payload["routes"])
    if any(len(payload[name]) != num_layers for name in list_fields):
        return False
    coverage_fields = {
        "assignment_count",
        "unique_token_count",
        "unique_sample_count",
        "affinity_mass",
    }
    for layer_idx in range(num_layers):
        layer_saliency = payload["saliency"][layer_idx]
        layer_fisher = payload["fisher"][layer_idx]
        layer_routes = payload["routes"][layer_idx]
        global_counts = payload["expert_global_assignment_counts"][layer_idx]
        coverage = payload["expert_global_coverage"][layer_idx]
        if (
            not isinstance(layer_saliency, dict)
            or any(
                not isinstance(name, str) or not torch.is_tensor(tensor)
                for name, tensor in layer_saliency.items()
            )
            or not torch.is_tensor(layer_fisher)
            or not torch.is_tensor(global_counts)
            or global_counts.dtype != torch.int64
            or global_counts.dim() != 1
        ):
            return False
        if isinstance(layer_routes, dict) and not layer_routes:
            if (
                int(global_counts.numel()) != 0
                or not isinstance(coverage, dict)
                or coverage
            ):
                return False
            continue
        if not isinstance(layer_routes, PackedLayerRoutes):
            return False
        try:
            layer_routes.validate()
        except (TypeError, ValueError, RuntimeError):
            return False
        num_experts = len(layer_routes)
        if num_experts == 0:
            return False
        if (
            not isinstance(coverage, dict)
            or set(coverage) != coverage_fields
        ):
            return False
        if (
            global_counts.dim() != 1
            or int(global_counts.numel()) != num_experts
            or global_counts.device != layer_routes.device
        ):
            return False
        for field in coverage_fields:
            tensor = coverage[field]
            expected_dtype = (
                torch.float64
                if field == "affinity_mass"
                else torch.int64
            )
            if (
                not torch.is_tensor(tensor)
                or tensor.dim() != 1
                or int(tensor.numel()) != num_experts
                or tensor.dtype != expected_dtype
                or tensor.device != layer_routes.device
            ):
                return False
        if not torch.equal(coverage["assignment_count"], global_counts):
            return False
    return True


def build_cache_key(cfg, world_size: int) -> str:
    """Stable string key for the precompute output.

    Inputs that DO change the cache contents (and so are part of the key):
        exact model artifact, dataset, nsamples, seq_len, rotation artifact,
        num_groups, saliency_clip_percentile, grad_hessian_topk,
        global_loss_bsz, calibration seed, generated-rotation seed,
        world_size.

    Excluded on purpose: anything that only affects the quantisation phase
    (grad_lr, blocksize, w_bits, A/V/K aware settings, ...).  Stage 0 is an
    FP teacher pass, so aware settings cannot change its output.
    """
    model_identity = artifact_identity(cfg.model)
    if cfg.rotate:
        rotation_identity = (
            artifact_identity(cfg.optimized_rotation_path)
            if cfg.optimized_rotation_path is not None
            else (
                "generated_hadamard:"
                f"rotation_seed={int(getattr(cfg, 'rotation_seed', 0))}"
            )
        )
    else:
        rotation_identity = "disabled"
    parts = [
        f"schema={_CACHE_SCHEMA_VERSION}",
        f"model={cfg.model_name}",
        f"modelid={model_identity}",
        f"ds={cfg.dataset}",
        f"n={cfg.nsamples}",
        f"sl={cfg.seq_len}",
        f"rot={int(bool(cfg.rotate))}",
        f"rotid={rotation_identity}",
        f"ng={cfg.num_groups}",
        f"salclip={cfg.saliency_clip_percentile:g}",
        f"topk={cfg.grad_hessian_topk}",
        f"glbsz={cfg.global_loss_bsz}",
        f"calibration_seed={cfg.seed}",
        f"world={world_size}",
    ]
    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"{cfg.model_name}_{cfg.dataset}_n{cfg.nsamples}_sl{cfg.seq_len}_{digest}"


def cache_path(cache_dir: str, key: str, world_size: int, rank: int) -> str:
    return os.path.join(cache_dir, f"{key}_world{world_size}_rank{rank}.pt")


def try_load(
    cache_dir: str,
    key: str,
    world_size: int,
    rank: int,
    *,
    map_location: torch.device | str | None = None,
) -> dict | None:
    path = cache_path(cache_dir, key, world_size, rank)
    if not os.path.isfile(path):
        return None
    try:
        payload = torch.load(
            path,
            weights_only=False,
            map_location=map_location,
        )
    except Exception as exc:
        # A cache read failure must be treated as a local miss.  In
        # distributed precompute every rank subsequently participates in a
        # hit-consensus collective; raising here on just one rank would strand
        # the other ranks in that collective.
        logging.warning(
            "[realq.precompute] ignoring unreadable cache file %s: %s",
            path,
            exc,
        )
        return None
    required = {
        "saliency",
        "fisher",
        "routes",
        "expert_global_assignment_counts",
        "expert_global_coverage",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        logging.warning(
            "[realq.precompute] ignoring invalid cache payload at %s "
            "(expected routed schema fields %s)",
            path,
            sorted(required),
        )
        return None
    if not _valid_routed_payload_structure(payload):
        logging.warning(
            "[realq.precompute] ignoring structurally invalid routed cache "
            "payload at %s",
            path,
        )
        return None
    return payload


def save(cache_dir: str, key: str, world_size: int, rank: int, payload: dict[str, Any]) -> str:
    """Atomically replace one rank's cache file.

    The temporary file is created in ``cache_dir`` so ``os.replace`` stays on
    the same filesystem and is atomic.  A unique temporary name also makes
    concurrent runs targeting the same cache key safe: readers see either the
    previous complete file or one complete new file, never a partially-written
    torch archive.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, key, world_size, rank)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_dir,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        # os.replace removes tmp_path on success.  On serialization/replace
        # failure, clean up only this invocation's unique temporary file.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
    return path
