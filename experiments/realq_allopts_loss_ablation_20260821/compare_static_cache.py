#!/usr/bin/env python3
"""Compare deterministic samples from two REAL-Q static-cache artifacts.

The checkpoints are memory mapped and only small, fixed slices from every
layer are materialized.  This makes the diagnostic safe to run while the main
GPU campaigns are active, while still distinguishing archive-only changes
from numerical Fisher/saliency changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


class CacheComparisonError(RuntimeError):
    pass


def _anchors(size: int, width: int) -> tuple[int, ...]:
    width = min(width, size)
    return tuple(sorted({0, max((size - width) // 2, 0), max(size - width, 0)}))


def _sample_fisher(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise CacheComparisonError(f"unexpected Fisher shape: {tuple(tensor.shape)}")
    width = min(16, tensor.shape[0])
    starts = _anchors(tensor.shape[0], width)
    chunks = [
        tensor[row : row + width, column : column + width].reshape(-1)
        for row in starts
        for column in starts
    ]
    diagonal_indices = torch.linspace(
        0, tensor.shape[0] - 1, steps=min(128, tensor.shape[0]), dtype=torch.long
    )
    chunks.append(tensor[diagonal_indices, diagonal_indices])
    return torch.cat(chunks).float()


def _sample_saliency(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise CacheComparisonError(
            f"unexpected saliency shape: {tuple(tensor.shape)}"
        )
    row_width = min(2, tensor.shape[0])
    token_width = min(16, tensor.shape[1])
    chunks = [
        tensor[
            row : row + row_width,
            token : token + token_width,
            :,
        ].reshape(-1)
        for row in _anchors(tensor.shape[0], row_width)
        for token in _anchors(tensor.shape[1], token_width)
    ]
    return torch.cat(chunks).float()


def _new_stats() -> dict[str, Any]:
    return {
        "sampled_values": 0,
        "unequal_values": 0,
        "abs_sum": 0.0,
        "square_sum": 0.0,
        "left_abs_sum": 0.0,
        "right_abs_sum": 0.0,
        "left_square_sum": 0.0,
        "right_square_sum": 0.0,
        "cross_sum": 0.0,
        "max_abs": 0.0,
        "max_abs_location": None,
    }


def _update(
    stats: dict[str, Any], left: torch.Tensor, right: torch.Tensor, location: str
) -> None:
    if left.shape != right.shape:
        raise CacheComparisonError(
            f"sample shape mismatch at {location}: {left.shape} != {right.shape}"
        )
    left_double = left.double()
    right_double = right.double()
    difference = left_double - right_double
    absolute = difference.abs()
    stats["sampled_values"] += difference.numel()
    stats["unequal_values"] += int(torch.count_nonzero(difference).item())
    stats["abs_sum"] += float(absolute.sum().item())
    stats["square_sum"] += float(difference.square().sum().item())
    stats["left_abs_sum"] += float(left_double.abs().sum().item())
    stats["right_abs_sum"] += float(right_double.abs().sum().item())
    stats["left_square_sum"] += float(left_double.square().sum().item())
    stats["right_square_sum"] += float(right_double.square().sum().item())
    stats["cross_sum"] += float((left_double * right_double).sum().item())
    maximum = float(absolute.max().item()) if absolute.numel() else 0.0
    if maximum > stats["max_abs"]:
        stats["max_abs"] = maximum
        stats["max_abs_location"] = location


def _finish(stats: dict[str, Any]) -> dict[str, Any]:
    count = stats["sampled_values"]
    left_rms = (stats["left_square_sum"] / count) ** 0.5 if count else 0.0
    right_rms = (stats["right_square_sum"] / count) ** 0.5 if count else 0.0
    difference_rms = (stats["square_sum"] / count) ** 0.5 if count else 0.0
    cosine_denominator = (
        stats["left_square_sum"] * stats["right_square_sum"]
    ) ** 0.5
    return {
        **stats,
        "unequal_fraction": stats["unequal_values"] / count if count else 0.0,
        "mean_abs": stats["abs_sum"] / count if count else 0.0,
        "rms": difference_rms,
        "left_mean_abs": stats["left_abs_sum"] / count if count else 0.0,
        "right_mean_abs": stats["right_abs_sum"] / count if count else 0.0,
        "left_rms": left_rms,
        "right_rms": right_rms,
        "relative_rms_to_left": difference_rms / left_rms if left_rms else None,
        "relative_rms_to_right": difference_rms / right_rms if right_rms else None,
        "cosine": (
            stats["cross_sum"] / cosine_denominator
            if cosine_denominator
            else None
        ),
    }


def compare(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = torch.load(left_path, map_location="cpu", mmap=True, weights_only=False)
    right = torch.load(right_path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise CacheComparisonError("static cache is not a dictionary")
    if set(left) != {"fisher", "saliency"} or set(right) != set(left):
        raise CacheComparisonError("static cache top-level schema mismatch")
    if len(left["fisher"]) != len(right["fisher"]):
        raise CacheComparisonError("Fisher layer count mismatch")
    if len(left["saliency"]) != len(right["saliency"]):
        raise CacheComparisonError("saliency layer count mismatch")

    fisher_stats = _new_stats()
    saliency_stats = _new_stats()
    saliency_by_module: dict[str, dict[str, Any]] = {}
    per_layer: list[dict[str, Any]] = []
    for index, (left_fisher, right_fisher) in enumerate(
        zip(left["fisher"], right["fisher"], strict=True)
    ):
        if left_fisher.shape != right_fisher.shape or left_fisher.dtype != right_fisher.dtype:
            raise CacheComparisonError(f"Fisher metadata mismatch at layer {index}")
        layer_fisher = _new_stats()
        left_sample = _sample_fisher(left_fisher)
        right_sample = _sample_fisher(right_fisher)
        _update(fisher_stats, left_sample, right_sample, f"layer{index}")
        _update(layer_fisher, left_sample, right_sample, f"layer{index}")

        left_saliency = left["saliency"][index]
        right_saliency = right["saliency"][index]
        if set(left_saliency) != set(right_saliency):
            raise CacheComparisonError(f"saliency key mismatch at layer {index}")
        layer_saliency = _new_stats()
        for name in sorted(left_saliency):
            if name not in saliency_by_module:
                saliency_by_module[name] = _new_stats()
            left_tensor = left_saliency[name]
            right_tensor = right_saliency[name]
            if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
                raise CacheComparisonError(
                    f"saliency metadata mismatch at layer {index}/{name}"
                )
            left_sample = _sample_saliency(left_tensor)
            right_sample = _sample_saliency(right_tensor)
            location = f"layer{index}/{name}"
            _update(saliency_stats, left_sample, right_sample, location)
            _update(layer_saliency, left_sample, right_sample, location)
            _update(
                saliency_by_module[name], left_sample, right_sample, location
            )
        per_layer.append(
            {
                "layer": index,
                "fisher": _finish(layer_fisher),
                "saliency": _finish(layer_saliency),
            }
        )

    return {
        "schema_version": 1,
        "left": {
            "path": str(left_path.resolve()),
            "size_bytes": left_path.stat().st_size,
        },
        "right": {
            "path": str(right_path.resolve()),
            "size_bytes": right_path.stat().st_size,
        },
        "layers": len(per_layer),
        "fisher": _finish(fisher_stats),
        "saliency": _finish(saliency_stats),
        "saliency_by_module": {
            name: _finish(stats)
            for name, stats in sorted(saliency_by_module.items())
        },
        "per_layer": per_layer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
