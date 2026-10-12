"""Measure memory and runtime of full vs token-chunked categorical sampling."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from gptq_utils.gptq_plus_utils import _deterministic_categorical_labels


def measure(logits: torch.Tensor, chunk_tokens: int, repeats: int):
    elapsed = []
    peaks = []
    labels = None
    for _ in range(repeats):
        torch.cuda.synchronize(logits.device)
        torch.cuda.reset_peak_memory_stats(logits.device)
        start = time.perf_counter()
        current = _deterministic_categorical_labels(
            logits, [7], base_seed=1701, chunk_tokens=chunk_tokens
        )
        torch.cuda.synchronize(logits.device)
        elapsed.append(time.perf_counter() - start)
        peaks.append(torch.cuda.max_memory_allocated(logits.device))
        if labels is not None and not torch.equal(labels, current):
            raise RuntimeError("categorical sampler is not deterministic for a fixed chunk size")
        labels = current
    return labels, peaks, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.chunk_tokens <= 0 or args.repeats <= 0:
        parser.error("--chunk-tokens and --repeats must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required for this memory benchmark")

    torch.manual_seed(123)
    logits = torch.randn(
        args.batch,
        args.tokens,
        args.vocab,
        device=device,
        dtype=torch.bfloat16,
    )
    full_labels, full_peaks, full_times = measure(logits, 0, args.repeats)
    chunk_labels, chunk_peaks, chunk_times = measure(
        logits, args.chunk_tokens, args.repeats
    )
    full_peak = max(full_peaks)
    chunk_peak = max(chunk_peaks)
    print(f"shape={tuple(logits.shape)} dtype={logits.dtype} repeats={args.repeats}")
    print(f"full_peak_allocated_bytes={full_peak}")
    print(f"chunk_peak_allocated_bytes={chunk_peak}")
    print(f"peak_saved_bytes={full_peak - chunk_peak}")
    print(f"peak_saved_fraction={(full_peak - chunk_peak) / full_peak:.6f}")
    print(f"full_seconds_median={statistics.median(full_times):.6f}")
    print(f"chunk_seconds_median={statistics.median(chunk_times):.6f}")
    print(f"time_ratio_chunk_over_full={statistics.median(chunk_times) / statistics.median(full_times):.6f}")
    print(f"label_mismatches={int((full_labels != chunk_labels).sum().item())}")


if __name__ == "__main__":
    main()
