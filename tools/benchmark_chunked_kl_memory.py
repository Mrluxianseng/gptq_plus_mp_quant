"""Compare full-tensor and token-chunked KL allocator peaks on CUDA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.loss_utils import (  # noqa: E402
    chunked_tokenwise_kl_from_logits,
    tokenwise_kl_from_logits,
)


def _run_case(shape, dtype, device, *, chunk_tokens):
    torch.manual_seed(20261001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20261001)
    student = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    teacher = torch.randn(shape, device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        base_allocated = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        base_allocated = 0

    started = time.perf_counter()
    if chunk_tokens > 0:
        loss = chunked_tokenwise_kl_from_logits(
            student, teacher, chunk_tokens=chunk_tokens
        )
    else:
        loss = tokenwise_kl_from_logits(student, teacher).mean()
    grad = torch.autograd.grad(loss, student)[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_allocated = 0
    elapsed_seconds = time.perf_counter() - started
    return {
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().float().cpu(),
        "base_allocated_bytes": base_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_bytes": max(0, peak_allocated - base_allocated),
        "elapsed_seconds": elapsed_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.chunk_tokens <= 0:
        parser.error("--chunk-tokens must be positive")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    shape = (args.batch_size, args.seq_len, args.vocab_size)
    free_bytes = total_bytes = None
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    reference = _run_case(shape, dtype, device, chunk_tokens=0)
    reference_grad = reference.pop("grad")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    chunked = _run_case(shape, dtype, device, chunk_tokens=args.chunk_tokens)
    chunked_grad = chunked.pop("grad")
    loss_diff = abs(chunked["loss"] - reference["loss"])
    grad_diff = (chunked_grad - reference_grad).abs().max().item()
    result = {
        "shape": shape,
        "dtype": args.dtype,
        "chunk_tokens": args.chunk_tokens,
        "device_free_before_bytes": free_bytes,
        "device_total_bytes": total_bytes,
        "reference": reference,
        "chunked": chunked,
        "abs_loss_diff": loss_diff,
        "max_grad_abs_diff": grad_diff,
        "peak_reduction_bytes": (
            reference["peak_allocated_bytes"] - chunked["peak_allocated_bytes"]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if loss_diff > 2e-6 or grad_diff > 2e-6:
        raise SystemExit("chunked KL failed the numerical equivalence check")


if __name__ == "__main__":
    main()
