#!/usr/bin/env python3
"""Compare RealQ's FA4 adapter with its historical masked-SDPA path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realq.attention import flash_attention_4_forward


def _repeat_kv(x: torch.Tensor, repetitions: int) -> torch.Tensor:
    batch, heads, sequence, width = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, heads, repetitions, sequence, width)
        .reshape(batch, heads * repetitions, sequence, width)
    )


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float()
    expected = expected.float()
    delta = actual - expected
    return {
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-30)).item(),
        "cosine": F.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
    }


def _run_fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out, _ = flash_attention_4_forward(
        SimpleNamespace(is_causal=True), q, k, v, None
    )
    (out.float() * grad).sum().backward()
    torch.cuda.synchronize()
    return out.detach(), (q.grad.detach(), k.grad.detach(), v.grad.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output-relative-l2-limit", type=float, default=5e-3)
    parser.add_argument("--grad-relative-l2-limit", type=float, default=1e-2)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        shape_q = (args.batch, args.q_heads, args.seq_len, args.head_dim)
        shape_kv = (args.batch, args.kv_heads, args.seq_len, args.head_dim)
        q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(
            args.batch,
            args.seq_len,
            args.q_heads,
            args.head_dim,
            device="cuda",
            dtype=torch.float32,
        )

        # This reproduces the old Transformers path: explicit 4-D causal mask
        # plus K/V expansion causes SDPA to select efficient attention.
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        repetitions = args.q_heads // args.kv_heads
        mask = torch.full(
            (1, 1, args.seq_len, args.seq_len),
            torch.finfo(torch.bfloat16).min,
            device="cuda",
            dtype=torch.bfloat16,
        ).triu(1)
        ref = F.scaled_dot_product_attention(
            q_ref,
            _repeat_kv(k_ref, repetitions),
            _repeat_kv(v_ref, repetitions),
            attn_mask=mask,
            is_causal=False,
        ).transpose(1, 2)
        (ref.float() * grad).sum().backward()
        torch.cuda.synchronize()

        out1, grads1 = _run_fa4(q, k, v, grad)
        out2, grads2 = _run_fa4(q, k, v, grad)
        report = {
            "shape": {
                "batch": args.batch,
                "seq_len": args.seq_len,
                "q_heads": args.q_heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
            },
            "device": torch.cuda.get_device_name(),
            "output": _metrics(out1, ref.detach()),
            "dq": _metrics(grads1[0], q_ref.grad),
            "dk": _metrics(grads1[1], k_ref.grad),
            "dv": _metrics(grads1[2], v_ref.grad),
            "deterministic_bitwise": {
                "output": torch.equal(out1, out2),
                "dq": torch.equal(grads1[0], grads2[0]),
                "dk": torch.equal(grads1[1], grads2[1]),
                "dv": torch.equal(grads1[2], grads2[2]),
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        failures = []
        if report["output"]["relative_l2"] > args.output_relative_l2_limit:
            failures.append(
                "output relative_l2 "
                f"{report['output']['relative_l2']:.6g} exceeds "
                f"{args.output_relative_l2_limit:.6g}"
            )
        for name in ("dq", "dk", "dv"):
            if report[name]["relative_l2"] > args.grad_relative_l2_limit:
                failures.append(
                    f"{name} relative_l2 {report[name]['relative_l2']:.6g} "
                    f"exceeds {args.grad_relative_l2_limit:.6g}"
                )
        for name, is_equal in report["deterministic_bitwise"].items():
            if not is_equal:
                failures.append(f"{name} is not bitwise deterministic")
        if failures:
            raise SystemExit(
                "FA4 numerical validation failed: " + "; ".join(failures)
            )
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=True)


if __name__ == "__main__":
    main()
