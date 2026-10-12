"""Memory/runtime demo for a third-order correction to REAL-Q's local KL gradient.

This is a mathematical/mechanism demo, not an end-to-end quantization claim.
It uses a small Transformer-like downstream tail (RMSNorm + SwiGLU + logits),
and compares the exact KL gradient against a local quadratic model and a
third-order correction estimated with Hessian-vector products (HVPs).
No third-order tensor is materialized in the scalable path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinySwiGLUTail(nn.Module):
    """A deterministic nonlinear downstream map from block output to logits."""

    def __init__(self, hidden: int, intermediate: int, vocab: int, seed: int):
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)
        self.head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            for module in (self.gate, self.up, self.down, self.head):
                module.weight.copy_(torch.randn(module.weight.shape, generator=gen) / math.sqrt(module.weight.shape[1]))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-6)
        ffn = self.down(F.silu(self.gate(h)) * self.up(h))
        return self.head(h + ffn)


def make_kl_fn(tail: nn.Module, teacher_prob: torch.Tensor):
    def loss_fn(h: torch.Tensor) -> torch.Tensor:
        logits = tail(h)
        logp = F.log_softmax(logits.float(), dim=-1)
        return (teacher_prob * (teacher_prob.clamp_min(1e-30).log() - logp)).sum(dim=-1).mean()
    return loss_fn


def grad_at(loss_fn, point: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
    x = point.detach().requires_grad_(True)
    return torch.autograd.grad(loss_fn(x), x, create_graph=create_graph)[0].detach()


def hvp(loss_fn, point: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Compute H(point) @ vector without storing H."""
    x = point.detach().requires_grad_(True)
    grad = torch.autograd.grad(loss_fn(x), x, create_graph=True)[0]
    hv = torch.autograd.grad((grad * vector.detach()).sum(), x)[0]
    return hv.detach()


@torch.no_grad()
def rel_error(estimate: torch.Tensor, target: torch.Tensor) -> float:
    denom = target.norm().clamp_min(1e-30)
    return float((estimate - target).norm() / denom)


def exact_small_dimension_third_contraction(loss_fn, point, delta):
    """Exact T[delta,delta] by differentiating delta^T H delta (small dim only)."""
    x = point.detach().requires_grad_(True)
    g = torch.autograd.grad(loss_fn(x), x, create_graph=True)[0]
    h_rows = []
    for i in range(x.numel()):
        h_rows.append(torch.autograd.grad(g.reshape(-1)[i], x, create_graph=True, retain_graph=True)[0])
    hessian = torch.stack(h_rows).reshape(x.numel(), x.numel())
    d = delta.reshape(-1).detach()
    directional_curvature = d @ hessian @ d
    contracted = torch.autograd.grad(directional_curvature, x)[0]
    return contracted.detach().reshape_as(point), hessian.detach().reshape(point.numel(), point.numel())


@dataclass
class TimingMemory:
    seconds: float
    peak_allocated_mib: float | None


def measure_cuda(fn, device: torch.device, repeats: int) -> TimingMemory:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak = None
    return TimingMemory((time.perf_counter() - start) / repeats, peak)


def run_case(hidden, batch_tokens, relative_delta, device, seed, vocab_factor, finite_diff_step, repeats):
    torch.manual_seed(seed)
    intermediate = 2 * hidden
    vocab = vocab_factor * hidden
    tail = TinySwiGLUTail(hidden, intermediate, vocab, seed + 17).to(device).eval()
    y0 = torch.randn(batch_tokens, hidden, device=device)
    with torch.no_grad():
        teacher_prob = F.softmax(tail(y0).float(), dim=-1)
    loss_fn = make_kl_fn(tail, teacher_prob)
    delta = torch.randn_like(y0)
    delta = delta * (relative_delta * y0.norm() / delta.norm())
    y1 = y0 + delta

    exact_grad = grad_at(loss_fn, y1)
    h0_delta = hvp(loss_fn, y0, delta)
    plus = hvp(loss_fn, y0 + finite_diff_step * delta, delta)
    minus = hvp(loss_fn, y0 - finite_diff_step * delta, delta)
    third = (plus - minus) / (2.0 * finite_diff_step)
    grad2 = h0_delta
    grad3 = h0_delta + 0.5 * third

    with torch.no_grad():
        exact_loss = float(loss_fn(y1))
        loss2 = float(0.5 * (delta * h0_delta).sum())
        loss3 = float(loss2 + (delta * third).sum() / 6.0)

    def run_quadratic():
        hvp(loss_fn, y0, delta)

    def run_cubic():
        base = hvp(loss_fn, y0, delta)
        p = hvp(loss_fn, y0 + finite_diff_step * delta, delta)
        m = hvp(loss_fn, y0 - finite_diff_step * delta, delta)
        _ = base + 0.5 * (p - m) / (2.0 * finite_diff_step)

    t2 = measure_cuda(run_quadratic, device, repeats)
    t3 = measure_cuda(run_cubic, device, repeats)
    full_tensor_bytes = hidden**3 * 4
    return {
        "hidden": hidden,
        "batch_tokens": batch_tokens,
        "vocab": vocab,
        "relative_delta_l2": relative_delta,
        "finite_diff_step": finite_diff_step,
        "exact_kl": exact_loss,
        "quadratic_kl_taylor": loss2,
        "cubic_kl_taylor": loss3,
        "quadratic_kl_abs_error": abs(loss2 - exact_loss),
        "cubic_kl_abs_error": abs(loss3 - exact_loss),
        "exact_grad_norm": float(exact_grad.norm()),
        "quadratic_grad_rel_error": rel_error(grad2, exact_grad),
        "cubic_grad_rel_error": rel_error(grad3, exact_grad),
        "quadratic_hvp_seconds": t2.seconds,
        "cubic_hvp_seconds": t3.seconds,
        "time_ratio_cubic_over_quadratic": t3.seconds / max(t2.seconds, 1e-12),
        "quadratic_peak_allocated_mib": t2.peak_allocated_mib,
        "cubic_peak_allocated_mib": t3.peak_allocated_mib,
        "explicit_third_tensor_gib": full_tensor_bytes / (1024**3),
        "extra_third_tensor_materialized": False,
    }


def run_small_exact_check(device):
    torch.manual_seed(101)
    h = 8
    tail = TinySwiGLUTail(h, 2 * h, 3 * h, 202).to(device).eval()
    y0 = torch.randn(1, h, device=device)
    with torch.no_grad():
        p = F.softmax(tail(y0).float(), dim=-1)
    loss_fn = make_kl_fn(tail, p)
    delta = torch.randn_like(y0) * 0.04
    exact_t, hessian = exact_small_dimension_third_contraction(loss_fn, y0, delta)
    eps = 0.25
    estimated_t = (hvp(loss_fn, y0 + eps * delta, delta) - hvp(loss_fn, y0 - eps * delta, delta)) / (2 * eps)
    return {
        "hidden": h,
        "third_tensor_bytes_if_materialized_fp32": h**3 * 4,
        "hessian_shape_reference_only": list(hessian.shape),
        "contracted_third_rel_error_vs_explicit_small_dim": rel_error(estimated_t, exact_t),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-sizes", default="128,512,1024")
    parser.add_argument("--relative-deltas", default="0.01,0.05,0.10,0.20")
    parser.add_argument("--batch-tokens", type=int, default=8)
    parser.add_argument("--vocab-factor", type=int, default=8)
    parser.add_argument("--finite-diff-steps", default="0.25",
                        help="comma-separated central-difference fractions of delta")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    results = {
        "title": "Third-order KL Taylor correction via central-difference HVP",
        "claim_scope": "Numerical mechanism demo; not an end-to-end quantization result.",
        "device": str(device),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "method": "T[delta,delta] ~= (H(y0+eps*delta)delta - H(y0-eps*delta)delta)/(2 eps)",
        "small_exact_check": run_small_exact_check(device),
        "cases": [],
    }
    for hidden in map(int, args.hidden_sizes.split(",")):
        for rel_delta in map(float, args.relative_deltas.split(",")):
            for step in map(float, args.finite_diff_steps.split(",")):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                results["cases"].append(
                    run_case(hidden, args.batch_tokens, rel_delta, device, args.seed + hidden,
                             args.vocab_factor, step, args.repeats)
                )
    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()
