"""Single-token Qwen3-0.6B downstream-KL/HVP feasibility probe.

This validates the local HVP contraction on the real model suffix. It uses
synthetic hidden-state perturbations, not REAL-Q's quantized residual, and it
does not update weights or claim an end-to-end quantization improvement.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class QwenSuffix(nn.Module):
    def __init__(self, model, first_layer, position_ids, cache_position, position_embeddings):
        super().__init__()
        self.model = model
        self.first_layer = first_layer
        self.position_ids = position_ids
        self.cache_position = cache_position
        self.position_embeddings = position_embeddings

    def forward(self, hidden):
        for layer in self.model.model.layers[self.first_layer:]:
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=self.position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=self.cache_position,
                position_embeddings=self.position_embeddings,
            )
        hidden = self.model.model.norm(hidden)
        return self.model.lm_head(hidden)


def make_loss(suffix, teacher_prob):
    log_teacher = teacher_prob.clamp_min(1e-30).log()

    def loss_fn(hidden):
        log_student = F.log_softmax(suffix(hidden).float(), dim=-1)
        return (teacher_prob * (log_teacher - log_student)).sum(dim=-1).mean()

    return loss_fn


def hvp(loss_fn, point, vector):
    x = point.detach().requires_grad_(True)
    grad = torch.autograd.grad(loss_fn(x), x, create_graph=True)[0]
    return torch.autograd.grad((grad * vector.detach()).sum(), x)[0].detach()


def rel_error(estimate, target):
    return float((estimate - target).norm() / target.norm().clamp_min(1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/d/llamaModels/Qwen3-0.6B")
    p.add_argument("--layer", type=int, default=20,
                   help="zero-based block whose output is y0; HVP traverses its suffix")
    p.add_argument("--token-id", type=int, default=42)
    p.add_argument("--relative-deltas", default="0.01,0.05,0.10")
    p.add_argument("--delta-seeds", default="9000,9001,9002",
                   help="comma-separated random directions for repeatability")
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--output", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This feasibility probe is intended for CUDA execution")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    n_layers = len(model.model.layers)
    if args.layer < 0 or args.layer >= n_layers - 1:
        raise ValueError(f"layer must be in [0, {n_layers - 2}] so a suffix remains")
    input_ids = torch.tensor([[args.token_id]], device=device)
    pos = torch.zeros((1, 1), dtype=torch.long, device=device)
    cache_position = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(input_ids)
        pos_emb = model.model.rotary_emb(embeddings, pos)
        full = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        y0 = full.hidden_states[args.layer + 1].detach()
        suffix = QwenSuffix(model, args.layer + 1, pos, cache_position, pos_emb)
        suffix_logits = suffix(y0).detach()
        full_logits = full.logits.detach()
        max_logit_diff = float((suffix_logits - full_logits).abs().max())
        teacher_prob = F.softmax(full_logits.float(), dim=-1)
    if max_logit_diff > 2e-3:
        raise RuntimeError(f"Suffix replay mismatch: max abs logit diff={max_logit_diff:.6g}")

    loss_fn = make_loss(suffix, teacher_prob)
    results = {
        "model": args.model,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "hidden_size": int(y0.shape[-1]),
        "vocab_size": int(full_logits.shape[-1]),
        "sequence_length": 1,
        "reference_layer_output": args.layer,
        "suffix_layers": n_layers - args.layer - 1,
        "full_vs_suffix_max_abs_logit_diff": max_logit_diff,
        "claim_scope": "Real Qwen suffix and KL; synthetic hidden perturbation only, not a quantization result.",
        "cases": [],
    }

    for relative_delta, delta_seed in product(
        map(float, args.relative_deltas.split(",")),
        map(int, args.delta_seeds.split(",")),
    ):
        torch.manual_seed(delta_seed)
        delta = torch.randn_like(y0)
        delta = delta * (relative_delta * y0.norm() / delta.norm())
        x1 = y0 + delta

        start = time.perf_counter()
        x = x1.detach().requires_grad_(True)
        exact = torch.autograd.grad(loss_fn(x), x)[0].detach()
        exact_seconds = time.perf_counter() - start

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        quad = hvp(loss_fn, y0, delta)
        quad_seconds = time.perf_counter() - start
        quad_peak = torch.cuda.max_memory_allocated(device) / (1024**2)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        plus = hvp(loss_fn, y0 + args.step * delta, delta)
        minus = hvp(loss_fn, y0 - args.step * delta, delta)
        cubic = quad + 0.5 * (plus - minus) / (2 * args.step)
        torch.cuda.synchronize(device)
        cubic_seconds = time.perf_counter() - start
        cubic_peak = torch.cuda.max_memory_allocated(device) / (1024**2)

        with torch.no_grad():
            exact_kl = float(loss_fn(x1))
            quad_kl = float(0.5 * (delta * quad).sum())
            cubic_kl = float(quad_kl + (delta * (plus - minus) / (2 * args.step)).sum() / 6)
        results["cases"].append({
            "relative_delta_l2": relative_delta,
            "delta_seed": delta_seed,
            "exact_kl": exact_kl,
            "quadratic_kl_taylor": quad_kl,
            "cubic_kl_taylor": cubic_kl,
            "quadratic_kl_abs_error": abs(quad_kl - exact_kl),
            "cubic_kl_abs_error": abs(cubic_kl - exact_kl),
            "exact_gradient_seconds": exact_seconds,
            "quadratic_hvp_seconds": quad_seconds,
            "cubic_two_extra_hvps_seconds": cubic_seconds,
            "quadratic_gradient_rel_error": rel_error(quad, exact),
            "cubic_gradient_rel_error": rel_error(cubic, exact),
            "quadratic_peak_allocated_mib": quad_peak,
            "cubic_peak_allocated_mib": cubic_peak,
        })
        del exact, quad, plus, minus, cubic, delta, x1
        torch.cuda.empty_cache()

    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()
