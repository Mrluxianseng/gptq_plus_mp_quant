"""One shared final-KL backward, with output and weight gradients for many layers.

Uses a saved single-module REAL-Q transition as the student perturbation, then
captures every suffix block's true KL output gradient in the same backward.
Static-Fisher matrices are reconstructed from same-Stage-0 sampled-label
gradients captured separately by capture_static_sequence_sketch.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realq.attention import configure_attention_backend
from utils import model_utils, quant_utils
from utils.loss_utils import tokenwise_kl_from_logits
from utils.reproducibility import configure_reproducibility


def hidden(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def tree_map(x, fn):
    if isinstance(x, torch.Tensor): return fn(x)
    if isinstance(x, tuple): return tuple(tree_map(v, fn) for v in x)
    if isinstance(x, list): return [tree_map(v, fn) for v in x]
    if isinstance(x, dict): return {k: tree_map(v, fn) for k, v in x.items()}
    return x


def select(x, i, b):
    if isinstance(x, torch.Tensor):
        return x[i:i+1] if x.ndim and x.shape[0] == b else x
    if isinstance(x, tuple): return tuple(select(v, i, b) for v in x)
    if isinstance(x, list): return [select(v, i, b) for v in x]
    if isinstance(x, dict): return {k: select(v, i, b) for k, v in x.items()}
    return x


def cosine(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).clamp(-1, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--target", required=True, help="transition key from summary.json")
    p.add_argument("--fisher-grad-dir", type=Path, required=True,
                   help="all-layer gradients from same Stage-0 backward")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weight-grad-layers", type=int, nargs="*", default=[3, 7, 15, 23],
                   help="also retain these down_proj weight gradients in the shared backward")
    p.add_argument("--microbatch-size", type=int, default=0,
                   help="student backward chunk size; 0 means the full calibration batch")
    p.add_argument("--save-gradients", action="store_true",
                   help="save output feedback/deltas and selected module weight gradients")
    a = p.parse_args()
    configure_reproducibility(0, deterministic=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")

    arm = json.loads((a.campaign / "summary.json").read_text())["results"][a.target]
    run_dir = Path(arm["adam"]["directory"])
    td = run_dir / "transition"
    context = torch.load(td / "context.pt", map_location="cpu", weights_only=False)
    refresh = torch.load(td / "refresh.pt", map_location="cpu", weights_only=False)
    after_update = torch.load(td / "after_update.pt", map_location="cpu", weights_only=True)
    target_idx = int(context["layer"])
    bsz = int(refresh["inps"].shape[0])
    chosen = [int(i) for i in refresh["selected"]]
    x = torch.cat([refresh["inps"][i:i+1] for i in chosen], dim=0).cuda().float()
    y0 = torch.cat([refresh["fp_target"][i:i+1] for i in chosen], dim=0).cuda().float()
    # Batch saved per-example kwargs in the same order as x / y0.
    kw = tree_map(
        refresh["kw"],
        lambda z: torch.cat([select(z, i, bsz) for i in chosen], dim=0).cuda().float()
        if z.ndim and z.shape[0] == bsz else z.cuda(),
    )

    analyzer = model_utils.ModelAnalyzer(a.model, 64, skip_state_dict=True)
    configure_attention_backend(analyzer.model, context["config"]["attention_backend"])
    quant_utils.add_actquant(analyzer)
    analyzer.model.eval().cuda()
    layers = analyzer.get_layers()
    for i, state in enumerate(context["prefix_layers"]):
        layers[i].load_state_dict(state, strict=True)
    if target_idx + 1 < len(layers):
        layers[target_idx + 1].load_state_dict(context["next_layer"], strict=True)

    # Smooth FP32 execution for stable comparison with the saved HVP audit.
    for block in layers[target_idx:]: block.float()
    analyzer.get_layernorm_before_head().float()
    analyzer.get_lm_head().float()
    target_layer = layers[target_idx]
    module = target_layer.get_submodule(context["module"])
    with torch.no_grad():
        module.weight.copy_(after_update.to(device="cuda", dtype=torch.float32))
    if hasattr(module, "module") and hasattr(module.module, "weight"):
        module.module.weight = module.weight

    analyzer.model.requires_grad_(False)
    requested_modules = {target_idx: context["module"]}
    for layer_idx in a.weight_grad_layers:
        if 0 <= int(layer_idx) < len(layers):
            requested_modules[int(layer_idx)] = "mlp.down_proj"
    grad_weights = {}
    for layer_idx, name in requested_modules.items():
        mod = layers[layer_idx].get_submodule(name)
        mod.weight.requires_grad_(True)
        grad_weights[f"layer{layer_idx}.{name}"] = mod.weight

    # Teacher trajectory from the saved FP output of the target block.
    teacher_outputs = {target_idx: y0.detach().cpu()}
    with torch.no_grad():
        h = y0
        for layer_idx in range(target_idx + 1, len(layers)):
            h = hidden(layers[layer_idx](h, **kw))
            teacher_outputs[layer_idx] = h.detach().float().cpu()
        teacher_logits = analyzer.get_lm_head()(
            analyzer.get_layernorm_before_head()(h)
        ).float().detach()

    true_outputs, true_grads = {}, {}
    handles = []

    def attach(index, out):
        tensor = hidden(out)
        true_outputs.setdefault(index, []).append(tensor.detach().float().cpu())
        def save_grad(g): true_grads.setdefault(index, []).append(g.detach().float().cpu())
        tensor.register_hook(save_grad)
        return out

    run_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    # Input to target block is saved by REAL-Q after the actual quantized prefix.
    # Recompute target output and then the full downstream suffix once.
    handles.append(target_layer.register_forward_hook(lambda m, i, o: attach(target_idx, o)))
    for layer_idx in range(target_idx + 1, len(layers)):
        handles.append(layers[layer_idx].register_forward_hook(
            lambda m, i, o, idx=layer_idx: attach(idx, o)
        ))
    x.requires_grad_(True)
    microbatch = int(a.microbatch_size) or bsz
    if microbatch <= 0:
        raise ValueError("microbatch-size must be non-negative")
    for param in grad_weights.values(): param.grad = None
    kl_total = 0.0
    try:
        for mb_start in range(0, bsz, microbatch):
            end = min(mb_start + microbatch, bsz)
            mb_kw = tree_map(kw, lambda z: z[mb_start:end] if z.ndim and z.shape[0] == bsz else z)
            h = hidden(target_layer(x[mb_start:end], **mb_kw))
            for layer_idx in range(target_idx + 1, len(layers)):
                h = hidden(layers[layer_idx](h, **mb_kw))
            student_logits = analyzer.get_lm_head()(
                analyzer.get_layernorm_before_head()(h)
            ).float()
            chunk_kl = tokenwise_kl_from_logits(
                student_logits, teacher_logits[mb_start:end]
            ).mean()
            weighted_loss = chunk_kl * ((end - mb_start) / bsz)
            kl_total += float(chunk_kl.detach()) * ((end - mb_start) / bsz)
            weighted_loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - run_start
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    finally:
        for handle in handles: handle.remove()

    true_outputs = {k: torch.cat(v, dim=0) for k, v in true_outputs.items()}
    true_grads = {k: torch.cat(v, dim=0) for k, v in true_grads.items()}

    fisher_results = {}
    fisher_manifest = json.loads((a.fisher_grad_dir / "manifest.json").read_text())
    for layer_idx in range(target_idx, len(layers)):
        gpath = a.fisher_grad_dir / f"layer{layer_idx}_sequence_gradients.pt"
        if not gpath.exists() or layer_idx not in true_grads: continue
        scores = torch.load(gpath, map_location="cpu", weights_only=True).float()
        # Match sampled sequence set size and token length.
        if scores.shape[1:] != true_grads[layer_idx].shape[1:]: continue
        n, t, hsize = scores.shape
        flat = scores.reshape(-1, hsize).double()
        fisher = (flat.T @ flat) / float(n * t)
        delta = true_outputs[layer_idx].double() - teacher_outputs[layer_idx].double()
        predicted = torch.einsum("hk,btk->bth", fisher, delta) / float(t * bsz)
        truth = true_grads[layer_idx].double()
        fisher_results[str(layer_idx)] = {
            "true_kl_feedback_vs_fisher_cosine": cosine(truth, predicted),
            "fisher_to_true_rms_ratio": float(predicted.square().mean().sqrt() / truth.square().mean().sqrt().clamp_min(1e-30)),
            "delta_rms": float(delta.square().mean().sqrt()),
            "true_feedback_rms": float(truth.square().mean().sqrt()),
        }

    weight_gradients = {
        name: {
            "shape": list(param.shape),
            "grad_rms": float(param.grad.detach().float().square().mean().sqrt()) if param.grad is not None else None,
            "grad_bytes": int(param.grad.numel() * param.grad.element_size()) if param.grad is not None else 0,
        }
        for name, param in grad_weights.items()
    }
    result = {
        "target": a.target,
        "target_layer": target_idx,
        "batch_samples": bsz,
        "sequence_length": int(x.shape[1]),
        "microbatch_size": microbatch,
        "single_student_kl_backward_seconds": elapsed,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "kl": kl_total,
        "captured_block_gradients": len(true_grads),
        "weight_gradients_from_same_backward": weight_gradients,
        "fisher_feedback_by_layer": fisher_results,
        "fisher_capture": fisher_manifest["collection"],
        "scope": "one final-KL backward for all suffix block outputs and selected module weights; one isolated post-Adam transition, diagnostic only",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    if a.save_gradients:
        torch.save({
            "block_output_gradients": true_grads,
            "block_output_deltas": {
                k: true_outputs[k] - teacher_outputs[k] for k in true_outputs
            },
            "weight_gradients": {
                name: param.grad.detach().float().cpu()
                for name, param in grad_weights.items() if param.grad is not None
            },
            "target": a.target,
        }, a.out.with_suffix(".pt"))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
