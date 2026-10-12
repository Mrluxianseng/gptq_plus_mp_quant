"""Probe a smooth FP32 target-block Fisher refresh against paired KL gradients.

This tests the precision hypothesis without adding an extra backward to any
production run: both proxy and true-KL gradients are computed offline from the
same saved refresh examples. Nothing in the quantization path is modified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realq.attention import configure_attention_backend
from realq.refresh.fisher_loss import fisher_mse_loss
from utils import model_utils, quant_utils
from utils.loss_utils import tokenwise_kl_from_logits
from utils.reproducibility import configure_reproducibility


def hidden(x): return x[0] if isinstance(x, tuple) else x


def tree_map(value, fn):
    if isinstance(value, torch.Tensor): return fn(value)
    if isinstance(value, tuple): return tuple(tree_map(x, fn) for x in value)
    if isinstance(value, list): return [tree_map(x, fn) for x in value]
    if isinstance(value, dict): return {k: tree_map(v, fn) for k, v in value.items()}
    return value


def sample(value, i, batch):
    if isinstance(value, torch.Tensor):
        return value[i:i + 1] if value.ndim and value.shape[0] == batch else value
    if isinstance(value, tuple): return tuple(sample(x, i, batch) for x in value)
    if isinstance(value, list): return [sample(x, i, batch) for x in value]
    if isinstance(value, dict): return {k: sample(v, i, batch) for k, v in value.items()}
    return value


def cosine(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).clamp(-1, 1))


def run_one(model_path, campaign, key):
    summary = json.loads((campaign / "summary.json").read_text())
    arms = summary["results"][key]
    d = Path(arms["adam"]["directory"]) / "transition"
    context = torch.load(d / "context.pt", map_location="cpu", weights_only=False)
    refresh = torch.load(d / "refresh.pt", map_location="cpu", weights_only=False)
    before = torch.load(d / "before_update.pt", map_location="cpu", weights_only=True)

    analyzer = model_utils.ModelAnalyzer(model_path, 64, skip_state_dict=True)
    configure_attention_backend(analyzer.model, context["config"]["attention_backend"])
    quant_utils.add_actquant(analyzer)
    analyzer.model.eval().requires_grad_(False).cuda()
    layers = analyzer.get_layers()
    for i, state in enumerate(context["prefix_layers"]): layers[i].load_state_dict(state, strict=True)
    layers[context["layer"] + 1].load_state_dict(context["next_layer"], strict=True)

    # Promote the current block, suffix, norm and head to a smooth FP32 path.
    # Weights originate from the loaded BF16 checkpoint, not an FP32 checkpoint.
    for block in layers[context["layer"]:]: block.float()
    analyzer.get_layernorm_before_head().float()
    analyzer.get_lm_head().float()
    layer = layers[context["layer"]]
    suffix = layers[context["layer"] + 1:]
    target = layer.get_submodule(context["module"])
    weight_name = refresh["weight_name"]
    leaf = before.to(device="cuda", dtype=torch.float32).detach().requires_grad_(True)
    teacher, proxy_sum, kl_sum = {}, 0.0, 0.0
    grad_proxy_sum = torch.zeros_like(before, dtype=torch.float64)
    grad_kl_sum = torch.zeros_like(before, dtype=torch.float64)
    batch = refresh["inps"].shape[0]

    # Same saved FP target-block output, propagated through the smooth suffix.
    with torch.no_grad():
        for i in refresh["selected"]:
            kw = tree_map(sample(refresh["kw"], i, batch),
                          lambda x: x.cuda().float() if x.is_floating_point() else x.cuda())
            h = refresh["fp_target"][i:i + 1].cuda().float()
            for block in suffix: h = hidden(block(h, **kw))
            teacher[i] = analyzer.get_lm_head()(
                analyzer.get_layernorm_before_head()(h)
            ).float().detach()

    for i in refresh["selected"]:
        x = refresh["inps"][i:i + 1].cuda().float()
        kw = tree_map(sample(refresh["kw"], i, batch),
                      lambda v: v.cuda().float() if v.is_floating_point() else v.cuda())
        fp_target = refresh["fp_target"][i:i + 1].cuda().float()
        with torch.enable_grad():
            out = functional_call(layer, {weight_name: leaf}, (x,), kw, strict=False)
            q = hidden(out)
            loss_proxy = fisher_mse_loss(
                q, fp_target, refresh["fisher"].cuda().float(),
                a_loss_ratio=float(refresh["a_loss_ratio"]),
                a_loss_threshold=refresh["a_loss_threshold"],
            )
            alpha = refresh["slide_alpha"]
            if alpha is not None and alpha < 1.0:
                next_q = hidden(layers[context["layer"] + 1](q, **kw))
                next_fp = refresh["next_fp_target"][i:i + 1].cuda().float()
                next_loss = fisher_mse_loss(
                    next_q, next_fp, refresh["next_fisher"].cuda().float(),
                    a_loss_ratio=float(refresh["a_loss_ratio"]),
                    a_loss_threshold=refresh["next_a_loss_threshold"],
                )
                loss_proxy = float(alpha) * loss_proxy + (1-float(alpha)) * next_loss
            h = q
            for block in suffix: h = hidden(block(h, **kw))
            logits = analyzer.get_lm_head()(analyzer.get_layernorm_before_head()(h)).float()
            loss_kl = tokenwise_kl_from_logits(logits, teacher[i]).mean()
            gp, = torch.autograd.grad(loss_proxy, leaf, retain_graph=True)
            gk, = torch.autograd.grad(loss_kl, leaf)
        grad_proxy_sum.add_(gp.detach().cpu().double())
        grad_kl_sum.add_(gk.detach().cpu().double())
        proxy_sum += float(loss_proxy.detach().item())
        kl_sum += float(loss_kl.detach().item())
        del out, q, loss_proxy, loss_kl, gp, gk

    n = len(refresh["selected"])
    proxy = (grad_proxy_sum / n).float()
    true_kl = (grad_kl_sum / n).float()
    perm = context["perm"]
    start = int(context["column_end"])
    if perm is not None:
        proxy, true_kl = proxy[:, perm.long()], true_kl[:, perm.long()]
    proxy, true_kl = proxy[:, start:], true_kl[:, start:]
    native = refresh["raw_gradient"].float()
    native_summary_path = campaign / "precision_proxy_alignment.json"
    native_results = json.loads(native_summary_path.read_text())
    native_metrics = native_results[key]
    result = {
        "target": key,
        "samples": refresh["selected"],
        "native_proxy_vs_native_paired_kl_cosine": native_metrics["native_proxy_vs_native_true_kl_gradient_cosine"],
        "fp32_block_proxy_vs_fp32_block_paired_kl_cosine": cosine(proxy, true_kl),
        "fp32_block_proxy_vs_native_proxy_cosine": cosine(proxy, native),
        "native_to_fp32_block_proxy_relative_l2": float((proxy-native).norm()/native.norm().clamp_min(1e-30)),
        "fp32_block_proxy_norm": float(proxy.norm()),
        "native_proxy_norm": float(native.norm()),
        "fp32_proxy_loss": proxy_sum / n,
        "fp32_paired_kl": kl_sum / n,
        "native_bf16_paired_kl": native_metrics["native_paired_kl"],
        "scope": "same selected examples; target block/suffix/norm/head all smoothly evaluated in FP32; weights promoted from BF16 checkpoint; diagnostic only",
    }
    if not math.isfinite(result["fp32_block_proxy_vs_fp32_block_paired_kl_cosine"]):
        raise RuntimeError("non-finite FP32 alignment")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--target", action="append", required=True)
    a = p.parse_args()
    configure_reproducibility(0, deterministic=True)
    torch.set_num_threads(4)
    out_path = a.campaign / "fp32_block_proxy_alignment.json"
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for key in a.target:
        print(f"[fp32-block] START {key}", flush=True)
        all_results[key] = run_one(a.model, a.campaign, key)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"[fp32-block] DONE {key}: " + json.dumps(all_results[key]), flush=True)
