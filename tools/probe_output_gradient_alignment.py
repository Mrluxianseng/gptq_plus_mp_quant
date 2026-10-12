"""Compare REAL-Q Fisher output feedback with paired downstream KL feedback.

The current block output is treated as a leaf tensor, isolating the
approximation of dL/dy from the target block's weight Jacobian. Diagnostic
only; uses saved refresh samples and changes no quantization path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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


def select(value, i, batch):
    if isinstance(value, torch.Tensor):
        return value[i:i + 1] if value.ndim and value.shape[0] == batch else value
    if isinstance(value, tuple): return tuple(select(x, i, batch) for x in value)
    if isinstance(value, list): return [select(x, i, batch) for x in value]
    if isinstance(value, dict): return {k: select(v, i, batch) for k, v in value.items()}
    return value


def cosine(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).clamp(-1, 1))


def run_one(model_path, campaign, key):
    arms = json.loads((campaign / "summary.json").read_text())["results"][key]
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
    target_layer = layers[context["layer"]]
    target = target_layer.get_submodule(context["module"])
    target.weight.copy_(before.to(target.weight))
    suffix = layers[context["layer"] + 1:]
    batch_size = refresh["inps"].shape[0]

    proxy_grads, current_grads, next_grads, true_grads = [], [], [], []
    proxy_losses, true_losses = [], []
    per_sample = []
    for i in refresh["selected"]:
        x = refresh["inps"][i:i + 1].cuda()
        kw = tree_map(select(refresh["kw"], i, batch_size), lambda v: v.cuda())
        with torch.no_grad():
            y0 = hidden(target_layer(x, **kw)).detach()
            ref = refresh["fp_target"][i:i + 1].cuda()
            for block in suffix: ref = hidden(block(ref, **kw))
            teacher_logits = analyzer.get_lm_head()(
                analyzer.get_layernorm_before_head()(ref)
            ).float().detach()
        y = y0.detach().requires_grad_(True)
        loss_current = fisher_mse_loss(
            y, refresh["fp_target"][i:i + 1].cuda(), refresh["fisher"].cuda(),
            a_loss_ratio=float(refresh["a_loss_ratio"]),
            a_loss_threshold=refresh["a_loss_threshold"],
        )
        alpha = refresh["slide_alpha"]
        loss_next = None
        if alpha is not None and alpha < 1.0:
            next_y = hidden(layers[context["layer"] + 1](y, **kw))
            loss_next = fisher_mse_loss(
                next_y, refresh["next_fp_target"][i:i + 1].cuda(),
                refresh["next_fisher"].cuda(),
                a_loss_ratio=float(refresh["a_loss_ratio"]),
                a_loss_threshold=refresh["next_a_loss_threshold"],
            )
            loss_proxy = float(alpha) * loss_current + (1.0-float(alpha)) * loss_next
        else:
            loss_proxy = loss_current

        h = y
        for block in suffix: h = hidden(block(h, **kw))
        student_logits = analyzer.get_lm_head()(
            analyzer.get_layernorm_before_head()(h)
        ).float()
        loss_true = tokenwise_kl_from_logits(student_logits, teacher_logits).mean()
        gc, = torch.autograd.grad(loss_current, y, retain_graph=True)
        if loss_next is not None:
            gn, = torch.autograd.grad(loss_next, y, retain_graph=True)
        else:
            gn = torch.zeros_like(gc)
        gp = float(alpha if alpha is not None and alpha < 1.0 else 1.0) * gc
        if loss_next is not None:
            gp = gp + (1.0-float(alpha)) * gn
        gt, = torch.autograd.grad(loss_true, y)
        gc32, gn32 = gc.detach().float(), gn.detach().float()
        gp32, gt32 = gp.detach().float(), gt.detach().float()
        proxy_grads.append(gp32.cpu())
        current_grads.append(gc32.cpu())
        next_grads.append(gn32.cpu())
        true_grads.append(gt32.cpu())
        proxy_losses.append(float(loss_proxy.detach().item()))
        true_losses.append(float(loss_true.detach().item()))
        per_sample.append({
            "sample": int(i),
            "cosine": cosine(gp32, gt32),
            "proxy_grad_rms": float(gp32.square().mean().sqrt()),
            "true_kl_grad_rms": float(gt32.square().mean().sqrt()),
        })
        del y, loss_proxy, loss_current, loss_true, gp, gc, gn, gt, student_logits, h

    gp = torch.cat(proxy_grads, dim=0).double()
    gc = torch.cat(current_grads, dim=0).double()
    gn = torch.cat(next_grads, dim=0).double()
    gt = torch.cat(true_grads, dim=0).double()
    result = {
        "target": key,
        "samples": refresh["selected"],
        "slide_alpha": refresh["slide_alpha"],
        "proxy_loss_replay": sum(proxy_losses) / len(proxy_losses),
        "stored_online_proxy_loss": refresh["actual_loss"],
        "paired_kl_mean": sum(true_losses) / len(true_losses),
        "output_gradient_cosine": cosine(gp.float(), gt.float()),
        "current_fisher_output_gradient_cosine": cosine(gc.float(), gt.float()),
        "next_fisher_output_gradient_cosine": cosine(gn.float(), gt.float()),
        "current_fisher_output_gradient_rms": float(gc.square().mean().sqrt()),
        "next_fisher_output_gradient_rms": float(gn.square().mean().sqrt()),
        "output_gradient_relative_l2": float((gp-gt).norm()/gt.norm().clamp_min(1e-30)),
        "proxy_output_gradient_rms": float(gp.square().mean().sqrt()),
        "true_kl_output_gradient_rms": float(gt.square().mean().sqrt()),
        "per_sample": per_sample,
        "scope": "current target-block output y is the differentiation leaf; same saved refresh examples; actual BF16 block/suffix forward; isolates dL/dy from dy/dW",
    }
    if not math.isclose(result["proxy_loss_replay"], refresh["actual_loss"], rel_tol=1e-5, abs_tol=1e-8):
        raise RuntimeError(f"proxy loss replay mismatch: {result['proxy_loss_replay']} vs {refresh['actual_loss']}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--target", action="append", required=True)
    a = p.parse_args()
    configure_reproducibility(0, deterministic=True)
    torch.set_num_threads(4)
    path = a.campaign / "output_gradient_alignment.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for key in a.target:
        print(f"[output-gradient] START {key}", flush=True)
        results[key] = run_one(a.model, a.campaign, key)
        path.write_text(json.dumps(results, indent=2))
        print(f"[output-gradient] DONE {key}: " + json.dumps(results[key]), flush=True)
