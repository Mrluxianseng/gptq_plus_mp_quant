"""Test whether FP32 target-Linear arithmetic improves REAL-Q proxy-gradient
alignment to paired downstream KL on the same saved refresh examples.

The experiment keeps BF16 activations and casts the candidate Linear output
back to BF16. It is an offline diagnostic, not a production change.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
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


def hidden(value):
    return value[0] if isinstance(value, tuple) else value


def select_sample(value, index, batch_size):
    if isinstance(value, torch.Tensor):
        return value[index:index + 1] if value.ndim and value.shape[0] == batch_size else value
    if isinstance(value, tuple): return tuple(select_sample(x, index, batch_size) for x in value)
    if isinstance(value, list): return [select_sample(x, index, batch_size) for x in value]
    if isinstance(value, dict): return {k: select_sample(v, index, batch_size) for k, v in value.items()}
    return value


def move(value):
    if isinstance(value, torch.Tensor): return value.cuda()
    if isinstance(value, tuple): return tuple(move(x) for x in value)
    if isinstance(value, list): return [move(x) for x in value]
    if isinstance(value, dict): return {k: move(v) for k, v in value.items()}
    return value


def cosine(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).clamp(-1, 1))


def run_one(model_path, campaign, key):
    arms = json.loads((campaign / "summary.json").read_text())["results"][key]
    directory = Path(arms["adam"]["directory"]) / "transition"
    context = torch.load(directory / "context.pt", map_location="cpu", weights_only=False)
    refresh = torch.load(directory / "refresh.pt", map_location="cpu", weights_only=False)
    before = torch.load(directory / "before_update.pt", map_location="cpu", weights_only=True)

    analyzer = model_utils.ModelAnalyzer(model_path, 64, skip_state_dict=True)
    configure_attention_backend(analyzer.model, context["config"]["attention_backend"])
    quant_utils.add_actquant(analyzer)
    analyzer.model.eval().requires_grad_(False).cuda()
    layers = analyzer.get_layers()
    for i, state in enumerate(context["prefix_layers"]): layers[i].load_state_dict(state, strict=True)
    layers[context["layer"] + 1].load_state_dict(context["next_layer"], strict=True)
    layer = layers[context["layer"]]
    suffix = layers[context["layer"] + 1:]
    target = layer.get_submodule(context["module"])
    linear = target.module
    weight_name = refresh["weight_name"]
    original_forward = linear.forward
    selected = refresh["selected"]
    batch_size = refresh["inps"].shape[0]

    # Paired teacher logits: saved FP output of this block, propagated only
    # through the same original BF16 suffix used by the REAL-Q refresh.
    teacher = {}
    with torch.no_grad():
        for i in selected:
            kw = move(select_sample(refresh["kw"], i, batch_size))
            h = refresh["fp_target"][i:i + 1].cuda()
            for block in suffix: h = hidden(block(h, **kw))
            teacher[i] = analyzer.get_lm_head()(analyzer.get_layernorm_before_head()(h)).float().detach()

    modes = ("native_bf16", "fp32_target_linear_output_bf16")
    proxy_grads, kl_grads = {}, {}
    proxy_losses, kl_losses = {}, {}
    target_outputs = {}
    for mode in modes:
        if mode == "native_bf16":
            linear.forward = original_forward
            leaf = before.to(device="cuda", dtype=torch.bfloat16).detach().requires_grad_(True)
        else:
            def fp32_linear_round(module, x, *args, **kwargs):
                if args or kwargs:
                    raise RuntimeError("unexpected extra target Linear arguments")
                bias = None if module.bias is None else module.bias.float()
                return F.linear(x.float(), module.weight.float(), bias).to(x.dtype)
            linear.forward = types.MethodType(fp32_linear_round, linear)
            leaf = before.to(device="cuda", dtype=torch.float32).detach().requires_grad_(True)

        grad_proxy_sum = torch.zeros_like(before, dtype=torch.float64)
        grad_kl_sum = torch.zeros_like(before, dtype=torch.float64)
        proxy_sum, kl_sum = 0.0, 0.0
        for i in selected:
            x = refresh["inps"][i:i + 1].cuda()
            kw = move(select_sample(refresh["kw"], i, batch_size))
            with torch.enable_grad():
                out = functional_call(layer, {weight_name: leaf}, (x,), kw, strict=False)
                q_out = hidden(out)
                if i == selected[0]: target_outputs[mode] = q_out.detach().float().cpu()

                loss_proxy = fisher_mse_loss(
                    q_out, refresh["fp_target"][i:i + 1].cuda(),
                    refresh["fisher"].cuda(),
                    a_loss_ratio=float(refresh["a_loss_ratio"]),
                    a_loss_threshold=refresh["a_loss_threshold"],
                )
                alpha = refresh["slide_alpha"]
                if alpha is not None and alpha < 1.0:
                    nxt = hidden(layers[context["layer"] + 1](q_out, **kw))
                    loss_next = fisher_mse_loss(
                        nxt, refresh["next_fp_target"][i:i + 1].cuda(),
                        refresh["next_fisher"].cuda(),
                        a_loss_ratio=float(refresh["a_loss_ratio"]),
                        a_loss_threshold=refresh["next_a_loss_threshold"],
                    )
                    loss_proxy = float(alpha) * loss_proxy + (1.0 - float(alpha)) * loss_next

                h = q_out
                for block in suffix: h = hidden(block(h, **kw))
                logits = analyzer.get_lm_head()(analyzer.get_layernorm_before_head()(h)).float()
                loss_kl = tokenwise_kl_from_logits(logits, teacher[i]).mean()

                grad_proxy, = torch.autograd.grad(loss_proxy, leaf, retain_graph=True)
                grad_kl, = torch.autograd.grad(loss_kl, leaf)
            grad_proxy_sum.add_(grad_proxy.detach().cpu().double())
            grad_kl_sum.add_(grad_kl.detach().cpu().double())
            proxy_sum += float(loss_proxy.detach().item())
            kl_sum += float(loss_kl.detach().item())
            del out, q_out, loss_proxy, loss_kl, grad_proxy, grad_kl
        proxy_grads[mode] = (grad_proxy_sum / len(selected)).float()
        kl_grads[mode] = (grad_kl_sum / len(selected)).float()
        proxy_losses[mode] = proxy_sum / len(selected)
        kl_losses[mode] = kl_sum / len(selected)
        del leaf

    linear.forward = original_forward
    perm = context["perm"]
    start = int(context["column_end"])
    stored_proxy = refresh["raw_gradient"].float()
    def active(g):
        if perm is not None: g = g[:, perm.long()]
        return g[:, start:]

    native_proxy, candidate_proxy = active(proxy_grads[modes[0]]), active(proxy_grads[modes[1]])
    native_kl, candidate_kl = active(kl_grads[modes[0]]), active(kl_grads[modes[1]])
    stored_cosine = cosine(native_proxy, stored_proxy)
    native_alignment = cosine(native_proxy, native_kl)
    candidate_alignment = cosine(candidate_proxy, candidate_kl)
    result = {
        "target": key,
        "samples": selected,
        "candidate_target_output_relative_l2_change": float(
            (target_outputs[modes[1]] - target_outputs[modes[0]]).norm() /
            target_outputs[modes[0]].norm().clamp_min(1e-30)
        ),
        "native_proxy_loss": proxy_losses[modes[0]],
        "candidate_proxy_loss": proxy_losses[modes[1]],
        "native_paired_kl": kl_losses[modes[0]],
        "candidate_paired_kl": kl_losses[modes[1]],
        "native_proxy_replay_cosine_vs_saved_gradient": stored_cosine,
        "native_proxy_vs_native_true_kl_gradient_cosine": native_alignment,
        "candidate_proxy_vs_candidate_true_kl_gradient_cosine": candidate_alignment,
        "candidate_proxy_vs_native_true_kl_gradient_cosine": cosine(candidate_proxy, native_kl),
        "native_vs_candidate_proxy_gradient_cosine": cosine(native_proxy, candidate_proxy),
        "native_vs_candidate_true_kl_gradient_cosine": cosine(native_kl, candidate_kl),
        "native_to_candidate_proxy_relative_l2": float((candidate_proxy-native_proxy).norm()/native_proxy.norm().clamp_min(1e-30)),
        "native_to_candidate_true_kl_relative_l2": float((candidate_kl-native_kl).norm()/native_kl.norm().clamp_min(1e-30)),
        "scope": "same selected refresh examples and BF16 block output dtype; candidate only computes target Linear matmul in FP32 then casts output back to BF16; paired downstream KL is also same-sample",
    }
    if not math.isfinite(result["candidate_proxy_vs_candidate_true_kl_gradient_cosine"]):
        raise RuntimeError("non-finite gradient alignment")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", action="append", required=True)
    args = parser.parse_args()
    configure_reproducibility(0, deterministic=True)
    torch.set_num_threads(4)
    out_path = args.campaign / "precision_proxy_alignment.json"
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for key in args.target:
        print(f"[precision-alignment] START {key}", flush=True)
        all_results[key] = run_one(args.model, args.campaign, key)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"[precision-alignment] DONE {key}: " + json.dumps(all_results[key]), flush=True)
