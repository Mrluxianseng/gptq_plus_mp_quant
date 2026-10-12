"""Compare REAL-Q Fisher, cubic-corrected, and exact downstream-KL FFN gradients.

This is an analysis-only wrapper around analyze_grad_cosine's GPTQ+ reference
quantization hook. It measures at the same post-quantization state and uses the
actual target-block output residual produced by the quantized prefix.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import analyze_grad_cosine as agc
from process_args import parse_gen
from gptq_utils.gptq_plus_utils import (
    compute_refresh_loss,
    hidden2logits,
    slice_layer_output_fisher_for_batch,
    temporary_requires_grad,
)


def _out(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def _tail(y, layers, layer_idx, attn, pos_ids, pos_emb):
    h = y
    for k in range(layer_idx + 1, len(layers)):
        h = _out(layers[k](
            h,
            attention_mask=attn,
            position_ids=pos_ids,
            position_embeddings=pos_emb,
        ))
    return h


def _cosine(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    den = a.norm() * b.norm()
    return float((torch.dot(a, b) / den.clamp_min(1e-30)).item())


def _relerr(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(((a - b).norm() / b.norm().clamp_min(1e-30)).item())


def _accumulate(dst, src):
    for name, value in src.items():
        if value is None:
            continue
        value = value.detach().float()
        dst[name] = dst.get(name, torch.zeros_like(value)) + value


def _sample_gradient(y, grad_y, params):
    names = list(params)
    grads = torch.autograd.grad(
        y,
        [params[name] for name in names],
        grad_outputs=grad_y.to(dtype=y.dtype),
        retain_graph=True,
        allow_unused=False,
    )
    return {name: grad.detach().float() for name, grad in zip(names, grads)}


def _measure(payload, target_idx, measure_samples, batch_size, output_path, args):
    layer = payload["layer"]
    layers = payload["layers"]
    analyzer = payload["analyzer"]
    dev = payload["dev"]
    fisher_all = payload["static_fisher_by_layer"][target_idx]
    if fisher_all is None:
        raise RuntimeError(f"No REAL-Q output Fisher collected for layer {target_idx}.")

    modules = analyzer.get_quantizable_modules(layer)
    named = {}
    for raw_name, module in modules.items():
        name = raw_name[:-7] if raw_name.endswith(".module") else raw_name
        if name in {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}:
            named[name] = module.weight
    expected = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    if set(named) != expected:
        raise RuntimeError(f"Expected dense SwiGLU FFN weights {sorted(expected)}, found {sorted(named)}")

    for k in range(target_idx + 1, len(layers)):
        layers[k].to(dev)

    accum = {kind: {} for kind in ("true", "fisher", "third")}
    out_accum = {kind: torch.zeros(1, device=dev) for kind in ("true", "fisher", "third")}
    residual_sq = delta_sq = 0.0
    kl_vals, g0_norms = [], []
    try:
        with torch.enable_grad(), temporary_requires_grad([analyzer.model], list(named.values())):
            for start in range(0, measure_samples, batch_size):
                end = start + batch_size
                ids = list(range(start, end))
                inp = payload["inps"][start:end].to(dev)
                y0 = payload["fp_inps"][start:end].to(dev)
                teacher_hidden = payload["fp_inps_final"][start:end].to(dev)
                b_attn, b_pos, b_emb = agc._expand_batch_kwargs(
                    payload["attention_mask"], payload["position_ids"],
                    payload["position_embeddings"], batch_size,
                )
                b_attn, b_pos = b_attn.to(dev), b_pos.to(dev)
                b_emb = tuple(x.to(dev) for x in b_emb)

                yq = _out(layer(
                    inp,
                    attention_mask=b_attn,
                    position_ids=b_pos,
                    position_embeddings=b_emb,
                ))
                # Keep the actual quantization/quantized-prefix residual in FP32
                # for the surrogate and symmetric finite-difference construction.
                delta = (yq.detach().float() - y0.detach().float())
                fisher = slice_layer_output_fisher_for_batch(
                    fisher_all, ids, dev, "fisher_diag_mse"
                )
                logits_teacher = hidden2logits(teacher_hidden, analyzer).detach()

                def downstream_loss(y):
                    h = _tail(y, layers, target_idx, b_attn, b_pos, b_emb)
                    logits = hidden2logits(h, analyzer)
                    return F.kl_div(
                        F.log_softmax(logits.float(), dim=-1),
                        F.softmax(logits_teacher.float(), dim=-1),
                        reduction="none",
                    ).sum(dim=-1).mean()

                # Exact downstream KL gradient at the actual quantized output.
                loss_true = downstream_loss(yq)
                g_true = torch.autograd.grad(loss_true, yq, retain_graph=True)[0]
                kl_vals.append(float(loss_true.detach().item()))

                # The production REAL-Q Fisher loss and its exact activation gradient.
                loss_fisher = compute_refresh_loss(
                    refresh_loss_type="fisher_diag_mse",
                    out_hidden=yq,
                    fp_hidden=y0,
                    analyzer=analyzer,
                    kl_topk=0,
                    layer_output_fisher=fisher,
                    a_loss_ratio=args.a_loss_ratio,
                    sink_size=(args.attention_sink_size if args.ignore_attention_sink else 0),
                )
                g_fisher = torch.autograd.grad(loss_fisher, yq, retain_graph=True)[0]

                # Estimate T[delta,delta] from the symmetric second difference
                # of the exact gradient field: g(y0+d)+g(y0-d)-2g(y0).
                # This is matrix-free and captures the third derivative term
                # in the Taylor expansion of the scalar KL.
                def grad_at(point):
                    x = point.detach().to(dtype=yq.dtype).requires_grad_(True)
                    loss = downstream_loss(x)
                    return torch.autograd.grad(loss, x)[0].detach().float()

                g0 = grad_at(y0)
                g0_norms.append(float(g0.norm().item()))
                gp = g_true.detach().float()
                gm = grad_at(y0.detach().float() - delta)
                t_delta = gp + gm - 2.0 * g0
                g_third = g0 + g_fisher.detach().float() + 0.5 * t_delta

                grads_true = _sample_gradient(yq, g_true, named)
                grads_fisher = _sample_gradient(yq, g_fisher, named)
                grads_third = _sample_gradient(yq, g_third, named)
                _accumulate(accum["true"], grads_true)
                _accumulate(accum["fisher"], grads_fisher)
                _accumulate(accum["third"], grads_third)
                for key, grad in (("true", g_true), ("fisher", g_fisher), ("third", g_third)):
                    out_accum[key] += grad.detach().float().square().sum()

                residual_sq += float((yq.detach().float() - y0.detach().float()).square().sum().item())
                delta_sq += float(y0.detach().float().square().sum().item())

                del yq, loss_true, loss_fisher, g_true, g_fisher, g0, gp, gm, t_delta
                torch.cuda.empty_cache()

        # Batch losses/grads are means over tokens; aggregate equal-size batches.
        rows = {}
        for name in sorted(expected):
            true = accum["true"][name]
            fisher_g = accum["fisher"][name]
            third_g = accum["third"][name]
            rows[name] = {
                "true_grad_norm": float(true.norm().item()),
                "fisher_grad_norm": float(fisher_g.norm().item()),
                "third_grad_norm": float(third_g.norm().item()),
                "fisher_cos_true": _cosine(fisher_g, true),
                "third_cos_true": _cosine(third_g, true),
                "fisher_rel_error": _relerr(fisher_g, true),
                "third_rel_error": _relerr(third_g, true),
            }
        result = {
            "layer": target_idx,
            "module_state": "real GPTQ+ W4 quantization after quantized prefix; FP suffix",
            "measure_samples": measure_samples,
            "batch_size": batch_size,
            "kl_topk": 0,
            "a_loss_ratio": args.a_loss_ratio,
            "mean_exact_downstream_kl": sum(kl_vals) / max(1, len(kl_vals)),
            "mean_fp_reference_gradient_norm": sum(g0_norms) / max(1, len(g0_norms)) if g0_norms else 0.0,
            "quantized_output_residual_to_fp_norm_ratio": (residual_sq / max(1.0, delta_sq)) ** 0.5,
            "activation_gradient_norms_sum_sq": {k: float(v.item() ** 0.5) for k, v in out_accum.items()},
            "modules": rows,
            "third_order_estimator": "g(y0+d)+g(y0-d)-2g(y0), then 0.5*T[d,d]",
            "caveat": "The symmetric gradient difference estimates the cubic Taylor gradient correction plus fourth-and-higher-order remainder; it is not an exact third derivative tensor contraction.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logging.info("THIRD_ORDER_REAL_QUANT_RESULT %s", json.dumps(result, sort_keys=True))
    finally:
        for k in range(target_idx + 1, len(layers)):
            layers[k] = layers[k].to(payload["orig_device"])


def main():
    args = parse_gen()
    # parse_gen's validation happens in analyze_grad_cosine.main. This wrapper
    # intentionally supports one target layer.
    target_ids = [int(x.strip()) for x in args.target_layers.split(",")]
    if len(target_ids) != 1:
        raise ValueError("Specify exactly one --target_layers entry for this probe.")
    target_idx = target_ids[0]
    output_path = Path(getattr(args, "third_order_output", None) or Path(args.output_dir) / "third_order_gradient_compare.json")

    def patched_builder(**kwargs):
        def hook(event, payload):
            if event == "before_layer":
                return payload["layer_idx"] == target_idx
            if event == "after_layer_quantized" and payload["layer_idx"] == target_idx:
                _measure(
                    payload,
                    target_idx,
                    args.measure_samples,
                    args.measure_batch_size,
                    output_path,
                    args,
                )
            return None

        return hook, (lambda: None)

    agc._build_cosine_analysis_hook = patched_builder
    # The existing diagnostic path turns on static Fisher collection when this
    # loss is requested; our added comparison uses the actual cached tensor.
    if "fisher_diag_mse" not in args.measure_losses:
        args.measure_losses = "fisher_diag_mse"
    agc.main(args)


if __name__ == "__main__":
    main()
