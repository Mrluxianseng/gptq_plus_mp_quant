"""Check whether Fisher feedback matches downstream KL only near the FP point.

For saved refresh inputs, interpolate the target block output from its saved
FP teacher activation toward the current quantized activation. At each point,
compare the Fisher refresh dL/dy with the true paired KL dL/dy under a smooth
FP32 suffix. This separates local-curvature mismatch from large-residual
nonlinearity; it is diagnostic only.
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


def move_float(value):
    if isinstance(value, torch.Tensor):
        value = value.cuda()
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple): return tuple(move_float(x) for x in value)
    if isinstance(value, list): return [move_float(x) for x in value]
    if isinstance(value, dict): return {k: move_float(v) for k, v in value.items()}
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


def run_one(model_path, campaign, key, alphas):
    summary = json.loads((campaign / "summary.json").read_text())
    arm = summary["results"][key]["adam"]
    d = Path(arm["directory"]) / "transition"
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
    for block in layers[context["layer"]:]: block.float()
    analyzer.get_layernorm_before_head().float()
    analyzer.get_lm_head().float()
    target_layer = layers[context["layer"]]
    target = target_layer.get_submodule(context["module"])
    target.weight.copy_(before.to(device="cuda", dtype=torch.float32))
    if hasattr(target, "module") and hasattr(target.module, "weight"):
        target.module.weight = target.weight
    suffix = layers[context["layer"] + 1:]
    batch = refresh["inps"].shape[0]
    collected = {str(a): {"proxy": [], "true": [], "cos_per_sample": [], "kl": []} for a in alphas}

    for i in refresh["selected"]:
        kw = move_float(select(refresh["kw"], i, batch))
        x = refresh["inps"][i:i + 1].cuda().float()
        with torch.no_grad():
            student_y = hidden(target_layer(x, **kw)).detach()
            teacher_y = refresh["fp_target"][i:i + 1].cuda().float()
            teacher_h = teacher_y
            for block in suffix: teacher_h = hidden(block(teacher_h, **kw))
            teacher_logits = analyzer.get_lm_head()(
                analyzer.get_layernorm_before_head()(teacher_h)
            ).float().detach()

        for alpha_out in alphas:
            y_value = teacher_y + float(alpha_out) * (student_y - teacher_y)
            y = y_value.detach().requires_grad_(True)
            loss_current = fisher_mse_loss(
                y, teacher_y, refresh["fisher"].cuda(),
                a_loss_ratio=float(refresh["a_loss_ratio"]),
                a_loss_threshold=refresh["a_loss_threshold"],
            )
            slide = refresh["slide_alpha"]
            if slide is not None and slide < 1.0:
                next_y = hidden(layers[context["layer"] + 1](y, **kw))
                next_fp = refresh["next_fp_target"][i:i + 1].cuda().float()
                loss_next = fisher_mse_loss(
                    next_y, next_fp, refresh["next_fisher"].cuda(),
                    a_loss_ratio=float(refresh["a_loss_ratio"]),
                    a_loss_threshold=refresh["next_a_loss_threshold"],
                )
                loss_proxy = float(slide) * loss_current + (1.0-float(slide)) * loss_next
            else:
                loss_proxy = loss_current
            h = y
            for block in suffix: h = hidden(block(h, **kw))
            logits = analyzer.get_lm_head()(analyzer.get_layernorm_before_head()(h)).float()
            loss_true = tokenwise_kl_from_logits(logits, teacher_logits).mean()
            gp, = torch.autograd.grad(loss_proxy, y, retain_graph=True)
            gt, = torch.autograd.grad(loss_true, y)
            gp, gt = gp.detach().float(), gt.detach().float()
            box = collected[str(alpha_out)]
            box["proxy"].append(gp.cpu())
            box["true"].append(gt.cpu())
            box["cos_per_sample"].append(cosine(gp, gt))
            box["kl"].append(float(loss_true.detach().item()))
            del y, loss_proxy, loss_true, gp, gt, logits, h

    result = {"target": key, "scales": {},
              "scope": "interpolates saved FP block output to actual student block output; smooth FP32 suffix; same selected samples"}
    for scale, box in collected.items():
        gp, gt = torch.cat(box["proxy"], 0).double(), torch.cat(box["true"], 0).double()
        result["scales"][scale] = {
            "output_gradient_cosine": cosine(gp.float(), gt.float()),
            "proxy_to_true_gradient_rms_ratio": float(gp.square().mean().sqrt()/gt.square().mean().sqrt().clamp_min(1e-30)),
            "gradient_relative_l2": float((gp-gt).norm()/gt.norm().clamp_min(1e-30)),
            "mean_per_sample_cosine": sum(box["cos_per_sample"])/len(box["cos_per_sample"]),
            "mean_paired_kl": sum(box["kl"])/len(box["kl"]),
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--target", action="append", required=True)
    p.add_argument("--scales", default="0.1,0.25,0.5,1.0")
    a = p.parse_args()
    configure_reproducibility(0, deterministic=True)
    torch.set_num_threads(4)
    scales = [float(x) for x in a.scales.split(",")]
    out_path = a.campaign / "fisher_locality_path.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for key in a.target:
        print(f"[fisher-locality] START {key}", flush=True)
        results[key] = run_one(a.model, a.campaign, key, scales)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[fisher-locality] DONE {key}: " + json.dumps(results[key]), flush=True)
