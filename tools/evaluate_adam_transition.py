"""Evaluate saved real REAL-Q transitions on fixed WikiText-2 validation windows."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from realq.attention import configure_attention_backend
from realq.refresh.adam_transition_probe import tensor_hash
from realq.refresh.fisher_loss import fisher_mse_loss
from utils import data_utils, model_utils, quant_utils
from utils.loss_utils import tokenwise_kl_from_logits
from utils.reproducibility import configure_reproducibility


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(v, device) for v in value)
    return value


@torch.no_grad()
def surrogate(analyzer, context, refresh):
    layer = analyzer.get_layers()[context["layer"]]
    next_layer = analyzer.get_layers()[context["layer"] + 1]
    values = []
    outputs = []
    for i in refresh["selected"]:
        x = refresh["inps"][i:i+1].cuda()
        output = layer(x, **to_device(refresh["kw"], "cuda"))
        output = output[0] if isinstance(output, tuple) else output
        target = refresh["fp_target"][i:i+1].cuda()
        loss = fisher_mse_loss(output, target, refresh["fisher"].cuda(), a_loss_ratio=refresh["a_loss_ratio"])
        alpha = refresh["slide_alpha"]
        if alpha is not None and alpha < 1:
            nxt = next_layer(output, **to_device(refresh["kw"], "cuda"))
            nxt = nxt[0] if isinstance(nxt, tuple) else nxt
            loss_next = fisher_mse_loss(nxt, refresh["next_fp_target"][i:i+1].cuda(), refresh["next_fisher"].cuda(), a_loss_ratio=refresh["a_loss_ratio"])
            loss = alpha * loss + (1-alpha) * loss_next
        values.append(loss.item())
        outputs.append(output.cpu())
    return {"mean": sum(values)/len(values), "per_sample": values}, torch.cat(outputs)


@torch.no_grad()
def metrics(analyzer, tokens, teacher):
    all_kl, all_nll = [], []
    for i, row in enumerate(tokens):
        hidden = analyzer.model.model(input_ids=row[None].cuda(), use_cache=False).last_hidden_state[0]
        kl_parts, ce_parts = [], []
        # Full-sequence head GEMM matches the project's evaluation convention.
        logits = analyzer.get_lm_head()(hidden).float()
        ref_logits = analyzer.get_lm_head()(teacher[i].cuda()).float()
        for j in range(0, len(row), 32):
            kl_parts.append(tokenwise_kl_from_logits(logits[j:j+32], ref_logits[j:j+32]).double().cpu())
        nll = F.cross_entropy(logits[:-1], row[1:].cuda(), reduction="none").double().mean().item()
        all_kl.append(torch.cat(kl_parts).mean().item())
        all_nll.append(nll)
    return {"kl": sum(all_kl)/len(all_kl), "ppl": math.exp(sum(all_nll)/len(all_nll)),
            "per_window_kl": all_kl, "per_window_nll": all_nll}


def evaluate(model_path, directory, eval_samples=16, eval_seq_len=128):
    configure_reproducibility(0, deterministic=True)
    started = time.monotonic()
    context = torch.load(directory / "context.pt", map_location="cpu", weights_only=False)
    refresh = torch.load(directory / "refresh.pt", map_location="cpu", weights_only=False)
    analyzer = model_utils.ModelAnalyzer(model_path, eval_seq_len, skip_state_dict=True)
    configure_attention_backend(analyzer.model, context["config"]["attention_backend"])
    quant_utils.add_actquant(analyzer)
    analyzer.model.eval().requires_grad_(False).cuda()
    texts = data_utils._get_dataset(analyzer.tokenizer, "wikitext2", "validation")
    encoded = analyzer.tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    tokens = encoded[:eval_samples*eval_seq_len].reshape(eval_samples, eval_seq_len)
    with torch.no_grad():
        teacher = [analyzer.model.model(input_ids=row[None].cuda(), use_cache=False).last_hidden_state[0].cpu() for row in tokens]
        fp_null = metrics(analyzer, tokens[:1], teacher[:1])
        if abs(fp_null["kl"]) > 1e-7:
            raise RuntimeError(f"FP self KL failed: {fp_null['kl']}")
        for i, state in enumerate(context["prefix_layers"]):
            analyzer.get_layers()[i].load_state_dict(state, strict=True)
        analyzer.get_layers()[context["layer"]+1].load_state_dict(context["next_layer"], strict=True)
        module = analyzer.get_layers()[context["layer"]].get_submodule(context["module"])
        weight_paths = ("before_update", "after_update", "after_next_quant")
        weights = {stage: torch.load(directory / f"{stage}.pt", map_location="cpu", weights_only=True) for stage in weight_paths}
        policy = os.environ.get("REALQ_ADAM_PROBE_RULE", "adam")
        result = {"context": {k: context[k] for k in ("layer", "module", "block", "column_end", "quant_state_hashes")},
                  "validation": {"split": "validation", "windows": eval_samples, "seq_len": eval_seq_len, "tokens_sha256": tensor_hash(tokens)},
                  "fp_null_kl": fp_null["kl"], "stages": {}}
        before_output = None
        for stage in weight_paths:
            module.weight.copy_(weights[stage].to(module.weight))
            local, output = surrogate(analyzer, context, refresh)
            measured = metrics(analyzer, tokens, teacher)
            result["stages"][stage] = {"surrogate": local, **measured}
            if before_output is None:
                before_output = output
            result["stages"][stage]["calibration_layer_output_mse_vs_before"] = (output.float()-before_output.float()).square().mean().item()
            print(f"[transition-eval] {directory.parent.name} {stage}: loss={local['mean']:.12g} KL={measured['kl']:.12g} PPL={measured['ppl']:.9g}", flush=True)
        observed_loss = result["stages"]["before_update"]["surrogate"]["mean"]
        result["surrogate_replay"] = {"online": refresh["actual_loss"], "offline": observed_loss,
                                      "absolute_difference": abs(observed_loss-refresh["actual_loss"])}
        if not math.isclose(observed_loss, refresh["actual_loss"], rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(f"surrogate replay mismatch: {result['surrogate_replay']}")
        perm = context["perm"]
        start = context["column_end"]
        raw_d = weights["before_update"] - weights["after_update"]
        actual_d = raw_d[:, perm][:, start:] if perm is not None else raw_d[:, start:]
        g = refresh["raw_gradient"]
        d = refresh["adam_update"]
        gc = refresh["clipped_gradient"]
        mask = d*gc > 0
        alpha = mask.numel()/(mask.sum().item()+1)
        grad_norm = torch.linalg.vector_norm(gc)
        norm_matched_gradient = (
            gc * (torch.linalg.vector_norm(d) / grad_norm)
            if grad_norm.item() > 0
            else torch.zeros_like(gc)
        )
        updates = {
            "adam": d,
            "grams_direction": torch.sign(gc) * d.abs(),
            "norm_matched_gradient": norm_matched_gradient,
            "cautious_mask": d * mask,
            "cautious_scale": d * alpha,
            "cautious": (d * mask) * alpha,
        }
        matched = []
        for name, candidate in updates.items():
            predicted = weights["before_update"][:, perm].clone() if perm is not None else weights["before_update"].clone()
            predicted[:, start:].sub_(candidate)
            predicted = predicted[:, torch.argsort(perm)] if perm is not None else predicted
            if torch.equal(predicted, weights["after_update"]):
                matched.append(name)
        if not matched:
            raise RuntimeError("observed after_update weight does not match any intended policy")
        result["update"] = {"exactly_matched_policies": matched, "alpha": alpha,
            "raw_gradient_dot_actual_delta": float((g.double()*actual_d.double()).sum()),
            "clipped_gradient_dot_actual_delta": float((gc.double()*actual_d.double()).sum()),
            "raw_vs_clipped_gradient_relative_l2": float((g-gc).norm()/g.norm().clamp_min(1e-30)),
            "fp32_weight_changed_fraction": float((weights['before_update']!=weights['after_update']).float().mean()),
            "bf16_weight_changed_fraction": float((weights['before_update'].bfloat16()!=weights['after_update'].bfloat16()).float().mean()),
            "predicted_first_order_loss_change": -float((g.double()*actual_d.double()).sum())}
        result["seconds"] = time.monotonic()-started
        result["peak_allocated"] = torch.cuda.max_memory_allocated()
        result["complete"] = True
        (directory / "evaluation.json").write_text(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--eval-samples", type=int, default=16)
    args = parser.parse_args()
    evaluate(args.model, args.directory, args.eval_samples)
