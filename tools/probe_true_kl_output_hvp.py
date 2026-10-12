"""Compare empirical-Fisher feedback with exact output-space KL HVP.

Offline diagnostic only. Computes H(y_fp) @ (y_student-y_fp) by double
autograd through the actual smooth FP32 downstream transformer suffix, then
compares it with the exact paired-KL output gradient at y_student. No training
or quantization code is changed.
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
from utils import model_utils, quant_utils
from utils.loss_utils import tokenwise_kl_from_logits
from utils.reproducibility import configure_reproducibility


def hidden(x):
    return x[0] if isinstance(x, tuple) else x


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
    return float(F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).clamp(-1, 1))


def run_one(model_path, campaign, key, fd_check=False, token_block_decomp=False,
            decomp_samples=2, save_vectors=False):
    arm = json.loads((campaign / "summary.json").read_text())["results"][key]
    d = Path(arm["adam"]["directory"]) / "transition"
    context = torch.load(d / "context.pt", map_location="cpu", weights_only=False)
    refresh = torch.load(d / "refresh.pt", map_location="cpu", weights_only=False)
    before = torch.load(d / "before_update.pt", map_location="cpu", weights_only=True)

    analyzer = model_utils.ModelAnalyzer(model_path, 64, skip_state_dict=True)
    configure_attention_backend(analyzer.model, context["config"]["attention_backend"])
    quant_utils.add_actquant(analyzer)
    analyzer.model.eval().requires_grad_(False).cuda()
    layers = analyzer.get_layers()
    for i, state in enumerate(context["prefix_layers"]):
        layers[i].load_state_dict(state, strict=True)
    layers[context["layer"] + 1].load_state_dict(context["next_layer"], strict=True)

    # Smooth FP32 reference path; it isolates the curvature approximation
    # from BF16's discrete rounding and permits a meaningful double backward.
    for block in layers[context["layer"]:]:
        block.float()
    analyzer.get_layernorm_before_head().float()
    analyzer.get_lm_head().float()
    layer = layers[context["layer"]]
    module = layer.get_submodule(context["module"])
    module.weight.copy_(before.to(device="cuda", dtype=torch.float32))
    if hasattr(module, "module") and hasattr(module.module, "weight"):
        module.module.weight = module.weight
    suffix = layers[context["layer"] + 1:]
    batch = refresh["inps"].shape[0]
    f_emp = refresh["fisher"].cuda().double()

    true_rows, hvp_rows, delta_rows, empirical_rows = [], [], [], []
    exact_kls, teacher_grad_norms = [], []
    fd_cosines, fd_relative_errors = [], []
    selected = refresh["selected"][:int(decomp_samples)] if token_block_decomp else refresh["selected"]
    blockdiag_cosines, cross_rms_ratios = [], []
    for i in selected:
        kw = tree_map(select(refresh["kw"], i, batch),
                      lambda x: x.cuda().float() if x.is_floating_point() else x.cuda())
        x = refresh["inps"][i:i + 1].cuda().float()
        y0 = refresh["fp_target"][i:i + 1].cuda().float().detach()
        with torch.no_grad():
            h = y0
            for block in suffix: h = hidden(block(h, **kw))
            teacher_logits = analyzer.get_lm_head()(
                analyzer.get_layernorm_before_head()(h)
            ).float().detach()
            ys = hidden(layer(x, **kw)).detach()

        # Exact KL gradient at the current student output.
        ys_leaf = ys.detach().requires_grad_(True)
        h = ys_leaf
        for block in suffix: h = hidden(block(h, **kw))
        student_logits = analyzer.get_lm_head()(
            analyzer.get_layernorm_before_head()(h)
        ).float()
        kl = tokenwise_kl_from_logits(student_logits, teacher_logits).mean()
        (g_student,) = torch.autograd.grad(kl, ys_leaf)

        # Exact Hessian-vector product at the FP teacher output.
        delta = (ys - y0).detach()
        y0_leaf = y0.detach().requires_grad_(True)
        h0 = y0_leaf
        for block in suffix: h0 = hidden(block(h0, **kw))
        logits0 = analyzer.get_lm_head()(
            analyzer.get_layernorm_before_head()(h0)
        ).float()
        kl0 = tokenwise_kl_from_logits(logits0, teacher_logits).mean()
        (g0,) = torch.autograd.grad(kl0, y0_leaf, create_graph=True)
        (hvp,) = torch.autograd.grad(
            g0, y0_leaf, grad_outputs=delta,
            retain_graph=token_block_decomp,
        )

        if token_block_decomp:
            block_size = 8
            blockdiag = torch.zeros_like(hvp)
            for start in range(0, int(delta.shape[1]), block_size):
                end = min(start + block_size, int(delta.shape[1]))
                mask = torch.zeros_like(delta)
                mask[:, start:end, :] = 1
                (hvp_from_block,) = torch.autograd.grad(
                    g0, y0_leaf, grad_outputs=delta * mask,
                    retain_graph=True,
                )
                blockdiag.add_(hvp_from_block * mask)
                del mask, hvp_from_block
            blockdiag_cosines.append(cosine(blockdiag, g_student.detach().float()))
            cross_rms_ratios.append(float(
                (hvp - blockdiag).square().mean().sqrt()
                / hvp.square().mean().sqrt().clamp_min(1e-30)
            ))
            del blockdiag

        if fd_check:
            directional_derivative = []
            for sign in (-1.0, 1.0):
                y_shift = (y0 + sign * 0.1 * delta).detach().requires_grad_(True)
                h_shift = y_shift
                for block in suffix: h_shift = hidden(block(h_shift, **kw))
                logits_shift = analyzer.get_lm_head()(
                    analyzer.get_layernorm_before_head()(h_shift)
                ).float()
                loss_shift = tokenwise_kl_from_logits(
                    logits_shift, teacher_logits
                ).mean()
                (grad_shift,) = torch.autograd.grad(loss_shift, y_shift)
                directional_derivative.append(grad_shift.detach().float())
            fd_hvp = (directional_derivative[1] - directional_derivative[0]) / 0.2
            fd_cosines.append(cosine(fd_hvp, hvp.detach().float()))
            fd_relative_errors.append(float(
                (fd_hvp - hvp.detach().float()).norm()
                / hvp.detach().float().norm().clamp_min(1e-30)
            ))
            del y_shift, h_shift, logits_shift, loss_shift, grad_shift, fd_hvp

        true_rows.append(g_student.detach().float().cpu())
        hvp_rows.append(hvp.detach().float().cpu())
        delta_rows.append(delta.detach().float().cpu())
        empirical_rows.append(torch.einsum("ij,btj->bti", f_emp, delta.double()).float().cpu())
        exact_kls.append(float(kl.detach()))
        teacher_grad_norms.append(float(g0.detach().norm()))
        del ys_leaf, y0_leaf, h, h0, logits0, student_logits, kl, kl0, g0, g_student, hvp

    gt = torch.cat(true_rows).double()
    gh = torch.cat(hvp_rows).double()
    delta = torch.cat(delta_rows).double()
    ge = torch.cat(empirical_rows).double()
    # Match mean-over-token output-gradient scale; saved Fisher feedback in the
    # alignment probe was divided by total token count (batch * sequence).
    norm = float(delta.shape[0] * delta.shape[1])
    ge /= norm
    out = {
        "target": key,
        "samples": selected,
        "full_refresh_sample_indices": refresh["selected"],
        "path": "smooth FP32 target block/suffix; exact full-sequence KL Hessian-vector product at FP target output",
        "paired_kl_mean": sum(exact_kls) / len(exact_kls),
        "teacher_point_gradient_norm_max": max(teacher_grad_norms),
        "token_block_decomposition": {
            "token_chunk_size": 8 if token_block_decomp else None,
            "heldout_sequences": len(blockdiag_cosines),
            "block_diagonal_hvp_vs_true_gradient_cosine_mean": (
                sum(blockdiag_cosines) / len(blockdiag_cosines) if blockdiag_cosines else None
            ),
            "off_block_hvp_rms_fraction_mean": (
                sum(cross_rms_ratios) / len(cross_rms_ratios) if cross_rms_ratios else None
            ),
        },
        "hvp_finite_difference_step_fraction": 0.1 if fd_check else None,
        "hvp_fd_cosine_mean": sum(fd_cosines) / len(fd_cosines) if fd_cosines else None,
        "hvp_fd_relative_l2_mean": sum(fd_relative_errors) / len(fd_relative_errors) if fd_relative_errors else None,
        "exact_hvp_vs_student_kl_gradient_cosine": cosine(gh.float(), gt.float()),
        "exact_hvp_to_student_gradient_rms_ratio": float(gh.square().mean().sqrt() / gt.square().mean().sqrt().clamp_min(1e-30)),
        "exact_hvp_relative_l2_to_student_gradient": float((gh-gt).norm()/gt.norm().clamp_min(1e-30)),
        "empirical_fisher_feedback_vs_student_kl_gradient_cosine": cosine(ge.float(), gt.float()),
        "empirical_fisher_to_student_gradient_rms_ratio": float(ge.square().mean().sqrt()/gt.square().mean().sqrt().clamp_min(1e-30)),
        "empirical_fisher_vs_exact_hvp_cosine": cosine(ge.float(), gh.float()),
        "scope_note": "HVP is exact for the smooth FP32 local Hessian; it is a diagnostic upper-bound reference, not a deployable low-cost estimator. Its comparison tests whether Taylor truncation or Fisher curvature is the main mismatch.",
    }
    if save_vectors:
        vector_path = campaign / f"{key.replace('/', '_')}_true_kl_vectors.pt"
        torch.save({
            "delta": torch.cat(delta_rows, dim=0),
            "true_kl_gradient": torch.cat(true_rows, dim=0),
            "exact_kl_hvp": torch.cat(hvp_rows, dim=0),
            "samples": selected,
            "target": key,
        }, vector_path)
        out["saved_vectors"] = str(vector_path)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--target", action="append", required=True)
    p.add_argument("--fd-check", action="store_true",
                   help="verify HVP by central finite differences of the output KL gradient")
    p.add_argument("--token-block-decomp", action="store_true",
                   help="split two examples into 8-token groups and measure block-diagonal vs cross-token HVP")
    p.add_argument("--decomp-samples", type=int, default=2,
                   help="number of refresh examples for token block decomposition")
    p.add_argument("--save-vectors", action="store_true",
                   help="save per-sequence delta, true KL gradient, and exact HVP tensors")
    a = p.parse_args()
    configure_reproducibility(0, deterministic=True)
    torch.set_num_threads(4)
    path = a.campaign / "true_kl_output_hvp.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for key in a.target:
        print(f"[true-kl-hvp] START {key}", flush=True)
        result_key = (f"{key}-tokenblock8-n{a.decomp_samples}"
                      if a.token_block_decomp else key)
        results[result_key] = run_one(
            a.model, a.campaign, key, fd_check=a.fd_check,
            token_block_decomp=a.token_block_decomp,
            decomp_samples=a.decomp_samples,
            save_vectors=a.save_vectors,
        )
        path.write_text(json.dumps(results, indent=2))
        print(f"[true-kl-hvp] DONE {result_key}: " + json.dumps(results[result_key]), flush=True)
