"""Pilot an output-space model-predictive Fisher for REAL-Q feedback.

At saved refresh states, estimate E_p[g g^T] by sampling next-token labels
from the FP teacher distribution, then compare F_model @ delta_y against the
paired downstream KL gradient. This is a single small offline pilot, not a
production change; the estimator is intentionally labeled Monte Carlo.
"""
from __future__ import annotations

import argparse
import json
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


def run_one(model_path, campaign, key, seed=0):
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
    for block in layers[context["layer"]:]: block.float()
    analyzer.get_layernorm_before_head().float(); analyzer.get_lm_head().float()
    target_layer = layers[context["layer"]]
    target = target_layer.get_submodule(context["module"])
    target.weight.copy_(before.to(device="cuda", dtype=torch.float32))
    if hasattr(target, "module") and hasattr(target.module, "weight"):
        target.module.weight = target.weight
    suffix = layers[context["layer"] + 1:]
    batch = refresh["inps"].shape[0]
    hidden_size = refresh["fp_target"].shape[-1]
    f_model = torch.zeros((hidden_size, hidden_size), dtype=torch.float64, device="cuda")
    # Diagnostic low-memory alternative: retain only each output position's
    # hidden-coordinate diagonal. This costs O(T*H), not O(H^2) per position.
    f_position_diag = torch.zeros(
        (int(refresh["fp_target"].shape[1]), hidden_size),
        dtype=torch.float64, device="cuda",
    )
    f_emp = refresh["fisher"].cuda().double()
    total_tokens = 0
    delta_rows, true_grad_rows, paired_kls = [], [], []
    sequence_score_rows = []
    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    for i in refresh["selected"]:
        kw = tree_map(select(refresh["kw"], i, batch),
                      lambda x: x.cuda().float() if x.is_floating_point() else x.cuda())
        y_fp = refresh["fp_target"][i:i + 1].cuda().float()
        with torch.no_grad():
            h = y_fp
            for block in suffix: h = hidden(block(h, **kw))
            teacher_logits = analyzer.get_lm_head()(
                analyzer.get_layernorm_before_head()(h)
            ).float()
            sampled = torch.multinomial(
                teacher_logits[:, :-1].reshape(-1, teacher_logits.shape[-1]).softmax(-1),
                1, replacement=True, generator=generator,
            ).reshape(1, -1)

        # Score gradients of sampled teacher tokens form one Monte Carlo
        # empirical estimate of the model Fisher at the FP teacher point.
        y_score = y_fp.detach().requires_grad_(True)
        h_score = y_score
        for block in suffix: h_score = hidden(block(h_score, **kw))
        score_logits = analyzer.get_lm_head()(
            analyzer.get_layernorm_before_head()(h_score)
        ).float()
        score_loss = F.cross_entropy(
            score_logits[:, :-1].reshape(-1, score_logits.shape[-1]),
            sampled.reshape(-1), reduction="sum",
        )
        (score_grad,) = torch.autograd.grad(score_loss, y_score)
        g = score_grad.detach().float().reshape(-1, hidden_size).double()
        sequence_score_rows.append(g.reshape(1, -1).cpu())
        f_model.add_(g.T @ g)
        f_position_diag.add_(g.reshape(-1, hidden_size).square())
        total_tokens += g.shape[0]
        del y_score, h_score, score_logits, score_loss, score_grad, g

        # Actual student residual and exact paired KL output feedback.
        x = refresh["inps"][i:i + 1].cuda().float()
        with torch.no_grad():
            y_student = hidden(target_layer(x, **kw)).detach()
        y_leaf = y_student.detach().requires_grad_(True)
        h_student = y_leaf
        for block in suffix: h_student = hidden(block(h_student, **kw))
        student_logits = analyzer.get_lm_head()(
            analyzer.get_layernorm_before_head()(h_student)
        ).float()
        loss_kl = tokenwise_kl_from_logits(student_logits, teacher_logits).mean()
        (true_grad,) = torch.autograd.grad(loss_kl, y_leaf)
        delta_rows.append((y_student-y_fp).detach().float().cpu())
        true_grad_rows.append(true_grad.detach().float().cpu())
        paired_kls.append(float(loss_kl.detach().item()))
        del y_leaf, h_student, student_logits, loss_kl, true_grad

    f_model.div_(float(total_tokens))
    f_position_diag.div_(float(len(refresh["selected"])))
    delta = torch.cat(delta_rows, 0).double().cuda()
    true_grad = torch.cat(true_grad_rows, 0).double().cuda()
    sequence_scores = torch.cat(sequence_score_rows, 0).double().cuda()
    normalization = float(delta.shape[0] * delta.shape[1])
    feedback_emp = torch.einsum("ij,btj->bti", f_emp, delta) / normalization
    feedback_model = torch.einsum("ij,btj->bti", f_model, delta) / normalization
    # Same mean-loss normalization as the shared Fisher loss. Position-wise
    # diagonal preserves the refresh token slot but drops hidden cross terms.
    feedback_position_diag = delta * f_position_diag[None, :, :]
    feedback_position_diag /= normalization
    # Held-out sequence-level low-rank sketch: fit score-gradient factors on
    # even refresh examples, then apply them to odd examples. This is a
    # mechanism screen for a bounded-rank sequence-space operator, not a
    # production estimator; factors are deliberately not fitted on eval rows.
    train_ids = torch.arange(0, sequence_scores.shape[0], 2, device="cuda")
    test_ids = torch.arange(1, sequence_scores.shape[0], 2, device="cuda")
    sequence_delta = torch.cat(delta_rows, 0).double().cuda().reshape(sequence_scores.shape[0], -1)
    sequence_true = torch.cat(true_grad_rows, 0).double().cuda().reshape(sequence_scores.shape[0], -1)
    sketch_metrics = {}
    for rank in (1, 2, 4):
        factors = sequence_scores[train_ids[:rank]]
        pred = (sequence_delta[test_ids] @ factors.T) @ factors
        pred /= float(rank * refresh["fp_target"].shape[1])
        truth = sequence_true[test_ids]
        sketch_metrics[str(rank)] = {
            "heldout_sequence_count": int(test_ids.numel()),
            "cosine": cosine(pred, truth),
            "rms_ratio": float(pred.square().mean().sqrt()/truth.square().mean().sqrt().clamp_min(1e-30)),
            "per_sequence_cosine": [cosine(pred[j], truth[j]) for j in range(pred.shape[0])],
        }
    # Separate token pooling from loss of within-hidden correlations. The
    # even refresh examples estimate (a) one shared HxH matrix and (b) one
    # full HxH matrix per token slot; odd examples are held out. All three
    # operators use the same score-gradient factors and mean-token scaling.
    train_g = sequence_scores[train_ids].reshape(-1, *delta.shape[1:])
    test_delta = torch.cat(delta_rows, 0).double().cuda()[test_ids]
    test_true = torch.cat(true_grad_rows, 0).double().cuda()[test_ids]
    r_train, n_tokens, hsize = train_g.shape
    f_shared_cv = torch.einsum("rth,rtk->hk", train_g, train_g) / float(r_train*n_tokens)
    feedback_shared_cv = torch.einsum("hk,vtk->vth", f_shared_cv, test_delta) / float(n_tokens)
    position_dot = torch.einsum("rth,vth->vrt", train_g, test_delta)
    feedback_position_full_cv = torch.einsum("rth,vrt->vth", train_g, position_dot) / float(r_train*n_tokens)
    sequence_dot = torch.einsum("rd,vd->vr", train_g.reshape(r_train, -1), test_delta.reshape(test_delta.shape[0], -1))
    feedback_sequence_cv = torch.einsum("rd,vr->vd", train_g.reshape(r_train, -1), sequence_dot) / float(r_train*n_tokens)
    feedback_sequence_cv = feedback_sequence_cv.reshape_as(test_true)
    token_structure_cv = {
        "shared_full_hidden": {
            "cosine": cosine(feedback_shared_cv, test_true),
            "rms_ratio": float(feedback_shared_cv.square().mean().sqrt()/test_true.square().mean().sqrt().clamp_min(1e-30)),
        },
        "position_specific_full_hidden": {
            "cosine": cosine(feedback_position_full_cv, test_true),
            "rms_ratio": float(feedback_position_full_cv.square().mean().sqrt()/test_true.square().mean().sqrt().clamp_min(1e-30)),
        },
        "full_sequence_lowrank": {
            "cosine": cosine(feedback_sequence_cv, test_true),
            "rms_ratio": float(feedback_sequence_cv.square().mean().sqrt()/test_true.square().mean().sqrt().clamp_min(1e-30)),
        },
        "train_refresh_indices": [int(x) for x in train_ids.cpu().tolist()],
        "heldout_refresh_indices": [int(x) for x in test_ids.cpu().tolist()],
        "monte_carlo_score_draws_per_operator": int(r_train),
    }
    result = {
        "target": key,
        "sampled_label_seed": seed,
        "sampled_model_fisher_tokens": total_tokens,
        "empirical_fisher_feedback_vs_true_kl_cosine": cosine(feedback_emp, true_grad),
        "predictive_fisher_feedback_vs_true_kl_cosine": cosine(feedback_model, true_grad),
        "position_diagonal_predictive_feedback_vs_true_kl_cosine": cosine(feedback_position_diag, true_grad),
        "empirical_feedback_to_true_rms_ratio": float(feedback_emp.square().mean().sqrt()/true_grad.square().mean().sqrt().clamp_min(1e-30)),
        "predictive_feedback_to_true_rms_ratio": float(feedback_model.square().mean().sqrt()/true_grad.square().mean().sqrt().clamp_min(1e-30)),
        "position_diagonal_predictive_feedback_to_true_rms_ratio": float(feedback_position_diag.square().mean().sqrt()/true_grad.square().mean().sqrt().clamp_min(1e-30)),
        "empirical_vs_predictive_feedback_cosine": cosine(feedback_emp, feedback_model),
        "empirical_vs_position_diagonal_feedback_cosine": cosine(feedback_emp, feedback_position_diag),
        "heldout_sequence_lowrank_feedback": sketch_metrics,
        "heldout_token_aggregation_ablation": token_structure_cv,
        "paired_kl_mean": sum(paired_kls)/len(paired_kls),
        "feedback_normalization": "divide by the number of sample-token positions to match mean-loss gradient scale",
        "scope": "FP32 smooth target block/suffix; sampled next-token labels from teacher predictive distribution; compares shared Fisher, O(T*H) position-aware diagonal, and held-out low-rank sequence score sketch; pilot only",
    }
    # Report alignment; not a replacement-quality claim. F_model is noisy at
    # this sample count and must be replicated before any further use.
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--target", action="append", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    configure_reproducibility(0, deterministic=True)
    torch.set_num_threads(4)
    path = a.campaign / "predictive_fisher_feedback.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for key in a.target:
        print(f"[predictive-fisher] START {key}", flush=True)
        result_key = f"{key}-predictive-seed{a.seed}"
        results[result_key] = run_one(a.model, a.campaign, key, a.seed)
        path.write_text(json.dumps(results, indent=2))
        print(f"[predictive-fisher] DONE {result_key}: " + json.dumps(results[result_key]), flush=True)
