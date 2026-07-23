# coding=utf-8
"""Gradient cosine diagnostic for Fisher-matrix / residual_kl surrogates.

Runs a reference quantization pass selected by `--analysis_quant_method`
(default: RTN) and, after each transformer block in `--target_layers` is
quantized, measures the cosine similarity between:
  * true KL gradient wrt this layer's linear weights (end-to-end backward)
  * Fisher-matrix surrogate gradient
  * residual_kl  surrogate gradient
  * layer-output MSE surrogate gradient
  * per-linear-output MSE surrogate gradient

Per target layer, batches of `--measure_batch_size` samples are averaged inside
each backward; cosine is computed per linear (q/k/v/o/gate/up/down_proj) then
averaged across batches. Measurement runs immediately after the target layer is
quantized: upstream layers and the target layer are quantized, downstream layers
are still FP.

The production GPTQ+ path only sees a default-off analysis hook. The RTN
reference path is implemented inside this diagnostic. Normal quantization runs
do not collect these diagnostic gradients.
"""

import os
import logging
import pprint

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from accelerate.hooks import remove_hook_from_module
from tqdm import tqdm

from process_args import parse_gen
from utils import (
    data_utils,
    dist_utils,
    model_utils,
    quant_utils,
    rotation_utils,
    memory_utils,
)
from utils.reproducibility import configure_reproducibility
from gptq_utils.gptq_plus_utils import gptq_fwrd as gptq_plus_fwrd
from gptq_utils.gptq_plus_utils import (
    clip_module_weight_to_quant_bounds_,
    collect_layer_output_grad_for_refined_mse,
    collect_static_end_to_end_saliency_and_fisher,
    compute_refresh_loss,
    hidden2logits,
    slice_layer_output_fisher_for_batch,
    temporary_requires_grad,
    _scale_delta_by_abs_quantile,
)
from gptq_utils.gptaq_utils import GPTAQ, FPInputsCache
from gptq_utils.quant_aware_utils import (
    configure_activation_quantizers_for_gptq,
    configure_k_cache_quantizers_for_gptq,
    disable_fp_path_quant,
)

torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------------------------
# target layer parsing
# ---------------------------------------------------------------------------

_VALID_MEASURE_LOSSES = {
    "fisher_diag_mse",
    "legacy_fisher_diag_mse",
    "residual_kl",
    "refined_residual_kl",
    "refined_diag_residual_kl",
    "refined_mse",
    "layer_mse",
    "module_mse",
}

_LOSS_REPORT_ORDER = (
    "fisher_diag_mse",
    "legacy_fisher_diag_mse",
    "residual_kl",
    "refined_residual_kl",
    "refined_diag_residual_kl",
    "refined_mse",
    "layer_mse",
    "module_mse",
)


def _parse_measure_losses(spec: str):
    """Parse --measure_losses into a set. Rejects unknown names explicitly so a
    typo doesn't silently drop a loss from the report."""
    if spec is None or spec.strip() == "":
        return set()
    names = {chunk.strip() for chunk in spec.split(",") if chunk.strip()}
    unknown = names - _VALID_MEASURE_LOSSES
    if unknown:
        raise ValueError(
            f"--measure_losses contains unknown names {sorted(unknown)}. "
            f"Valid choices: {sorted(_VALID_MEASURE_LOSSES)}."
        )
    return names


def _parse_target_layers(spec: str, num_layers: int):
    """Parse --target_layers."""
    if spec is None or spec.strip() == "":
        return []
    if spec.strip().lower() == "all":
        ids = list(range(num_layers))
    else:
        ids = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            idx = int(chunk)
            if not (0 <= idx < num_layers):
                raise ValueError(
                    f"target_layers contains {idx} outside [0, {num_layers})."
                )
            ids.append(idx)
    ids = sorted(set(ids))
    return ids


# ---------------------------------------------------------------------------
# measurement core
# ---------------------------------------------------------------------------

def _layer_out(out):
    """Qwen3DecoderLayer returns bare tensor; Llama returns (hidden, attn_weights).
    Unwrap uniformly. Using `out[0]` indiscriminately would index batch dim for Qwen3."""
    return out[0] if isinstance(out, (tuple, list)) else out


def _forward_layer_and_tail(layer, layer_idx, layers, h,
                            attention_mask, position_ids, position_embeddings):
    """h -> target_layer (with grad) -> downstream (FP, no grad) -> final hidden."""
    h = _layer_out(layer(
        h,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    ))
    for k in range(layer_idx + 1, len(layers)):
        h = _layer_out(layers[k](
            h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        ))
    return h


def _expand_batch_kwargs(attention_mask, position_ids, position_embeddings, bsz):
    """Expand BS=1 captured kwargs to match measurement batch size (gptq_plus pattern)."""
    return (
        attention_mask.expand(bsz, -1, -1, -1),
        position_ids.expand(bsz, -1),
        (
            position_embeddings[0].expand(bsz, -1, -1),
            position_embeddings[1].expand(bsz, -1, -1),
        ),
    )


def _zero_grads(param_list):
    for p in param_list:
        if p.grad is not None:
            p.grad.zero_()


def _capture_grads(name_to_weight, grad_clip=None):
    """Copy `.grad` off each weight into a fresh fp32 tensor.

    If `grad_clip` is provided and positive, apply the same element-wise clamp
    that block_gd does in `apply_dense_optimizer_step` (see gptq_plus_utils.py).
    This makes the measured cosine/norm reflect what the optimizer actually
    sees, not the raw autograd output — useful when grad_clip is a meaningful
    hyper-parameter in production runs (default 1.0 is a wide permissive band;
    small values like 5e-6 used in the lr sweep aggressively reshape grads)."""
    grads = {}
    for name, w in name_to_weight.items():
        if w.grad is None:
            raise RuntimeError(f"weight `{name}` received no gradient.")
        g = w.grad.detach().float().clone()
        if grad_clip is not None and grad_clip > 0:
            g.clamp_(min=-grad_clip, max=grad_clip)
        grads[name] = g
    return grads


def _capture_one_grad(weight, name, grad_clip=None):
    if weight.grad is None:
        raise RuntimeError(f"weight `{name}` received no gradient.")
    g = weight.grad.detach().float().clone()
    if grad_clip is not None and grad_clip > 0:
        g.clamp_(min=-grad_clip, max=grad_clip)
    return g


def _cosine_per_linear(grads_true, grads_surrogate):
    out = {}
    for name, g_true in grads_true.items():
        g_sur = grads_surrogate[name]
        out[name] = F.cosine_similarity(
            g_true.flatten().unsqueeze(0),
            g_sur.flatten().unsqueeze(0),
            dim=1,
        ).item()
    return out


def _summarize_loss_series(per_loss_values):
    summary = {}
    for loss_name, values in per_loss_values.items():
        if not values:
            continue
        t = torch.tensor(values, dtype=torch.float32)
        summary[loss_name] = {
            "mean": t.mean().item(),
            "std": t.std(unbiased=False).item() if len(t) > 1 else 0.0,
            "n_batches": int(t.numel()),
            "per_batch": t.tolist(),
        }
    return summary


def _analysis_loss_value(loss_name, loss_tensor, batch_size):
    return loss_tensor.item()


def _tensor_stats(values):
    t = torch.tensor(values, dtype=torch.float32)
    return (
        t.mean().item(),
        t.std(unbiased=False).item() if t.numel() > 1 else 0.0,
    )


_ENTRY_SERIES_TO_STATS = {
    "per_batch_true_kl_grad_norm": ("true_kl_grad_norm_mean", None),
    "per_batch_reg_cos": ("reg_cos_mean", "reg_cos_std"),
    "per_batch_fisher": ("fisher_mean", "fisher_std"),
    "per_batch_fisher_grad_norm": ("fisher_grad_norm_mean", None),
    "per_batch_legacy_fisher_diag_mse": (
        "legacy_fisher_diag_mse_mean",
        "legacy_fisher_diag_mse_std",
    ),
    "per_batch_legacy_fisher_diag_mse_grad_norm": (
        "legacy_fisher_diag_mse_grad_norm_mean",
        None,
    ),
    "per_batch_residual_kl": ("residual_kl_mean", "residual_kl_std"),
    "per_batch_residual_kl_grad_norm": ("residual_kl_grad_norm_mean", None),
    "per_batch_refined_residual_kl": (
        "refined_residual_kl_mean", "refined_residual_kl_std",
    ),
    "per_batch_refined_residual_kl_grad_norm": (
        "refined_residual_kl_grad_norm_mean", None,
    ),
    "per_batch_refined_diag_residual_kl": (
        "refined_diag_residual_kl_mean", "refined_diag_residual_kl_std",
    ),
    "per_batch_refined_diag_residual_kl_grad_norm": (
        "refined_diag_residual_kl_grad_norm_mean", None,
    ),
    "per_batch_refined_mse": ("refined_mse_mean", "refined_mse_std"),
    "per_batch_refined_mse_grad_norm": ("refined_mse_grad_norm_mean", None),
    "per_batch_layer_mse": ("layer_mse_mean", "layer_mse_std"),
    "per_batch_layer_mse_grad_norm": ("layer_mse_grad_norm_mean", None),
    "per_batch_module_mse": ("module_mse_mean", "module_mse_std"),
    "per_batch_module_mse_grad_norm": ("module_mse_grad_norm_mean", None),
}
for _loss_name in _LOSS_REPORT_ORDER:
    _ENTRY_SERIES_TO_STATS[f"per_batch_{_loss_name}_combined_cos"] = (
        f"{_loss_name}_combined_cos_mean",
        f"{_loss_name}_combined_cos_std",
    )
    _ENTRY_SERIES_TO_STATS[f"per_batch_{_loss_name}_combined_grad_norm"] = (
        f"{_loss_name}_combined_grad_norm_mean",
        None,
    )


def _merge_cosine_results_across_ranks(per_rank_results):
    merged = {}
    for rank_result in per_rank_results:
        if not rank_result:
            continue
        for layer_idx, per_layer in rank_result.items():
            layer_out = merged.setdefault(layer_idx, {})
            for module_name, entry in per_layer.items():
                out_entry = layer_out.setdefault(module_name, {})
                for key, value in entry.items():
                    if key.startswith("per_batch_"):
                        out_entry.setdefault(key, []).extend(list(value))
                    elif key not in out_entry:
                        out_entry[key] = value
    for per_layer in merged.values():
        for entry in per_layer.values():
            true_norms = entry.get("per_batch_true_kl_grad_norm", [])
            entry["n_batches"] = len(true_norms)
            for per_batch_key, (mean_key, std_key) in _ENTRY_SERIES_TO_STATS.items():
                values = entry.get(per_batch_key, [])
                if not values:
                    continue
                mean, std = _tensor_stats(values)
                entry[mean_key] = mean
                if std_key is not None:
                    entry[std_key] = std
            if not entry.get("reg_enabled", False):
                entry["reg_cos_mean"] = float("nan")
                entry["reg_cos_std"] = float("nan")
                entry.setdefault("per_batch_reg_cos", [])
    return merged


def _merge_layer_losses_across_ranks(per_rank_losses):
    merged_values = {}
    for rank_losses in per_rank_losses:
        if not rank_losses:
            continue
        for layer_idx, per_loss in rank_losses.items():
            layer_out = merged_values.setdefault(layer_idx, {})
            for loss_name, stats in per_loss.items():
                layer_out.setdefault(loss_name, []).extend(stats.get("per_batch", []))
    merged = {}
    for layer_idx, per_loss in merged_values.items():
        merged[layer_idx] = _summarize_loss_series(per_loss)
    return merged


def _gather_analysis_results(cosine_results, layer_loss_results):
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return cosine_results, layer_loss_results
    payload = {
        "cosine_results": cosine_results,
        "layer_loss_results": layer_loss_results,
    }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    if not dist_utils.is_main():
        return cosine_results, layer_loss_results
    return (
        _merge_cosine_results_across_ranks(
            [item["cosine_results"] for item in gathered if item is not None]
        ),
        _merge_layer_losses_across_ranks(
            [item["layer_loss_results"] for item in gathered if item is not None]
        ),
    )


def _format_layer_loss_summary(loss_summary):
    bits = []
    for loss_name in _LOSS_REPORT_ORDER:
        stats = loss_summary.get(loss_name)
        if stats is not None:
            bits.append(f"{loss_name}={stats['mean']:.6g}")
    return " ".join(bits) if bits else "<no selected loss evaluated>"


def _select_refined_A_for_batch(refined_A_list, samples_per_A, start, loss_name):
    a_idx = 0 if samples_per_A <= 0 else (start // samples_per_A)
    if a_idx >= len(refined_A_list):
        raise RuntimeError(
            f"{loss_name}: batch start={start} resolves to a_idx={a_idx} "
            f"which exceeds num_A={len(refined_A_list)}."
        )
    refined_A_slot = refined_A_list[a_idx]
    if refined_A_slot is None:
        raise RuntimeError(
            f"{loss_name}: missing A for batch start={start}, a_idx={a_idx}."
        )
    return refined_A_slot


def _canonical_module_dict(analyzer, layer):
    raw_modules = analyzer.get_quantizable_modules(layer)
    out = {}
    for raw_name, mod in raw_modules.items():
        out[raw_name[:-7] if raw_name.endswith(".module") else raw_name] = mod
    return out


def _unwrap_module_output(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def _collect_fp_module_outputs(
    *,
    analyzer,
    layer,
    inps,
    attention_mask,
    position_ids,
    position_embeddings,
    measure_samples,
    measure_batch_size,
    dev,
):
    if measure_samples % measure_batch_size != 0:
        raise ValueError(
            f"measure_samples ({measure_samples}) must be divisible by "
            f"measure_batch_size ({measure_batch_size})."
        )
    modules = _canonical_module_dict(analyzer, layer)
    outputs = {name: [] for name in modules}
    handles = []

    def _hook(name):
        def _tmp(_, _inp, out):
            outputs[name].append(_unwrap_module_output(out).detach().cpu())
        return _tmp

    for name, module in modules.items():
        handles.append(module.register_forward_hook(_hook(name)))

    try:
        b_attn, b_pos_ids, b_pos_emb = _expand_batch_kwargs(
            attention_mask, position_ids, position_embeddings, measure_batch_size,
        )
        with disable_fp_path_quant(layer):
            for start in range(0, measure_samples, measure_batch_size):
                _ = _layer_out(layer(
                    inps[start:start + measure_batch_size].to(dev),
                    attention_mask=b_attn,
                    position_ids=b_pos_ids,
                    position_embeddings=b_pos_emb,
                ))
    finally:
        for h in handles:
            h.remove()

    for name, chunks in outputs.items():
        if not chunks:
            raise RuntimeError(
                f"module_mse: no FP output captured for module {name!r}. "
                "This diagnostic currently expects every quantizable module to run "
                "on the measurement samples."
            )
        expected_chunks = measure_samples // measure_batch_size
        if len(chunks) != expected_chunks:
            raise RuntimeError(
                f"module_mse: expected {expected_chunks} FP output chunks for "
                f"module {name!r}, captured {len(chunks)}. This usually means the "
                "module is conditionally executed (for example an MoE expert); "
                "the per-module MSE diagnostic currently requires dense modules."
            )
        for chunk in chunks:
            if chunk.dim() != 3 or chunk.shape[0] != measure_batch_size or chunk.shape[1] != inps.shape[1]:
                raise RuntimeError(
                    f"module_mse: module {name!r} produced FP output shape "
                    f"{tuple(chunk.shape)}; expected (batch, seq, hidden) with "
                    f"batch={measure_batch_size}, seq={inps.shape[1]}. This "
                    "diagnostic currently supports dense transformer linear modules."
                )
    return outputs


def run_cosine_measurement(
    *,
    analyzer,
    layer,
    layer_idx,
    layers,
    inps,
    fp_inps,
    fp_inps_final,
    fisher_tensor,
    legacy_fisher_diag_tensor,
    refined_A_list,
    refined_diag_A_list,
    samples_per_A,
    attention_mask,
    position_ids,
    position_embeddings,
    measure_samples,
    measure_batch_size,
    sample_offset=0,
    kl_topk,
    dev,
    measure_losses,
    a_loss_ratio=1.0,
    grad_clip=None,
    refined_mse_pool_ids=None,
    refined_mse_grad_pool=None,
    refined_mse_mean_grad=None,
    fp_module_outputs=None,
    fp_weights=None,
    hessians=None,
    reg_strategy="none",
    reg_lambda=0.0,
):
    if measure_samples % measure_batch_size != 0:
        raise ValueError(
            f"measure_samples ({measure_samples}) must be divisible by "
            f"measure_batch_size ({measure_batch_size})."
        )
    if measure_samples > inps.shape[0]:
        raise ValueError(
            f"measure_samples ({measure_samples}) > calibration pool "
            f"({inps.shape[0]})."
        )
    # refined_A_list is the per-sub-A list for THIS layer (None for last layer).
    # If multi-A is in play, each measurement batch picks A[sample_idx//samples_per_A].
    # Require batch to sit entirely within one sub-A so the routing is
    # unambiguous.
    want_fisher = "fisher_diag_mse" in measure_losses
    want_legacy_fisher_diag = "legacy_fisher_diag_mse" in measure_losses
    want_residual = "residual_kl" in measure_losses
    want_refined_full = "refined_residual_kl" in measure_losses
    want_refined_diag = "refined_diag_residual_kl" in measure_losses
    want_refined_mse = "refined_mse" in measure_losses
    want_layer_mse = "layer_mse" in measure_losses
    want_module_mse = "module_mse" in measure_losses
    # refined_mse's second-order term is fisher_diag_mse, so it shares the
    # same batch-level fisher slice. Use a broader "needs fisher slice" gate
    # for the per-batch indexing below without touching `want_fisher` (which
    # still drives the dedicated fisher-surrogate measurement branch).
    need_fisher_slice = want_fisher or want_refined_mse
    has_refined = (
        want_refined_full
        and refined_A_list is not None
        and len(refined_A_list) > 0
        and any(a is not None for a in refined_A_list)
    )
    has_refined_diag = (
        want_refined_diag
        and refined_diag_A_list is not None
        and len(refined_diag_A_list) > 0
        and any(a is not None for a in refined_diag_A_list)
    )
    has_refined_mse = (
        want_refined_mse
        and refined_mse_mean_grad is not None
    )
    # refined_mse pool lookup: local_idx -> pool_pos. Built once per measurement.
    refined_mse_pool_lookup = {}
    if has_refined_mse and refined_mse_pool_ids is not None:
        refined_mse_pool_lookup = {
            int(li): int(pp) for pp, li in enumerate(refined_mse_pool_ids.tolist())
        }
    if (has_refined or has_refined_diag) and samples_per_A > 0 and samples_per_A % measure_batch_size != 0:
        raise ValueError(
            f"refined_rkl: samples_per_A ({samples_per_A}) must be divisible by "
            f"measure_batch_size ({measure_batch_size}) so each measurement batch "
            f"lands in one sub-A bucket."
        )
    if need_fisher_slice and fisher_tensor is None:
        raise RuntimeError(
            "measure_losses requested a Fisher-backed loss "
            "(fisher_diag_mse / refined_mse) but "
            "fisher was not collected (check collect_fisher flag)."
        )
    if want_legacy_fisher_diag and legacy_fisher_diag_tensor is None:
        raise RuntimeError(
            "measure_losses requested legacy_fisher_diag_mse but per-token "
            "legacy Fisher diagonal was not collected."
        )

    # Canonicalise linear names ("q_proj" not "q_proj.module" after ActQuantWrapper).
    raw_modules = analyzer.get_quantizable_modules(layer)
    name_to_module = {}
    for raw_name, mod in raw_modules.items():
        canon = raw_name[:-7] if raw_name.endswith(".module") else raw_name
        name_to_module[canon] = mod
    name_to_weight = {n: m.weight for n, m in name_to_module.items()}
    target_params = list(name_to_weight.values())

    # Precompute per-module regularization gradient once (independent of batch).
    # reg_grad shape matches weight: (out_features, in_features), fp32 on dev.
    # l2:      reg_grad = λ · (W_q - W_fp)
    # hessian: reg_grad = λ · (W_q - W_fp) · H, with H the full Fisher/Hessian matrix.
    # We do NOT multiply by grad_lr — cosine is scale-invariant, and for the
    # combined `surrogate + reg` the raw-gradient sum is what measures how reg
    # reshapes the effective direction.
    if reg_strategy not in ("none", "l2", "hessian"):
        raise ValueError(
            f"run_cosine_measurement: reg_strategy must be one of "
            f"{{none, l2, hessian}}, got {reg_strategy!r}."
        )
    reg_enabled = reg_strategy != "none" and reg_lambda > 0
    reg_grads = {}
    reg_grad_norm = {n: float("nan") for n in name_to_weight}
    if reg_enabled:
        if fp_weights is None:
            raise RuntimeError("reg_strategy != none requires fp_weights to be provided.")
        if reg_strategy == "hessian" and hessians is None:
            raise RuntimeError("reg_strategy == hessian requires hessians to be provided.")
        for canon, mod in name_to_module.items():
            if canon not in fp_weights:
                raise RuntimeError(
                    f"fp_weights missing entry for canonical name {canon!r}. "
                    f"Have: {sorted(fp_weights.keys())}."
                )
            W_q = mod.weight.detach().float()
            W_fp = fp_weights[canon].to(dev).float()
            diff = W_q - W_fp
            if reg_strategy == "l2":
                rg = reg_lambda * diff
            else:  # hessian
                if canon not in hessians:
                    raise RuntimeError(
                        f"hessians missing entry for canonical name {canon!r}. "
                        f"Have: {sorted(hessians.keys())}."
                    )
                H = hessians[canon].to(dev).float()
                if H.dim() == 3:
                    rows = diff.shape[0]
                    if H.shape[1] != H.shape[2] or H.shape[1] != diff.shape[1]:
                        raise ValueError(
                            f"grouped hessian for {canon!r} must have shape "
                            f"(G, in_features, in_features); got {tuple(H.shape)} "
                            f"vs diff {tuple(diff.shape)}"
                        )
                    if rows % H.shape[0] != 0:
                        raise ValueError(
                            f"rows ({rows}) must be divisible by hessian groups "
                            f"({H.shape[0]}) for {canon!r}."
                        )
                    rows_per_group = rows // H.shape[0]
                    rg_parts = []
                    for g in range(H.shape[0]):
                        row_start = g * rows_per_group
                        row_end = row_start + rows_per_group
                        rg_parts.append(diff[row_start:row_end].matmul(H[g]))
                    rg = reg_lambda * torch.cat(rg_parts, dim=0)
                    reg_grads[canon] = rg
                    reg_grad_norm[canon] = rg.flatten().norm(p=2).item()
                    continue
                if H.dim() != 2 or H.shape[0] != H.shape[1] or H.shape[0] != diff.shape[1]:
                    raise ValueError(
                        f"hessian matrix for {canon!r} must be square and match in_features; got {tuple(H.shape)} vs diff {tuple(diff.shape)}"
                    )
                rg = reg_lambda * diff.matmul(H)
            reg_grads[canon] = rg
            reg_grad_norm[canon] = rg.flatten().norm(p=2).item()

    per_batch_fisher_cos = {n: [] for n in name_to_weight}
    per_batch_legacy_fisher_diag_cos = {n: [] for n in name_to_weight}
    per_batch_residual_cos = {n: [] for n in name_to_weight}
    per_batch_refined_cos = {n: [] for n in name_to_weight}
    per_batch_refined_diag_cos = {n: [] for n in name_to_weight}
    per_batch_refined_mse_cos = {n: [] for n in name_to_weight}
    per_batch_layer_mse_cos = {n: [] for n in name_to_weight}
    per_batch_module_mse_cos = {n: [] for n in name_to_weight}
    per_batch_true_norm = {n: [] for n in name_to_weight}
    per_batch_fisher_norm = {n: [] for n in name_to_weight}
    per_batch_legacy_fisher_diag_norm = {n: [] for n in name_to_weight}
    per_batch_residual_norm = {n: [] for n in name_to_weight}
    per_batch_refined_norm = {n: [] for n in name_to_weight}
    per_batch_refined_diag_norm = {n: [] for n in name_to_weight}
    per_batch_refined_mse_norm = {n: [] for n in name_to_weight}
    per_batch_layer_mse_norm = {n: [] for n in name_to_weight}
    per_batch_module_mse_norm = {n: [] for n in name_to_weight}
    # Per-batch cos(true, reg_grad). Varies per batch because `grads_true`
    # changes even though reg_grad is constant; same value shared across all
    # loss types within a (layer, module, batch) cell.
    per_batch_reg_cos = {n: [] for n in name_to_weight}
    # Per-batch cos(true, surrogate + reg) and norm(surrogate + reg), one
    # series per (module, loss) pair.
    per_batch_combined_cos = {
        loss: {n: [] for n in name_to_weight}
        for loss in ("fisher_diag_mse", "legacy_fisher_diag_mse",
                     "residual_kl",
                     "refined_residual_kl", "refined_diag_residual_kl",
                     "refined_mse", "layer_mse", "module_mse")
    }
    per_batch_combined_norm = {
        loss: {n: [] for n in name_to_weight}
        for loss in ("fisher_diag_mse", "legacy_fisher_diag_mse",
                     "residual_kl",
                     "refined_residual_kl", "refined_diag_residual_kl",
                     "refined_mse", "layer_mse", "module_mse")
    }
    # Per-batch loss VALUES (the scalar each loss reduces to in this batch).
    # One series per loss type; only filled for losses we actually evaluated.
    # Used to print "loss after each layer" alongside the cosine/grad-norm view.
    per_batch_fisher_loss = []
    per_batch_legacy_fisher_diag_loss = []
    per_batch_residual_loss = []
    per_batch_refined_loss = []
    per_batch_refined_diag_loss = []
    per_batch_refined_mse_loss = []
    per_batch_layer_mse_loss = []
    per_batch_module_mse_loss = []

    def _call_layer(h_in):
        return _layer_out(layer(
            h_in,
            attention_mask=b_attn,
            position_ids=b_pos_ids,
            position_embeddings=b_pos_emb,
        ))

    n_batches = measure_samples // measure_batch_size
    b_attn, b_pos_ids, b_pos_emb = _expand_batch_kwargs(
        attention_mask, position_ids, position_embeddings, measure_batch_size,
    )
    # Freeze everything except the target layer's linear weights.
    with temporary_requires_grad([analyzer.model], target_params):
        for b in tqdm(range(n_batches), ncols=100, desc=f"cosine@layer{layer_idx}", leave=False):
            start = b * measure_batch_size
            end = start + measure_batch_size
            inp_batch = inps[start:end].to(dev)
            # `fp_inps` is expected to already hold the FP OUTPUT of the target
            # layer (i.e. fp_inps[target+1] after the caller advanced the FP
            # rolling buffer). Use it directly as `fp_hidden` to match production
            # block_gd semantics — there `fp_hidden = fp_inps[batch_indices]`, not
            # a fresh forward of fp input through the (now-quantized) target.
            fp_hidden_cached = fp_inps[start:end].to(dev)
            # Keep fp_final in model dtype (bf16) so hidden2logits can run through
            # the bf16 norm + lm_head without a dtype mismatch.
            if fp_inps_final is None:
                raise RuntimeError(
                    "run_cosine_measurement requires fp_inps_final for true KL "
                    "measurement, but GPTQ+ did not provide it."
                )
            fp_final_batch = fp_inps_final[start:end].to(dev)
            batch_indices = list(range(start, end))
            fisher_batch = (
                slice_layer_output_fisher_for_batch(
                    fisher_tensor,
                    batch_indices,
                    dev,
                    "fisher_diag_mse",
                )
                if need_fisher_slice and fisher_tensor is not None else None
            )
            legacy_fisher_diag_batch = (
                slice_layer_output_fisher_for_batch(
                    legacy_fisher_diag_tensor,
                    batch_indices,
                    dev,
                    "legacy_fisher_diag_mse",
                )
                if want_legacy_fisher_diag else None
            )

            # ---------- (1) true KL ----------
            _zero_grads(target_params)
            h_student = _forward_layer_and_tail(
                layer, layer_idx, layers, inp_batch,
                b_attn, b_pos_ids, b_pos_emb,
            )
            logits_student = hidden2logits(h_student, analyzer)
            logits_teacher = hidden2logits(fp_final_batch, analyzer).detach()
            if kl_topk > 0:
                logits_teacher, idx = logits_teacher.topk(kl_topk, dim=-1, sorted=False)
                logits_student = logits_student.gather(-1, idx)
            kl_loss = F.kl_div(
                F.log_softmax(logits_student, dim=-1),
                F.softmax(logits_teacher, dim=-1),
                reduction="none",
            ).sum(dim=-1).mean()
            kl_loss.backward()
            grads_true = _capture_grads(name_to_weight, grad_clip=grad_clip)
            del h_student, logits_student, logits_teacher, kl_loss

            # ---------- (2) fisher_diag_mse ----------
            grads_fisher = None
            if want_fisher:
                _zero_grads(target_params)
                out_hidden = _call_layer(inp_batch)
                fisher_loss = compute_refresh_loss(
                    refresh_loss_type="fisher_diag_mse",
                    out_hidden=out_hidden,
                    fp_hidden=fp_hidden_cached,
                    analyzer=analyzer,
                    kl_topk=kl_topk,
                    layer_output_fisher=fisher_batch,
                    fp_final_hidden=None,
                    a_loss_ratio=a_loss_ratio,
                )
                fisher_loss.backward()
                per_batch_fisher_loss.append(fisher_loss.item())
                grads_fisher = _capture_grads(name_to_weight, grad_clip=grad_clip)
                del out_hidden, fisher_loss

            # ---------- (3) legacy_fisher_diag_mse ----------
            grads_legacy_fisher_diag = None
            if want_legacy_fisher_diag:
                _zero_grads(target_params)
                out_hidden = _call_layer(inp_batch)
                legacy_fisher_diag_loss = compute_refresh_loss(
                    refresh_loss_type="legacy_fisher_diag_mse",
                    out_hidden=out_hidden,
                    fp_hidden=fp_hidden_cached,
                    analyzer=analyzer,
                    kl_topk=kl_topk,
                    layer_output_fisher=legacy_fisher_diag_batch,
                    fp_final_hidden=None,
                    a_loss_ratio=a_loss_ratio,
                )
                legacy_fisher_diag_loss.backward()
                per_batch_legacy_fisher_diag_loss.append(
                    _analysis_loss_value(
                        "legacy_fisher_diag_mse",
                        legacy_fisher_diag_loss,
                        measure_batch_size,
                    )
                )
                grads_legacy_fisher_diag = _capture_grads(
                    name_to_weight, grad_clip=grad_clip
                )
                del out_hidden, legacy_fisher_diag_loss

            # ---------- (4) residual_kl ----------
            grads_residual = None
            if want_residual:
                _zero_grads(target_params)
                out_hidden = _call_layer(inp_batch)
                residual_loss = compute_refresh_loss(
                    refresh_loss_type="residual_kl",
                    out_hidden=out_hidden,
                    fp_hidden=fp_hidden_cached,
                    analyzer=analyzer,
                    kl_topk=kl_topk,
                    layer_output_fisher=None,
                    fp_final_hidden=fp_final_batch,
                )
                residual_loss.backward()
                per_batch_residual_loss.append(residual_loss.item())
                grads_residual = _capture_grads(name_to_weight, grad_clip=grad_clip)
                del out_hidden, residual_loss

            # ---------- (5) refined_residual_kl ----------
            grads_refined = None
            if has_refined:
                # Pick the sub-A that owns this batch's samples. We enforce
                # `samples_per_A % measure_batch_size == 0` above so the batch
                # sits fully inside one bucket.
                global_start = int(sample_offset) + start
                a_idx = 0 if samples_per_A <= 0 else (global_start // samples_per_A)
                if a_idx >= len(refined_A_list):
                    raise RuntimeError(
                        f"refined_rkl: global batch start={global_start} resolves to a_idx={a_idx} "
                        f"which exceeds num_A={len(refined_A_list)}."
                    )
                refined_A_slot = refined_A_list[a_idx]
                if refined_A_slot is None:
                    raise RuntimeError(
                        f"refined_rkl: missing A for (layer={layer_idx}, a={a_idx}). "
                        "Check that measurement samples overlap the fit's sample range."
                    )
                refined_A_dev = refined_A_slot.to(dev)
                _zero_grads(target_params)
                out_hidden = _call_layer(inp_batch)
                refined_loss = compute_refresh_loss(
                    refresh_loss_type="refined_residual_kl",
                    out_hidden=out_hidden,
                    fp_hidden=fp_hidden_cached,
                    analyzer=analyzer,
                    kl_topk=kl_topk,
                    layer_output_fisher=None,
                    fp_final_hidden=fp_final_batch,
                    refined_A=refined_A_dev,
                )
                refined_loss.backward()
                per_batch_refined_loss.append(refined_loss.item())
                grads_refined = _capture_grads(name_to_weight, grad_clip=grad_clip)
                del out_hidden, refined_loss, refined_A_dev

            # ---------- (6) refined_diag_residual_kl ----------
            grads_refined_diag = None
            if has_refined_diag:
                global_start = int(sample_offset) + start
                a_idx_d = 0 if samples_per_A <= 0 else (global_start // samples_per_A)
                if a_idx_d >= len(refined_diag_A_list):
                    raise RuntimeError(
                        f"refined_diag_rkl: global batch start={global_start} resolves to "
                        f"a_idx={a_idx_d} which exceeds num_A={len(refined_diag_A_list)}."
                    )
                refined_diag_slot = refined_diag_A_list[a_idx_d]
                if refined_diag_slot is None:
                    raise RuntimeError(
                        f"refined_diag_rkl: missing A_diag for (layer={layer_idx}, "
                        f"a={a_idx_d})."
                    )
                # (seq, H) — shared across the batch via broadcast; no bmm.
                refined_diag_dev = refined_diag_slot.to(dev)
                _zero_grads(target_params)
                out_hidden = _call_layer(inp_batch)
                refined_diag_loss = compute_refresh_loss(
                    refresh_loss_type="refined_diag_residual_kl",
                    out_hidden=out_hidden,
                    fp_hidden=fp_hidden_cached,
                    analyzer=analyzer,
                    kl_topk=kl_topk,
                    layer_output_fisher=None,
                    fp_final_hidden=fp_final_batch,
                    refined_A=refined_diag_dev,
                )
                refined_diag_loss.backward()
                per_batch_refined_diag_loss.append(refined_diag_loss.item())
                grads_refined_diag = _capture_grads(name_to_weight, grad_clip=grad_clip)
                del out_hidden, refined_diag_loss, refined_diag_dev

            # ---------- (7) refined_mse ----------
            grads_refined_mse = None
            if has_refined_mse and fisher_tensor is not None:
                # Which measurement samples land in the pre-collected pool?
                # (Measurement batch is contiguous; we just look up each row.)
                pool_pos_list = []
                batch_row_list = []
                for p_in_batch in range(measure_batch_size):
                    li = start + p_in_batch
                    pp = refined_mse_pool_lookup.get(int(li))
                    if pp is not None:
                        pool_pos_list.append(pp)
                        batch_row_list.append(p_in_batch)
                if batch_row_list:
                    layer_output_grad_exact = (
                        refined_mse_grad_pool[pool_pos_list].to(dev, dtype=torch.float32)
                    )
                    pool_positions = torch.tensor(
                        batch_row_list, dtype=torch.long, device=dev
                    )
                else:
                    layer_output_grad_exact = None
                    pool_positions = None
                _zero_grads(target_params)
                out_hidden = _call_layer(inp_batch)
                refined_mse_loss = compute_refresh_loss(
                    refresh_loss_type="refined_mse",
                    out_hidden=out_hidden,
                    fp_hidden=fp_hidden_cached,
                    analyzer=analyzer,
                    kl_topk=kl_topk,
                    layer_output_fisher=fisher_batch,
                    fp_final_hidden=None,
                    layer_output_grad_exact=layer_output_grad_exact,
                    layer_output_grad_mean=refined_mse_mean_grad,
                    pool_positions=pool_positions,
                    a_loss_ratio=a_loss_ratio,
                )
                refined_mse_loss.backward()
                per_batch_refined_mse_loss.append(refined_mse_loss.item())
                grads_refined_mse = _capture_grads(name_to_weight, grad_clip=grad_clip)
                del out_hidden, refined_mse_loss
                if layer_output_grad_exact is not None:
                    del layer_output_grad_exact

            # ---------- (8) layer_mse ----------
            grads_layer_mse = None
            if want_layer_mse:
                _zero_grads(target_params)
                out_hidden = _call_layer(inp_batch)
                layer_mse_loss = compute_refresh_loss(
                    refresh_loss_type="hidden_mse",
                    out_hidden=out_hidden,
                    fp_hidden=fp_hidden_cached,
                    analyzer=analyzer,
                    kl_topk=kl_topk,
                    layer_output_fisher=None,
                    fp_final_hidden=None,
                    a_loss_ratio=a_loss_ratio,
                )
                layer_mse_loss.backward()
                per_batch_layer_mse_loss.append(layer_mse_loss.item())
                grads_layer_mse = _capture_grads(name_to_weight, grad_clip=grad_clip)
                del out_hidden, layer_mse_loss

            # ---------- (9) module_mse ----------
            grads_module_mse = None
            if want_module_mse:
                if fp_module_outputs is None:
                    raise RuntimeError(
                        "measure_losses requested module_mse but FP module outputs "
                        "were not collected."
                    )
                grads_module_mse = {}
                module_loss_sum = 0.0
                for module_name, module in name_to_module.items():
                    if module_name not in fp_module_outputs:
                        raise RuntimeError(
                            f"module_mse: missing FP output for module {module_name!r}."
                        )
                    _zero_grads(target_params)
                    captured = []

                    def _capture_out(_module, _inp, out):
                        captured.append(_unwrap_module_output(out))

                    handle = module.register_forward_hook(_capture_out)
                    try:
                        _ = _call_layer(inp_batch)
                    finally:
                        handle.remove()
                    if len(captured) != 1:
                        raise RuntimeError(
                            f"module_mse: expected exactly one output from {module_name!r}, "
                            f"captured {len(captured)}."
                        )
                    fp_module_out = fp_module_outputs[module_name][b].to(
                        dev, dtype=captured[0].dtype
                    )
                    module_delta = captured[0] - fp_module_out
                    if a_loss_ratio < 1.0:
                        module_delta = _scale_delta_by_abs_quantile(
                            module_delta,
                            a_loss_ratio,
                        )
                    module_mse_loss = 0.5 * module_delta.square().sum(dim=-1).mean()
                    module_mse_loss.backward()
                    module_loss_sum += module_mse_loss.item()
                    grads_module_mse[module_name] = _capture_one_grad(
                        name_to_weight[module_name],
                        module_name,
                        grad_clip=grad_clip,
                    )
                    del captured, fp_module_out, module_delta, module_mse_loss
                per_batch_module_mse_loss.append(
                    module_loss_sum / max(len(name_to_module), 1)
                )

            # ---------- cosine + grad L2 per linear ----------
            cos_f = _cosine_per_linear(grads_true, grads_fisher) if grads_fisher is not None else None
            cos_lfd = (
                _cosine_per_linear(grads_true, grads_legacy_fisher_diag)
                if grads_legacy_fisher_diag is not None else None
            )
            cos_r = _cosine_per_linear(grads_true, grads_residual) if grads_residual is not None else None
            cos_rf = _cosine_per_linear(grads_true, grads_refined) if grads_refined is not None else None
            cos_rfd = _cosine_per_linear(grads_true, grads_refined_diag) if grads_refined_diag is not None else None
            cos_rm = _cosine_per_linear(grads_true, grads_refined_mse) if grads_refined_mse is not None else None
            cos_lm = _cosine_per_linear(grads_true, grads_layer_mse) if grads_layer_mse is not None else None
            cos_mm = _cosine_per_linear(grads_true, grads_module_mse) if grads_module_mse is not None else None
            # cos(true, reg_grad) per module. Constant reg_grad but varying
            # true_grad → recompute per batch.
            cos_reg = (
                _cosine_per_linear(grads_true, reg_grads)
                if reg_enabled else None
            )
            # (loss_name, grads_surrogate) pairs for the combined metric loop.
            _combined_pairs = [
                ("fisher_diag_mse", grads_fisher),
                ("legacy_fisher_diag_mse", grads_legacy_fisher_diag),
                ("residual_kl", grads_residual),
                ("refined_residual_kl", grads_refined),
                ("refined_diag_residual_kl", grads_refined_diag),
                ("refined_mse", grads_refined_mse),
                ("layer_mse", grads_layer_mse),
                ("module_mse", grads_module_mse),
            ]
            for n in name_to_weight:
                if cos_f is not None:
                    per_batch_fisher_cos[n].append(cos_f[n])
                if cos_lfd is not None:
                    per_batch_legacy_fisher_diag_cos[n].append(cos_lfd[n])
                if cos_r is not None:
                    per_batch_residual_cos[n].append(cos_r[n])
                if cos_rf is not None:
                    per_batch_refined_cos[n].append(cos_rf[n])
                if cos_rfd is not None:
                    per_batch_refined_diag_cos[n].append(cos_rfd[n])
                if cos_rm is not None:
                    per_batch_refined_mse_cos[n].append(cos_rm[n])
                if cos_lm is not None:
                    per_batch_layer_mse_cos[n].append(cos_lm[n])
                if cos_mm is not None:
                    per_batch_module_mse_cos[n].append(cos_mm[n])
                # Per-batch grad Frobenius / L2 norms (flattened). Useful to see
                # not just whether surrogate grads point the right way (cosine)
                # but also how their magnitude compares to the true KL grad's.
                per_batch_true_norm[n].append(grads_true[n].flatten().norm(p=2).item())
                if grads_fisher is not None:
                    per_batch_fisher_norm[n].append(grads_fisher[n].flatten().norm(p=2).item())
                if grads_legacy_fisher_diag is not None:
                    per_batch_legacy_fisher_diag_norm[n].append(
                        grads_legacy_fisher_diag[n].flatten().norm(p=2).item()
                    )
                if grads_residual is not None:
                    per_batch_residual_norm[n].append(grads_residual[n].flatten().norm(p=2).item())
                if grads_refined is not None:
                    per_batch_refined_norm[n].append(grads_refined[n].flatten().norm(p=2).item())
                if grads_refined_diag is not None:
                    per_batch_refined_diag_norm[n].append(grads_refined_diag[n].flatten().norm(p=2).item())
                if cos_reg is not None:
                    per_batch_reg_cos[n].append(cos_reg[n])
                # Combined surrogate + reg_grad metrics. Compute per (loss, module).
                # reg_grads[n] is fp32 on dev; grads_surrogate[n] is also fp32 (via
                # `_capture_grads`). The sum is the "effective" gradient seen if
                # we added reg on top of the surrogate signal.
                if reg_enabled:
                    for loss_name, g_sur in _combined_pairs:
                        if g_sur is None:
                            continue
                        combined = g_sur[n] + reg_grads[n]
                        per_batch_combined_cos[loss_name][n].append(
                            F.cosine_similarity(
                                grads_true[n].flatten().unsqueeze(0),
                                combined.flatten().unsqueeze(0),
                                dim=1,
                            ).item()
                        )
                        per_batch_combined_norm[loss_name][n].append(
                            combined.flatten().norm(p=2).item()
                        )
                if grads_refined_mse is not None:
                    per_batch_refined_mse_norm[n].append(grads_refined_mse[n].flatten().norm(p=2).item())
                if grads_layer_mse is not None:
                    per_batch_layer_mse_norm[n].append(grads_layer_mse[n].flatten().norm(p=2).item())
                if grads_module_mse is not None:
                    per_batch_module_mse_norm[n].append(grads_module_mse[n].flatten().norm(p=2).item())
            del grads_true
            if grads_fisher is not None:
                del grads_fisher
            if grads_legacy_fisher_diag is not None:
                del grads_legacy_fisher_diag
            if grads_residual is not None:
                del grads_residual
            if grads_refined is not None:
                del grads_refined
            if grads_refined_diag is not None:
                del grads_refined_diag
            if grads_refined_mse is not None:
                del grads_refined_mse
            if grads_layer_mse is not None:
                del grads_layer_mse
            if grads_module_mse is not None:
                del grads_module_mse
            torch.cuda.empty_cache()

        _zero_grads(target_params)

    loss_summary = _summarize_loss_series({
        "fisher_diag_mse": per_batch_fisher_loss,
        "legacy_fisher_diag_mse": per_batch_legacy_fisher_diag_loss,
        "residual_kl": per_batch_residual_loss,
        "refined_residual_kl": per_batch_refined_loss,
        "refined_diag_residual_kl": per_batch_refined_diag_loss,
        "refined_mse": per_batch_refined_mse_loss,
        "layer_mse": per_batch_layer_mse_loss,
        "module_mse": per_batch_module_mse_loss,
    })

    results = {}
    for n in name_to_weight:
        tn = torch.tensor(per_batch_true_norm[n])
        entry = {
            "n_batches": len(tn),
            # Grad L2 norms (mean over batches). Reported alongside cosines so
            # we can judge both direction (cosine) and magnitude (norm ratio).
            "true_kl_grad_norm_mean": tn.mean().item(),
            "per_batch_true_kl_grad_norm": tn.tolist(),
            # Regularizer meta. Present on every entry for consistency — reg
            # disabled gives `nan` cosines / norms so downstream table code can
            # print them uniformly.
            "reg_strategy": reg_strategy,
            "reg_lambda": float(reg_lambda),
            "reg_enabled": bool(reg_enabled),
            "reg_grad_norm": reg_grad_norm[n],
        }
        if reg_enabled and per_batch_reg_cos[n]:
            rc = torch.tensor(per_batch_reg_cos[n])
            entry["reg_cos_mean"] = rc.mean().item()
            entry["reg_cos_std"] = rc.std(unbiased=False).item() if len(rc) > 1 else 0.0
            entry["per_batch_reg_cos"] = rc.tolist()
        else:
            entry["reg_cos_mean"] = float("nan")
            entry["reg_cos_std"] = float("nan")
            entry["per_batch_reg_cos"] = []
        if want_fisher and per_batch_fisher_cos[n]:
            fs = torch.tensor(per_batch_fisher_cos[n])
            fn = torch.tensor(per_batch_fisher_norm[n])
            entry["fisher_mean"] = fs.mean().item()
            entry["fisher_std"] = fs.std(unbiased=False).item() if len(fs) > 1 else 0.0
            entry["per_batch_fisher"] = fs.tolist()
            entry["fisher_grad_norm_mean"] = fn.mean().item()
            entry["per_batch_fisher_grad_norm"] = fn.tolist()
        if want_legacy_fisher_diag and per_batch_legacy_fisher_diag_cos[n]:
            lfd = torch.tensor(per_batch_legacy_fisher_diag_cos[n])
            lfdn = torch.tensor(per_batch_legacy_fisher_diag_norm[n])
            entry["legacy_fisher_diag_mse_mean"] = lfd.mean().item()
            entry["legacy_fisher_diag_mse_std"] = (
                lfd.std(unbiased=False).item() if len(lfd) > 1 else 0.0
            )
            entry["per_batch_legacy_fisher_diag_mse"] = lfd.tolist()
            entry["legacy_fisher_diag_mse_grad_norm_mean"] = lfdn.mean().item()
            entry["per_batch_legacy_fisher_diag_mse_grad_norm"] = lfdn.tolist()
        if want_residual and per_batch_residual_cos[n]:
            rs = torch.tensor(per_batch_residual_cos[n])
            rn = torch.tensor(per_batch_residual_norm[n])
            entry["residual_kl_mean"] = rs.mean().item()
            entry["residual_kl_std"] = rs.std(unbiased=False).item() if len(rs) > 1 else 0.0
            entry["per_batch_residual_kl"] = rs.tolist()
            entry["residual_kl_grad_norm_mean"] = rn.mean().item()
            entry["per_batch_residual_kl_grad_norm"] = rn.tolist()
        if has_refined and per_batch_refined_cos[n]:
            rf = torch.tensor(per_batch_refined_cos[n])
            rfn = torch.tensor(per_batch_refined_norm[n])
            entry["refined_residual_kl_mean"] = rf.mean().item()
            entry["refined_residual_kl_std"] = rf.std(unbiased=False).item() if len(rf) > 1 else 0.0
            entry["per_batch_refined_residual_kl"] = rf.tolist()
            entry["refined_residual_kl_grad_norm_mean"] = rfn.mean().item()
            entry["per_batch_refined_residual_kl_grad_norm"] = rfn.tolist()
        if has_refined_diag and per_batch_refined_diag_cos[n]:
            rfd = torch.tensor(per_batch_refined_diag_cos[n])
            rfdn = torch.tensor(per_batch_refined_diag_norm[n])
            entry["refined_diag_residual_kl_mean"] = rfd.mean().item()
            entry["refined_diag_residual_kl_std"] = rfd.std(unbiased=False).item() if len(rfd) > 1 else 0.0
            entry["per_batch_refined_diag_residual_kl"] = rfd.tolist()
            entry["refined_diag_residual_kl_grad_norm_mean"] = rfdn.mean().item()
            entry["per_batch_refined_diag_residual_kl_grad_norm"] = rfdn.tolist()
        if has_refined_mse and per_batch_refined_mse_cos[n]:
            rm = torch.tensor(per_batch_refined_mse_cos[n])
            rmn = torch.tensor(per_batch_refined_mse_norm[n])
            entry["refined_mse_mean"] = rm.mean().item()
            entry["refined_mse_std"] = rm.std(unbiased=False).item() if len(rm) > 1 else 0.0
            entry["per_batch_refined_mse"] = rm.tolist()
            entry["refined_mse_grad_norm_mean"] = rmn.mean().item()
            entry["per_batch_refined_mse_grad_norm"] = rmn.tolist()
        if want_layer_mse and per_batch_layer_mse_cos[n]:
            lm = torch.tensor(per_batch_layer_mse_cos[n])
            lmn = torch.tensor(per_batch_layer_mse_norm[n])
            entry["layer_mse_mean"] = lm.mean().item()
            entry["layer_mse_std"] = lm.std(unbiased=False).item() if len(lm) > 1 else 0.0
            entry["per_batch_layer_mse"] = lm.tolist()
            entry["layer_mse_grad_norm_mean"] = lmn.mean().item()
            entry["per_batch_layer_mse_grad_norm"] = lmn.tolist()
        if want_module_mse and per_batch_module_mse_cos[n]:
            mm = torch.tensor(per_batch_module_mse_cos[n])
            mmn = torch.tensor(per_batch_module_mse_norm[n])
            entry["module_mse_mean"] = mm.mean().item()
            entry["module_mse_std"] = mm.std(unbiased=False).item() if len(mm) > 1 else 0.0
            entry["per_batch_module_mse"] = mm.tolist()
            entry["module_mse_grad_norm_mean"] = mmn.mean().item()
            entry["per_batch_module_mse_grad_norm"] = mmn.tolist()
        # Combined (surrogate + reg_grad) metrics, one set per loss that ran.
        if reg_enabled:
            for loss_name in (
                "fisher_diag_mse", "legacy_fisher_diag_mse", "residual_kl",
                "refined_residual_kl", "refined_diag_residual_kl",
                "refined_mse", "layer_mse", "module_mse",
            ):
                cos_list = per_batch_combined_cos[loss_name][n]
                norm_list = per_batch_combined_norm[loss_name][n]
                if not cos_list:
                    continue
                cc = torch.tensor(cos_list)
                nn_ = torch.tensor(norm_list)
                entry[f"{loss_name}_combined_cos_mean"] = cc.mean().item()
                entry[f"{loss_name}_combined_cos_std"] = (
                    cc.std(unbiased=False).item() if len(cc) > 1 else 0.0
                )
                entry[f"per_batch_{loss_name}_combined_cos"] = cc.tolist()
                entry[f"{loss_name}_combined_grad_norm_mean"] = nn_.mean().item()
                entry[f"per_batch_{loss_name}_combined_grad_norm"] = nn_.tolist()
        results[n] = entry
    return results, loss_summary


def _collect_refined_mse_state_for_loss_report(
    *,
    args,
    analyzer,
    layer,
    layer_idx,
    layers,
    inps,
    fp_inps_final,
    attention_mask,
    position_ids,
    position_embeddings,
    dev,
    measure_samples_local,
):
    import random as _rng_mod

    n_pool = int(getattr(args, "num_samples_for_refined_mse", 32))
    if n_pool <= 0:
        raise ValueError(
            f"num_samples_for_refined_mse must be > 0 when measuring refined_mse, got {n_pool}."
        )
    n_pool = min(n_pool, measure_samples_local, inps.shape[0])
    if n_pool % args.measure_batch_size != 0:
        n_pool = (n_pool // args.measure_batch_size) * args.measure_batch_size
    if n_pool <= 0:
        raise ValueError(
            "num_samples_for_refined_mse is smaller than the local measurement "
            f"batch size ({args.measure_batch_size}) after DP sharding."
        )
    if n_pool > inps.shape[0]:
        raise ValueError(
            f"num_samples_for_refined_mse ({n_pool}) exceeds calibration pool ({inps.shape[0]})."
        )
    rm_bwd_bsz = max(1, args.global_loss_bsz // max(dist_utils.get_world_size(), 1))
    rm_bwd_bsz = min(rm_bwd_bsz, n_pool)
    if n_pool % rm_bwd_bsz != 0:
        rm_bwd_bsz = args.measure_batch_size
    if n_pool % rm_bwd_bsz != 0:
        raise ValueError(
            f"num_samples_for_refined_mse ({n_pool}) must be divisible by "
            f"local refined-mse backward batch size ({rm_bwd_bsz})."
        )

    rng = _rng_mod.Random(
        int(getattr(args, "refresh_seed", 0)) + layer_idx
    )
    sample_ids_local = sorted(rng.sample(range(measure_samples_local), n_pool))
    (
        refined_mse_grad_pool,
        refined_mse_mean_grad,
        _unused_grad_pool_next,
        _unused_mean_grad_next,
    ) = collect_layer_output_grad_for_refined_mse(
        analyzer=analyzer,
        layer=layer,
        layer_idx=layer_idx,
        layers=layers,
        inps=inps,
        fp_inps_final=fp_inps_final,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        sample_ids_local=sample_ids_local,
        backward_bsz=rm_bwd_bsz,
        kl_topk=args.kl_topk,
        dev=dev,
    )
    return (
        torch.tensor(sample_ids_local, dtype=torch.long),
        refined_mse_grad_pool,
        refined_mse_mean_grad,
    )


@torch.no_grad()
def measure_layer_losses_after_quant(
    *,
    analyzer,
    layer,
    layer_idx,
    inps,
    fp_inps,
    fp_inps_final,
    fisher_tensor,
    legacy_fisher_diag_tensor,
    refined_A_list,
    refined_diag_A_list,
    samples_per_A,
    attention_mask,
    position_ids,
    position_embeddings,
    measure_samples,
    measure_batch_size,
    sample_offset=0,
    kl_topk,
    dev,
    measure_losses,
    a_loss_ratio=1.0,
    refined_mse_pool_ids=None,
    refined_mse_grad_pool=None,
    refined_mse_mean_grad=None,
    fp_module_outputs=None,
):
    if measure_samples % measure_batch_size != 0:
        raise ValueError(
            f"measure_samples ({measure_samples}) must be divisible by "
            f"measure_batch_size ({measure_batch_size})."
        )
    if measure_samples > inps.shape[0]:
        raise ValueError(
            f"measure_samples ({measure_samples}) > calibration pool ({inps.shape[0]})."
        )

    want_fisher = "fisher_diag_mse" in measure_losses
    want_legacy_fisher_diag = "legacy_fisher_diag_mse" in measure_losses
    want_refined_mse = "refined_mse" in measure_losses
    need_fisher = want_fisher or want_refined_mse
    want_layer_mse = "layer_mse" in measure_losses
    want_module_mse = "module_mse" in measure_losses
    want_residual = "residual_kl" in measure_losses
    has_refined = (
        "refined_residual_kl" in measure_losses
        and refined_A_list is not None
        and len(refined_A_list) > 0
        and any(a is not None for a in refined_A_list)
    )
    has_refined_diag = (
        "refined_diag_residual_kl" in measure_losses
        and refined_diag_A_list is not None
        and len(refined_diag_A_list) > 0
        and any(a is not None for a in refined_diag_A_list)
    )
    has_refined_mse = want_refined_mse and refined_mse_mean_grad is not None
    refined_mse_pool_lookup = {}
    if has_refined_mse and refined_mse_pool_ids is not None:
        refined_mse_pool_lookup = {
            int(li): int(pp) for pp, li in enumerate(refined_mse_pool_ids.tolist())
        }
    if need_fisher and fisher_tensor is None:
        raise RuntimeError(
            "measure_losses requested a Fisher-backed loss "
            "(fisher_diag_mse / refined_mse) but "
            "fisher was not collected (check collect_fisher flag)."
        )
    if want_legacy_fisher_diag and legacy_fisher_diag_tensor is None:
        raise RuntimeError(
            "measure_losses requested legacy_fisher_diag_mse but per-token "
            "legacy Fisher diagonal was not collected."
        )
    if want_refined_mse and not has_refined_mse:
        raise RuntimeError(
            "measure_losses requested refined_mse but refined_mse grad state was not provided."
        )
    if (has_refined or has_refined_diag) and samples_per_A > 0 and samples_per_A % measure_batch_size != 0:
        raise ValueError(
            f"refined_rkl: samples_per_A ({samples_per_A}) must be divisible by "
            f"measure_batch_size ({measure_batch_size}) so each measurement batch "
            f"lands in one sub-A bucket."
        )

    per_loss_values = {loss_name: [] for loss_name in _LOSS_REPORT_ORDER}
    b_attn, b_pos_ids, b_pos_emb = _expand_batch_kwargs(
        attention_mask, position_ids, position_embeddings, measure_batch_size,
    )
    n_batches = measure_samples // measure_batch_size

    for b in tqdm(range(n_batches), ncols=100, desc=f"loss@layer{layer_idx}", leave=False):
        start = b * measure_batch_size
        end = start + measure_batch_size
        batch_indices = list(range(start, end))
        inp_batch = inps[start:end].to(dev)
        fp_hidden_cached = fp_inps[start:end].to(dev)
        fp_final_batch = (
            None if fp_inps_final is None else fp_inps_final[start:end].to(dev)
        )
        if (
            fp_final_batch is None
            and (want_residual or has_refined or has_refined_diag)
        ):
            raise RuntimeError(
                "measure_layer_losses_after_quant requires fp_inps_final for "
                "residual/refined residual losses, but GPTQ+ did not provide it."
            )
        fisher_batch = (
            slice_layer_output_fisher_for_batch(
                fisher_tensor,
                batch_indices,
                dev,
                "fisher_diag_mse",
            )
            if need_fisher else None
        )
        legacy_fisher_diag_batch = (
            slice_layer_output_fisher_for_batch(
                legacy_fisher_diag_tensor,
                batch_indices,
                dev,
                "legacy_fisher_diag_mse",
            )
            if want_legacy_fisher_diag else None
        )

        out_hidden = _layer_out(layer(
            inp_batch,
            attention_mask=b_attn,
            position_ids=b_pos_ids,
            position_embeddings=b_pos_emb,
        ))

        if want_fisher:
            loss = compute_refresh_loss(
                refresh_loss_type="fisher_diag_mse",
                out_hidden=out_hidden,
                fp_hidden=fp_hidden_cached,
                analyzer=analyzer,
                kl_topk=kl_topk,
                layer_output_fisher=fisher_batch,
                fp_final_hidden=None,
                a_loss_ratio=a_loss_ratio,
            )
            per_loss_values["fisher_diag_mse"].append(loss.item())
        if want_legacy_fisher_diag:
            loss = compute_refresh_loss(
                refresh_loss_type="legacy_fisher_diag_mse",
                out_hidden=out_hidden,
                fp_hidden=fp_hidden_cached,
                analyzer=analyzer,
                kl_topk=kl_topk,
                layer_output_fisher=legacy_fisher_diag_batch,
                fp_final_hidden=None,
                a_loss_ratio=a_loss_ratio,
            )
            per_loss_values["legacy_fisher_diag_mse"].append(
                _analysis_loss_value(
                    "legacy_fisher_diag_mse",
                    loss,
                    measure_batch_size,
                )
            )
        if want_residual:
            loss = compute_refresh_loss(
                refresh_loss_type="residual_kl",
                out_hidden=out_hidden,
                fp_hidden=fp_hidden_cached,
                analyzer=analyzer,
                kl_topk=kl_topk,
                layer_output_fisher=None,
                fp_final_hidden=fp_final_batch,
            )
            per_loss_values["residual_kl"].append(loss.item())
        if has_refined:
            refined_A = _select_refined_A_for_batch(
                refined_A_list, samples_per_A, int(sample_offset) + start,
                "refined_residual_kl",
            ).to(dev)
            loss = compute_refresh_loss(
                refresh_loss_type="refined_residual_kl",
                out_hidden=out_hidden,
                fp_hidden=fp_hidden_cached,
                analyzer=analyzer,
                kl_topk=kl_topk,
                layer_output_fisher=None,
                fp_final_hidden=fp_final_batch,
                refined_A=refined_A,
            )
            per_loss_values["refined_residual_kl"].append(loss.item())
            del refined_A
        if has_refined_diag:
            refined_diag_A = _select_refined_A_for_batch(
                refined_diag_A_list, samples_per_A, int(sample_offset) + start,
                "refined_diag_residual_kl",
            ).to(dev)
            loss = compute_refresh_loss(
                refresh_loss_type="refined_diag_residual_kl",
                out_hidden=out_hidden,
                fp_hidden=fp_hidden_cached,
                analyzer=analyzer,
                kl_topk=kl_topk,
                layer_output_fisher=None,
                fp_final_hidden=fp_final_batch,
                refined_A=refined_diag_A,
            )
            per_loss_values["refined_diag_residual_kl"].append(loss.item())
            del refined_diag_A
        if has_refined_mse:
            pool_pos_list = []
            batch_row_list = []
            for p_in_batch in range(measure_batch_size):
                li = start + p_in_batch
                pp = refined_mse_pool_lookup.get(int(li))
                if pp is not None:
                    pool_pos_list.append(pp)
                    batch_row_list.append(p_in_batch)
            if batch_row_list:
                layer_output_grad_exact = refined_mse_grad_pool[
                    pool_pos_list
                ].to(dev, dtype=torch.float32)
                pool_positions = torch.tensor(
                    batch_row_list, dtype=torch.long, device=dev
                )
            else:
                layer_output_grad_exact = None
                pool_positions = None
            loss = compute_refresh_loss(
                refresh_loss_type="refined_mse",
                out_hidden=out_hidden,
                fp_hidden=fp_hidden_cached,
                analyzer=analyzer,
                kl_topk=kl_topk,
                layer_output_fisher=fisher_batch,
                fp_final_hidden=None,
                layer_output_grad_exact=layer_output_grad_exact,
                layer_output_grad_mean=refined_mse_mean_grad,
                pool_positions=pool_positions,
                a_loss_ratio=a_loss_ratio,
            )
            per_loss_values["refined_mse"].append(loss.item())
            if layer_output_grad_exact is not None:
                del layer_output_grad_exact
        if want_layer_mse:
            loss = compute_refresh_loss(
                refresh_loss_type="hidden_mse",
                out_hidden=out_hidden,
                fp_hidden=fp_hidden_cached,
                analyzer=analyzer,
                kl_topk=kl_topk,
                layer_output_fisher=None,
                fp_final_hidden=None,
                a_loss_ratio=a_loss_ratio,
            )
            per_loss_values["layer_mse"].append(loss.item())
        if want_module_mse:
            if fp_module_outputs is None:
                raise RuntimeError(
                    "measure_losses requested module_mse but FP module outputs "
                    "were not collected."
                )
            module_losses = []
            captured = {}
            handles = []
            modules = _canonical_module_dict(analyzer, layer)

            def _capture(name):
                def _tmp(_module, _inp, out):
                    captured[name] = _unwrap_module_output(out).detach()
                return _tmp

            for name, module in modules.items():
                handles.append(module.register_forward_hook(_capture(name)))
            try:
                _ = _layer_out(layer(
                    inp_batch,
                    attention_mask=b_attn,
                    position_ids=b_pos_ids,
                    position_embeddings=b_pos_emb,
                ))
            finally:
                for h in handles:
                    h.remove()
            missing = sorted(set(modules) - set(captured))
            if missing:
                raise RuntimeError(
                    f"module_mse: quantized forward did not capture outputs for "
                    f"modules {missing}. This diagnostic currently requires every "
                    "quantizable module to run exactly once per measurement batch."
                )
            for name, out_q in captured.items():
                if name not in fp_module_outputs:
                    raise RuntimeError(
                        f"module_mse: missing FP output for module {name!r}."
                    )
                out_fp = fp_module_outputs[name][b].to(dev, dtype=out_q.dtype)
                module_delta = out_q - out_fp
                if a_loss_ratio < 1.0:
                    module_delta = _scale_delta_by_abs_quantile(
                        module_delta,
                        a_loss_ratio,
                    )
                module_losses.append(
                    0.5 * module_delta.square().sum(dim=-1).mean().item()
                )
            per_loss_values["module_mse"].append(
                sum(module_losses) / max(len(module_losses), 1)
            )

        del inp_batch, fp_hidden_cached, fp_final_batch, out_hidden

    return _summarize_loss_series(per_loss_values)


# ---------------------------------------------------------------------------
# analysis hook shared by reference quantization paths
# ---------------------------------------------------------------------------

def _build_cosine_analysis_hook(
    *,
    args,
    analyzer,
    layers,
    target_layers,
    measure_losses,
    measure_samples_local,
    sample_offset,
    dev,
    cosine_results,
    layer_loss_results,
):
    fp_module_outputs_by_layer = {}
    refined_rkl_num_A = int(getattr(args, "refined_rkl_num_A", 1))
    samples_per_A = (
        args.nsamples // refined_rkl_num_A
        if refined_rkl_num_A > 0 else args.nsamples
    )
    want_module_mse = "module_mse" in measure_losses

    def _layer_grad_clip(layer_idx):
        final_layer_idx = len(layers) - 1
        fl_clip = getattr(args, "final_layer_grad_clip", None)
        return (
            fl_clip
            if layer_idx == final_layer_idx and fl_clip is not None
            else getattr(args, "grad_clip", None)
        )

    def _collect_refined_mse_for_layer(payload):
        if "refined_mse" not in measure_losses:
            return (None, None, None)
        return _collect_refined_mse_state_for_loss_report(
            args=args,
            analyzer=analyzer,
            layer=payload["layer"],
            layer_idx=payload["layer_idx"],
            layers=payload["layers"],
            inps=payload["inps"],
            fp_inps_final=payload["fp_inps_final"],
            attention_mask=payload["attention_mask"],
            position_ids=payload["position_ids"],
            position_embeddings=payload["position_embeddings"],
            dev=payload["dev"],
            measure_samples_local=measure_samples_local,
        )

    def _log_layer_summary(layer_idx, layer_loss_summary):
        logging.info(
            "Layer %d loss after quantization (rank-local batch-avg over %d samples): %s",
            layer_idx,
            measure_samples_local,
            _format_layer_loss_summary(layer_loss_summary),
        )

    def _fmt_entry(name, r):
        loss_specs = [
            ("fisher_mean", "fisher"),
            ("legacy_fisher_diag_mse_mean", "legacy_fisher_diag"),
            ("residual_kl_mean", "res_kl"),
            ("refined_residual_kl_mean", "refined_res_kl"),
            ("refined_diag_residual_kl_mean", "refined_diag_res_kl"),
            ("refined_mse_mean", "refined_mse"),
            ("layer_mse_mean", "layer_mse"),
            ("module_mse_mean", "module_mse"),
        ]
        norm_specs = [
            ("fisher_grad_norm_mean", "fisher"),
            ("legacy_fisher_diag_mse_grad_norm_mean", "legacy_fisher_diag"),
            ("residual_kl_grad_norm_mean", "res_kl"),
            ("refined_residual_kl_grad_norm_mean", "refined_res_kl"),
            ("refined_diag_residual_kl_grad_norm_mean", "refined_diag_res_kl"),
            ("refined_mse_grad_norm_mean", "refined_mse"),
            ("layer_mse_grad_norm_mean", "layer_mse"),
            ("module_mse_grad_norm_mean", "module_mse"),
        ]
        cos_bits = [f"{label}={r[key]:.4f}" for key, label in loss_specs if key in r]
        if r.get("reg_enabled"):
            cos_bits.append(f"reg={r['reg_cos_mean']:.4f}")
            for loss_key, loss_short in (
                ("fisher_diag_mse", "fisher+reg"),
                ("legacy_fisher_diag_mse", "legacy_fisher_diag+reg"),
                ("residual_kl", "res_kl+reg"),
                ("refined_residual_kl", "refined_res_kl+reg"),
                ("refined_diag_residual_kl", "refined_diag_res_kl+reg"),
                ("refined_mse", "refined_mse+reg"),
                ("layer_mse", "layer_mse+reg"),
                ("module_mse", "module_mse+reg"),
            ):
                key = f"{loss_key}_combined_cos_mean"
                if key in r:
                    cos_bits.append(f"{loss_short}={r[key]:.4f}")
        norm_bits = [f"true={r['true_kl_grad_norm_mean']:.3e}"]
        norm_bits.extend(
            f"{label}={r[key]:.3e}" for key, label in norm_specs if key in r
        )
        if r.get("reg_enabled"):
            norm_bits.append(f"reg={r['reg_grad_norm']:.3e}")
        return f"{name} cos[{' '.join(cos_bits)}] norm[{' '.join(norm_bits)}]"

    def _analysis_hook(event, payload):
        layer_idx = payload["layer_idx"]
        if event == "before_layer":
            return layer_idx in target_layers

        if event == "before_fp_reference":
            if want_module_mse:
                fp_module_outputs_by_layer[layer_idx] = _collect_fp_module_outputs(
                    analyzer=analyzer,
                    layer=payload["layer"],
                    inps=payload["fp_inps"],
                    attention_mask=payload["attention_mask"],
                    position_ids=payload["position_ids"],
                    position_embeddings=payload["position_embeddings"],
                    measure_samples=measure_samples_local,
                    measure_batch_size=args.measure_batch_size,
                    dev=payload["dev"],
                )
            return None

        if event != "after_layer_quantized":
            return None

        layer = payload["layer"]
        layers_local = payload["layers"]
        is_target = bool(payload.get("is_target", False))
        refined_A_list_i = (
            payload["static_refined_A_by_layer"][layer_idx]
            if payload.get("static_refined_A_by_layer") is not None
            else None
        )
        refined_diag_A_list_i = (
            payload["static_refined_diag_A_by_layer"][layer_idx]
            if payload.get("static_refined_diag_A_by_layer") is not None
            else None
        )
        fp_module_outputs_i = fp_module_outputs_by_layer.pop(layer_idx, None)
        refined_mse_pool_ids_i = None
        refined_mse_grad_pool_i = None
        refined_mse_mean_grad_i = None

        if is_target:
            for k in range(layer_idx + 1, len(layers_local)):
                layers_local[k].to(payload["dev"])
            try:
                with torch.enable_grad():
                    (
                        refined_mse_pool_ids_i,
                        refined_mse_grad_pool_i,
                        refined_mse_mean_grad_i,
                    ) = _collect_refined_mse_for_layer(payload)
                    cosine_results[layer_idx], layer_loss_summary = run_cosine_measurement(
                        analyzer=analyzer,
                        layer=layer,
                        layer_idx=layer_idx,
                        layers=layers_local,
                        inps=payload["inps"],
                        fp_inps=payload["fp_inps"],
                        fp_inps_final=payload["fp_inps_final"],
                        fisher_tensor=payload["static_fisher_by_layer"][layer_idx],
                        legacy_fisher_diag_tensor=(
                            payload.get("static_legacy_fisher_diag_by_layer", [None] * len(layers_local))[layer_idx]
                        ),
                        refined_A_list=refined_A_list_i,
                        refined_diag_A_list=refined_diag_A_list_i,
                        samples_per_A=payload.get("samples_per_A", samples_per_A),
                        attention_mask=payload["attention_mask"],
                        position_ids=payload["position_ids"],
                        position_embeddings=payload["position_embeddings"],
                        measure_samples=measure_samples_local,
                        measure_batch_size=args.measure_batch_size,
                        sample_offset=sample_offset,
                        kl_topk=args.kl_topk,
                        dev=payload["dev"],
                        measure_losses=measure_losses,
                        a_loss_ratio=args.a_loss_ratio,
                        grad_clip=_layer_grad_clip(layer_idx),
                        refined_mse_pool_ids=refined_mse_pool_ids_i,
                        refined_mse_grad_pool=refined_mse_grad_pool_i,
                        refined_mse_mean_grad=refined_mse_mean_grad_i,
                        fp_module_outputs=fp_module_outputs_i,
                        fp_weights=payload.get("fp_weights"),
                        hessians=payload.get("hessians"),
                        reg_strategy=args.grad_reg_strategy,
                        reg_lambda=args.grad_reg_lambda,
                    )
                layer_loss_results[layer_idx] = layer_loss_summary
                _log_layer_summary(layer_idx, layer_loss_summary)
                logging.info(
                    "Layer %d cosine+grad-norm (rank-local batch-avg): %s",
                    layer_idx,
                    ", ".join(
                        _fmt_entry(n, r)
                        for n, r in cosine_results[layer_idx].items()
                    ),
                )
            finally:
                for k in range(layer_idx + 1, len(layers_local)):
                    layers_local[k] = layers_local[k].to(payload["orig_device"])
                memory_utils.cleanup_memory()
        else:
            if "refined_mse" in measure_losses:
                for k in range(layer_idx + 1, len(layers_local)):
                    layers_local[k].to(payload["dev"])
                try:
                    (
                        refined_mse_pool_ids_i,
                        refined_mse_grad_pool_i,
                        refined_mse_mean_grad_i,
                    ) = _collect_refined_mse_for_layer(payload)
                finally:
                    for k in range(layer_idx + 1, len(layers_local)):
                        layers_local[k] = layers_local[k].to(payload["orig_device"])
            layer_loss_summary = measure_layer_losses_after_quant(
                analyzer=analyzer,
                layer=layer,
                layer_idx=layer_idx,
                inps=payload["inps"],
                fp_inps=payload["fp_inps"],
                fp_inps_final=payload["fp_inps_final"],
                fisher_tensor=payload["static_fisher_by_layer"][layer_idx],
                legacy_fisher_diag_tensor=(
                    payload.get("static_legacy_fisher_diag_by_layer", [None] * len(layers_local))[layer_idx]
                ),
                refined_A_list=refined_A_list_i,
                refined_diag_A_list=refined_diag_A_list_i,
                samples_per_A=payload.get("samples_per_A", samples_per_A),
                attention_mask=payload["attention_mask"],
                position_ids=payload["position_ids"],
                position_embeddings=payload["position_embeddings"],
                measure_samples=measure_samples_local,
                measure_batch_size=args.measure_batch_size,
                sample_offset=sample_offset,
                kl_topk=args.kl_topk,
                dev=payload["dev"],
                measure_losses=measure_losses,
                a_loss_ratio=args.a_loss_ratio,
                refined_mse_pool_ids=refined_mse_pool_ids_i,
                refined_mse_grad_pool=refined_mse_grad_pool_i,
                refined_mse_mean_grad=refined_mse_mean_grad_i,
                fp_module_outputs=fp_module_outputs_i,
            )
            layer_loss_results[layer_idx] = layer_loss_summary
            _log_layer_summary(layer_idx, layer_loss_summary)
            memory_utils.cleanup_memory()
        return None

    def _clear():
        fp_module_outputs_by_layer.clear()

    return _analysis_hook, _clear


# ---------------------------------------------------------------------------
# reference quantization helpers
# ---------------------------------------------------------------------------

def _apply_rtn_quant_to_module(args, module):
    quantizer = quant_utils.WeightQuantizer()
    quantizer.configure(
        args.w_bits,
        perchannel=True,
        sym=not args.w_asym,
        mse=args.w_clip,
        weight_groupsize=args.w_groupsize,
    )
    W = module.weight.data
    quantizer.find_params(W)
    q, _int_weight, _scale = quantizer.fake_quantize(W)
    module.weight.data = q.to(module.weight.data.dtype)
    return quantizer.cpu()


def _collect_layer_hessians_for_reg(args, analyzer, layer, inps, attention_mask,
                                    position_ids, position_embeddings, dev):
    if getattr(args, "grad_reg_strategy", "none") != "hessian" or getattr(args, "grad_reg_lambda", 0.0) <= 0:
        return None
    hessians = {}
    modules = _canonical_module_dict(analyzer, layer)
    accum = {name: None for name in modules}
    counts = {name: 0 for name in modules}
    handles = []

    def _hook(name):
        def _tmp(_module, inp, _out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 2:
                pass
            else:
                raise RuntimeError(
                    f"hessian regularizer capture expected 2D/3D input for {name}, "
                    f"got {tuple(x.shape)}."
                )
            x = x.float()
            block = x.t().matmul(x)
            if accum[name] is None:
                accum[name] = block
            else:
                accum[name].add_(block)
            counts[name] += int(x.shape[0])
        return _tmp

    for name, module in modules.items():
        handles.append(module.register_forward_hook(_hook(name)))
    try:
        bsz = int(getattr(args, "bsz", 1))
        bsz = max(1, min(bsz, inps.shape[0]))
        with disable_fp_path_quant(layer):
            for start in range(0, inps.shape[0], bsz):
                cur_bsz = min(bsz, inps.shape[0] - start)
                b_attn, b_pos_ids, b_pos_emb = _expand_batch_kwargs(
                    attention_mask, position_ids, position_embeddings, cur_bsz,
                )
                _ = _layer_out(layer(
                    inps[start:start + cur_bsz].to(dev),
                    attention_mask=b_attn,
                    position_ids=b_pos_ids,
                    position_embeddings=b_pos_emb,
                ))
    finally:
        for handle in handles:
            handle.remove()

    for name, block in accum.items():
        if block is None or counts[name] <= 0:
            raise RuntimeError(f"Failed to collect hessian regularizer state for {name}.")
        dist_utils.allreduce_sum_(block)
        total_count = dist_utils.allreduce_sum_scalar(counts[name])
        hessians[name] = (block / float(total_count)).detach().cpu()
    return hessians


def _load_or_collect_static_analysis_stats(
    *,
    args,
    analyzer,
    trainloader,
    dev,
    layers,
    want_fisher,
    want_legacy_fisher_diag,
    want_refined_full,
    want_refined_diag,
    need_fp_final,
):
    model = analyzer.model
    if (
        bool(getattr(args, "global_loss", False))
        and bool(getattr(args, "static_cache_path", None))
    ):
        static_cache_dir = getattr(args, "static_cache_path", None)
    else:
        static_cache_dir = None

    static_cache_file = None
    if static_cache_dir is not None:
        import hashlib

        dataset_id = getattr(args, "dataset", "unknown")
        rotate_flag = int(bool(getattr(args, "rotate", False)))
        rkl_na = int(getattr(args, "refined_rkl_num_A", 1))
        sink_size = (
            int(getattr(args, "attention_sink_size", 256))
            if bool(getattr(args, "ignore_attention_sink", False))
            else 0
        )
        key_bits = [
            getattr(args, "model_name", "model"),
            dataset_id,
            f"s{args.nsamples}",
            f"blk{args.seq_len}",
            f"rot{rotate_flag}",
            f"g{args.num_groups}",
            f"ghtk{args.grad_hessian_topk}",
            f"glbsz{args.global_loss_bsz}",
            f"seed{args.seed}",
            f"salclip{getattr(args, 'saliency_clip_percentile', 0.99)}",
            f"radk{int(getattr(args, 'fisher_rademacher_k', 0))}",
            f"ngrad{int(getattr(args, 'num_samples_for_grad', 0))}",
            f"rklNA{rkl_na}",
            f"fisher{int(want_fisher)}",
            f"legacydiag{int(want_legacy_fisher_diag)}",
            f"rkl{int(want_refined_full)}",
            f"diagrkl{int(want_refined_diag)}",
            f"fpfinal{int(need_fp_final)}",
            f"sink{sink_size}",
            f"world{dist_utils.get_world_size()}",
            f"rank{dist_utils.get_rank()}",
        ]
        key = "anagrad_" + hashlib.sha1("|".join(map(str, key_bits)).encode()).hexdigest()
        os.makedirs(static_cache_dir, exist_ok=True)
        static_cache_file = os.path.join(static_cache_dir, f"{key}.pt")
        if os.path.exists(static_cache_file):
            logging.info("Loading analyze static fisher/fp-final cache from %s", static_cache_file)
            loaded = torch.load(static_cache_file, map_location="cpu", weights_only=True)
            static_fisher_by_layer = loaded.get("fisher", [None] * len(layers))
            static_legacy_fisher_diag_by_layer = loaded.get(
                "legacy_fisher_diag", [None] * len(layers)
            )
            static_refined_A_by_layer = loaded.get("refined_A", None)
            static_refined_diag_A_by_layer = loaded.get("refined_diag_A", None)
            fp_inps_final_cpu = loaded.get("fp_inps_final", None)
            if want_fisher and not static_fisher_by_layer:
                raise RuntimeError(
                    f"Cached analyze stats at {static_cache_file} do not contain fisher."
                )
            if (
                want_legacy_fisher_diag
                and not any(f is not None for f in static_legacy_fisher_diag_by_layer)
            ):
                raise RuntimeError(
                    f"Cached analyze stats at {static_cache_file} do not contain "
                    "per-token legacy Fisher diagonal."
                )
            if want_refined_full and not static_refined_A_by_layer:
                raise RuntimeError(
                    f"Cached analyze stats at {static_cache_file} do not contain refined_A."
                )
            if want_refined_diag and not static_refined_diag_A_by_layer:
                raise RuntimeError(
                    f"Cached analyze stats at {static_cache_file} do not contain refined_diag_A."
                )
            if need_fp_final and fp_inps_final_cpu is None:
                raise RuntimeError(
                    f"Cached analyze stats at {static_cache_file} do not contain fp_inps_final."
                )
            del loaded
            return (
                static_fisher_by_layer,
                static_legacy_fisher_diag_by_layer,
                static_refined_A_by_layer,
                static_refined_diag_A_by_layer,
                fp_inps_final_cpu,
            )

    if want_fisher or want_legacy_fisher_diag or want_refined_full or want_refined_diag or need_fp_final:
        logging.info("Collecting analyze static fisher/fp-final caches for reference quantization.")
        (
            _static_saliency_unused,
            static_fisher_by_layer,
            static_legacy_fisher_diag_by_layer,
            static_refined_A_by_layer,
            static_refined_diag_A_by_layer,
            fp_inps_final_cpu,
            _static_dynsal_unused,
        ) = collect_static_end_to_end_saliency_and_fisher(
            model=model,
            analyzer=analyzer,
            dataloader=trainloader,
            dev=dev,
            saliency_num_groups=args.num_groups,
            grad_hessian_topk=args.grad_hessian_topk,
            batch_size=args.global_loss_bsz,
            collect_fisher=want_fisher,
            collect_legacy_fisher_diag=want_legacy_fisher_diag,
            collect_refined_rkl=want_refined_full,
            refined_rkl_damp=getattr(args, "refined_rkl_damp", 0.01),
            refined_rkl_num_A=int(getattr(args, "refined_rkl_num_A", 1)),
            collect_refined_diag_rkl=want_refined_diag,
            use_fsdp=bool(getattr(args, "fsdp_precompute", False)),
            fsdp_cpu_offload=bool(getattr(args, "fsdp_cpu_offload", False)),
            saliency_clip_percentile=getattr(args, "saliency_clip_percentile", 0.99),
            capture_fp_final=need_fp_final,
            collect_saliency=False,
            collect_dynsal=False,
            sink_size=(
                int(getattr(args, "attention_sink_size", 256))
                if bool(getattr(args, "ignore_attention_sink", False))
                else 0
            ),
            fisher_rademacher_k=int(getattr(args, "fisher_rademacher_k", 0)),
            rademacher_seed=int(getattr(args, "refresh_seed", 0)) + 1701,
            num_samples_for_grad=int(getattr(args, "num_samples_for_grad", 0)),
        )
        if bool(getattr(args, "exit_after_precompute", False)):
            logging.info(
                "exit_after_precompute=1 -> finished analyze static precompute, exiting."
            )
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            raise SystemExit(0)
    else:
        static_fisher_by_layer = [None] * len(layers)
        static_legacy_fisher_diag_by_layer = [None] * len(layers)
        static_refined_A_by_layer = None
        static_refined_diag_A_by_layer = None
        fp_inps_final_cpu = None

    if static_cache_file is not None:
        logging.info("Saving analyze static fisher/fp-final cache to %s", static_cache_file)
        payload = {
            "fisher": static_fisher_by_layer,
            "legacy_fisher_diag": static_legacy_fisher_diag_by_layer,
        }
        if static_refined_A_by_layer is not None:
            payload["refined_A"] = static_refined_A_by_layer
        if static_refined_diag_A_by_layer is not None:
            payload["refined_diag_A"] = static_refined_diag_A_by_layer
        if fp_inps_final_cpu is not None:
            payload["fp_inps_final"] = fp_inps_final_cpu
        torch.save(payload, static_cache_file)
        del payload

    return (
        static_fisher_by_layer,
        static_legacy_fisher_diag_by_layer,
        static_refined_A_by_layer,
        static_refined_diag_A_by_layer,
        fp_inps_final_cpu,
    )


@torch.no_grad()
def _rtn_fwrd_with_analysis(
    args,
    analyzer,
    trainloader,
    dev,
    analysis_hook,
    *,
    want_fisher,
    want_legacy_fisher_diag,
    want_refined_full,
    want_refined_diag,
    need_fp_final,
):
    logging.info("-----RTN + grad-cosine analysis quantization-----")
    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device
    if args.offload_inps:
        raise NotImplementedError(
            "analyze_grad_cosine --analysis_quant_method=rtn does not support "
            "--offload_inps yet. Disable offload_inps or use gptq_plus."
        )
    if bool(getattr(args, "act_quant_aware_gptq", False)) or bool(
        getattr(args, "k_cache_quant_aware_gptq", False)
    ):
        raise ValueError(
            "RTN reference quantization in analyze_grad_cosine does not support "
            "act/K-cache quant-aware GPTQ options."
        )

    if not getattr(args, "global_loss", False):
        logging.info(
            "analyze_grad_cosine RTN reference needs static end-to-end stats for "
            "the selected losses; forcing --global_loss for this diagnostic run."
        )
        args.global_loss = True

    (
        static_fisher_by_layer,
        static_legacy_fisher_diag_by_layer,
        static_refined_A_by_layer,
        static_refined_diag_A_by_layer,
        fp_inps_final_cpu,
    ) = _load_or_collect_static_analysis_stats(
        args=args,
        analyzer=analyzer,
        trainloader=trainloader,
        dev=dev,
        layers=layers,
        want_fisher=want_fisher,
        want_legacy_fisher_diag=want_legacy_fisher_diag,
        want_refined_full=want_refined_full,
        want_refined_diag=want_refined_diag,
        need_fp_final=need_fp_final,
    )

    per_layer_runtime_modules = list(analyzer.get_pre_block_modules())
    per_layer_runtime_modules.extend(
        [analyzer.get_layernorm_before_head(), analyzer.get_lm_head()]
    )
    for module in per_layer_runtime_modules:
        module.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    dp_world = dist_utils.get_world_size()
    dp_rank = dist_utils.get_rank()
    if args.nsamples % dp_world != 0:
        raise ValueError(
            f"nsamples ({args.nsamples}) must be divisible by world_size ({dp_world}) for DP."
        )
    n_local = args.nsamples // dp_world
    dp_shard = slice(dp_rank * n_local, (dp_rank + 1) * n_local)
    inps = torch.zeros(
        (n_local, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"global_i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            global_i = cache["global_i"]
            if dp_shard.start <= global_i < dp_shard.stop:
                inps[global_i - dp_shard.start] = inp
            cache["global_i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in trainloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory(False)

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    fp_inps = inps.clone()
    fp_inps_final = (
        None
        if fp_inps_final_cpu is None
        else fp_inps_final_cpu.to(device=fp_inps.device, dtype=fp_inps.dtype)
    )
    fp_inps_final_cpu = None

    quantizers = {}
    refined_rkl_num_A = int(getattr(args, "refined_rkl_num_A", 1))
    samples_per_A = (
        args.nsamples // refined_rkl_num_A
        if refined_rkl_num_A > 1 else 0
    )
    preclip_enabled = bool(args.w_clip and getattr(args, "pre_clip", True))

    pbar = tqdm(range(len(layers)), ncols=120, desc="RTN Quantizing Layers", position=0)
    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)
        analysis_is_target = bool(analysis_hook("before_layer", {
            "args": args,
            "analyzer": analyzer,
            "layer": layer,
            "layer_idx": i,
            "layers": layers,
            "full": full,
            "inps": inps,
            "fp_inps": fp_inps,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "dev": dev,
            "orig_device": orig_device,
        }))

        if preclip_enabled:
            for name, module in full.items():
                if module is None or "lm_head" in name:
                    continue
                clip_module_weight_to_quant_bounds_(
                    module,
                    bits=args.w_bits,
                    sym=not args.w_asym,
                    mse=args.w_clip,
                )

        analysis_fp_weights = None
        if analysis_is_target:
            analysis_fp_weights = {
                raw_name[:-7] if raw_name.endswith(".module") else raw_name:
                mod.weight.detach().clone().cpu()
                for raw_name, mod in full.items()
                if mod is not None and hasattr(mod, "weight")
            }
        analysis_hessians = _collect_layer_hessians_for_reg(
            args,
            analyzer,
            layer,
            inps,
            attention_mask,
            position_ids,
            position_embeddings,
            dev,
        ) if analysis_is_target else None

        analysis_hook("before_fp_reference", {
            "args": args,
            "analyzer": analyzer,
            "layer": layer,
            "layer_idx": i,
            "layers": layers,
            "full": full,
            "inps": inps,
            "fp_inps": fp_inps,
            "fp_inps_final": fp_inps_final,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "dev": dev,
            "orig_device": orig_device,
            "is_target": analysis_is_target,
        })

        with disable_fp_path_quant(layer):
            for j in range(inps.shape[0]):
                fp_inps[j] = _layer_out(layer(
                    fp_inps[j].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )).squeeze(0).to(fp_inps.device)

        for name, module in full.items():
            if module is None or "lm_head" in name:
                continue
            pbar.set_postfix(module=f"layers.{i}." + name)
            quantizers["model.layers.%d.%s" % (i, name)] = _apply_rtn_quant_to_module(
                args,
                module,
            )

        analysis_hook("after_layer_quantized", {
            "args": args,
            "analyzer": analyzer,
            "layer": layer,
            "layer_idx": i,
            "layers": layers,
            "full": full,
            "inps": inps,
            "fp_inps": fp_inps,
            "fp_inps_final": fp_inps_final,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "dev": dev,
            "orig_device": orig_device,
            "static_fisher_by_layer": static_fisher_by_layer,
            "static_legacy_fisher_diag_by_layer": static_legacy_fisher_diag_by_layer,
            "static_refined_A_by_layer": static_refined_A_by_layer,
            "static_refined_diag_A_by_layer": static_refined_diag_A_by_layer,
            "samples_per_A": samples_per_A,
            "fp_weights": analysis_fp_weights,
            "hessians": analysis_hessians,
            "is_target": analysis_is_target,
        })

        for j in range(inps.shape[0]):
            inps[j] = _layer_out(layer(
                inps[j].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )).squeeze(0).to(inps.device)

        layers[i] = layer.to(orig_device)
        del layer, full, analysis_fp_weights, analysis_hessians
        memory_utils.cleanup_memory()
        dist_utils.barrier()

    for module in per_layer_runtime_modules:
        module.to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----RTN + grad-cosine analysis done-----")
    return quantizers


@torch.no_grad()
def _gptaq_fwrd_with_analysis(
    args,
    analyzer,
    trainloader,
    dev,
    analysis_hook,
    *,
    want_fisher,
    want_legacy_fisher_diag,
    want_refined_full,
    want_refined_diag,
    need_fp_final,
):
    logging.info("-----GPTAQ + grad-cosine analysis quantization-----")
    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    if not getattr(args, "global_loss", False):
        logging.info(
            "analyze_grad_cosine GPTAQ reference needs static end-to-end stats for "
            "the selected losses; forcing --global_loss for this diagnostic run."
        )
        args.global_loss = True

    (
        static_fisher_by_layer,
        static_legacy_fisher_diag_by_layer,
        static_refined_A_by_layer,
        static_refined_diag_A_by_layer,
        fp_inps_final_cpu,
    ) = _load_or_collect_static_analysis_stats(
        args=args,
        analyzer=analyzer,
        trainloader=trainloader,
        dev=dev,
        layers=layers,
        want_fisher=want_fisher,
        want_legacy_fisher_diag=want_legacy_fisher_diag,
        want_refined_full=want_refined_full,
        want_refined_diag=want_refined_diag,
        need_fp_final=need_fp_final,
    )

    per_layer_runtime_modules = list(analyzer.get_pre_block_modules())
    per_layer_runtime_modules.extend(
        [analyzer.get_layernorm_before_head(), analyzer.get_lm_head()]
    )
    for module in per_layer_runtime_modules:
        module.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    dp_world = dist_utils.get_world_size()
    dp_rank = dist_utils.get_rank()
    if args.nsamples % dp_world != 0:
        raise ValueError(
            f"nsamples ({args.nsamples}) must be divisible by world_size ({dp_world}) for DP."
        )
    n_local = args.nsamples // dp_world
    dp_shard = slice(dp_rank * n_local, (dp_rank + 1) * n_local)
    inps = torch.zeros(
        (n_local, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"global_i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            global_i = cache["global_i"]
            if dp_shard.start <= global_i < dp_shard.stop:
                inps[global_i - dp_shard.start] = inp
            cache["global_i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in trainloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory(False)

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    quantizers = {}
    sequential = analyzer.get_sequential_quantizable_module_names()
    fp_inputs_cache = FPInputsCache(sequential)
    fp_inps = inps.clone()
    fp_inps_final = (
        None
        if fp_inps_final_cpu is None
        else fp_inps_final_cpu.to(device=fp_inps.device, dtype=fp_inps.dtype)
    )
    fp_inps_final_cpu = None

    if bool(getattr(args, "act_quant_aware_gptq", False)):
        input_q_count, v_q_count = configure_activation_quantizers_for_gptq(args, model)
        logging.info(
            "act_quant_aware_gptq enabled for GPTAQ analysis: student paths use "
            "A/V fake quantization after FP teacher caches are captured "
            "(input_wrappers=%d v_out_wrappers=%d, a_bits=%d, v_bits=%d).",
            input_q_count,
            v_q_count,
            args.a_bits,
            args.v_bits,
        )

    if bool(getattr(args, "k_cache_quant_aware_gptq", False)):
        k_q_count = configure_k_cache_quantizers_for_gptq(args, analyzer)
        logging.info(
            "k_cache_quant_aware_gptq enabled for GPTAQ analysis: student paths use "
            "online K fake quantization after RoPE/QK rotation "
            "(qk_wrappers=%d, k_bits=%d).",
            k_q_count,
            args.k_bits,
        )

    if args.offload_inps:
        inps = inps.cpu()
        fp_inps = fp_inps.cpu()
        if fp_inps_final is not None:
            fp_inps_final = fp_inps_final.cpu()

    refined_rkl_num_A = int(getattr(args, "refined_rkl_num_A", 1))
    samples_per_A = (
        args.nsamples // refined_rkl_num_A
        if refined_rkl_num_A > 1 else 0
    )

    pbar = tqdm(range(len(layers)), ncols=120, desc="GPTAQ Quantizing Layers", position=0)
    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)
        analysis_is_target = bool(analysis_hook("before_layer", {
            "args": args,
            "analyzer": analyzer,
            "layer": layer,
            "layer_idx": i,
            "layers": layers,
            "full": full,
            "inps": inps,
            "fp_inps": fp_inps,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "dev": dev,
            "orig_device": orig_device,
        }))

        analysis_hook("before_fp_reference", {
            "args": args,
            "analyzer": analyzer,
            "layer": layer,
            "layer_idx": i,
            "layers": layers,
            "full": full,
            "inps": inps,
            "fp_inps": fp_inps,
            "fp_inps_final": fp_inps_final,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "dev": dev,
            "orig_device": orig_device,
            "is_target": analysis_is_target,
        })

        with disable_fp_path_quant(layer):
            fp_inputs_cache.add_hook(full)
            for j in range(fp_inps.shape[0]):
                fp_inps[j] = _layer_out(layer(
                    fp_inps[j].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )).to(fp_inps.device)
            fp_inputs_cache.clear_hook()

        analysis_fp_weights = None
        if analysis_is_target:
            analysis_fp_weights = {
                raw_name[:-7] if raw_name.endswith(".module") else raw_name:
                mod.weight.detach().clone().cpu()
                for raw_name, mod in full.items()
                if mod is not None and hasattr(mod, "weight")
            }
        analysis_hessians = _collect_layer_hessians_for_reg(
            args,
            analyzer,
            layer,
            inps,
            attention_mask,
            position_ids,
            position_embeddings,
            dev,
        ) if analysis_is_target else None

        for names in sequential:
            subset = {n: full.get(n, full.get(n + ".module", None)) for n in names}
            subset = {n: m for n, m in subset.items() if m is not None and "lm_head" not in n}
            if not subset:
                continue

            gptq = {}
            for name, module in subset.items():
                gptq[name] = GPTAQ(module)
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    args.w_bits,
                    perchannel=True,
                    sym=not args.w_asym,
                    mse=args.w_clip,
                )
                gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            first_module_name = list(subset.keys())[0]
            handle = subset[first_module_name].register_forward_hook(
                add_batch(first_module_name)
            )
            for j in range(inps.shape[0]):
                _ = _layer_out(layer(
                    inps[j].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                ))
            handle.remove()

            if dp_world > 1:
                dist_utils.allreduce_sum_(gptq[first_module_name].H)
                dist_utils.allreduce_sum_(gptq[first_module_name].dXXT)
                gptq[first_module_name].H.div_(float(dp_world))
                gptq[first_module_name].dXXT.div_(float(dp_world))

            for name in subset:
                if name != first_module_name:
                    gptq[name].H = gptq[first_module_name].H
                    gptq[name].dXXT = gptq[first_module_name].dXXT

            for name in subset:
                pbar.set_postfix(module=f"layers.{i}." + name)
                gptq[name].fasterquant(
                    blocksize=args.blocksize,
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    actorder=args.act_order,
                    static_groups=args.act_order,
                )
                if getattr(args, "enable_debug", False):
                    dist_utils.assert_bit_exact(
                        subset[name].weight.data,
                        tag=f"gptaq_analysis.layer{i}.{name}.weight_after_fasterquant",
                    )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        analysis_hook("after_layer_quantized", {
            "args": args,
            "analyzer": analyzer,
            "layer": layer,
            "layer_idx": i,
            "layers": layers,
            "full": full,
            "inps": inps,
            "fp_inps": fp_inps,
            "fp_inps_final": fp_inps_final,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "dev": dev,
            "orig_device": orig_device,
            "static_fisher_by_layer": static_fisher_by_layer,
            "static_legacy_fisher_diag_by_layer": static_legacy_fisher_diag_by_layer,
            "static_refined_A_by_layer": static_refined_A_by_layer,
            "static_refined_diag_A_by_layer": static_refined_diag_A_by_layer,
            "samples_per_A": samples_per_A,
            "fp_weights": analysis_fp_weights,
            "hessians": analysis_hessians,
            "is_target": analysis_is_target,
        })

        for j in range(inps.shape[0]):
            inps[j] = _layer_out(layer(
                inps[j].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )).to(inps.device)

        fp_inputs_cache.clear_cache()
        layers[i] = layer.to(orig_device)
        del layer, full, gptq, analysis_fp_weights, analysis_hessians
        memory_utils.cleanup_memory()
        dist_utils.barrier()

    for module in per_layer_runtime_modules:
        module.to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----GPTAQ + grad-cosine analysis done-----")
    return quantizers


# ---------------------------------------------------------------------------
# main pipeline — selectable reference quantization with measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def quantize_and_measure(args, analyzer, trainloader, dev, target_layers, measure_losses):
    analysis_quant_method = getattr(args, "analysis_quant_method", "rtn")
    logging.info(
        "----- %s + grad-cosine analysis -----",
        analysis_quant_method.upper(),
    )
    if analysis_quant_method not in {"rtn", "gptaq", "gptq_plus"}:
        raise ValueError(
            "--analysis_quant_method currently supports {'rtn', 'gptaq', 'gptq_plus'} "
            f"for target-layer cosine measurement, got {analysis_quant_method!r}."
        )
    layers = analyzer.get_layers()
    cosine_results = {}
    layer_loss_results = {}
    dp_world = dist_utils.get_world_size()
    if args.measure_samples % dp_world != 0:
        raise ValueError(
            f"measure_samples ({args.measure_samples}) must be divisible by "
            f"WORLD_SIZE ({dp_world}) for DP cosine analysis."
        )
    measure_samples_local = args.measure_samples // dp_world
    if measure_samples_local <= 0:
        raise ValueError(
            f"measure_samples ({args.measure_samples}) gives zero samples per rank "
            f"with WORLD_SIZE={dp_world}."
        )
    if measure_samples_local % args.measure_batch_size != 0:
        raise ValueError(
            f"local measure_samples ({measure_samples_local} = global "
            f"{args.measure_samples} / WORLD_SIZE {dp_world}) must be divisible by "
            f"measure_batch_size ({args.measure_batch_size})."
        )
    if args.nsamples % dp_world != 0:
        raise ValueError(
            f"nsamples ({args.nsamples}) must be divisible by WORLD_SIZE "
            f"({dp_world}) for DP cosine analysis."
        )
    rank_sample_start = dist_utils.get_rank() * (args.nsamples // dp_world)
    logging.info(
        "Cosine measurement uses %d global samples = %d rank-local samples "
        "per rank (world_size=%d, rank%d shard starts at global sample %d).",
        args.measure_samples,
        measure_samples_local,
        dp_world,
        dist_utils.get_rank(),
        rank_sample_start,
    )
    sample_offset = rank_sample_start

    want_fisher = bool(
        measure_losses.intersection({"fisher_diag_mse", "refined_mse"})
    )
    want_legacy_fisher_diag = "legacy_fisher_diag_mse" in measure_losses
    want_refined_full = "refined_residual_kl" in measure_losses
    want_refined_diag = "refined_diag_residual_kl" in measure_losses
    want_module_mse = "module_mse" in measure_losses
    need_fp_final = bool(
        target_layers
        or measure_losses.intersection(
            {"residual_kl", "refined_residual_kl", "refined_diag_residual_kl", "refined_mse"}
        )
    )
    if (
        want_fisher
        or want_legacy_fisher_diag
        or want_refined_full
        or want_refined_diag
    ) and not args.global_loss:
        logging.info(
            "analyze_grad_cosine requires global static stats for selected losses; "
            "forcing --global_loss for this diagnostic run."
        )
        args.global_loss = True
    if want_legacy_fisher_diag:
        if int(getattr(args, "fisher_rademacher_k", 0)) > 0:
            raise ValueError(
                "legacy_fisher_diag_mse grad-cosine uses per-token g^2 from a "
                "sum-reduced NLL backward and cannot be used with "
                "--fisher_rademacher_k > 0."
            )
        if int(getattr(args, "num_samples_for_grad", 0)) > 0:
            raise ValueError(
                "legacy_fisher_diag_mse grad-cosine needs per-token Fisher "
                "diagonals for all calibration samples and cannot be used with "
                "--num_samples_for_grad > 0."
            )
    if want_module_mse and measure_samples_local % args.measure_batch_size != 0:
        raise ValueError(
            f"local measure_samples ({measure_samples_local}) must be divisible by "
            f"measure_batch_size ({args.measure_batch_size}) for module_mse."
        )
    if args.measure_samples > args.nsamples:
        raise ValueError(
            f"measure_samples ({args.measure_samples}) must be <= nsamples "
            f"({args.nsamples})."
        )
    analysis_hook, clear_analysis_state = _build_cosine_analysis_hook(
        args=args,
        analyzer=analyzer,
        layers=layers,
        target_layers=target_layers,
        measure_losses=measure_losses,
        measure_samples_local=measure_samples_local,
        sample_offset=sample_offset,
        dev=dev,
        cosine_results=cosine_results,
        layer_loss_results=layer_loss_results,
    )

    if analysis_quant_method == "rtn":
        quantizers = _rtn_fwrd_with_analysis(
            args,
            analyzer,
            trainloader,
            dev,
            analysis_hook,
            want_fisher=want_fisher,
            want_legacy_fisher_diag=want_legacy_fisher_diag,
            want_refined_full=want_refined_full,
            want_refined_diag=want_refined_diag,
            need_fp_final=need_fp_final,
        )
        clear_analysis_state()
        memory_utils.cleanup_memory()
        logging.info("----- %s + grad-cosine done -----", analysis_quant_method.upper())
        return quantizers, cosine_results, layer_loss_results

    if analysis_quant_method == "gptaq":
        if getattr(args, "w_method", None) != "gptaq":
            logging.info(
                "analysis_quant_method=gptaq requires GPTAQ reference path; "
                "overriding w_method=%s -> gptaq.",
                getattr(args, "w_method", None),
            )
            args.w_method = "gptaq"
        quantizers = _gptaq_fwrd_with_analysis(
            args,
            analyzer,
            trainloader,
            dev,
            analysis_hook,
            want_fisher=want_fisher,
            want_legacy_fisher_diag=want_legacy_fisher_diag,
            want_refined_full=want_refined_full,
            want_refined_diag=want_refined_diag,
            need_fp_final=need_fp_final,
        )
        clear_analysis_state()
        memory_utils.cleanup_memory()
        logging.info("----- %s + grad-cosine done -----", analysis_quant_method.upper())
        return quantizers, cosine_results, layer_loss_results

    if getattr(args, "w_method", None) != "gptq_plus":
        logging.info(
            "analysis_quant_method=gptq_plus requires GPTQ+ reference path; "
            "overriding w_method=%s -> gptq_plus.",
            getattr(args, "w_method", None),
        )
        args.w_method = "gptq_plus"

    old_attrs = {}
    for name, value in {
        "_analysis_hook": analysis_hook,
        "_analysis_collect_fisher": want_fisher,
        "_analysis_collect_legacy_fisher_diag": want_legacy_fisher_diag,
        "_analysis_collect_refined_rkl": want_refined_full,
        "_analysis_collect_refined_diag_rkl": want_refined_diag,
        "_analysis_need_fp_inps_final": need_fp_final,
    }.items():
        old_attrs[name] = getattr(args, name, None)
        setattr(args, name, value)

    try:
        quantizers = gptq_plus_fwrd(args, analyzer, trainloader, dev)
    finally:
        for name, old_value in old_attrs.items():
            if old_value is None:
                try:
                    delattr(args, name)
                except AttributeError:
                    pass
            else:
                setattr(args, name, old_value)
        clear_analysis_state()
        memory_utils.cleanup_memory()

    logging.info("----- GPTQ+ + grad-cosine done -----")
    return quantizers, cosine_results, layer_loss_results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    configure_reproducibility(args.refresh_seed, deterministic=True)
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist_utils.init_process_group()
    dp_world = dist_utils.get_world_size()
    if args.measure_samples % dp_world != 0:
        raise ValueError(
            f"measure_samples ({args.measure_samples}) must be divisible by "
            f"WORLD_SIZE ({dp_world}) for DP cosine analysis."
        )
    _measure_samples_local = args.measure_samples // dp_world
    if _measure_samples_local % args.measure_batch_size != 0:
        raise ValueError(
            f"local measure_samples ({_measure_samples_local} = global "
            f"{args.measure_samples} / WORLD_SIZE {dp_world}) must be divisible "
            f"by measure_batch_size ({args.measure_batch_size})."
        )

    analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model
    tokenizer = analyzer.tokenizer
    model_pre_rotated = bool(getattr(model, "_gptqplus_checkpoint_is_rotated", False))

    target_layers = set(_parse_target_layers(args.target_layers, analyzer.num_layers))
    if not target_layers:
        logging.warning("No valid target layers after parsing; nothing to measure.")
    else:
        logging.info("Will measure cosine at layers: %s", sorted(target_layers))

    measure_losses = _parse_measure_losses(getattr(args, "measure_losses", None))
    if not measure_losses:
        raise ValueError(
            "--measure_losses parsed to empty set. Specify at least one of: "
            f"{sorted(_VALID_MEASURE_LOSSES)}"
        )
    logging.info("Will measure surrogate losses: %s", sorted(measure_losses))

    # Regularization: only the additive variants (l2 / hessian) are supported
    # here. `quant_error_gate` / `quant_error_gate_optimized` are gates on the
    # optimizer update (multiplicative), not additive first-order gradients, so
    # "cos(true, reg_grad)" is not defined for them. Reject early.
    if getattr(args, "grad_reg_strategy", "none") in (
        "quant_error_gate", "quant_error_gate_optimized",
    ):
        raise ValueError(
            f"analyze_grad_cosine only supports grad_reg_strategy in "
            f"{{none, l2, hessian}}. Got {args.grad_reg_strategy!r}. Gate "
            f"variants multiply the optimizer update, not the gradient, and "
            f"have no additive-gradient interpretation."
        )
    if getattr(args, "grad_reg_lambda", 0.0) < 0:
        raise ValueError(
            f"--grad_reg_lambda must be >= 0, got {args.grad_reg_lambda}."
        )
    logging.info(
        "Regularization for cosine analysis: strategy=%s lambda=%s",
        args.grad_reg_strategy, args.grad_reg_lambda,
    )

    if args.rotate and not model_pre_rotated:
        rotation_utils.fuse_layer_norms(analyzer)
        rotation_utils.rotate_model(args, analyzer)
        memory_utils.cleanup_memory()

        if dist_utils.get_world_size() > 1:
            _cuda_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            for p in model.parameters():
                data = p.data
                if not data.is_contiguous():
                    data = data.contiguous()
                if not data.is_cuda:
                    data = data.to(_cuda_dev)
                dist.broadcast(data, src=0)
                if data.data_ptr() != p.data.data_ptr():
                    p.data = data

        rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
    elif args.rotate and model_pre_rotated:
        logging.info(
            "Model was loaded from a pre-rotated checkpoint; installing rotation "
            "wrappers and skipping in-process rotation."
        )
        rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
    else:
        quant_utils.add_actquant(analyzer)

    model.cpu()
    remove_hook_from_module(model, recurse=True)

    # Calibration tokens.
    trainloader = data_utils.get_tokens(
        args.dataset, "train", tokenizer, args.seq_len, args.nsamples,
        args.tokens_cache_path, args.seed,
    )
    if isinstance(trainloader[0], torch.Tensor):
        assert trainloader[0].dim() == 1
        trainloader = [(x.unsqueeze(0), None) for x in trainloader]

    dp_dev = (
        f"cuda:{torch.cuda.current_device()}"
        if torch.cuda.is_available() else "cpu"
    )

    _, cosine_results, layer_loss_results = quantize_and_measure(
        args, analyzer, trainloader, dp_dev, target_layers, measure_losses,
    )
    cosine_results, layer_loss_results = _gather_analysis_results(
        cosine_results,
        layer_loss_results,
    )

    if dist_utils.is_main():
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        out_pt = os.path.join(out_dir, "grad_cosine_results.pt")
        torch.save(
            {
                "results": cosine_results,
                "layer_losses": layer_loss_results,
                "args": vars(args),
            },
            out_pt,
        )
        logging.info("Saved cosine results to %s", out_pt)

        # ---------- TSV table (long format) ----------
        # Columns: layer  module  loss  cos_no_reg  cos_reg  cos_with_reg
        #          norm_true  norm_loss  norm_reg  norm_with_reg
        # One row per (layer, module, loss). `cos_reg` and `norm_reg` repeat
        # across loss rows within the same (layer, module) — they depend on
        # the weight diff and H, not the loss.
        out_tsv = os.path.join(out_dir, "grad_cosine_table.txt")
        # Match the measure-loss order the user asked for on the CLI
        # (deterministic iteration instead of set order).
        cli_order = []
        if getattr(args, "measure_losses", None):
            seen = set()
            for chunk in args.measure_losses.split(","):
                name = chunk.strip()
                if name and name not in seen and name in _VALID_MEASURE_LOSSES:
                    cli_order.append(name)
                    seen.add(name)
        _LOSS_COL_SPEC = [
            ("fisher_diag_mse", "fisher_mean", "fisher_grad_norm_mean"),
            ("legacy_fisher_diag_mse", "legacy_fisher_diag_mse_mean",
             "legacy_fisher_diag_mse_grad_norm_mean"),
            ("residual_kl", "residual_kl_mean", "residual_kl_grad_norm_mean"),
            ("refined_residual_kl", "refined_residual_kl_mean",
             "refined_residual_kl_grad_norm_mean"),
            ("refined_diag_residual_kl", "refined_diag_residual_kl_mean",
             "refined_diag_residual_kl_grad_norm_mean"),
            ("refined_mse", "refined_mse_mean", "refined_mse_grad_norm_mean"),
            ("layer_mse", "layer_mse_mean", "layer_mse_grad_norm_mean"),
            ("module_mse", "module_mse_mean", "module_mse_grad_norm_mean"),
        ]
        _LOSS_BY_NAME = {name: (cos_key, norm_key) for name, cos_key, norm_key in _LOSS_COL_SPEC}
        if cli_order:
            ordered_losses = cli_order
        else:
            ordered_losses = [name for name, _, _ in _LOSS_COL_SPEC]

        def _fmt_cos(x):
            return "nan" if x is None or not isinstance(x, (int, float)) or x != x else f"{x:.6f}"

        def _fmt_norm(x):
            if x is None or not isinstance(x, (int, float)) or x != x:
                return "nan"
            return f"{x:.6e}"

        header_lines = [
            f"# analyze_grad_cosine results",
            f"# model={args.model}  exp={args.exp}",
            f"# analysis_quant_method={args.analysis_quant_method}",
            f"# reg_strategy={args.grad_reg_strategy}  reg_lambda={args.grad_reg_lambda}",
            f"# a_loss_ratio={args.a_loss_ratio}",
            f"# measure_losses={args.measure_losses}  target_layers={args.target_layers}",
            f"# measure_samples={args.measure_samples}  measure_batch_size={args.measure_batch_size}",
            f"# columns: cos_* are global batch-mean cosine(true_KL_grad, ·); "
            f"norm_* are batch-mean L2 norm of the flattened grad.",
            f"# cos_reg/norm_reg repeat across loss rows within the same (layer, module).",
            f"# cos_with_reg = cos(true, surrogate + reg_grad); "
            f"norm_with_reg = ||surrogate + reg_grad||_2.",
        ]
        cols = [
            "layer", "module", "loss",
            "cos_no_reg", "cos_reg", "cos_with_reg",
            "norm_true", "norm_loss", "norm_reg", "norm_with_reg",
        ]
        rows = []
        layer_loss_cos_values = {}
        layer_loss_cos_with_reg_values = {}
        layer_loss_norm_values = {}
        for layer_idx in sorted(cosine_results.keys()):
            per_layer = cosine_results[layer_idx]
            for module_name in per_layer:
                r = per_layer[module_name]
                reg_enabled = bool(r.get("reg_enabled", False))
                reg_cos = r.get("reg_cos_mean", float("nan")) if reg_enabled else float("nan")
                reg_norm = r.get("reg_grad_norm", float("nan")) if reg_enabled else float("nan")
                norm_true = r.get("true_kl_grad_norm_mean", float("nan"))
                for loss_name in ordered_losses:
                    if loss_name not in _LOSS_BY_NAME:
                        continue
                    cos_key, norm_key = _LOSS_BY_NAME[loss_name]
                    cos_no_reg = r.get(cos_key, float("nan"))
                    norm_loss = r.get(norm_key, float("nan"))
                    combined_cos_key = f"{loss_name}_combined_cos_mean"
                    combined_norm_key = f"{loss_name}_combined_grad_norm_mean"
                    cos_with = r.get(combined_cos_key, float("nan")) if reg_enabled else float("nan")
                    norm_with = r.get(combined_norm_key, float("nan")) if reg_enabled else float("nan")
                    # Skip rows where this loss wasn't measured at this layer
                    # (keeps the table tight instead of filling with NaN).
                    if cos_key not in r:
                        continue
                    layer_loss_cos_values.setdefault((layer_idx, loss_name), []).append(cos_no_reg)
                    layer_loss_norm_values.setdefault((layer_idx, loss_name), []).append(norm_loss)
                    if reg_enabled and isinstance(cos_with, (int, float)) and cos_with == cos_with:
                        layer_loss_cos_with_reg_values.setdefault((layer_idx, loss_name), []).append(cos_with)
                    rows.append([
                        str(layer_idx), module_name, loss_name,
                        _fmt_cos(cos_no_reg), _fmt_cos(reg_cos), _fmt_cos(cos_with),
                        _fmt_norm(norm_true), _fmt_norm(norm_loss),
                        _fmt_norm(reg_norm), _fmt_norm(norm_with),
                    ])
        with open(out_tsv, "w") as f:
            for line in header_lines:
                f.write(line + "\n")
            f.write("\t".join(cols) + "\n")
            for row in rows:
                f.write("\t".join(row) + "\n")
        logging.info("Wrote cosine table to %s (%d rows)", out_tsv, len(rows))

        layer_cos_rows = []
        layer_cos_summary = {}
        for layer_idx in sorted(cosine_results.keys()):
            for loss_name in ordered_losses:
                values = [
                    float(v) for v in layer_loss_cos_values.get((layer_idx, loss_name), [])
                    if isinstance(v, (int, float)) and v == v
                ]
                if not values:
                    continue
                norm_values = [
                    float(v) for v in layer_loss_norm_values.get((layer_idx, loss_name), [])
                    if isinstance(v, (int, float)) and v == v
                ]
                with_reg_values = layer_loss_cos_with_reg_values.get((layer_idx, loss_name), [])
                cos_mean = sum(values) / len(values)
                norm_mean = (
                    sum(norm_values) / len(norm_values)
                    if norm_values else float("nan")
                )
                cos_with_reg_mean = (
                    sum(with_reg_values) / len(with_reg_values)
                    if with_reg_values else float("nan")
                )
                layer_cos_rows.append([
                    str(layer_idx),
                    loss_name,
                    f"{cos_mean:.8g}",
                    _fmt_cos(cos_with_reg_mean),
                    f"{norm_mean:.8g}" if norm_mean == norm_mean else "nan",
                    str(len(values)),
                ])
                layer_cos_summary.setdefault(layer_idx, {})[loss_name] = {
                    "cos_mean": cos_mean,
                    "cos_with_reg_mean": cos_with_reg_mean,
                    "norm_loss_mean": norm_mean,
                    "n_modules": len(values),
                }

        out_layer_cos_tsv = os.path.join(out_dir, "layer_module_cosine_mean_table.txt")
        with open(out_layer_cos_tsv, "w") as f:
            f.write("# analyze_grad_cosine per-layer module-average cosine\n")
            f.write(f"# analysis_quant_method={args.analysis_quant_method}\n")
            f.write(f"# a_loss_ratio={args.a_loss_ratio}\n")
            f.write(f"# measure_samples={args.measure_samples}  measure_batch_size={args.measure_batch_size}\n")
            f.write(
                "\t".join([
                    "layer", "loss", "cos_module_mean", "cos_with_reg_module_mean",
                    "norm_loss_module_mean", "n_modules",
                ]) + "\n"
            )
            for row in layer_cos_rows:
                f.write("\t".join(row) + "\n")
        logging.info(
            "Wrote per-layer module-mean cosine table to %s (%d rows)",
            out_layer_cos_tsv,
            len(layer_cos_rows),
        )
        with open(out_tsv, "a") as f:
            f.write("\n")
            f.write("# per-layer module-mean cosine across measured modules\n")
            f.write(
                "# columns: layer loss cos_module_mean "
                "cos_with_reg_module_mean norm_loss_module_mean n_modules\n"
            )
            for row in layer_cos_rows:
                f.write("\t".join(row) + "\n")
        logging.info(
            "Appended per-layer module-mean cosine summary to %s",
            out_tsv,
        )

        out_loss_tsv = os.path.join(out_dir, "layer_loss_table.txt")
        loss_rows = []
        for layer_idx in sorted(layer_loss_results.keys()):
            for loss_name in ordered_losses:
                stats = layer_loss_results[layer_idx].get(loss_name)
                if stats is None:
                    continue
                loss_rows.append([
                    str(layer_idx),
                    loss_name,
                    f"{stats['mean']:.8g}",
                    f"{stats['std']:.8g}",
                    str(stats["n_batches"]),
        ])
        with open(out_loss_tsv, "w") as f:
            f.write("# analyze_grad_cosine per-layer post-quantization losses\n")
            f.write(f"# analysis_quant_method={args.analysis_quant_method}\n")
            f.write(f"# a_loss_ratio={args.a_loss_ratio}\n")
            f.write(f"# measure_samples={args.measure_samples}  measure_batch_size={args.measure_batch_size}\n")
            f.write("\t".join(["layer", "loss", "mean", "std", "n_batches"]) + "\n")
            for row in loss_rows:
                f.write("\t".join(row) + "\n")
        logging.info("Wrote layer-loss table to %s (%d rows)", out_loss_tsv, len(loss_rows))

        logging.info("==== cosine summary (global merged batch-avg) ====")
        def _summary_entry(r):
            out = {"gnorm_true": f"{r['true_kl_grad_norm_mean']:.3e}"}
            if "fisher_mean" in r:
                out["cos_fisher"] = f"{r['fisher_mean']:.4f}"
                out["gnorm_fisher"] = f"{r['fisher_grad_norm_mean']:.3e}"
            if "legacy_fisher_diag_mse_mean" in r:
                out["cos_legacy_fisher_diag"] = (
                    f"{r['legacy_fisher_diag_mse_mean']:.4f}"
                )
                out["gnorm_legacy_fisher_diag"] = (
                    f"{r['legacy_fisher_diag_mse_grad_norm_mean']:.3e}"
                )
            if "residual_kl_mean" in r:
                out["cos_res_kl"] = f"{r['residual_kl_mean']:.4f}"
                out["gnorm_res_kl"] = f"{r['residual_kl_grad_norm_mean']:.3e}"
            if "refined_residual_kl_mean" in r:
                out["cos_refined_res_kl"] = f"{r['refined_residual_kl_mean']:.4f}"
                out["gnorm_refined_res_kl"] = f"{r['refined_residual_kl_grad_norm_mean']:.3e}"
            if "refined_diag_residual_kl_mean" in r:
                out["cos_refined_diag_res_kl"] = f"{r['refined_diag_residual_kl_mean']:.4f}"
                out["gnorm_refined_diag_res_kl"] = f"{r['refined_diag_residual_kl_grad_norm_mean']:.3e}"
            if "refined_mse_mean" in r:
                out["cos_refined_mse"] = f"{r['refined_mse_mean']:.4f}"
                out["gnorm_refined_mse"] = f"{r['refined_mse_grad_norm_mean']:.3e}"
            if "layer_mse_mean" in r:
                out["cos_layer_mse"] = f"{r['layer_mse_mean']:.4f}"
                out["gnorm_layer_mse"] = f"{r['layer_mse_grad_norm_mean']:.3e}"
            if "module_mse_mean" in r:
                out["cos_module_mse"] = f"{r['module_mse_mean']:.4f}"
                out["gnorm_module_mse"] = f"{r['module_mse_grad_norm_mean']:.3e}"
            return out
        logging.info(pprint.pformat({
            i: {n: _summary_entry(r) for n, r in per_layer.items()}
            for i, per_layer in cosine_results.items()
        }))
        logging.info("==== per-layer module-mean cosine (global merged batch-avg) ====")
        logging.info(pprint.pformat({
            i: {
                loss_name: {
                    "cos": f"{stats['cos_mean']:.4f}",
                    "cos_with_reg": (
                        "nan"
                        if stats["cos_with_reg_mean"] != stats["cos_with_reg_mean"]
                        else f"{stats['cos_with_reg_mean']:.4f}"
                    ),
                    "gnorm": f"{stats['norm_loss_mean']:.3e}",
                    "n_modules": stats["n_modules"],
                }
                for loss_name, stats in per_layer.items()
            }
            for i, per_layer in layer_cos_summary.items()
        }))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_gen()
    main(args)
