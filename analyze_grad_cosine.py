# coding=utf-8
"""Gradient cosine diagnostic for fisher_diag_mse / residual_kl surrogates.

Runs a full GPTAQ quantization pass (rotate + w_clip + act_order, aligned with
scripts/gptaq.sh) and, immediately BEFORE quantizing each transformer block in
`--target_layers`, measures the cosine similarity between:
  * true KL gradient wrt this layer's linear weights (end-to-end backward)
  * fisher_diag_mse surrogate gradient
  * residual_kl  surrogate gradient

Per target layer, batches of `--measure_batch_size` samples are averaged inside
each backward; cosine is computed per linear (q/k/v/o/gate/up/down_proj) then
averaged across batches. Target layer is left in FP during the measurement
(upstream layers already quantized, downstream layers in FP).

This script does NOT touch gptq_utils/gptaq_utils.py — the GPTAQ inner loop is
copied locally so we can interleave measurement without risk of breaking the
production path.
"""

import os
import logging
import pprint
import math
import copy
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers
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
    hadamard_utils,
)
from gptq_utils.gptaq_utils import GPTAQ, FPInputsCache
from gptq_utils.gptq_plus_utils import (
    collect_static_end_to_end_saliency_and_fisher,
    compute_refresh_loss,
    hidden2logits,
    temporary_requires_grad,
)

torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------------------------
# target layer parsing
# ---------------------------------------------------------------------------

def _parse_target_layers(spec: str, num_layers: int):
    """Parse --target_layers. Strips 0 with a warning (delta=0 there)."""
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
    if 0 in ids:
        logging.warning(
            "target_layers contains 0; at layer 0 there is no upstream quantization so "
            "inps == fp_inps, delta = 0, and all three gradients are zero (cosine = nan). "
            "Dropping 0 from the target set."
        )
        ids = [i for i in ids if i != 0]
    return ids


# ---------------------------------------------------------------------------
# forward-kwargs catcher (re-implementation of gptaq_utils pattern)
# ---------------------------------------------------------------------------

def _prepare_calib_inps(analyzer, trainloader, dev):
    """Run embed + first-block-catcher to extract inps / attention_mask / positions."""
    model = analyzer.model
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    for module in analyzer.get_pre_block_modules():
        module.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    nsamples = len(trainloader)
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
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

    return (
        inps,
        cache["attention_mask"],
        cache["position_ids"],
        cache["position_embeddings"],
        orig_device,
    )


# ---------------------------------------------------------------------------
# fp_inps_final precompute (copy of gptq_plus_utils:3657-3692 pattern)
# ---------------------------------------------------------------------------

def _precompute_fp_inps_final(analyzer, inps, attention_mask, position_ids,
                              position_embeddings, dev, orig_device):
    """Per-sample per-layer FP forward. Each layer restored to orig_device after."""
    layers = analyzer.get_layers()
    logging.info(
        "Precomputing FP final-layer hidden states for residual_kl "
        "(nsamples=%d, layers=%d).", inps.shape[0], len(layers),
    )
    scratch = inps.detach().clone().to(dev)
    for idx in range(len(layers)):
        lay = layers[idx].to(dev)
        bits_cfg = quant_utils.disable_act_quant(lay)
        for j in range(scratch.shape[0]):
            scratch[j] = lay(
                scratch[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0].squeeze(0)
        quant_utils.enable_act_quant(lay, bits_cfg)
        layers[idx] = lay.to(orig_device)
    return scratch.to(inps.device)


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


def _capture_grads(name_to_weight):
    grads = {}
    for name, w in name_to_weight.items():
        if w.grad is None:
            raise RuntimeError(f"weight `{name}` received no gradient.")
        grads[name] = w.grad.detach().float().clone()
    return grads


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
    refined_A_list,
    samples_per_A,
    attention_mask,
    position_ids,
    position_embeddings,
    measure_samples,
    measure_batch_size,
    kl_topk,
    dev,
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
    has_refined = (
        refined_A_list is not None
        and len(refined_A_list) > 0
        and any(a is not None for a in refined_A_list)
    )
    if has_refined and samples_per_A > 0 and samples_per_A % measure_batch_size != 0:
        raise ValueError(
            f"refined_rkl: samples_per_A ({samples_per_A}) must be divisible by "
            f"measure_batch_size ({measure_batch_size}) so each measurement batch "
            f"lands in one sub-A bucket."
        )

    # Canonicalise linear names ("q_proj" not "q_proj.module" after ActQuantWrapper).
    raw_modules = analyzer.get_quantizable_modules(layer)
    name_to_module = {}
    for raw_name, mod in raw_modules.items():
        canon = raw_name[:-7] if raw_name.endswith(".module") else raw_name
        name_to_module[canon] = mod
    name_to_weight = {n: m.weight for n, m in name_to_module.items()}
    target_params = list(name_to_weight.values())

    per_batch_fisher_cos = {n: [] for n in name_to_weight}
    per_batch_residual_cos = {n: [] for n in name_to_weight}
    per_batch_refined_cos = {n: [] for n in name_to_weight}
    per_batch_true_norm = {n: [] for n in name_to_weight}
    per_batch_fisher_norm = {n: [] for n in name_to_weight}
    per_batch_residual_norm = {n: [] for n in name_to_weight}
    per_batch_refined_norm = {n: [] for n in name_to_weight}

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
            fp_batch = fp_inps[start:end].to(dev)
            # Keep fp_final in model dtype (bf16) so hidden2logits can run through
            # the bf16 norm + lm_head without a dtype mismatch.
            fp_final_batch = fp_inps_final[start:end].to(dev)
            fisher_batch = fisher_tensor[start:end].to(dev).float()

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
            grads_true = _capture_grads(name_to_weight)
            del h_student, logits_student, logits_teacher, kl_loss

            # ---------- (2) fisher_diag_mse ----------
            _zero_grads(target_params)
            out_hidden = _call_layer(inp_batch)
            with torch.no_grad():
                fp_hidden = _call_layer(fp_batch)
            fisher_loss = compute_refresh_loss(
                refresh_loss_type="fisher_diag_mse",
                out_hidden=out_hidden,
                fp_hidden=fp_hidden,
                analyzer=analyzer,
                kl_topk=kl_topk,
                layer_output_fisher=fisher_batch,
                fp_final_hidden=None,
            )
            fisher_loss.backward()
            grads_fisher = _capture_grads(name_to_weight)
            del out_hidden, fp_hidden, fisher_loss

            # ---------- (3) residual_kl ----------
            _zero_grads(target_params)
            out_hidden = _call_layer(inp_batch)
            with torch.no_grad():
                fp_hidden = _call_layer(fp_batch)
            residual_loss = compute_refresh_loss(
                refresh_loss_type="residual_kl",
                out_hidden=out_hidden,
                fp_hidden=fp_hidden,
                analyzer=analyzer,
                kl_topk=kl_topk,
                layer_output_fisher=None,
                fp_final_hidden=fp_final_batch,
            )
            residual_loss.backward()
            grads_residual = _capture_grads(name_to_weight)
            del out_hidden, fp_hidden, residual_loss

            # ---------- (4) refined_residual_kl ----------
            grads_refined = None
            if has_refined:
                # Pick the sub-A that owns this batch's samples. We enforce
                # `samples_per_A % measure_batch_size == 0` above so the batch
                # sits fully inside one bucket.
                a_idx = 0 if samples_per_A <= 0 else (start // samples_per_A)
                if a_idx >= len(refined_A_list):
                    raise RuntimeError(
                        f"refined_rkl: batch start={start} resolves to a_idx={a_idx} "
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
                with torch.no_grad():
                    fp_hidden = _call_layer(fp_batch)
                refined_loss = compute_refresh_loss(
                    refresh_loss_type="refined_residual_kl",
                    out_hidden=out_hidden,
                    fp_hidden=fp_hidden,
                    analyzer=analyzer,
                    kl_topk=kl_topk,
                    layer_output_fisher=None,
                    fp_final_hidden=fp_final_batch,
                    refined_A=refined_A_dev,
                )
                refined_loss.backward()
                grads_refined = _capture_grads(name_to_weight)
                del out_hidden, fp_hidden, refined_loss, refined_A_dev

            # ---------- cosine + grad L2 per linear ----------
            cos_f = _cosine_per_linear(grads_true, grads_fisher)
            cos_r = _cosine_per_linear(grads_true, grads_residual)
            cos_rf = _cosine_per_linear(grads_true, grads_refined) if grads_refined is not None else None
            for n in name_to_weight:
                per_batch_fisher_cos[n].append(cos_f[n])
                per_batch_residual_cos[n].append(cos_r[n])
                if cos_rf is not None:
                    per_batch_refined_cos[n].append(cos_rf[n])
                # Per-batch grad Frobenius / L2 norms (flattened). Useful to see
                # not just whether surrogate grads point the right way (cosine)
                # but also how their magnitude compares to the true KL grad's.
                per_batch_true_norm[n].append(grads_true[n].flatten().norm(p=2).item())
                per_batch_fisher_norm[n].append(grads_fisher[n].flatten().norm(p=2).item())
                per_batch_residual_norm[n].append(grads_residual[n].flatten().norm(p=2).item())
                if grads_refined is not None:
                    per_batch_refined_norm[n].append(grads_refined[n].flatten().norm(p=2).item())
            del grads_true, grads_fisher, grads_residual
            if grads_refined is not None:
                del grads_refined
            torch.cuda.empty_cache()

        _zero_grads(target_params)

    results = {}
    for n in name_to_weight:
        fs = torch.tensor(per_batch_fisher_cos[n])
        rs = torch.tensor(per_batch_residual_cos[n])
        tn = torch.tensor(per_batch_true_norm[n])
        fn = torch.tensor(per_batch_fisher_norm[n])
        rn = torch.tensor(per_batch_residual_norm[n])
        entry = {
            "fisher_mean": fs.mean().item(),
            "fisher_std": fs.std(unbiased=False).item() if len(fs) > 1 else 0.0,
            "residual_kl_mean": rs.mean().item(),
            "residual_kl_std": rs.std(unbiased=False).item() if len(rs) > 1 else 0.0,
            "per_batch_fisher": fs.tolist(),
            "per_batch_residual_kl": rs.tolist(),
            "n_batches": len(fs),
            # Grad L2 norms (mean over batches). Reported alongside cosines so
            # we can judge both direction (cosine) and magnitude (norm ratio).
            "true_kl_grad_norm_mean": tn.mean().item(),
            "fisher_grad_norm_mean": fn.mean().item(),
            "residual_kl_grad_norm_mean": rn.mean().item(),
            "per_batch_true_kl_grad_norm": tn.tolist(),
            "per_batch_fisher_grad_norm": fn.tolist(),
            "per_batch_residual_kl_grad_norm": rn.tolist(),
        }
        if has_refined and per_batch_refined_cos[n]:
            rf = torch.tensor(per_batch_refined_cos[n])
            rfn = torch.tensor(per_batch_refined_norm[n])
            entry["refined_residual_kl_mean"] = rf.mean().item()
            entry["refined_residual_kl_std"] = rf.std(unbiased=False).item() if len(rf) > 1 else 0.0
            entry["per_batch_refined_residual_kl"] = rf.tolist()
            entry["refined_residual_kl_grad_norm_mean"] = rfn.mean().item()
            entry["per_batch_refined_residual_kl_grad_norm"] = rfn.tolist()
        results[n] = entry
    return results


# ---------------------------------------------------------------------------
# main pipeline — GPTAQ per-layer loop with measurement hook
# ---------------------------------------------------------------------------

@torch.no_grad()
def quantize_and_measure(args, analyzer, trainloader, dev, target_layers):
    logging.info("----- GPTAQ + grad-cosine analysis -----")
    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()

    inps, attention_mask, position_ids, position_embeddings, orig_device = \
        _prepare_calib_inps(analyzer, trainloader, dev)

    # Offload the big calibration activation buffer to CPU. For 7B at
    # nsamples=1024/seq=2048/hidden=4096/bf16 this is 16 GB per buffer; we have
    # three (inps/fp_inps/fp_inps_final) so 48 GB of headroom is freed. Per-use
    # slices are moved to `dev` in the inner loops (already the pattern).
    inps = inps.cpu()

    memory_utils.cleanup_memory(False)

    # fisher (end-to-end sampled-NLL empirical Fisher) + refined_residual_kl A (LS fit).
    refined_rkl_num_A = int(getattr(args, "refined_rkl_num_A", 1))
    samples_per_A = (
        args.nsamples // refined_rkl_num_A
        if refined_rkl_num_A > 0 else args.nsamples
    )
    static_saliency, static_fisher_by_layer, static_refined_A_by_layer = \
        collect_static_end_to_end_saliency_and_fisher(
            model=model,
            analyzer=analyzer,
            dataloader=trainloader,
            dev=dev,
            saliency_num_groups=args.num_groups,
            fisher_num_groups=args.fisher_num_groups,
            grad_hessian_topk=args.grad_hessian_topk,
            batch_size=args.global_loss_bsz,
            collect_fisher=True,
            collect_refined_rkl=True,
            refined_rkl_damp=args.refined_rkl_damp,
            refined_rkl_num_A=refined_rkl_num_A,
            use_fsdp=False,
            fsdp_cpu_offload=False,
            saliency_clip_percentile=args.saliency_clip_percentile,
        )
    del static_saliency
    memory_utils.cleanup_memory()

    # Pre-block modules back to dev (collect_static... left them on CPU).
    for module in analyzer.get_pre_block_modules():
        module.to(dev)
    # Final norm + lm_head on dev so hidden2logits works during measurement.
    analyzer.get_layernorm_before_head().to(dev)
    analyzer.get_lm_head().to(dev)

    fp_inps_final = _precompute_fp_inps_final(
        analyzer, inps, attention_mask, position_ids, position_embeddings,
        dev, orig_device,
    )

    quantizers = {}
    sequential = analyzer.get_sequential_quantizable_module_names()
    fp_inputs_cache = FPInputsCache(sequential)
    fp_inps = inps.clone()
    cosine_results = {}

    pbar = tqdm(range(len(layers)), ncols=120, desc="Quantizing Layers")
    for i in pbar:
        layer = layers[i].to(dev)
        full = analyzer.get_quantizable_modules(layer)

        # ---------- MEASUREMENT (before quantization) ----------
        if i in target_layers:
            # Downstream layers to dev for tail forward.
            for k in range(i + 1, len(layers)):
                layers[k].to(dev)
            try:
                with torch.enable_grad():
                    # `static_refined_A_by_layer[i]` is a list of A matrices
                    # (length = refined_rkl_num_A). Pass the whole list; the
                    # measurement routine picks per-batch by sample id.
                    refined_A_list_i = (
                        static_refined_A_by_layer[i]
                        if static_refined_A_by_layer is not None
                        else None
                    )
                    cosine_results[i] = run_cosine_measurement(
                        analyzer=analyzer,
                        layer=layer,
                        layer_idx=i,
                        layers=layers,
                        inps=inps,
                        fp_inps=fp_inps,
                        fp_inps_final=fp_inps_final,
                        fisher_tensor=static_fisher_by_layer[i],
                        refined_A_list=refined_A_list_i,
                        samples_per_A=samples_per_A,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        measure_samples=args.measure_samples,
                        measure_batch_size=args.measure_batch_size,
                        kl_topk=args.kl_topk,
                        dev=dev,
                    )
                def _fmt_entry(name, r):
                    base = (
                        f"{name} "
                        f"cos[fisher={r['fisher_mean']:.4f} "
                        f"res_kl={r['residual_kl_mean']:.4f}"
                    )
                    if "refined_residual_kl_mean" in r:
                        base += f" refined_res_kl={r['refined_residual_kl_mean']:.4f}"
                    base += (
                        f"] norm[true={r['true_kl_grad_norm_mean']:.3e} "
                        f"fisher={r['fisher_grad_norm_mean']:.3e} "
                        f"res_kl={r['residual_kl_grad_norm_mean']:.3e}"
                    )
                    if "refined_residual_kl_grad_norm_mean" in r:
                        base += f" refined_res_kl={r['refined_residual_kl_grad_norm_mean']:.3e}"
                    base += "]"
                    return base
                logging.info(
                    "Layer %d cosine+grad-norm (batch-avg): %s",
                    i,
                    ", ".join(_fmt_entry(n, r) for n, r in cosine_results[i].items()),
                )
            finally:
                for k in range(i + 1, len(layers)):
                    layers[k] = layers[k].to(orig_device)
                memory_utils.cleanup_memory()

        # ---------- GPTAQ quantization (copy of gptaq_utils.gptq_fwrd body) ----------
        bits_config = quant_utils.disable_act_quant(layer)
        fp_inputs_cache.add_hook(full)
        for j in range(args.nsamples):
            fp_inps[j] = layer(
                fp_inps[j].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0].to(fp_inps.device)
        fp_inputs_cache.clear_hook()
        quant_utils.enable_act_quant(layer, bits_config)

        for names in sequential:
            subset = {n: full.get(n, full.get(n + ".module", None)) for n in names}
            gptq = {}
            for name in subset:
                layer_weight_bits = args.w_bits
                layer_weight_sym = not args.w_asym
                if "lm_head" in name:
                    continue
                gptq[name] = GPTAQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
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
            for j in range(args.nsamples):
                _ = layer(
                    inps[j].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            handle.remove()

            for name in subset:
                if name != first_module_name:
                    gptq[name].H = gptq[first_module_name].H
                    gptq[name].dXXT = gptq[first_module_name].dXXT

            for name in subset:
                pbar.set_postfix(module=f"layers.{i}." + name)
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    actorder=args.act_order,
                    static_groups=args.act_order,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            inps[j] = layer(
                inps[j].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0].to(inps.device)

        fp_inputs_cache.clear_cache()
        layers[i] = layer.to(orig_device)
        del layer, gptq
        memory_utils.cleanup_memory()

    # Restore module residency.
    for module in analyzer.get_pre_block_modules():
        module.to(orig_device)
    analyzer.get_layernorm_before_head().to(orig_device)
    analyzer.get_lm_head().to(orig_device)
    model.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("----- GPTAQ + grad-cosine done -----")
    return quantizers, cosine_results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist_utils.init_process_group()

    analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model
    tokenizer = analyzer.tokenizer

    target_layers = set(_parse_target_layers(args.target_layers, analyzer.num_layers))
    if not target_layers:
        logging.warning("No valid target layers after parsing; nothing to measure.")
    else:
        logging.info("Will measure cosine at layers: %s", sorted(target_layers))

    if args.rotate:
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

        quant_utils.add_actquant(analyzer)
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = False
    else:
        quant_utils.add_actquant(analyzer)

    transformers.set_seed(args.seed)
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

    _, cosine_results = quantize_and_measure(
        args, analyzer, trainloader, dp_dev, target_layers,
    )

    if dist_utils.is_main():
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        out_pt = os.path.join(out_dir, "grad_cosine_results.pt")
        torch.save({"results": cosine_results, "args": vars(args)}, out_pt)
        logging.info("Saved cosine results to %s", out_pt)
        logging.info("==== cosine summary (batch-avg) ====")
        def _summary_entry(r):
            out = {
                "cos_fisher": f"{r['fisher_mean']:.4f}",
                "cos_res_kl": f"{r['residual_kl_mean']:.4f}",
                "gnorm_true": f"{r['true_kl_grad_norm_mean']:.3e}",
                "gnorm_fisher": f"{r['fisher_grad_norm_mean']:.3e}",
                "gnorm_res_kl": f"{r['residual_kl_grad_norm_mean']:.3e}",
            }
            if "refined_residual_kl_mean" in r:
                out["cos_refined_res_kl"] = f"{r['refined_residual_kl_mean']:.4f}"
            if "refined_residual_kl_grad_norm_mean" in r:
                out["gnorm_refined_res_kl"] = f"{r['refined_residual_kl_grad_norm_mean']:.3e}"
            return out
        logging.info(pprint.pformat({
            i: {n: _summary_entry(r) for n, r in per_layer.items()}
            for i, per_layer in cosine_results.items()
        }))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_gen()
    main(args)
