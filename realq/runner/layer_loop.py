"""Per-layer quantisation driver.

For each transformer layer:
    fp_outs = layer(state.fp_inps)         # save BEFORE any quant for next layer's fp_inps
    for each module group in (attn_in, attn_out, mlp_in, mlp_out):
        forward layer(state.inps) with hooks → captures inputs for hessian
        for each module in group:
            build RealQLayer (Hessian, quantizer)
            quantise (mutates linear.weight in place; grad_refresh_fn fires
                      after each block when block_gd is enabled)
    new_inps = layer(state.inps)            # forward with all-quantised layer weights
    state.inps = new_inps
    state.fp_inps = fp_outs                 # advance the FP stream too

Sub-task 5 adds the block_gd refresh closure (Adam + grad_clip + cosine
layer schedule + fisher_mse loss) wired per linear.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm import tqdm

from realq.parallel import env as parallel_env
from realq.parallel.cpu_master import CpuMasterLayerManager
from realq.quant.realq_layer import RealQLayer
from realq.refresh.block_gd import (
    RefreshContext,
    _SharedSampleScheduler,
    layer_lr_for_schedule,
    make_grad_refresh_fn,
)
from realq.refresh.kl_loss import make_kl_refresh_fn
from realq.runner import module_groups, streams
from realq.utils import memory as mem_utils
from utils import dist_utils, quant_utils

if TYPE_CHECKING:
    from realq.config import Config
    from realq.precompute import StaticStats
    from utils.model_utils import ModelAnalyzer


def _make_quantizer(cfg: "Config"):
    """Fresh per-linear quantizer matching old code's WeightQuantizer config."""
    q = quant_utils.WeightQuantizer()
    q.configure(
        bits=cfg.w_bits,
        perchannel=True,
        sym=not cfg.w_asym,
        mse=cfg.w_clip,   # MSE clip search (sub-task 4 turns this on)
        weight_groupsize=cfg.w_groupsize,
    )
    return q


def _accumulate_hessian_for_group(
    layer: nn.Module,
    group_modules: dict[str, nn.Module],
    realqs: dict[str, RealQLayer],
    state: streams.LayerInputs,
    hessian_accum_bsz: int,
) -> None:
    """Forward layer once with a hook on each module in the group; the hook
    feeds ``add_batch`` with the captured input."""
    handles = []

    def make_hook(name: str):
        def hook(_mod, inputs, _out):
            x = inputs[0]
            realqs[name].add_batch(x.detach())
        return hook

    for name, mod in group_modules.items():
        handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        # Reset every realq's index so add_batch sees ordered sample feeding.
        for r in realqs.values():
            r.index = 0
            r.token_count = 0
        # Drive the forward; we don't need the output here, only the hook
        # side effect.
        n = state.inps.shape[0]
        am = state.attention_mask
        pi = state.position_ids
        pe = state.position_embeddings
        for j in range(0, n, hessian_accum_bsz):
            b = min(hessian_accum_bsz, n - j)
            kw = {}
            if am is not None:
                kw["attention_mask"] = am.expand(b, *am.shape[1:]) if am.shape[0] != b else am
            if pi is not None:
                kw["position_ids"] = pi.expand(b, -1) if pi.shape[0] != b else pi
            if pe is not None:
                kw["position_embeddings"] = (
                    pe[0].expand(b, *pe[0].shape[1:]) if pe[0].shape[0] != b else pe[0],
                    pe[1].expand(b, *pe[1].shape[1:]) if pe[1].shape[0] != b else pe[1],
                )
            _ = layer(state.inps[j : j + b], **kw)
    finally:
        for h in handles:
            h.remove()
    for r in realqs.values():
        r.finalize_hessian()


@torch.no_grad()
def quantize_one_layer(
    cfg: "Config",
    layer_idx: int,
    layer: nn.Module,
    static: "StaticStats",
    state: streams.LayerInputs,
    dev: torch.device,
    num_layers: int,
    sample_scheduler: "_SharedSampleScheduler | None" = None,
    next_layer: "nn.Module | None" = None,
    next_fp_inps: "torch.Tensor | None" = None,
    analyzer: "ModelAnalyzer | None" = None,
    layer_manager: "CpuMasterLayerManager | None" = None,
) -> streams.LayerInputs:
    """Quantise one transformer layer, return updated input state for the
    next layer (= output of this layer with all-quantised weights).

    ``next_layer`` / ``next_fp_inps`` enable the loss_slide_window blend:
    per-block α linearly interpolates the refresh loss between current-layer
    fisher_mse and next-layer fisher_mse. Pass None for the LAST layer (or
    when loss_slide_window=False). Next-layer fisher is fetched directly
    from ``static.fisher[layer_idx + 1]`` inside.

    ``layer_manager`` is consulted for the cpu_master path: rank>0's model
    has meta-tensor weights so plain ``.to(dev)`` would crash. None or
    ``layer_manager.enabled=False`` ⇒ identical to plain ``.to`` calls.
    """
    use_manager = layer_manager is not None and layer_manager.enabled
    if use_manager:
        layer = layer_manager.materialize_layer(layer_idx)
    else:
        layer.to(dev)
    saliency_for_layer = static.saliency[layer_idx]

    # Pre-compute the FP-reference forward of THIS layer BEFORE any of its
    # weights get mutated. ``fp_outs`` becomes ``state.fp_inps`` for the
    # next layer; refresh closures slice it per backward batch as the
    # fisher_mse target. ``bsz=1`` matches old GPTQ+
    # ``layer.fp_reference_forward`` (gptq_plus_utils.py:8262-8268) which
    # runs ``fp_inps[j] = layer(fp_inps[j].unsqueeze(0))`` per sample.
    # cuBLAS attention picks a different bf16 kernel for batch>1 vs batch=1
    # so ``fp_outs`` ULP-drifts from old's whenever bsz>1; the drift then
    # propagates into every refresh's ``fp_target``.
    fp_outs = streams.replay_layer(layer, state, bsz=1, inps=state.fp_inps)

    # loss_slide_window: pre-stage next_layer to GPU and forward
    # ``fp_outs`` (the FP forward output of THIS layer) through it once to
    # cache the FP next-layer reference. Old GPTQ+ ``slide_fp_inps_next``
    # (lines 8291-8300, 8353-8361) updates ``fp_inps`` in place to
    # ``layer_FP(fp_inps)`` at the start of each layer's iteration, then
    # computes ``slide_next_layer(fp_inps)`` — i.e. the next-layer FP
    # baseline already includes the current layer's FP forward. ``fp_outs``
    # in our code is the same quantity, so feeding it to next_layer here
    # mirrors old exactly. Earlier RealQ versions fed ``state.fp_inps``
    # directly (skipping the current layer's FP forward), producing a
    # next-layer baseline 10×+ smaller than old's and the slide-arm loss
    # 4-5 orders of magnitude too large.
    next_fp_outs = None
    if cfg.loss_slide_window and next_layer is not None and next_fp_inps is not None:
        if use_manager:
            next_layer = layer_manager.materialize_layer(layer_idx + 1)
        else:
            next_layer.to(dev)
        # bsz=1: match old GPTQ+ ``slide_fp_inps_next`` per-sample loop
        # (gptq_plus_utils.py:8355-8361). cuBLAS attention picks a different
        # bf16 kernel for batch>1 vs batch=1; using ``hessian_accum_bsz``
        # here lets next-layer FP target drift ULP-wise from old's, which
        # propagates through every slide-arm refresh gradient.
        next_fp_outs = streams.replay_layer(
            next_layer, state, bsz=1, inps=fp_outs,
        )

    # Layer-wise lr (cosine schedule) — same for every module in this layer.
    # The last transformer layer can opt in to a separate base lr via
    # ``cfg.final_layer_grad_lr``; loss formula still falls back to
    # fisher_mse for sub-task simplicity (KL-vs-ref_logits override is
    # documented as a follow-up in REFACTOR_NOTES.md).
    is_final_layer = (layer_idx == num_layers - 1)
    base_lr = cfg.grad_lr
    if is_final_layer and cfg.final_layer_grad_lr is not None:
        base_lr = float(cfg.final_layer_grad_lr)
    layer_lr = layer_lr_for_schedule(
        base_lr, layer_idx, num_layers,
        cfg.grad_lr_layer_base_ratio, cfg.grad_lr_layer_schedule,
    )
    block_gd_enabled = base_lr > 0
    # Final layer drops the fisher_mse loss for the more accurate
    # KL-vs-real-time-FP-logits loss (matches old GPTQ+ ``--grad_refresh_loss=kl``
    # semantics for the final layer when ``final_layer_grad_lr`` is set).
    # Requires lm_head + final norm on the same device as ``layer``.
    use_kl_refresh = is_final_layer and block_gd_enabled and analyzer is not None
    if use_kl_refresh:
        if use_manager:
            layer_manager.materialize_runtime_modules(
                [analyzer.get_layernorm_before_head(), analyzer.get_lm_head()]
            )
        else:
            analyzer.get_layernorm_before_head().to(dev)
            analyzer.get_lm_head().to(dev)

    # Slide α schedule: at the FIRST refresh in the layer α=1, at the LAST
    # α=0. Total refreshes in the layer = sum over modules of
    # (cols/blocksize - 1).
    slide_total_refreshes = 0
    if cfg.loss_slide_window and next_layer is not None:
        for grp in module_groups.GROUP_ORDER:
            mods = module_groups.get_group_modules(layer, grp)
            for _, mod in mods.items():
                cols = mod.weight.shape[1]
                n_blocks = (cols + cfg.blocksize - 1) // cfg.blocksize
                slide_total_refreshes += max(n_blocks - 1, 0)
    slide_cursor = {"n": 0}  # advances by 1 per refresh CALL across all modules

    if block_gd_enabled:
        # functional_call inside the refresh closure carries the override
        # weight's requires_grad; module params themselves stay False so
        # autograd doesn't waste time computing their grads. (Any param
        # left requires_grad=True would still produce the right gradient
        # for module.weight, but at the cost of wasted backward work.)
        for p in layer.parameters():
            p.requires_grad_(False)
        if next_layer is not None and cfg.loss_slide_window:
            for p in next_layer.parameters():
                p.requires_grad_(False)
        if use_kl_refresh:
            for p in analyzer.get_layernorm_before_head().parameters():
                p.requires_grad_(False)
            for p in analyzer.get_lm_head().parameters():
                p.requires_grad_(False)

    # Pre-stage the per-layer Fisher to GPU ONCE (reused across all modules in
    # this layer's refresh closures). Calling ``.to(dev)`` lazily inside each
    # ``make_grad_refresh_fn`` invocation perturbs the GPU memory pool (each
    # call mints a fresh 2 MB tensor right before ``realq.quantize`` runs the
    # next module's inner block + outer compensation), and the resulting
    # cuBLAS workspace shifts produce ULP-level diffs in the bmm-driven outer
    # compensation that diverge from the legacy reference (~1e-2 max-diff
    # propagated through later refreshes). Hoisting the alloc here keeps the
    # GPU memory layout deterministic across all per-module quantize calls.
    fisher_dev = static.fisher[layer_idx].to(dev) if block_gd_enabled else None

    for grp in module_groups.GROUP_ORDER:
        modules = module_groups.get_group_modules(layer, grp)
        # Build one RealQLayer per module in this group, sharing the layer's
        # static saliency entry for that module.
        realqs = {}
        for name, mod in modules.items():
            r = RealQLayer(
                linear=mod,
                saliency=saliency_for_layer[name],
                quantizer=_make_quantizer(cfg),
                num_groups=cfg.num_groups,
                dev=dev,
                group_parallel_quant=cfg.group_parallel_quant,
            )
            realqs[name] = r
        _accumulate_hessian_for_group(
            layer, modules, realqs, state, cfg.hessian_accum_bsz,
        )
        # Quantise each module in the group; subsequent groups will see the
        # mutated weights when they re-forward the layer.
        for name, realq in realqs.items():
            grad_refresh_fn = None
            if block_gd_enabled:
                # Single shared sample scheduler across all layers + modules.
                # Old code creates one ``BackwardSampleScheduler`` for the
                # whole model (line 7882) and every refresh draws the next
                # chunk from the same cursor. Replicating per-(layer,module)
                # made every module replay [0..bsz), which gave the wrong
                # gradient signal.
                if sample_scheduler is None:
                    raise RuntimeError(
                        "block_gd is enabled but no shared sample scheduler "
                        "was passed in. Quantize via quantize_all_layers, "
                        "which constructs the scheduler once."
                    )
                ctx = RefreshContext(
                    module=realq.linear,
                    layer_lr=layer_lr,
                    grad_clip=cfg.grad_clip,
                    backward_bsz=cfg.backward_bsz,
                    scheduler=sample_scheduler,
                )
                # slide_alpha closure: returns CURRENT α and advances the
                # layer-shared cumulative refresh cursor. Must be called
                # exactly once per refresh; tying the cursor advance to the
                # alpha read keeps the count honest.
                def _alpha_advance(_cursor=slide_cursor, _total=slide_total_refreshes):
                    n = _cursor["n"]
                    alpha = 1.0 - n / max(_total - 1, 1) if _total > 1 else 1.0
                    _cursor["n"] = n + 1
                    return alpha

                if use_kl_refresh:
                    grad_refresh_fn = make_kl_refresh_fn(
                        layer=layer,
                        module=realq.linear,
                        layer_state=state,
                        fp_out_for_this_layer=fp_outs,
                        analyzer=analyzer,
                        kl_topk=cfg.kl_topk,
                        ctx=ctx,
                    )
                else:
                    grad_refresh_fn = make_grad_refresh_fn(
                        layer=layer,
                        module=realq.linear,
                        layer_state=state,
                        fp_out_for_this_layer=fp_outs,
                        fisher=fisher_dev,
                        ctx=ctx,
                        next_layer=next_layer if cfg.loss_slide_window else None,
                        next_fp_out=next_fp_outs if cfg.loss_slide_window else None,
                        next_fisher=(
                            static.fisher[layer_idx + 1].to(dev)
                            if cfg.loss_slide_window and next_layer is not None
                            else None
                        ),
                        slide_alpha_fn=(
                            _alpha_advance
                            if cfg.loss_slide_window and next_layer is not None
                            else None
                        ),
                        a_loss_ratio=cfg.a_loss_ratio,
                    )
            realq.quantize(
                blocksize=cfg.blocksize,
                percdamp=cfg.percdamp,
                act_order=cfg.act_order,
                w_clip=cfg.w_clip,
                grad_refresh_fn=grad_refresh_fn,
                group_parallel_quant=cfg.group_parallel_quant,
            )
            realq.free()
        del realqs

    if block_gd_enabled:
        for p in layer.parameters():
            p.requires_grad_(False)
            p.grad = None

    # Final forward → produces input for next layer (all weights quantised).
    new_inps = streams.replay_layer(layer, state, bsz=cfg.hessian_accum_bsz)
    if use_manager:
        layer_manager.release_layer(layer_idx, layer, orig_device=torch.device("cpu"))
    else:
        layer.cpu()
    if next_layer is not None and cfg.loss_slide_window:
        # Free the next-layer GPU copy; it'll be re-streamed when its turn
        # comes (and quantised at that point — the FP forward we did up
        # there was on un-mutated weights).
        if use_manager:
            layer_manager.release_layer(
                layer_idx + 1, next_layer, orig_device=torch.device("cpu"),
            )
        else:
            next_layer.cpu()
    if use_kl_refresh:
        if use_manager:
            layer_manager.release_runtime_modules(
                [analyzer.get_layernorm_before_head(), analyzer.get_lm_head()],
                torch.device("cpu"),
            )
        else:
            analyzer.get_layernorm_before_head().cpu()
            analyzer.get_lm_head().cpu()
    mem_utils.cleanup_memory()
    return streams.LayerInputs(
        inps=new_inps,
        fp_inps=fp_outs,
        attention_mask=state.attention_mask,
        position_ids=state.position_ids,
        position_embeddings=state.position_embeddings,
    )


def quantize_all_layers(
    cfg: "Config",
    analyzer: "ModelAnalyzer",
    static: "StaticStats",
    trainloader: list[torch.Tensor],
) -> None:
    """Drive the full layer-by-layer GPTQ loop. Mutates the model in place."""
    rank = parallel_env.get_rank()
    world = parallel_env.get_world_size()
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    sl = dist_utils.shard_slice(len(trainloader), rank=rank, world=world)
    rank_samples = [trainloader[i] for i in range(sl.start, sl.stop)]

    layers = analyzer.get_layers()
    # Build the cpu_master layer manager. enabled=False ⇒ falls through to
    # plain .to(dev)/.cpu() so the legacy fsdp=True path is unaffected.
    layer_manager = CpuMasterLayerManager(analyzer, dev, layers)

    state = streams.capture_layer0_inputs(
        analyzer, rank_samples, dev, layer_manager=layer_manager,
    )

    # capture may have refreshed layers[0] (Catcher unwrap + manager release),
    # so re-fetch the canonical list.
    layers = analyzer.get_layers()
    n_layers = len(layers)
    if cfg.quant_stop_layer is not None:
        n_layers = min(n_layers, cfg.quant_stop_layer + 1)

    # Single GLOBAL scheduler shared across ALL block_gd refreshes for the
    # whole quantisation pass — matches old gptq_plus_utils.py
    # ``--dp_global_shuffle=True`` branch (lines 7885-7895). chunk_size is
    # ``backward_samples`` (NOT divided by world); ``next_indices()``
    # returns the same global id list on every rank, and each refresh
    # closure filters to its own ``[rank * n_local, (rank+1) * n_local)``
    # shard (block_gd / kl_loss). Constructed once here, threaded into
    # each ``quantize_one_layer`` call so its cursor advances across
    # modules.
    sample_scheduler = None
    if cfg.grad_lr > 0 or (
        cfg.final_layer_grad_lr is not None and cfg.final_layer_grad_lr > 0
    ):
        if cfg.nsamples % cfg.backward_samples != 0:
            raise ValueError(
                f"nsamples ({cfg.nsamples}) must be divisible by "
                f"backward_samples ({cfg.backward_samples}) so the global "
                f"round-robin scheduler exhausts each shuffled epoch "
                f"cleanly (matches old BackwardSampleScheduler check at "
                f"gptq_plus_utils.py:594)."
            )
        sample_scheduler = _SharedSampleScheduler(
            n_total=cfg.nsamples,
            chunk_size=cfg.backward_samples,
            seed=cfg.seed,
        )

    for layer_idx in tqdm(range(n_layers), ncols=100, desc="Quantising layers",
                          disable=not parallel_env.is_main()):
        # loss_slide_window needs the NEXT transformer block on GPU during
        # this layer's refreshes. Old GPTQ+ (lines 8284-8287) gates slide
        # on ``i <= final_layer_idx - 2``: i.e. the last TWO layers
        # (final-1 and final) DON'T slide. Match exactly.
        next_layer = None
        if (
            cfg.loss_slide_window
            and layer_idx <= len(layers) - 3
            and layer_idx + 1 < n_layers + 1  # tolerate final-stop runs
        ):
            next_layer = layers[layer_idx + 1]
        state = quantize_one_layer(
            cfg, layer_idx, layers[layer_idx], static, state, dev,
            num_layers=len(layers), sample_scheduler=sample_scheduler,
            next_layer=next_layer,
            next_fp_inps=state.fp_inps if next_layer is not None else None,
            analyzer=analyzer,
            layer_manager=layer_manager,
        )

    if cfg.quant_stop_layer is not None:
        logging.info(
            "[realq] quant_stop_layer=%d reached; remaining layers stay FP.",
            cfg.quant_stop_layer,
        )
