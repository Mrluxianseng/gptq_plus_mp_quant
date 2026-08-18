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

import json
import logging
import os
import socket
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
from tqdm import tqdm

from realq.alignment import RefreshTraceWriter, default_refresh_trace_config
from realq.parallel import env as parallel_env
from realq.parallel.cpu_master import CpuMasterLayerManager
from realq.quant.realq_layer import RealQLayer
from realq.refresh.block_gd import (
    BlockRefreshState,
    RefreshContext,
    _SharedSampleScheduler,
    layer_lr_for_schedule,
    make_grad_refresh_fn,
)
from realq.refresh.kl_loss import make_kl_refresh_fn
from realq.runner import module_groups, streams
from realq.utils import memory as mem_utils
from realq.utils import nvtx
from gptq_utils.quant_aware_utils import disable_fp_path_quant
from utils import dist_utils, quant_utils

if TYPE_CHECKING:
    from realq.config import Config
    from realq.precompute import StaticStats
    from utils.model_utils import ModelAnalyzer


_T = TypeVar("_T")


def _stage_fisher_for_refresh(
    fisher: torch.Tensor,
    dev: torch.device,
    *,
    fp32_cache: bool,
) -> torch.Tensor:
    """Move one persisted Fisher matrix to the refresh device.

    The default branch deliberately retains the historical ``fisher.to(dev)``
    expression and BF16 residency.  The opt-in branch performs the same
    BF16-to-FP32 value conversion once at the layer boundary that
    :func:`fisher_mse_loss` otherwise repeats for every backward chunk.
    """
    if fp32_cache:
        return fisher.to(device=dev, dtype=torch.float32)
    return fisher.to(dev)


def _should_stage_fisher_for_refresh(
    *,
    block_gd_enabled: bool,
    use_kl_refresh: bool,
    fp32_cache: bool,
) -> bool:
    """Whether layer entry should materialize its Fisher tensor.

    KL does not consume Fisher. The legacy/default-off branch nevertheless
    keeps its historical BF16 allocation for exact orchestration compatibility;
    only the opt-in cache removes that otherwise wasted allocation.
    """

    return block_gd_enabled and not (fp32_cache and use_kl_refresh)


def _utc_now_iso() -> str:
    """Return an unambiguous UTC timestamp for performance artifacts."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json_dump(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish one rank's measurement in its destination folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _canonical_cuda_device_uuid(value: object | None) -> str | None:
    """Match PyTorch device UUIDs to NVML/nvidia-smi's canonical spelling.

    PyTorch 2.6 on the L20C image returns the hexadecimal UUID without the
    ``GPU-`` prefix, while ``nvidia-smi --query-gpu=uuid`` includes it.  The
    value identifies the same device; normalize only the representation so
    the timing harness can validate the physical rank mapping.
    """

    if value is None:
        return None
    uuid = str(value)
    if uuid and not uuid.startswith(("GPU-", "MIG-")):
        uuid = f"GPU-{uuid}"
    return uuid


def _measure_quantize_one_layer(
    cfg: "Config",
    layer_idx: int,
    dev: torch.device,
    quantize_call: Callable[[], _T],
) -> _T:
    """Measure exactly one synchronized ``quantize_one_layer`` invocation.

    This helper is called only when ``cfg.perf_measure_layer`` selects the
    current layer. The ordinary path does not enter it, so a default
    ``perf_measure_layer=None`` performs no barriers, timing calls, CUDA
    queries/stat resets, or artifact writes.
    """
    rank = parallel_env.get_rank()
    world = parallel_env.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device_index = (
        dev.index if dev.index is not None else torch.cuda.current_device()
    )
    properties = torch.cuda.get_device_properties(device_index)
    device_name = str(properties.name)
    device_uuid_value = getattr(properties, "uuid", None)
    device_uuid = _canonical_cuda_device_uuid(device_uuid_value)

    # GPU metadata/context discovery happens above so it cannot contaminate
    # either the timed interval or the reset peak counters.
    # Drain every rank's preceding CUDA work *before* admitting it to the
    # rendezvous.  A bare NCCL barrier may otherwise be enqueued while work on
    # another CUDA stream is still outstanding, making that work leak into the
    # measured layer on some ranks.  The second synchronization also makes the
    # post-rendezvous boundary explicit across backends.
    torch.cuda.synchronize(dev)
    parallel_env.barrier()
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    start_allocated = int(torch.cuda.memory_allocated(dev))
    start_reserved = int(torch.cuda.memory_reserved(dev))
    start_utc = _utc_now_iso()
    start_perf_counter_ns = time.perf_counter_ns()

    result = quantize_call()

    torch.cuda.synchronize(dev)
    end_perf_counter_ns = time.perf_counter_ns()
    end_utc = _utc_now_iso()
    end_allocated = int(torch.cuda.memory_allocated(dev))
    end_reserved = int(torch.cuda.memory_reserved(dev))
    peak_allocated = int(torch.cuda.max_memory_allocated(dev))
    peak_reserved = int(torch.cuda.max_memory_reserved(dev))
    # Keep the post-boundary synchronized as well. It is deliberately outside
    # elapsed_ns: max(per-rank elapsed_ns) is the distributed critical path,
    # while a slow rank's wait time must not inflate faster ranks.
    parallel_env.barrier()

    output_dir = Path(cfg.output_dir).resolve()
    output_path = (
        output_dir
        / cfg.exp
        / f"perf_measure_layer_{layer_idx}_rank{rank}.json"
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "metric_name": "quant_layer_critical_wall",
        "output_dir": str(output_dir),
        "exp": cfg.exp,
        "global_rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "layer_idx": layer_idx,
        "cuda_device_index": device_index,
        "cuda_device_name": device_name,
        "cuda_device_uuid": device_uuid,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "start_perf_counter_ns": start_perf_counter_ns,
        "end_perf_counter_ns": end_perf_counter_ns,
        "elapsed_ns": end_perf_counter_ns - start_perf_counter_ns,
        "cuda_start_allocated_bytes": start_allocated,
        "cuda_end_allocated_bytes": end_allocated,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_allocated_delta_bytes": peak_allocated - start_allocated,
        "cuda_start_reserved_bytes": start_reserved,
        "cuda_end_reserved_bytes": end_reserved,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_reserved_delta_bytes": peak_reserved - start_reserved,
    }
    _atomic_json_dump(output_path, payload)
    return result


def _make_quantizer(cfg: "Config"):
    """Fresh per-linear quantizer matching old code's WeightQuantizer config."""
    if cfg.w_bits < 16 and cfg.w_asym:
        raise ValueError(
            "RealQ weight quantization only supports symmetric weights; "
            "set w_asym=False. Asymmetric weights are not part of the paper "
            "algorithm and the previous refactor silently applied symmetric "
            "fake quantization despite w_asym=True."
        )
    q = quant_utils.WeightQuantizer()
    q.configure(
        bits=cfg.w_bits,
        perchannel=True,
        # W16 is a strict no-op, so its nominal symmetry setting is immaterial.
        # Preserve the requested setting there while rejecting low-bit asymmetry
        # above; this matches the legacy argument validation.
        sym=not cfg.w_asym,
        mse=cfg.w_clip,   # MSE clip search (sub-task 4 turns this on)
        weight_groupsize=cfg.w_groupsize,
        w_clip_search_impl=getattr(
            cfg, "w_clip_search_impl", "cartesian_legacy"
        ),
        w_clip_update_impl=getattr(cfg, "w_clip_update_impl", "guarded"),
        w_group_param_layout=getattr(
            cfg, "w_group_param_layout", "expanded"
        ),
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
        with nvtx.nvtx_range("hessian.forward_loop"):
            chunk_idx = 0
            for j in range(0, n, hessian_accum_bsz):
                with nvtx.nvtx_range(f"hessian.chunk_{chunk_idx}"):
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
                    chunk_idx += 1
    finally:
        for h in handles:
            h.remove()
    with nvtx.nvtx_range("hessian.finalize"):
        for r in realqs.values():
            r.finalize_hessian()


def _replay_fp_layer(
    layer: nn.Module,
    state: streams.LayerInputs,
    *,
    inps: torch.Tensor,
    bsz: int = 1,
) -> torch.Tensor:
    """Replay a teacher layer with all runtime A/V/K fake quant disabled.

    The context is deliberately scoped to this helper.  Once it exits the
    quantisers are restored, so Hessian accumulation, gradient refresh and
    the final student replay still see the configured aware path.
    """
    with disable_fp_path_quant(layer):
        return streams.replay_layer(layer, state, bsz=bsz, inps=inps)


def _all_quantizable_modules(
    layer: nn.Module,
) -> list[tuple[str, nn.Module]]:
    """Return the exact sequential linear order for a transformer block."""

    ordered: list[tuple[str, nn.Module]] = []
    for group_name in module_groups.GROUP_ORDER:
        ordered.extend(
            module_groups.get_group_modules(layer, group_name).items()
        )
    return ordered


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
    trace_writer: "RefreshTraceWriter | None" = None,
    block_refresh_states: dict[int, BlockRefreshState] | None = None,
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
    with nvtx.nvtx_range(f"layer_{layer_idx}"):
        use_manager = layer_manager is not None and layer_manager.enabled
        with nvtx.nvtx_range("layer.materialize"):
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
        with nvtx.nvtx_range("layer.fp_replay"):
            # A/K/V aware fake quantisation is a student-path feature.  The
            # teacher stream must stay FP even though the same layer object is
            # already configured for aware Hessian accumulation / refreshes.
            fp_outs = _replay_fp_layer(
                layer, state, bsz=1, inps=state.fp_inps,
            )

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
            with nvtx.nvtx_range("layer.next_fp_replay"):
                if use_manager:
                    next_layer = layer_manager.materialize_layer(layer_idx + 1)
                else:
                    next_layer.to(dev)
                # bsz=1: match old GPTQ+ ``slide_fp_inps_next`` per-sample loop
                # (gptq_plus_utils.py:8355-8361). cuBLAS attention picks a different
                # bf16 kernel for batch>1 vs batch=1; using ``hessian_accum_bsz``
                # here lets next-layer FP target drift ULP-wise from old's, which
                # propagates through every slide-arm refresh gradient.
                next_fp_outs = _replay_fp_layer(
                    next_layer, state, bsz=1, inps=fp_outs,
                )

        # Layer-wise reverse-cosine lr — same for every module in this layer.
        # The last transformer layer can opt in to a separate base lr via
        # ``cfg.final_layer_grad_lr`` and, when Block-GD is enabled, uses the
        # full-vocabulary KL refresh constructed below rather than Fisher MSE.
        is_final_layer = (layer_idx == num_layers - 1)
        base_lr = cfg.grad_lr
        if is_final_layer and cfg.final_layer_grad_lr is not None:
            base_lr = float(cfg.final_layer_grad_lr)
        layer_lr = layer_lr_for_schedule(
            base_lr, layer_idx, num_layers,
            cfg.grad_lr_layer_base_ratio, cfg.grad_lr_layer_schedule,
            # The final transformer block always keeps its separately tuned
            # LR. For earlier blocks, paper activation-aware rows use the
            # configured/reported LR as a constant (no base-ratio interpolation).
            activation_aware=(
                cfg.activation_aware_quantization_enabled
                and not is_final_layer
            ),
        )
        effective_grad_clip = (
            cfg.final_layer_grad_clip
            if is_final_layer and cfg.final_layer_grad_clip is not None
            else cfg.grad_clip
        )
        world = parallel_env.get_world_size()
        refresh_bsz_global = (
            cfg.final_layer_backward_bsz
            if is_final_layer
            else cfg.backward_bsz
        )
        if refresh_bsz_global % world != 0:
            raise ValueError(
                f"refresh batch size ({refresh_bsz_global}) must be divisible "
                f"by world_size ({world}) for layer {layer_idx}."
            )
        refresh_bsz_local = refresh_bsz_global // world
        block_gd_enabled = base_lr > 0
        # Final layer drops the fisher_mse loss for the more accurate
        # KL-vs-real-time-FP-logits loss (matches old GPTQ+ ``--grad_refresh_loss=kl``
        # semantics for the final layer when ``final_layer_grad_lr`` is set).
        # Requires lm_head + final norm on the same device as ``layer``.
        use_kl_refresh = is_final_layer and block_gd_enabled and analyzer is not None
        if use_kl_refresh:
            with nvtx.nvtx_range("layer.materialize_lm_head"):
                if use_manager:
                    layer_manager.materialize_runtime_modules(
                        [analyzer.get_layernorm_before_head(), analyzer.get_lm_head()]
                    )
                else:
                    analyzer.get_layernorm_before_head().to(dev)
                    analyzer.get_lm_head().to(dev)

        block_state = None
        next_block_state = None
        if block_gd_enabled and cfg.full_block_refresh:
            if block_refresh_states is None:
                block_refresh_states = {}
            current_named_modules = _all_quantizable_modules(layer)
            block_state = block_refresh_states.get(layer_idx)
            if block_state is None:
                block_state = BlockRefreshState(
                    layer, current_named_modules
                )
                block_refresh_states[layer_idx] = block_state
            else:
                block_state.rebind(layer, current_named_modules)
            if cfg.loss_slide_window and next_layer is not None:
                next_named_modules = _all_quantizable_modules(next_layer)
                next_block_state = block_refresh_states.get(layer_idx + 1)
                if next_block_state is None:
                    next_block_state = BlockRefreshState(
                        next_layer, next_named_modules
                    )
                    block_refresh_states[layer_idx + 1] = next_block_state
                else:
                    next_block_state.rebind(
                        next_layer, next_named_modules
                    )

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
        with nvtx.nvtx_range("layer.fisher_to_gpu"):
            # Keep the default-off path byte-for-byte faithful, including its
            # historical (unused) BF16 allocation on a final KL layer. The
            # opt-in FP32 cache may safely omit that allocation because the KL
            # closure never reads Fisher.
            stage_fisher = _should_stage_fisher_for_refresh(
                block_gd_enabled=block_gd_enabled,
                use_kl_refresh=use_kl_refresh,
                fp32_cache=cfg.fisher_fp32_cache,
            )
            fisher_dev = (
                _stage_fisher_for_refresh(
                    static.fisher[layer_idx],
                    dev,
                    fp32_cache=cfg.fisher_fp32_cache,
                )
                if stage_fisher
                else None
            )
            # next-layer fisher for slide_window. Hoisted to layer entry so
            # the H2D copy happens ONCE per layer instead of once per module
            # (block_gd reused 6 modules × per-module ``static.fisher[...].to(dev)``
            # before this hoist, perturbing cuBLAS workspace allocations and
            # wasting ~12 MB H2D bandwidth per layer for Qwen3-0.6B).
            next_fisher_dev = None
            if (
                block_gd_enabled
                and cfg.loss_slide_window
                and next_layer is not None
            ):
                next_fisher_dev = _stage_fisher_for_refresh(
                    static.fisher[layer_idx + 1],
                    dev,
                    fp32_cache=cfg.fisher_fp32_cache,
                )

        for grp in module_groups.GROUP_ORDER:
            with nvtx.nvtx_range(f"group_{grp}"):
                modules = module_groups.get_group_modules(layer, grp)
                # Build one RealQLayer per module in this group, sharing the layer's
                # static saliency entry for that module.
                with nvtx.nvtx_range("group.build_realqs"):
                    realqs = {}
                    for name, mod in modules.items():
                        r = RealQLayer(
                            linear=mod,
                            saliency=saliency_for_layer[name],
                            quantizer=_make_quantizer(cfg),
                            num_groups=cfg.num_groups,
                            dev=dev,
                            group_parallel_quant=cfg.group_parallel_quant,
                            hessian_tf32=cfg.hessian_tf32,
                        )
                        realqs[name] = r
                with nvtx.nvtx_range("group.hessian_accum"):
                    _accumulate_hessian_for_group(
                        layer, modules, realqs, state, cfg.hessian_accum_bsz,
                    )
                # Quantise each module in the group; subsequent groups will see the
                # mutated weights when they re-forward the layer.
                for name, realq in realqs.items():
                    with nvtx.nvtx_range(f"module_{name}"):
                        grad_refresh_fn = None
                        if block_gd_enabled:
                            with nvtx.nvtx_range("module.build_refresh_fn"):
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
                                    grad_clip=effective_grad_clip,
                                    backward_bsz=refresh_bsz_local,
                                    scheduler=sample_scheduler,
                                    trace_writer=trace_writer,
                                    trace_layer=layer_idx,
                                    trace_module=name,
                                    blocksize=cfg.blocksize,
                                    log_column_block_loss=(
                                        cfg.log_column_block_loss
                                    ),
                                    fused_block_adam=cfg.fused_block_adam,
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
                                        module_name=(
                                            name
                                            if cfg.full_block_refresh
                                            else None
                                        ),
                                        block_state=(
                                            block_state
                                            if cfg.full_block_refresh
                                            else None
                                        ),
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
                                        module_name=(
                                            name
                                            if cfg.full_block_refresh
                                            else None
                                        ),
                                        block_state=(
                                            block_state
                                            if cfg.full_block_refresh
                                            else None
                                        ),
                                        layer_state=state,
                                        fp_out_for_this_layer=fp_outs,
                                        fisher=fisher_dev,
                                        ctx=ctx,
                                        next_layer=next_layer if cfg.loss_slide_window else None,
                                        next_block_state=(
                                            next_block_state
                                            if (
                                                cfg.full_block_refresh
                                                and cfg.loss_slide_window
                                            )
                                            else None
                                        ),
                                        next_fp_out=next_fp_outs if cfg.loss_slide_window else None,
                                        next_fisher=next_fisher_dev,
                                        slide_alpha_fn=(
                                            _alpha_advance
                                            if cfg.loss_slide_window and next_layer is not None
                                            else None
                                        ),
                                        a_loss_ratio=cfg.a_loss_ratio,
                                        a_loss_clip_scope=cfg.a_loss_clip_scope,
                                    )
                        initial_weight_fp32 = (
                            block_state.begin_quantization(name)
                            if block_state is not None
                            else None
                        )
                        with nvtx.nvtx_range("module.quantize"):
                            realq.quantize(
                                blocksize=cfg.blocksize,
                                percdamp=cfg.percdamp,
                                act_order=cfg.act_order,
                                w_clip=cfg.w_clip,
                                grad_refresh_fn=grad_refresh_fn,
                                initial_weight_fp32=initial_weight_fp32,
                                group_parallel_quant=cfg.group_parallel_quant,
                                quantizer_inner_fastpath=(
                                    cfg.quantizer_inner_fastpath
                                ),
                                act_order_stitch_impl=(
                                    cfg.act_order_stitch_impl
                                ),
                                prepared_clamp_bound_cache=(
                                    cfg.prepared_clamp_bound_cache
                                ),
                                triton_column_block=(
                                    cfg.triton_column_block
                                ),
                            )
                        if block_state is not None:
                            block_state.finish_quantization(name)
                        realq.free()
                del realqs

        if block_gd_enabled:
            for p in layer.parameters():
                p.requires_grad_(False)
                p.grad = None
            if cfg.full_block_refresh:
                if block_state is None or block_refresh_states is None:
                    raise RuntimeError(
                        "full-block refresh completed without block state"
                    )
                block_state.assert_complete()
                block_state.release()
                block_refresh_states.pop(layer_idx, None)
        if cfg.fisher_fp32_cache:
            # The final loop locals otherwise retain the last refresh closure,
            # which in turn retains both FP32 Fisher matrices through the
            # final replay and teardown. Drop every direct/closure reference
            # once the last module refresh has completed. Do not empty the
            # allocator cache or synchronize: later work may reuse the freed
            # blocks without changing the mathematical operation sequence.
            grad_refresh_fn = None
            ctx = None
            fisher_dev = None
            next_fisher_dev = None

        # Final forward → produces input for next layer (all weights quantised).
        # Legacy GPTQ+ replays one calibration sample at a time here.  Keeping
        # that batch shape is numerically significant in bf16: fused attention
        # and GEMM may select different kernels for batch>1, and even an ULP
        # drift is amplified by every later Hessian and refresh.
        with nvtx.nvtx_range("layer.final_replay"):
            new_inps = streams.replay_layer(layer, state, bsz=1)
        with nvtx.nvtx_range("layer.teardown"):
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

    profiler_capture = None
    if cfg.nsys_capture_start_layer is not None:
        if cfg.nsys_capture_end_layer >= n_layers:
            raise ValueError(
                "nsys capture window exceeds the layers selected for "
                f"quantization: end={cfg.nsys_capture_end_layer}, "
                f"selected_layers={n_layers}."
            )
        profiler_capture = nvtx.CudaProfilerLayerCapture(
            cfg.nsys_capture_start_layer,
            cfg.nsys_capture_end_layer,
        )

    alignment_trace_config = {}
    if cfg.alignment_trace_path is not None:
        alignment_trace_config = default_refresh_trace_config(cfg)
        alignment_trace_config.update(
            {
                "global_loss": True,
                "grad_refresh_loss": "fisher_diag_mse",
                "g_update_mode": "block_gd",
                "weight_update_scope": (
                    "full_transformer_block"
                    if cfg.full_block_refresh
                    else "current_linear_trailing_columns"
                ),
                "grad_optimizer": "adam",
                "final_layer_grad_optimizer": "adam",
                "analytical_first_order_enabled": False,
                "second_order_scale": 1.0,
                "block_atomic_quant": False,
                "pre_clip": False,
                "fused_block_adam": cfg.fused_block_adam,
                # Refactored refresh sampling always uses one shared global
                # scheduler and rank-local filtering.
                "dp_global_shuffle": True,
            }
        )
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
            seed=cfg.refresh_seed,
        )

    trace_writer = RefreshTraceWriter(
        cfg.alignment_trace_path,
        implementation="realq",
        run_id=cfg.alignment_run_id,
        config=alignment_trace_config,
    )
    block_refresh_states: dict[int, BlockRefreshState] = {}
    try:
        for layer_idx in tqdm(
            range(n_layers),
            ncols=100,
            desc="Quantising layers",
            disable=not parallel_env.is_main(),
        ):
            if profiler_capture is not None:
                profiler_capture.before_layer(layer_idx)
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
            if (
                cfg.perf_measure_layer is not None
                and layer_idx == cfg.perf_measure_layer
            ):
                # The lambda exists only in the opt-in branch. In particular,
                # the default-off path below remains a direct call with the
                # original arguments and execution path.
                state = _measure_quantize_one_layer(
                    cfg,
                    layer_idx,
                    dev,
                    lambda: quantize_one_layer(
                        cfg, layer_idx, layers[layer_idx], static, state, dev,
                        num_layers=len(layers),
                        sample_scheduler=sample_scheduler,
                        next_layer=next_layer,
                        next_fp_inps=(
                            state.fp_inps if next_layer is not None else None
                        ),
                        analyzer=analyzer,
                        layer_manager=layer_manager,
                        trace_writer=trace_writer,
                        block_refresh_states=block_refresh_states,
                    ),
                )
            else:
                state = quantize_one_layer(
                    cfg, layer_idx, layers[layer_idx], static, state, dev,
                    num_layers=len(layers), sample_scheduler=sample_scheduler,
                    next_layer=next_layer,
                    next_fp_inps=(
                        state.fp_inps if next_layer is not None else None
                    ),
                    analyzer=analyzer,
                    layer_manager=layer_manager,
                    trace_writer=trace_writer,
                    block_refresh_states=block_refresh_states,
                )
            if profiler_capture is not None:
                profiler_capture.after_layer(layer_idx)
    finally:
        if profiler_capture is not None:
            profiler_capture.close()
        for pending_state in block_refresh_states.values():
            pending_state.release()
        block_refresh_states.clear()
        trace_writer.close()
    if cfg.quant_stop_layer is not None:
        logging.info(
            "[realq] quant_stop_layer=%d reached; remaining layers stay FP.",
            cfg.quant_stop_layer,
        )
