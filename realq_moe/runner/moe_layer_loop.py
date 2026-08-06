"""Qwen3-MoE all-expert joint quantisation driver.

Attention remains in :mod:`realq_moe.runner.layer_loop` and executes the
copied dense REAL-Q path.  This module owns only sparse expert work:

* capture the post-attention MLP input once;
* gather teacher-routed inputs in stable assignment order;
* build all expert Hessians for one projection on CUDA;
* quantise the same column block for every expert in lockstep;
* run one natural MoE forward/backward to refresh all expert suffixes jointly;
* quantise projections in ``all-up -> all-gate -> all-down`` order.

No helper in this file is used by a dense layer.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from realq_moe import model_adapter
from realq_moe.precompute.routed_stats import (
    PackedLayerRoutes,
    RouteCaptureManager,
)
from realq_moe.quant.routed_realq_layer import RoutedRealQLayer
from realq_moe.refresh.block_gd import RefreshContext
from realq_moe.runner import streams
from realq_moe.utils import nvtx
from realq_moe.utils.module_capture import (
    capture_inner_linear_input_without_gemm,
)
from utils import dist_utils

if TYPE_CHECKING:
    from realq_moe.alignment import RefreshTraceWriter
    from realq_moe.config import Config
    from realq_moe.precompute.static_e2e import StaticStats
    from realq_moe.refresh.block_gd import _SharedSampleScheduler
    from utils.model_utils import ModelAnalyzer


@dataclass
class SparseReplay:
    """GPU-resident current-MoE replay state.

    ``mlp_residuals`` is the post-attention residual before
    ``post_attention_layernorm``.  Together with ``mlp_inputs`` it lets a
    joint refresh run only ``layer.mlp`` and reconstruct the exact layer
    boundary as ``residual + mlp(input)`` without replaying current attention.
    """

    mlp_inputs: torch.Tensor
    mlp_residuals: torch.Tensor | None
    layer_outputs: torch.Tensor | None
    routes: PackedLayerRoutes | None


class _MlpInputCaptured(RuntimeError):
    """Private control-flow exception used to skip experts when no replay is needed."""


class _StudentRoutesCaptured(RuntimeError):
    """Stop a joint replay immediately after the student router was captured."""


def _layer_kwargs(
    state: streams.LayerInputs,
    batch_size: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    am = state.attention_mask
    pi = state.position_ids
    pe = state.position_embeddings
    if am is not None:
        kwargs["attention_mask"] = (
            am.expand(batch_size, *am.shape[1:])
            if am.shape[0] != batch_size
            else am
        )
    if pi is not None:
        kwargs["position_ids"] = (
            pi.expand(batch_size, -1) if pi.shape[0] != batch_size else pi
        )
    if pe is not None:
        kwargs["position_embeddings"] = (
            (
                pe[0].expand(batch_size, *pe[0].shape[1:])
                if pe[0].shape[0] != batch_size
                else pe[0]
            ),
            (
                pe[1].expand(batch_size, *pe[1].shape[1:])
                if pe[1].shape[0] != batch_size
                else pe[1]
            ),
        )
    return kwargs


@torch.no_grad()
def _capture_sparse_replay(
    layer: nn.Module,
    state: streams.LayerInputs,
    *,
    batch_size: int,
    capture_student_routes: bool,
    capture_layer_outputs: bool,
) -> SparseReplay:
    """Replay attention once and capture the tensor passed to sparse MLP.

    With refresh disabled the MLP pre-hook raises after copying its input, so
    the 128 experts are not evaluated merely to obtain a Hessian source.  The
    joint path stops immediately after the natural student router has run,
    avoiding a full expert replay whose output it does not consume.  The
    serial contribution oracle requests ``capture_layer_outputs=True`` and
    therefore still completes the natural layer replay to obtain ``Y_running``.
    """

    if capture_layer_outputs and not capture_student_routes:
        raise ValueError(
            "capture_layer_outputs requires capture_student_routes so the "
            "completed sparse replay and its contribution routes stay aligned."
        )
    n_local, seq_len, _hidden = state.inps.shape
    mlp_inputs = torch.empty_like(state.inps)
    mlp_residuals = (
        torch.empty_like(state.inps) if capture_student_routes else None
    )
    layer_outputs = (
        torch.empty_like(state.inps) if capture_layer_outputs else None
    )
    route_manager = (
        RouteCaptureManager([layer]) if capture_student_routes else None
    )
    active_start = {"value": None}
    seen_start = {"value": None}
    seen_residual_start = {"value": None}

    def capture_residual_hook(_module, inputs):
        if mlp_residuals is None:
            return
        start = active_start["value"]
        if start is None:
            raise RuntimeError(
                "post-attention residual fired outside an active replay chunk."
            )
        if seen_residual_start["value"] == start:
            raise RuntimeError(
                "post-attention residual fired more than once for replay "
                f"start={start}."
            )
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError(
                "post_attention_layernorm received no tensor input."
            )
        residual = inputs[0]
        end = int(start) + int(residual.shape[0])
        mlp_residuals[int(start) : end].copy_(residual.detach())
        seen_residual_start["value"] = start

    def capture_hook(_module, inputs):
        start = active_start["value"]
        if start is None:
            raise RuntimeError("sparse MLP fired outside an active replay chunk.")
        if seen_start["value"] == start:
            raise RuntimeError(
                f"sparse MLP fired more than once for replay start={start}."
            )
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError("sparse MLP did not receive a tensor input.")
        hidden = inputs[0]
        if hidden.dim() != 3 or hidden.shape[1] != seq_len:
            raise RuntimeError(
                "unexpected sparse MLP input shape "
                f"{tuple(hidden.shape)}; expected (*,{seq_len},H)."
            )
        end = int(start) + int(hidden.shape[0])
        mlp_inputs[int(start) : end].copy_(hidden.detach())
        seen_start["value"] = start
        if not capture_student_routes:
            raise _MlpInputCaptured

    handle = layer.mlp.register_forward_pre_hook(capture_hook)
    residual_handle = layer.post_attention_layernorm.register_forward_pre_hook(
        capture_residual_hook
    )
    route_stop_handle = None
    if capture_student_routes and not capture_layer_outputs:
        # RouteCaptureManager registered its router hook during construction,
        # before this hook.  Raising here therefore preserves the exact natural
        # route pack while skipping softmax/top-k duplication, one-hot
        # construction, every expert GEMM and the unused layer-output copy.
        def stop_after_router(_module, _inputs, _outputs):
            raise _StudentRoutesCaptured

        route_stop_handle = layer.mlp.gate.register_forward_hook(
            stop_after_router
        )
    try:
        for start in range(0, n_local, batch_size):
            bsz = min(batch_size, n_local - start)
            active_start["value"] = start
            if route_manager is not None:
                route_manager.begin_batch(
                    local_start=start,
                    batch_size=bsz,
                    seq_len=seq_len,
                )
            try:
                out = layer(
                    state.inps[start : start + bsz],
                    **_layer_kwargs(state, bsz),
                )
            except _MlpInputCaptured:
                if capture_student_routes:
                    raise
            except _StudentRoutesCaptured:
                if route_manager is None or capture_layer_outputs:
                    raise RuntimeError(
                        "student-route early stop fired outside the joint "
                        "capture-only path."
                    )
                route_manager.release_current()
            else:
                if layer_outputs is None:
                    raise RuntimeError(
                        "sparse layer completed although capture-only replay "
                        "should have stopped before expert execution."
                    )
                out_tensor = out[0] if isinstance(out, tuple) else out
                layer_outputs[start : start + bsz].copy_(out_tensor)
                if route_manager is not None and hasattr(
                    route_manager, "release_current"
                ):
                    route_manager.release_current()
        if seen_start["value"] != (max(0, n_local - 1) // batch_size) * batch_size:
            raise RuntimeError("sparse MLP input capture did not cover final chunk.")
    finally:
        handle.remove()
        residual_handle.remove()
        if route_stop_handle is not None:
            route_stop_handle.remove()
        active_start["value"] = None
        if route_manager is not None:
            route_manager.remove()

    routes = None
    if route_manager is not None:
        route_layers = route_manager.finalize()
        if len(route_layers) != 1:
            raise RuntimeError(
                f"student route capture returned {len(route_layers)} layers."
            )
        routes = route_layers[0]
    return SparseReplay(
        mlp_inputs=mlp_inputs,
        mlp_residuals=mlp_residuals,
        layer_outputs=layer_outputs,
        routes=routes,
    )


def _validate_route_payload(
    route: dict[str, torch.Tensor],
    *,
    n_local: int,
    seq_len: int,
    label: str,
) -> int:
    required = {
        "flat_token_indices",
        "topk_slots",
        "route_weights",
    }
    if not required.issubset(route):
        raise RuntimeError(
            f"{label} route is missing {sorted(required.difference(route))}."
        )
    count = int(route["flat_token_indices"].numel())
    if (
        int(route["topk_slots"].numel()) != count
        or int(route["route_weights"].numel()) != count
    ):
        raise RuntimeError(f"{label} route field lengths do not match.")
    flat_ids = route["flat_token_indices"]
    if count:
        if int(flat_ids.min().item()) < 0:
            raise RuntimeError(f"{label} route contains a negative token id.")
        if int(flat_ids.max().item()) >= n_local * seq_len:
            raise RuntimeError(
                f"{label} route token id exceeds local calibration shard."
            )
    return count


def _validate_student_coverage(
    cfg: "Config",
    routes: dict[int, dict[str, torch.Tensor]],
    *,
    n_local: int,
    seq_len: int,
    dev: torch.device,
    layer_idx: int,
) -> torch.Tensor:
    """Validate routes and return the globally zero-route expert mask."""

    num_experts = len(routes)
    local = torch.zeros((num_experts, 3), dtype=torch.int64, device=dev)
    for expert_idx in range(num_experts):
        route = routes[expert_idx]
        count = _validate_route_payload(
            route,
            n_local=n_local,
            seq_len=seq_len,
            label=f"student layer={layer_idx} expert={expert_idx}",
        )
        flat_ids = route["flat_token_indices"]
        local[expert_idx, 0] = count
        if count:
            local[expert_idx, 1] = int(torch.unique(flat_ids).numel())
            local[expert_idx, 2] = int(
                torch.unique(
                    torch.div(flat_ids, seq_len, rounding_mode="floor")
                ).numel()
            )
    dist_utils.allreduce_sum_(local)
    thresholds = torch.tensor(
        [
            cfg.moe_min_expert_assignments,
            cfg.moe_min_expert_unique_tokens,
            cfg.moe_min_expert_unique_samples,
        ],
        dtype=local.dtype,
        device=local.device,
    )
    zero_routed = local[:, 0] == 0
    inconsistent_zero = zero_routed & (local[:, 1:] != 0).any(dim=1)
    if bool(inconsistent_zero.any().item()):
        experts = (
            inconsistent_zero.nonzero(as_tuple=False).flatten().tolist()
        )
        raise RuntimeError(
            "REAL-Q MoE student coverage is inconsistent at "
            f"layer={layer_idx}: zero-assignment experts have nonzero "
            f"token/sample counts; experts={experts}."
        )
    failing = (
        (local < thresholds.unsqueeze(0)).any(dim=1)
        & ~zero_routed
    )
    if bool(failing.any().item()):
        experts = failing.nonzero(as_tuple=False).flatten().tolist()
        details = {
            int(expert_idx): local[expert_idx].tolist()
            for expert_idx in experts
        }
        raise RuntimeError(
            "REAL-Q MoE student routes failed the accepted fail-closed "
            f"coverage gate at layer={layer_idx}; experts={details}, "
            f"thresholds={thresholds.tolist()}."
        )
    if bool(zero_routed.any().item()):
        experts = zero_routed.nonzero(
            as_tuple=False
        ).flatten().tolist()
        logging.warning(
            "[realq_moe.routes] layer=%d student zero-route experts=%s "
            "fallback=%s",
            layer_idx,
            experts,
            cfg.moe_zero_route_fallback,
        )
    logging.info(
        "[realq_moe.routes] layer=%d student assignments min=%d max=%d; "
        "unique_samples min=%d max=%d",
        layer_idx,
        int(local[:, 0].min().item()),
        int(local[:, 0].max().item()),
        int(local[:, 2].min().item()),
        int(local[:, 2].max().item()),
    )
    return zero_routed


def _down_input_for_hessian(
    *,
    expert: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute the exact down-proj input without running the down GEMM.

    ``gate_proj`` and ``up_proj`` are accessed through the live expert so any
    activation-quantization wrappers and already-quantized weights remain in
    force.  The raw product is then passed through the live ``down_proj``
    module far enough to capture the exact input to its underlying
    ``nn.Linear``.  This includes wrapper transforms such as
    ``online_full_had``; a private control-flow exception stops before the
    otherwise-discarded down GEMM.
    """

    gate_proj = getattr(expert, "gate_proj", None)
    up_proj = getattr(expert, "up_proj", None)
    down_proj = getattr(expert, "down_proj", None)
    act_fn = getattr(expert, "act_fn", None)
    if not isinstance(gate_proj, nn.Module) or not isinstance(
        up_proj, nn.Module
    ) or not isinstance(down_proj, nn.Module) or not callable(act_fn):
        raise TypeError(
            "Qwen3-MoE expert must expose gate_proj, up_proj, down_proj and "
            "callable act_fn for direct down-input Hessian accumulation."
        )
    raw_down_input = act_fn(gate_proj(x)) * up_proj(x)
    if isinstance(down_proj, nn.Linear):
        down_linear = down_proj
    else:
        down_linear = getattr(down_proj, "module", None)
        if not isinstance(down_linear, nn.Linear):
            raise TypeError(
                "Qwen3-MoE expert.down_proj must be nn.Linear or a one-level "
                f"wrapper around nn.Linear, got {type(down_proj).__name__}."
            )
    return capture_inner_linear_input_without_gemm(
        down_proj,
        down_linear,
        raw_down_input,
        label="Qwen3-MoE expert.down_proj Hessian input",
    )


@torch.no_grad()
def _build_routed_realq(
    *,
    cfg: "Config",
    layer_idx: int,
    expert_idx: int,
    projection: str,
    layer: nn.Module,
    mlp_inputs: torch.Tensor,
    teacher_route: dict[str, torch.Tensor],
    saliency: torch.Tensor,
    dev: torch.device,
    rtn_fallback_reason: str | None = None,
) -> RoutedRealQLayer:
    """Accumulate one expert projection's Hessian in teacher-route order."""

    from realq_moe.runner.layer_loop import _make_quantizer

    n_local, seq_len, hidden = mlp_inputs.shape
    count = _validate_route_payload(
        teacher_route,
        n_local=n_local,
        seq_len=seq_len,
        label=(
            f"teacher layer={layer_idx} expert={expert_idx} "
            f"projection={projection}"
        ),
    )
    if int(saliency.shape[0]) != count:
        raise RuntimeError(
            "teacher route/saliency assignment mismatch for "
            f"layer={layer_idx} expert={expert_idx} projection={projection}: "
            f"{count} != {saliency.shape[0]}."
        )

    path = model_adapter.expert_projection_path(expert_idx, projection)
    linear = model_adapter.resolve_linear(layer, path)
    expert = layer.mlp.experts[expert_idx]
    realq = RoutedRealQLayer(
        linear=linear,
        saliency=saliency,
        quantizer=_make_quantizer(cfg),
        num_groups=cfg.num_groups,
        dev=dev,
        normalization_token_count=cfg.nsamples * cfg.seq_len,
        group_parallel_quant=cfg.group_parallel_quant,
    )
    if rtn_fallback_reason is not None:
        realq.finalize_rtn_fallback(rtn_fallback_reason)
        return realq

    flat_mlp = mlp_inputs.reshape(-1, hidden)
    flat_ids = teacher_route["flat_token_indices"]
    route_device = torch.device(dev)
    if route_device.type == "cuda" and route_device.index is None:
        route_device = torch.device("cuda", torch.cuda.current_device())
    if flat_ids.device != route_device:
        raise RuntimeError(
            f"{path} route indices must already be on {route_device}; "
            f"got {flat_ids.device}."
        )
    if projection in ("up_proj", "gate_proj"):
        # H = X^T diag(s) X only depends on the linear input.  The historical
        # hook path evaluated the full expert projection merely to capture X,
        # wasting one GEMM for every expert and calibration chunk.  Direct
        # accumulation is mathematically identical and is MoE-specific.
        for start in range(0, count, cfg.moe_expert_chunk_assignments):
            end = min(start + cfg.moe_expert_chunk_assignments, count)
            # Canonical packed routes use CUDA int32, which index_select
            # accepts directly.  Do not allocate/cast an int64 copy per
            # expert/chunk.
            ids = flat_ids[start:end]
            realq.add_batch(flat_mlp.index_select(0, ids))
    else:
        for start in range(0, count, cfg.moe_expert_chunk_assignments):
            end = min(start + cfg.moe_expert_chunk_assignments, count)
            ids = flat_ids[start:end]
            x = flat_mlp.index_select(0, ids)
            realq.add_batch(
                _down_input_for_hessian(expert=expert, x=x).detach()
            )
    realq.finalize_hessian()
    return realq


@torch.no_grad()
def _compute_expert_outputs(
    *,
    expert: nn.Module,
    mlp_inputs: torch.Tensor,
    route: dict[str, torch.Tensor],
    chunk_assignments: int,
    dev: torch.device,
) -> torch.Tensor:
    flat_ids = route["flat_token_indices"]
    count = int(flat_ids.numel())
    hidden = int(mlp_inputs.shape[-1])
    outputs = torch.empty(
        (count, hidden),
        device=dev,
        dtype=mlp_inputs.dtype,
    )
    flat_mlp = mlp_inputs.reshape(-1, hidden)
    for start in range(0, count, chunk_assignments):
        end = min(start + chunk_assignments, count)
        ids = flat_ids[start:end].to(device=dev, dtype=torch.long)
        outputs[start:end].copy_(expert(flat_mlp.index_select(0, ids)))
    return outputs


@torch.no_grad()
def _update_running_output_and_replace_old_(
    *,
    y_running: torch.Tensor,
    old_outputs: torch.Tensor,
    expert: nn.Module,
    mlp_inputs: torch.Tensor,
    route: dict[str, torch.Tensor],
    chunk_assignments: int,
    dev: torch.device,
) -> None:
    """Apply ``scatter(weight * (new-old))`` and reuse old buffer for new."""

    flat_ids_cpu = route["flat_token_indices"]
    weights_cpu = route["route_weights"]
    count = int(flat_ids_cpu.numel())
    hidden = int(mlp_inputs.shape[-1])
    flat_mlp = mlp_inputs.reshape(-1, hidden)
    flat_y = y_running.reshape(-1, hidden)
    for start in range(0, count, chunk_assignments):
        end = min(start + chunk_assignments, count)
        ids = flat_ids_cpu[start:end].to(device=dev, dtype=torch.long)
        weights = weights_cpu[start:end].to(
            device=dev, dtype=y_running.dtype
        )
        new_outputs = expert(flat_mlp.index_select(0, ids))
        delta = new_outputs - old_outputs[start:end]
        delta.mul_(weights.unsqueeze(-1))
        flat_y.index_add_(0, ids, delta)
        old_outputs[start:end].copy_(new_outputs)


@torch.no_grad()
def _quantize_sparse_experts_serial_reference(
    *,
    cfg: "Config",
    layer_idx: int,
    layer: nn.Module,
    static: "StaticStats",
    state: streams.LayerInputs,
    fp_outs: torch.Tensor,
    fisher: torch.Tensor | None,
    layer_lr: float,
    grad_clip: float,
    refresh_bsz_local: int,
    sample_scheduler: "_SharedSampleScheduler | None",
    block_gd_enabled: bool,
    use_kl_refresh: bool,
    analyzer: "ModelAnalyzer | None",
    trace_writer: "RefreshTraceWriter | None",
    dev: torch.device,
) -> None:
    """Serial expert/contribution oracle retained for correctness comparison."""

    plan = model_adapter.describe_layer(layer)
    if not plan.is_sparse:
        raise ValueError("quantize_sparse_experts received a dense layer.")
    if cfg.moe_expert_loss_slide_window:
        raise RuntimeError(
            "expert next-layer slide is forbidden by the accepted policy."
        )
    if block_gd_enabled and sample_scheduler is None:
        raise RuntimeError(
            "expert Block-GD is enabled but no shared sample scheduler exists."
        )
    if block_gd_enabled and cfg.a_loss_ratio != 1.0:
        raise NotImplementedError(
            "MoE contribution refresh currently supports only "
            "a_loss_ratio=1.0; exact global clipping must be implemented "
            "before enabling another value."
        )

    with nvtx.nvtx_range("moe.capture_post_attention"):
        replay = _capture_sparse_replay(
            layer,
            state,
            batch_size=cfg.hessian_accum_bsz,
            capture_student_routes=block_gd_enabled,
            capture_layer_outputs=block_gd_enabled,
        )
    n_local, seq_len, _hidden = replay.mlp_inputs.shape

    student_routes = replay.routes
    y_running = replay.layer_outputs
    if block_gd_enabled:
        if student_routes is None or y_running is None:
            raise RuntimeError("expert refresh requires student routes/output.")
        _validate_student_coverage(
            cfg,
            student_routes,
            n_local=n_local,
            seq_len=seq_len,
            dev=dev,
            layer_idx=layer_idx,
        )
    teacher_routes = static.routes[layer_idx]
    if len(teacher_routes) != plan.num_experts:
        raise RuntimeError(
            f"layer {layer_idx} teacher routes contain {len(teacher_routes)} "
            f"experts; expected {plan.num_experts}."
        )

    for expert_idx in range(plan.num_experts):
        expert = layer.mlp.experts[expert_idx]
        old_student_outputs = None
        student_route = None
        if block_gd_enabled:
            student_route = student_routes[expert_idx]
            with nvtx.nvtx_range(
                f"moe.expert_{expert_idx}.initial_contribution"
            ):
                old_student_outputs = _compute_expert_outputs(
                    expert=expert,
                    mlp_inputs=replay.mlp_inputs,
                    route=student_route,
                    chunk_assignments=cfg.moe_expert_chunk_assignments,
                    dev=dev,
                )

        for projection in model_adapter.EXPERT_PROJECTION_ORDER:
            path = model_adapter.expert_projection_path(
                expert_idx, projection
            )
            with nvtx.nvtx_range(
                f"moe.expert_{expert_idx}.{projection}"
            ):
                realq = _build_routed_realq(
                    cfg=cfg,
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    projection=projection,
                    layer=layer,
                    mlp_inputs=replay.mlp_inputs,
                    teacher_route=teacher_routes[expert_idx],
                    saliency=static.saliency[layer_idx][path],
                    dev=dev,
                )
                grad_refresh_fn = None
                ctx = None
                if block_gd_enabled:
                    from realq_moe.refresh.moe_block_gd import (
                        make_moe_expert_refresh_fn,
                    )

                    ctx = RefreshContext(
                        module=realq.linear,
                        layer_lr=layer_lr,
                        grad_clip=grad_clip,
                        backward_bsz=refresh_bsz_local,
                        scheduler=sample_scheduler,
                        trace_writer=trace_writer,
                        trace_layer=layer_idx,
                        trace_module=path,
                        blocksize=cfg.blocksize,
                        log_column_block_loss=cfg.log_column_block_loss,
                    )
                    global_sample_start = dist_utils.shard_slice(
                        cfg.nsamples
                    ).start
                    grad_refresh_fn = make_moe_expert_refresh_fn(
                        expert=expert,
                        target_linear=realq.linear,
                        student_route=student_route,
                        mlp_inputs=replay.mlp_inputs,
                        y_running=y_running,
                        old_expert_outputs=old_student_outputs,
                        fp_out_for_this_layer=fp_outs,
                        fisher=fisher,
                        ctx=ctx,
                        seq_len=seq_len,
                        global_sample_start=global_sample_start,
                        hit_sample_probability=1.0,
                        a_loss_ratio=cfg.a_loss_ratio,
                        final_layer_analyzer=(
                            analyzer if use_kl_refresh else None
                        ),
                        kl_topk=cfg.kl_topk,
                    )
                realq.quantize(
                    blocksize=cfg.blocksize,
                    percdamp=cfg.percdamp,
                    act_order=cfg.act_order,
                    w_clip=cfg.w_clip,
                    grad_refresh_fn=grad_refresh_fn,
                    group_parallel_quant=cfg.group_parallel_quant,
                    quantizer_inner_fastpath=cfg.quantizer_inner_fastpath,
                    act_order_stitch_impl=cfg.act_order_stitch_impl,
                    prepared_clamp_bound_cache=(
                        getattr(cfg, "prepared_clamp_bound_cache", False)
                    ),
                    triton_column_block=getattr(
                        cfg, "triton_column_block", False
                    ),
                )
                realq.free()
                realq = None
                grad_refresh_fn = None
                ctx = None

                if block_gd_enabled:
                    with nvtx.nvtx_range(
                        f"moe.expert_{expert_idx}.{projection}.update_contribution"
                    ):
                        _update_running_output_and_replace_old_(
                            y_running=y_running,
                            old_outputs=old_student_outputs,
                            expert=expert,
                            mlp_inputs=replay.mlp_inputs,
                            route=student_route,
                            chunk_assignments=(
                                cfg.moe_expert_chunk_assignments
                            ),
                            dev=dev,
                        )
        old_student_outputs = None

    # Do not carry contribution buffers into the natural final layer replay.
    replay.layer_outputs = None
    replay.routes = None
    replay.mlp_inputs = torch.empty(0, device=dev)


@torch.no_grad()
def quantize_sparse_experts(
    *,
    cfg: "Config",
    layer_idx: int,
    layer: nn.Module,
    static: "StaticStats",
    state: streams.LayerInputs,
    fp_outs: torch.Tensor,
    fisher: torch.Tensor | None,
    layer_lr: float,
    grad_clip: float,
    refresh_bsz_local: int,
    sample_scheduler: "_SharedSampleScheduler | None",
    block_gd_enabled: bool,
    use_kl_refresh: bool,
    analyzer: "ModelAnalyzer | None",
    trace_writer: "RefreshTraceWriter | None",
    dev: torch.device,
    next_layer: nn.Module | None = None,
    next_fp_outs: torch.Tensor | None = None,
    next_fisher: torch.Tensor | None = None,
    slide_alpha_fn: Callable[[], float] | None = None,
) -> None:
    """Quantize every expert jointly, one projection/column block at a time.

    For each of ``up_proj``, ``gate_proj`` and ``down_proj`` this function
    retains all expert Hessians, advances the analytic GPTQ solver over the
    expert dimension, and runs one tuple-gradient Block-GD refresh per
    non-final column block.  The serial contribution path above is an oracle
    only and is intentionally unreachable from the accepted runtime config.
    """

    from realq_moe.quant.joint_column_quant import (
        JointMoeProjectionStepper,
    )
    from realq_moe.refresh.joint_moe_block_gd import (
        make_joint_moe_block_gd_refresh_fn,
    )

    plan = model_adapter.describe_layer(layer)
    if not plan.is_sparse:
        raise ValueError("quantize_sparse_experts received a dense layer.")
    if not cfg.moe_joint_column_block:
        raise RuntimeError(
            "The accepted MoE runtime requires joint column-block "
            "quantization; the serial implementation is an oracle only."
        )
    if dist_utils.get_world_size() != 1:
        raise NotImplementedError(
            "The first joint MoE runtime is intentionally single-GPU."
        )
    if block_gd_enabled and sample_scheduler is None:
        raise RuntimeError(
            "joint expert Block-GD requires the run-wide sample scheduler."
        )

    slide_args = (
        next_layer is not None,
        next_fp_outs is not None,
        next_fisher is not None,
        slide_alpha_fn is not None,
    )
    if any(slide_args) and not all(slide_args):
        raise ValueError(
            "joint expert slide requires next_layer, next_fp_outs, "
            "next_fisher and slide_alpha_fn together."
        )
    slide_enabled = all(slide_args)
    if slide_enabled and not cfg.moe_expert_loss_slide_window:
        raise RuntimeError(
            "joint expert slide inputs were supplied while the MoE slide "
            "policy is disabled."
        )
    if slide_enabled and use_kl_refresh:
        raise ValueError(
            "final-layer KL and next-layer slide are mutually exclusive."
        )

    with nvtx.nvtx_range("moe_joint.capture_post_attention"):
        replay = _capture_sparse_replay(
            layer,
            state,
            batch_size=cfg.hessian_accum_bsz,
            capture_student_routes=block_gd_enabled,
            capture_layer_outputs=False,
        )
    n_local, seq_len, _hidden = replay.mlp_inputs.shape
    student_zero_routed = torch.zeros(
        plan.num_experts, dtype=torch.bool, device=dev
    )
    if block_gd_enabled:
        if replay.mlp_residuals is None or replay.routes is None:
            raise RuntimeError(
                "joint refresh requires captured MLP residuals and routes."
            )
        student_zero_routed = _validate_student_coverage(
            cfg,
            replay.routes,
            n_local=n_local,
            seq_len=seq_len,
            dev=dev,
            layer_idx=layer_idx,
        )

    teacher_routes = static.routes[layer_idx]
    if len(teacher_routes) != plan.num_experts:
        raise RuntimeError(
            f"layer {layer_idx} teacher routes contain "
            f"{len(teacher_routes)} experts; expected {plan.num_experts}."
        )
    if len(static.expert_global_assignment_counts) <= layer_idx:
        raise RuntimeError(
            "Stage-0 payload is missing global expert assignment counts for "
            f"layer {layer_idx}."
        )
    teacher_counts = static.expert_global_assignment_counts[layer_idx]
    expected_device = torch.device(dev)
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda", torch.cuda.current_device())
    if (
        teacher_counts.dim() != 1
        or int(teacher_counts.numel()) != plan.num_experts
        or teacher_counts.device != expected_device
    ):
        raise RuntimeError(
            "invalid Stage-0 global expert assignment counts for "
            f"layer={layer_idx}: shape={tuple(teacher_counts.shape)}, "
            f"device={teacher_counts.device}, expected "
            f"({plan.num_experts},) on {expected_device}."
        )
    if bool((teacher_counts < 0).any().item()):
        raise RuntimeError(
            f"negative Stage-0 expert assignment count at layer={layer_idx}."
        )
    teacher_zero_routed = teacher_counts == 0
    rtn_fallback_mask = teacher_zero_routed | student_zero_routed
    if bool(rtn_fallback_mask.any().item()):
        expert_ids = rtn_fallback_mask.nonzero(
            as_tuple=False
        ).flatten().tolist()
        logging.warning(
            "[realq_moe.rtn_fallback] layer=%d experts=%s "
            "teacher_zero=%s student_zero=%s",
            layer_idx,
            expert_ids,
            teacher_zero_routed.nonzero(
                as_tuple=False
            ).flatten().tolist(),
            student_zero_routed.nonzero(
                as_tuple=False
            ).flatten().tolist(),
        )

    expected_refreshes = 0
    completed_refreshes = 0
    for projection in model_adapter.EXPERT_PROJECTION_ORDER:
        with nvtx.nvtx_range(f"moe_joint.projection_{projection}"):
            projection_expected_refreshes = 0
            projection_completed_refreshes = 0
            realqs: list[RoutedRealQLayer] = []
            stepper = None
            joint_refresh = None
            boundary = None
            updates = None
            contexts: list[RefreshContext] = []
            try:
                with nvtx.nvtx_range(
                    f"moe_joint.{projection}.hessian_all_experts"
                ):
                    for expert_idx in range(plan.num_experts):
                        path = model_adapter.expert_projection_path(
                            expert_idx, projection
                        )
                        fallback_reason = None
                        if bool(rtn_fallback_mask[expert_idx].item()):
                            reasons = []
                            if bool(
                                teacher_zero_routed[expert_idx].item()
                            ):
                                reasons.append("teacher_zero_route")
                            if bool(
                                student_zero_routed[expert_idx].item()
                            ):
                                reasons.append("student_zero_route")
                            fallback_reason = "+".join(reasons)
                        realqs.append(
                            _build_routed_realq(
                                cfg=cfg,
                                layer_idx=layer_idx,
                                expert_idx=expert_idx,
                                projection=projection,
                                layer=layer,
                                mlp_inputs=replay.mlp_inputs,
                                teacher_route=teacher_routes[expert_idx],
                                saliency=static.saliency[layer_idx][path],
                                dev=dev,
                                rtn_fallback_reason=fallback_reason,
                            )
                        )

                with nvtx.nvtx_range(
                    f"moe_joint.{projection}.prepare_stepper"
                ):
                    stepper = JointMoeProjectionStepper(
                        realqs,
                        blocksize=cfg.blocksize,
                        percdamp=cfg.percdamp,
                        act_order=cfg.act_order,
                        w_clip=cfg.w_clip,
                        triton_column_block=getattr(
                            cfg, "triton_column_block", False
                        ),
                    )

                if block_gd_enabled:
                    contexts = [
                        RefreshContext(
                            module=realq.linear,
                            layer_lr=layer_lr,
                            grad_clip=grad_clip,
                            backward_bsz=refresh_bsz_local,
                            scheduler=sample_scheduler,
                            trace_writer=trace_writer,
                            trace_layer=layer_idx,
                            trace_module=(
                                model_adapter.expert_projection_path(
                                    expert_idx, projection
                                )
                            ),
                            blocksize=cfg.blocksize,
                            log_column_block_loss=(
                                cfg.log_column_block_loss
                            ),
                        )
                        for expert_idx, realq in enumerate(realqs)
                    ]
                    joint_refresh = make_joint_moe_block_gd_refresh_fn(
                        layer=layer,
                        target_linears=[
                            realq.linear for realq in realqs
                        ],
                        layer_state=state,
                        fp_out_for_this_layer=fp_outs,
                        fisher=fisher,
                        contexts=contexts,
                        current_mlp_inputs=replay.mlp_inputs,
                        current_mlp_residuals=replay.mlp_residuals,
                        student_routes=replay.routes,
                        projection=projection,
                        expert_modules=layer.mlp.experts,
                        next_layer=next_layer if slide_enabled else None,
                        next_fp_out=(
                            next_fp_outs if slide_enabled else None
                        ),
                        next_fisher=(
                            next_fisher if slide_enabled else None
                        ),
                        slide_alpha_fn=(
                            slide_alpha_fn if slide_enabled else None
                        ),
                        final_layer_analyzer=(
                            analyzer if use_kl_refresh else None
                        ),
                        kl_topk=cfg.kl_topk,
                        a_loss_ratio=cfg.a_loss_ratio,
                    )

                projection_refreshes = max(
                    (stepper.columns + cfg.blocksize - 1)
                    // cfg.blocksize
                    - 1,
                    0,
                )
                projection_expected_refreshes = (
                    projection_refreshes
                    if block_gd_enabled
                    else 0
                )
                expected_refreshes += projection_expected_refreshes
                while not stepper.finished:
                    boundary = stepper.advance_block()
                    if boundary is None:
                        continue
                    if joint_refresh is not None:
                        updates = joint_refresh(
                            boundary.weights_fp32,
                            [boundary.trailing_col_start]
                            * plan.num_experts,
                            boundary.perms,
                        )
                        stepper.apply_updates(updates)
                        projection_completed_refreshes += 1
                        completed_refreshes += 1
                stepper.writeback()
                logging.info(
                    "[realq_moe.joint_complete] scope=projection "
                    "layer_idx=%d projection=%s actual_refreshes=%d "
                    "expected_refreshes=%d",
                    layer_idx,
                    projection,
                    projection_completed_refreshes,
                    projection_expected_refreshes,
                )
            finally:
                # Drop projection-local replay/candidate tensors before the
                # next projection starts building its 128 Hessians.  In
                # particular, never overlap the previous fixed-route cache
                # with the next projection's Hessian/Hinv preparation.
                updates = None
                boundary = None
                joint_refresh = None
                contexts.clear()
                if stepper is not None:
                    stepper.free()
                for realq in realqs:
                    realq.free()
                realqs.clear()

    if completed_refreshes != expected_refreshes:
        raise RuntimeError(
            "joint MoE refresh count mismatch: "
            f"{completed_refreshes} != {expected_refreshes}."
        )
    logging.info(
        "[realq_moe.joint_complete] scope=layer layer_idx=%d "
        "total_actual_refreshes=%d total_expected_refreshes=%d",
        layer_idx,
        completed_refreshes,
        expected_refreshes,
    )

    # Release only transient references.  Every model/stat/state tensor stays
    # on CUDA; there is deliberately no CPU offload or host staging here.
    replay.layer_outputs = None
    replay.routes = None
    replay.mlp_residuals = None
    replay.mlp_inputs = torch.empty(0, device=dev)
