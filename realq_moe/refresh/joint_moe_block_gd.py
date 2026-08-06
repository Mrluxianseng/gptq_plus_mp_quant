"""Joint/Jacobi Block-GD refresh for one MoE projection.

All experts of one projection advance the same GPTQ column block in lockstep.
At a block boundary this module installs every stitched expert weight in one
``torch.func.functional_call`` and asks one ``torch.autograd.grad`` call for
the tuple of all expert-weight gradients.  The speed path replays only the
complete current MoE MLP from cached post-attention-normalization inputs and
adds the cached post-attention residual.  A full-current-layer path remains
available as an oracle.  In either mode the router and sparse MLP are evaluated
once per backward batch, rather than once per expert.

This is deliberately a MoE-only primitive.  The inherited dense refresh and
the serial contribution-decomposition reference are not changed.

Important semantics
-------------------

* All expert gradients are evaluated at the same joint candidate state, then
  all expert Adam updates are returned together.  This is a Jacobi update, not
  the serial expert-by-expert Gauss--Seidel trajectory.
* The first implementation is single-rank and fully GPU resident.  Inputs,
  targets, Fisher matrices, candidate weights, Adam states, current layer,
  optional next layer, and optional final-head modules must already be CUDA
  resident.  There is no CPU route/CSR construction and no CPU offload.
* Every :class:`RefreshContext` must reference the same scheduler.  One joint
  refresh consumes exactly one ``next_indices()`` result and advances every
  expert's Adam step, including experts with an unused/zero gradient.
* With the protocol setting ``backward_samples == backward_bsz`` there is one
  layer forward and one tuple-valued autograd call per joint column block.
  Smaller ``backward_bsz`` values retain the established sample-weighted
  gradient accumulation and run one joint forward/backward per chunk.
* Supplying ``next_layer`` enables the usual current/next Fisher blend.  The
  next layer receives the current candidate output directly, so its router is
  naturally recomputed.  Final-layer KL reuses ``kl_topk_loss`` and is
  intentionally mutually exclusive with the slide arm.
* Supplying both ``current_mlp_inputs`` and ``current_mlp_residuals`` selects
  the speed path.  Omitting both selects the full-layer oracle.  The two cache
  tensors must represent the exact current student attention state immediately
  before and after ``post_attention_layernorm``, respectively.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.func import functional_call

from realq_moe.refresh.fisher_loss import fisher_mse_loss
from realq_moe.refresh.joint_batched_adam import JointBatchedAdamState
from realq_moe.refresh.kl_loss import kl_topk_loss
from realq_moe.utils import nvtx
from utils import dist_utils

if TYPE_CHECKING:
    from realq_moe.precompute.routed_stats import PackedLayerRoutes
    from realq_moe.refresh.block_gd import RefreshContext
    from realq_moe.runner.streams import LayerInputs
    from utils.model_utils import ModelAnalyzer


def _tensor_output(value, *, source: str) -> torch.Tensor:
    output = value[0] if isinstance(value, (tuple, list)) else value
    if not torch.is_tensor(output):
        raise RuntimeError(
            f"{source} must return a Tensor or a tuple/list whose first item "
            f"is a Tensor, got {type(output).__name__}."
        )
    return output


def _linear_weight_name(layer: nn.Module, linear: nn.Linear) -> str:
    """Resolve the canonical functional-call key for an underlying Linear.

    Resolving the module object, rather than only parameter identity, handles
    ``ActQuantWrapper`` correctly: its forward reads ``module.weight`` even
    when the wrapper exposes a direct weight alias.
    """

    matches = [
        name
        for name, module in layer.named_modules()
        if module is linear
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "target linear must occur exactly once in layer.named_modules(); "
            f"found {len(matches)} matches."
        )
    module_name = matches[0]
    return f"{module_name}.weight" if module_name else "weight"


def _require_cuda_tensor(
    value: torch.Tensor,
    *,
    name: str,
    device: torch.device | None = None,
) -> torch.device:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Tensor, got {type(value).__name__}.")
    if value.device.type != "cuda":
        raise ValueError(
            f"{name} must remain CUDA resident; got device={value.device}."
        )
    if device is not None and value.device != device:
        raise ValueError(
            f"{name} must be on {device}, got {value.device}."
        )
    return value.device


def _require_module_on_device(
    module: nn.Module,
    *,
    name: str,
    device: torch.device,
) -> None:
    for parameter_name, parameter in module.named_parameters():
        if parameter.device != device:
            raise ValueError(
                f"{name}.{parameter_name} must remain on {device}, "
                f"got {parameter.device}."
            )
    for buffer_name, buffer in module.named_buffers():
        if buffer.device != device:
            raise ValueError(
                f"{name}.{buffer_name} must remain on {device}, "
                f"got {buffer.device}."
            )


def _require_optional_state_tensor(
    value,
    *,
    name: str,
    device: torch.device,
) -> None:
    if value is None:
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_optional_state_tensor(
                item,
                name=f"{name}[{index}]",
                device=device,
            )
        return
    _require_cuda_tensor(value, name=name, device=device)


def _select_or_expand_batch_tensor(
    value: torch.Tensor,
    *,
    sample_indices: torch.Tensor,
    n_local: int,
    batch_size: int,
    name: str,
) -> torch.Tensor:
    """Select per-sample state or expand a shared leading dimension."""

    if value.dim() == 0:
        return value
    leading = int(value.shape[0])
    if leading == n_local:
        return value.index_select(0, sample_indices)
    if leading == 1:
        return value.expand(batch_size, *value.shape[1:])
    if leading == batch_size and batch_size == n_local:
        return value
    raise ValueError(
        f"{name} leading dimension must be 1 or n_local={n_local}, "
        f"got {leading}."
    )


def _batch_kwargs(
    layer_state: "LayerInputs",
    *,
    sample_indices: torch.Tensor,
    n_local: int,
    batch_size: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    attention_mask = layer_state.attention_mask
    position_ids = layer_state.position_ids
    position_embeddings = layer_state.position_embeddings
    if attention_mask is not None:
        kwargs["attention_mask"] = _select_or_expand_batch_tensor(
            attention_mask,
            sample_indices=sample_indices,
            n_local=n_local,
            batch_size=batch_size,
            name="layer_state.attention_mask",
        )
    if position_ids is not None:
        kwargs["position_ids"] = _select_or_expand_batch_tensor(
            position_ids,
            sample_indices=sample_indices,
            n_local=n_local,
            batch_size=batch_size,
            name="layer_state.position_ids",
        )
    if position_embeddings is not None:
        if (
            not isinstance(position_embeddings, (tuple, list))
            or len(position_embeddings) != 2
        ):
            raise ValueError(
                "layer_state.position_embeddings must be a two-tensor "
                "tuple/list when provided."
            )
        kwargs["position_embeddings"] = tuple(
            _select_or_expand_batch_tensor(
                item,
                sample_indices=sample_indices,
                n_local=n_local,
                batch_size=batch_size,
                name=f"layer_state.position_embeddings[{index}]",
            )
            for index, item in enumerate(position_embeddings)
        )
    return kwargs


def _validate_slide(
    *,
    next_layer: nn.Module | None,
    next_fp_out: torch.Tensor | None,
    next_fisher: torch.Tensor | None,
    slide_alpha_fn: Callable[[], float] | None,
) -> bool:
    supplied = (
        next_layer is not None,
        next_fp_out is not None,
        next_fisher is not None,
        slide_alpha_fn is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "next-layer slide requires next_layer, next_fp_out, next_fisher, "
            "and slide_alpha_fn together."
        )
    return all(supplied)


def make_joint_moe_block_gd_refresh_fn(
    *,
    layer: nn.Module,
    target_linears: Sequence[nn.Linear],
    layer_state: "LayerInputs",
    fp_out_for_this_layer: torch.Tensor,
    fisher: torch.Tensor | None,
    contexts: Sequence["RefreshContext"],
    current_mlp_inputs: torch.Tensor | None = None,
    current_mlp_residuals: torch.Tensor | None = None,
    student_routes: "PackedLayerRoutes | None" = None,
    projection: str | None = None,
    expert_modules: Sequence[nn.Module] | None = None,
    next_layer: nn.Module | None = None,
    next_fp_out: torch.Tensor | None = None,
    next_fisher: torch.Tensor | None = None,
    slide_alpha_fn: Callable[[], float] | None = None,
    final_layer_analyzer: "ModelAnalyzer | None" = None,
    kl_topk: int = -1,
    a_loss_ratio: float = 1.0,
) -> Callable[
    [
        torch.Tensor | Sequence[torch.Tensor],
        Sequence[int],
        Sequence[torch.Tensor | None] | None,
    ],
    torch.Tensor | tuple[torch.Tensor, ...],
]:
    """Build one all-expert refresh closure for a synchronized column block.

    The returned closure has the calling convention::

        updates = refresh(
            stitched_weights_fp32,
            trailing_col_starts,
            perms=None,
        )

    Inputs and outputs are aligned with ``target_linears``.  Each returned
    tensor is FP32 and covers only that expert's permuted trailing column
    slice; the joint GPTQ state subtracts it from its FP32 working master.
    The closure mutates only the independent Adam states in ``contexts``.

    ``current_mlp_inputs`` and ``current_mlp_residuals`` are an all-or-none
    pair. Supplying the additional all-or-none
    ``student_routes``/``projection``/``expert_modules`` triple selects the
    fixed-route evaluator: it caches projection-specific operands once, then
    computes only scheduler-selected samples with one candidate ``(E,R,C)``
    bmm and never invokes the current router. Omitting that triple retains the
    complete current-MLP functional-call oracle.
    """

    if dist_utils.get_world_size() != 1:
        raise NotImplementedError(
            "joint MoE Block-GD currently implements the accepted single-GPU "
            "path only."
        )
    if not isinstance(layer, nn.Module):
        raise TypeError(f"layer must be nn.Module, got {type(layer).__name__}.")
    target_linears = tuple(target_linears)
    contexts = tuple(contexts)
    if not target_linears:
        raise ValueError("target_linears must contain at least one expert.")
    if len(contexts) != len(target_linears):
        raise ValueError(
            "contexts must align one-for-one with target_linears: "
            f"{len(contexts)} != {len(target_linears)}."
        )
    if any(not isinstance(linear, nn.Linear) for linear in target_linears):
        bad = [
            type(linear).__name__
            for linear in target_linears
            if not isinstance(linear, nn.Linear)
        ]
        raise TypeError(
            "target_linears must contain underlying nn.Linear modules; "
            f"bad entries={bad}."
        )
    if len({id(linear) for linear in target_linears}) != len(target_linears):
        raise ValueError("target_linears must not contain duplicates.")

    device = _require_cuda_tensor(
        target_linears[0].weight,
        name="target_linears[0].weight",
    )
    for expert_idx, (linear, ctx) in enumerate(
        zip(target_linears, contexts)
    ):
        _require_cuda_tensor(
            linear.weight,
            name=f"target_linears[{expert_idx}].weight",
            device=device,
        )
        if ctx.module is not linear:
            raise ValueError(
                f"contexts[{expert_idx}].module is not its target linear."
            )
        _require_cuda_tensor(
            ctx.exp_avg,
            name=f"contexts[{expert_idx}].exp_avg",
            device=device,
        )
        _require_cuda_tensor(
            ctx.exp_avg_sq,
            name=f"contexts[{expert_idx}].exp_avg_sq",
            device=device,
        )
        if tuple(ctx.exp_avg.shape) != tuple(linear.weight.shape):
            raise ValueError(
                f"contexts[{expert_idx}] Adam shape does not match weight: "
                f"{tuple(ctx.exp_avg.shape)} != {tuple(linear.weight.shape)}."
            )
        if ctx.backward_bsz <= 0:
            raise ValueError(
                f"contexts[{expert_idx}].backward_bsz must be positive."
            )
    scheduler = contexts[0].scheduler
    if any(ctx.scheduler is not scheduler for ctx in contexts[1:]):
        raise ValueError(
            "all joint expert RefreshContexts must share the same scheduler "
            "object."
        )
    backward_bsz = int(contexts[0].backward_bsz)
    if any(int(ctx.backward_bsz) != backward_bsz for ctx in contexts[1:]):
        raise ValueError(
            "all joint expert RefreshContexts must use one backward_bsz."
        )

    inps = layer_state.inps
    _require_cuda_tensor(inps, name="layer_state.inps", device=device)
    _require_cuda_tensor(
        fp_out_for_this_layer,
        name="fp_out_for_this_layer",
        device=device,
    )
    _require_optional_state_tensor(
        layer_state.attention_mask,
        name="layer_state.attention_mask",
        device=device,
    )
    _require_optional_state_tensor(
        layer_state.position_ids,
        name="layer_state.position_ids",
        device=device,
    )
    _require_optional_state_tensor(
        layer_state.position_embeddings,
        name="layer_state.position_embeddings",
        device=device,
    )
    if inps.dim() != 3:
        raise ValueError(
            f"layer_state.inps must have shape (N,T,H), got {tuple(inps.shape)}."
        )
    if tuple(fp_out_for_this_layer.shape) != tuple(inps.shape):
        raise ValueError(
            "fp_out_for_this_layer must match layer_state.inps shape: "
            f"{tuple(fp_out_for_this_layer.shape)} != {tuple(inps.shape)}."
        )
    _require_module_on_device(layer, name="layer", device=device)

    fast_current_supplied = (
        current_mlp_inputs is not None,
        current_mlp_residuals is not None,
    )
    if any(fast_current_supplied) and not all(fast_current_supplied):
        raise ValueError(
            "current_mlp_inputs and current_mlp_residuals must be supplied "
            "together."
        )
    fast_current = all(fast_current_supplied)
    if fast_current:
        current_module = getattr(layer, "mlp", None)
        if not isinstance(current_module, nn.Module):
            raise TypeError(
                "fast current replay requires layer.mlp to be nn.Module."
            )
        _require_cuda_tensor(
            current_mlp_inputs,
            name="current_mlp_inputs",
            device=device,
        )
        _require_cuda_tensor(
            current_mlp_residuals,
            name="current_mlp_residuals",
            device=device,
        )
        if tuple(current_mlp_inputs.shape) != tuple(inps.shape):
            raise ValueError(
                "current_mlp_inputs must match layer_state.inps shape: "
                f"{tuple(current_mlp_inputs.shape)} != {tuple(inps.shape)}."
            )
        if tuple(current_mlp_residuals.shape) != tuple(inps.shape):
            raise ValueError(
                "current_mlp_residuals must match layer_state.inps shape: "
                f"{tuple(current_mlp_residuals.shape)} != "
                f"{tuple(inps.shape)}."
            )
    else:
        current_module = layer

    fixed_route_args = (
        student_routes is not None,
        projection is not None,
        expert_modules is not None,
    )
    if any(fixed_route_args) and not all(fixed_route_args):
        raise ValueError(
            "fixed current replay requires student_routes, projection and "
            "expert_modules together."
        )
    fixed_route_enabled = all(fixed_route_args)
    fixed_route_cache = None
    fixed_route_evaluator = None
    if fixed_route_enabled:
        if not fast_current:
            raise ValueError(
                "fixed current replay also requires current_mlp_inputs and "
                "current_mlp_residuals."
            )
        from realq_moe.refresh.fixed_route_moe import (
            build_fixed_route_moe_cache,
            evaluate_fixed_route_moe,
        )

        fixed_route_cache = build_fixed_route_moe_cache(
            student_routes=student_routes,
            projection=projection,
            expert_modules=expert_modules,
            target_linears=target_linears,
            mlp_inputs=current_mlp_inputs,
            seq_len=int(inps.shape[1]),
        )
        fixed_route_evaluator = evaluate_fixed_route_moe

    slide_enabled = _validate_slide(
        next_layer=next_layer,
        next_fp_out=next_fp_out,
        next_fisher=next_fisher,
        slide_alpha_fn=slide_alpha_fn,
    )
    if final_layer_analyzer is not None and slide_enabled:
        raise ValueError(
            "final-layer KL and next-layer slide are mutually exclusive."
        )
    if final_layer_analyzer is None:
        if fisher is None:
            raise ValueError("fisher is required for current-layer refresh.")
        _require_cuda_tensor(fisher, name="fisher", device=device)
    else:
        if not isinstance(kl_topk, int) or isinstance(kl_topk, bool):
            raise ValueError(f"kl_topk must be an integer, got {kl_topk!r}.")
        final_norm = final_layer_analyzer.get_layernorm_before_head()
        final_lm_head = final_layer_analyzer.get_lm_head()
        if not isinstance(final_norm, nn.Module) or not isinstance(
            final_lm_head, nn.Module
        ):
            raise TypeError(
                "final_layer_analyzer must return nn.Module norm and lm_head."
            )
        _require_module_on_device(
            final_norm,
            name="final_layer_analyzer.norm",
            device=device,
        )
        _require_module_on_device(
            final_lm_head,
            name="final_layer_analyzer.lm_head",
            device=device,
        )

    if slide_enabled:
        _require_module_on_device(
            next_layer,
            name="next_layer",
            device=device,
        )
        _require_cuda_tensor(
            next_fp_out,
            name="next_fp_out",
            device=device,
        )
        _require_cuda_tensor(
            next_fisher,
            name="next_fisher",
            device=device,
        )
        if tuple(next_fp_out.shape) != tuple(inps.shape):
            raise ValueError(
                "next_fp_out must match layer_state.inps shape: "
                f"{tuple(next_fp_out.shape)} != {tuple(inps.shape)}."
            )
    if not 0.0 < float(a_loss_ratio) <= 1.0:
        raise ValueError(
            f"a_loss_ratio must be in (0, 1], got {a_loss_ratio!r}."
        )

    weight_names = tuple(
        _linear_weight_name(current_module, linear)
        for linear in target_linears
    )
    if len(set(weight_names)) != len(weight_names):
        raise RuntimeError(
            f"joint functional weight names must be unique: {weight_names}."
        )
    n_local = int(inps.shape[0])
    objective = (
        "kl" if final_layer_analyzer is not None else "fisher_mse"
    )
    adam_signatures = {
        (
            float(ctx.beta1),
            float(ctx.beta2),
            float(ctx.eps),
            float(ctx.layer_lr),
            float(ctx.grad_clip),
        )
        for ctx in contexts
    }
    batched_adam = (
        JointBatchedAdamState(contexts)
        if len(adam_signatures) == 1
        else None
    )

    @torch.no_grad()
    def refresh(
        stitched_weights_fp32: torch.Tensor | Sequence[torch.Tensor],
        trailing_col_starts: Sequence[int],
        perms: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        candidate_batch = (
            stitched_weights_fp32
            if torch.is_tensor(stitched_weights_fp32)
            else None
        )
        if candidate_batch is not None and candidate_batch.dim() != 3:
            raise ValueError(
                "a batched stitched_weights_fp32 tensor must have shape "
                f"(E,R,C), got {tuple(candidate_batch.shape)}."
            )
        if fixed_route_enabled and candidate_batch is None:
            raise ValueError(
                "fixed-route joint refresh requires one batched "
                "stitched_weights_fp32 tensor with shape (E,R,C)."
            )
        candidates = tuple(stitched_weights_fp32)
        trailing_starts = tuple(trailing_col_starts)
        permutation_batch = perms if torch.is_tensor(perms) else None
        if perms is None:
            permutations: tuple[torch.Tensor | None, ...] = (
                None,
            ) * len(target_linears)
        else:
            permutations = tuple(perms)
        expected_count = len(target_linears)
        for name, values in (
            ("stitched_weights_fp32", candidates),
            ("trailing_col_starts", trailing_starts),
            ("perms", permutations),
        ):
            if len(values) != expected_count:
                raise ValueError(
                    f"{name} must contain {expected_count} entries, "
                    f"got {len(values)}."
                )

        current_steps = {int(ctx.adam_step) for ctx in contexts}
        if len(current_steps) != 1:
            raise RuntimeError(
                "joint expert Adam steps diverged before refresh: "
                f"{sorted(current_steps)}."
            )
        if len(set(trailing_starts)) != 1:
            raise ValueError(
                "all experts must advance the same trailing column boundary "
                f"in lockstep, got {trailing_starts}."
            )

        for expert_idx, (
            candidate,
            trailing_start,
            permutation,
            linear,
        ) in enumerate(
            zip(
                candidates,
                trailing_starts,
                permutations,
                target_linears,
            )
        ):
            _require_cuda_tensor(
                candidate,
                name=f"stitched_weights_fp32[{expert_idx}]",
                device=device,
            )
            if candidate.dtype != torch.float32:
                raise ValueError(
                    f"stitched_weights_fp32[{expert_idx}] must be FP32, "
                    f"got {candidate.dtype}."
                )
            if tuple(candidate.shape) != tuple(linear.weight.shape):
                raise ValueError(
                    f"candidate {expert_idx} shape must match target weight: "
                    f"{tuple(candidate.shape)} != "
                    f"{tuple(linear.weight.shape)}."
                )
            columns = int(linear.weight.shape[1])
            if (
                not isinstance(trailing_start, int)
                or isinstance(trailing_start, bool)
                or not 0 <= trailing_start <= columns
            ):
                raise ValueError(
                    f"trailing_col_starts[{expert_idx}] must be in "
                    f"[0, {columns}], got {trailing_start!r}."
                )
            if permutation is not None:
                _require_cuda_tensor(
                    permutation,
                    name=f"perms[{expert_idx}]",
                    device=device,
                )
                if permutation.dim() != 1 or permutation.numel() != columns:
                    raise ValueError(
                        f"perms[{expert_idx}] must have shape ({columns},), "
                        f"got {tuple(permutation.shape)}."
                    )
                if permutation.dtype not in (
                    torch.int32,
                    torch.int64,
                ):
                    raise ValueError(
                        f"perms[{expert_idx}] must be integer, "
                        f"got {permutation.dtype}."
                    )

        with nvtx.nvtx_range("moe_joint_refresh.setup"):
            for ctx in contexts:
                ctx.adam_step += 1
            # The defining joint scheduling rule: exactly one scheduler call,
            # shared by every expert at this column boundary.
            selected_global = [
                int(index) for index in contexts[0].next_indices()
            ]
            if not selected_global:
                raise RuntimeError(
                    "joint MoE refresh scheduler returned no samples."
                )
            if min(selected_global) < 0 or max(selected_global) >= n_local:
                raise ValueError(
                    "single-GPU joint scheduler indices must address the "
                    f"local [0, {n_local}) sample range; got "
                    f"[{min(selected_global)}, {max(selected_global)}]."
                )
            slide_alpha: float | None = None
            if slide_enabled:
                slide_alpha = float(slide_alpha_fn())
                if (
                    not math.isfinite(slide_alpha)
                    or not 0.0 <= slide_alpha <= 1.0
                ):
                    raise ValueError(
                        "slide_alpha_fn must return a finite value in [0,1], "
                        f"got {slide_alpha!r}."
                    )
            target_dtypes = {
                linear.weight.dtype for linear in target_linears
            }
            if candidate_batch is not None and len(target_dtypes) == 1:
                # The expert-batched analytic stepper already owns one
                # contiguous (E,R,C) FP32 tensor.  Cast it with one GPU kernel
                # and expose expert views to functional_call/autograd instead
                # of launching E independent full-weight casts.
                with torch.enable_grad():
                    override_batch = (
                        candidate_batch.detach()
                        .to(dtype=target_linears[0].weight.dtype)
                        .requires_grad_(True)
                    )
                    override_weights = tuple(
                        override_batch.unbind(dim=0)
                    )
            else:
                override_batch = None
                override_weights = tuple(
                    candidate.detach()
                    .to(dtype=linear.weight.dtype)
                    .requires_grad_(True)
                    for candidate, linear in zip(
                        candidates, target_linears
                    )
                )
            overrides = (
                None
                if fixed_route_enabled
                else dict(zip(weight_names, override_weights))
            )
            if candidate_batch is not None:
                partial_grad_batch = torch.zeros_like(candidate_batch)
                partial_grad_sums = tuple(
                    partial_grad_batch.unbind(dim=0)
                )
            else:
                partial_grad_batch = None
                partial_grad_sums = tuple(
                    torch.zeros_like(candidate)
                    for candidate in candidates
                )
            partial_count = 0
            observe_loss = any(
                ctx.loss_observation_enabled for ctx in contexts
            )
            has_next_loss = (
                slide_alpha is not None and slide_alpha < 1.0
            )
            partial_loss_sums = (
                torch.zeros(
                    3 if has_next_loss else 1,
                    dtype=torch.float64,
                    device=device,
                )
                if observe_loss
                else None
            )

        for iteration, start in enumerate(
            range(0, len(selected_global), backward_bsz)
        ):
            with nvtx.nvtx_range(
                f"moe_joint_refresh.iter_{iteration}"
            ):
                batch_indices = selected_global[
                    start : start + backward_bsz
                ]
                batch_size = len(batch_indices)
                sample_indices = torch.tensor(
                    batch_indices,
                    dtype=torch.int64,
                    device=device,
                )
                fp_target = fp_out_for_this_layer.index_select(
                    0, sample_indices
                )
                kwargs = _batch_kwargs(
                    layer_state,
                    sample_indices=sample_indices,
                    n_local=n_local,
                    batch_size=batch_size,
                )
                with torch.enable_grad():
                    with nvtx.nvtx_range(
                        "moe_joint_refresh.current_forward"
                    ):
                        if fast_current:
                            residual = current_mlp_residuals.index_select(
                                0, sample_indices
                            )
                            if fixed_route_enabled:
                                fixed_output = fixed_route_evaluator(
                                    fixed_route_cache,
                                    override_batch,
                                    sample_indices,
                                )
                                mlp_out = fixed_output.hidden_states
                            else:
                                mlp_input = current_mlp_inputs.index_select(
                                    0, sample_indices
                                )
                                package = functional_call(
                                    current_module,
                                    overrides,
                                    (mlp_input,),
                                    strict=False,
                                )
                                mlp_out = _tensor_output(
                                    package,
                                    source="joint current MLP",
                                )
                            if tuple(mlp_out.shape) != tuple(
                                residual.shape
                            ):
                                raise RuntimeError(
                                    "joint current MLP output shape changed: "
                                    f"{tuple(mlp_out.shape)} != "
                                    f"{tuple(residual.shape)}."
                                )
                            q_out = residual + mlp_out
                        else:
                            x = inps.index_select(0, sample_indices)
                            package = functional_call(
                                current_module,
                                overrides,
                                (x,),
                                kwargs,
                                strict=False,
                            )
                            q_out = _tensor_output(
                                package,
                                source="joint current layer",
                            )
                    if tuple(q_out.shape) != tuple(fp_target.shape):
                        raise RuntimeError(
                            "joint current output shape changed: "
                            f"{tuple(q_out.shape)} != "
                            f"{tuple(fp_target.shape)}."
                        )
                    with nvtx.nvtx_range(
                        "moe_joint_refresh.loss"
                    ):
                        if final_layer_analyzer is not None:
                            loss_curr = kl_topk_loss(
                                q_out,
                                fp_target,
                                final_layer_analyzer,
                                kl_topk,
                            )
                        else:
                            loss_curr = fisher_mse_loss(
                                q_out,
                                fp_target,
                                fisher,
                                a_loss_ratio=float(a_loss_ratio),
                            )
                        loss_next = None
                        if has_next_loss:
                            # This is a complete next-layer replay.  In
                            # particular, a sparse next layer runs its router
                            # on q_out rather than reusing cached assignments.
                            with nvtx.nvtx_range(
                                "moe_joint_refresh.next_forward"
                            ):
                                next_package = next_layer(
                                    q_out,
                                    **kwargs,
                                )
                                next_q_out = _tensor_output(
                                    next_package,
                                    source="joint next layer",
                                )
                            fp_target_next = next_fp_out.index_select(
                                0, sample_indices
                            )
                            if tuple(next_q_out.shape) != tuple(
                                fp_target_next.shape
                            ):
                                raise RuntimeError(
                                    "joint next output shape changed: "
                                    f"{tuple(next_q_out.shape)} != "
                                    f"{tuple(fp_target_next.shape)}."
                                )
                            loss_next = fisher_mse_loss(
                                next_q_out,
                                fp_target_next,
                                next_fisher,
                                a_loss_ratio=float(a_loss_ratio),
                            )
                            loss = (
                                slide_alpha * loss_curr
                                + (1.0 - slide_alpha) * loss_next
                            )
                        else:
                            loss = loss_curr
                    with nvtx.nvtx_range(
                        "moe_joint_refresh.backward"
                    ):
                        batch_grad_batch = None
                        batch_grads = None
                        if loss.requires_grad:
                            if override_batch is not None:
                                batch_grad_batch = torch.autograd.grad(
                                    loss,
                                    override_batch,
                                    retain_graph=False,
                                    allow_unused=True,
                                )[0]
                            else:
                                batch_grads = torch.autograd.grad(
                                    loss,
                                    override_weights,
                                    retain_graph=False,
                                    allow_unused=True,
                                )
                        elif override_batch is None:
                            batch_grads = (None,) * len(
                                override_weights
                            )

                if override_batch is not None:
                    if batch_grad_batch is not None:
                        partial_grad_batch.add_(
                            batch_grad_batch.detach().float(),
                            alpha=float(batch_size),
                        )
                else:
                    for partial_grad_sum, batch_grad in zip(
                        partial_grad_sums,
                        batch_grads,
                    ):
                        if batch_grad is not None:
                            partial_grad_sum.add_(
                                batch_grad.detach().float(),
                                alpha=float(batch_size),
                            )
                partial_count += batch_size
                if partial_loss_sums is not None:
                    partial_loss_sums[0].add_(
                        loss.detach().float(),
                        alpha=float(batch_size),
                    )
                    if has_next_loss:
                        partial_loss_sums[1].add_(
                            loss_curr.detach().float(),
                            alpha=float(batch_size),
                        )
                        partial_loss_sums[2].add_(
                            loss_next.detach().float(),
                            alpha=float(batch_size),
                        )

        if partial_count != len(selected_global):
            raise RuntimeError(
                "joint refresh sample coverage mismatch: "
                f"{partial_count} != {len(selected_global)}."
            )
        if partial_count <= 0:
            raise RuntimeError("joint refresh produced zero samples.")
        if partial_grad_batch is not None:
            accumulated_grad_batch = (
                partial_grad_batch / float(partial_count)
            )
            if (
                batched_adam is not None
                and (perms is None or permutation_batch is not None)
            ):
                accumulated_grads = None
            else:
                accumulated_grads = tuple(
                    accumulated_grad_batch.unbind(dim=0)
                )
        else:
            accumulated_grad_batch = None
            accumulated_grads = tuple(
                grad_sum / float(partial_count)
                for grad_sum in partial_grad_sums
            )

        if partial_loss_sums is not None:
            for ctx in contexts:
                if ctx.loss_observation_enabled:
                    ctx.record_loss_observation(
                        global_loss_sums=partial_loss_sums,
                        global_count=partial_count,
                        sample_indices=selected_global,
                        slide_alpha=slide_alpha,
                        has_next_loss=has_next_loss,
                        objective=objective,
                    )

        with nvtx.nvtx_range("moe_joint_refresh.adam"):
            if (
                batched_adam is not None
                and accumulated_grad_batch is not None
                and (perms is None or permutation_batch is not None)
            ):
                return batched_adam.step(
                    accumulated_grad_batch,
                    trailing_starts[0],
                    permutation_batch,
                )

            updates: list[torch.Tensor] = []
            for expert_idx, (
                ctx,
                accum_grad,
                trailing_start,
                permutation,
            ) in enumerate(
                zip(
                    contexts,
                    accumulated_grads,
                    trailing_starts,
                    permutations,
                )
            ):
                if permutation is not None:
                    accum_grad = accum_grad.index_select(
                        1, permutation.to(dtype=torch.int64)
                    )
                grad_slice = accum_grad[:, trailing_start:]
                if ctx.grad_clip > 0:
                    grad_slice = grad_slice.clamp(
                        min=-ctx.grad_clip,
                        max=ctx.grad_clip,
                    )
                exp_avg = ctx.exp_avg[:, trailing_start:]
                exp_avg_sq = ctx.exp_avg_sq[:, trailing_start:]
                exp_avg.mul_(ctx.beta1).add_(
                    grad_slice,
                    alpha=1.0 - ctx.beta1,
                )
                exp_avg_sq.mul_(ctx.beta2).addcmul_(
                    grad_slice,
                    grad_slice,
                    value=1.0 - ctx.beta2,
                )
                bias_correction1 = 1.0 - ctx.beta1 ** ctx.adam_step
                bias_correction2 = 1.0 - ctx.beta2 ** ctx.adam_step
                denominator = exp_avg_sq.sqrt() / math.sqrt(
                    bias_correction2
                )
                denominator.add_(ctx.eps)
                step_size = ctx.layer_lr / bias_correction1
                update = step_size * (exp_avg / denominator)
                expected_shape = (
                    int(target_linears[expert_idx].weight.shape[0]),
                    int(target_linears[expert_idx].weight.shape[1])
                    - trailing_start,
                )
                if tuple(update.shape) != expected_shape:
                    raise RuntimeError(
                        f"joint update {expert_idx} shape mismatch: "
                        f"{tuple(update.shape)} != {expected_shape}."
                    )
                updates.append(update)
        return tuple(updates)

    if fixed_route_cache is not None:
        refresh.fixed_route_padding_stats = fixed_route_cache.padding
        # Keep diagnostics lightweight.  Attaching the multi-GiB cache to the
        # returned function and mutating the function from inside itself would
        # create a reference cycle that can retain one projection's cache while
        # the next projection builds all expert Hessians.
        refresh.fixed_route_last_batch_stats = None
    return refresh
