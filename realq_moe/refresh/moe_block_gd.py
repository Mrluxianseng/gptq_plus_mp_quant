"""MoE expert Block-GD with a fixed-route contribution decomposition.

This module is intentionally separate from :mod:`realq_moe.refresh.block_gd`.
The dense/attention refresh path replays a complete transformer layer and must
remain unchanged for the frozen dense baseline.  A routed expert cannot afford
that replay for every projection and every column block, so its current-layer
output is reconstructed as

``y_running + scatter(route_weight * (expert_candidate - expert_old))``.

The decomposition is algebraically exact while the router, attention, other
experts, and all non-target projections are fixed.  It is not guaranteed to be
bit-exact to a fresh Hugging Face MoE forward in BF16: recomputing one expert
can select a different GEMM shape, and subtract/scatter-add has a different
floating-point accumulation order from the original expert loop.  The caller
must use tolerance-based equivalence tests and retain a final natural layer
replay as the authoritative stream update.

Input contract
--------------

* ``expert`` is one complete expert MLP. ``target_linear`` is the underlying
  ``nn.Linear`` for the projection currently being quantized, including when
  that linear is nested inside an ``ActQuantWrapper``.
* ``student_route`` contains rank-local flattened token indices and route
  weights in the same order.  Indices address ``mlp_inputs.reshape(-1, H)``.
  One expert may have zero local assignments.
* ``old_expert_outputs`` is the *unweighted* output of ``expert`` immediately
  before this projection is quantized, aligned one-for-one with
  ``student_route``.  Route weights are applied exactly once in this module.
* ``mlp_inputs``, ``y_running``, and ``fp_out_for_this_layer`` have leading
  shape ``(N_local, seq_len)``.  They may be CUDA tensors or CPU/pinned staging
  tensors. ``y_running`` is the full current student layer output before the
  candidate projection override.
* ``ctx.scheduler.next_indices()`` returns the same GLOBAL sample-id sequence
  on every rank. ``global_sample_start`` maps this rank's local sample zero
  into that global space.  Every selected sample is counted, even if it has no
  local route assignment; such a sample contributes an exact zero gradient.
* For expert-hit conditional sampling, pass
  ``hit_sample_probability = num_global_hit_samples / num_global_samples``.
  Pass ``1.0`` for an unconditional scheduler.
* Other expert parameters should be frozen.  The closure differentiates only
  the supplied stitched target weight.

Only the current-layer Fisher objective is used by default.  Supplying
``final_layer_analyzer`` switches to the existing real-time ``norm + lm_head``
KL objective for a final transformer layer.  There is deliberately no
next-layer/sliding arm.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.func import functional_call

from realq_moe.refresh.block_gd import (
    _aggregate_loss_sums_for_logging,
    _aggregate_refresh_sums,
)
from realq_moe.refresh.fisher_loss import fisher_mse_loss
from realq_moe.refresh.kl_loss import kl_topk_loss
from realq_moe.utils import nvtx
from utils import dist_utils

if TYPE_CHECKING:
    from realq_moe.refresh.block_gd import RefreshContext
    from utils.model_utils import ModelAnalyzer


@dataclass(frozen=True)
class StudentExpertRoute:
    """Rank-local natural student assignments for one expert.

    ``flat_token_indices`` indexes the flattened ``(N_local, T)`` token axis.
    ``route_weights`` is either ``(A,)`` or ``(A, 1)`` and is aligned with the
    assignment order.
    """

    flat_token_indices: torch.Tensor
    route_weights: torch.Tensor


def _route_tensors(
    route: StudentExpertRoute | Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(route, StudentExpertRoute):
        return route.flat_token_indices, route.route_weights
    if not isinstance(route, Mapping):
        raise TypeError(
            "student_route must be StudentExpertRoute or a tensor mapping, "
            f"got {type(route).__name__}."
        )
    flat = route.get("flat_token_indices")
    if flat is None:
        # Accept the spelling used in some CSR staging code, while emitting
        # one canonical contract from this module.
        flat = route.get("flat_token_ids")
    weights = route.get("route_weights")
    if flat is None or weights is None:
        raise KeyError(
            "student_route requires 'flat_token_indices' (or "
            "'flat_token_ids') and 'route_weights'."
        )
    return flat, weights


def _underlying_weight_name(
    expert: nn.Module,
    target_linear: nn.Linear,
) -> str:
    """Return the functional-call key that the expert forward actually reads.

    ``ActQuantWrapper`` registers a direct ``weight`` alias as well as its
    inner ``module.weight``.  Resolving by parameter identity can choose the
    wrapper alias even though ``forward`` calls the inner linear.  Resolving
    the module object first produces the canonical inner key and avoids that
    ambiguity.
    """

    matches = [
        module_name
        for module_name, module in expert.named_modules()
        if module is target_linear
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "target_linear must occur exactly once inside expert.named_modules; "
            f"found {len(matches)} matches."
        )
    module_name = matches[0]
    return f"{module_name}.weight" if module_name else "weight"


def _index_select_to(
    source: torch.Tensor,
    indices_cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Detached index-select that supports CPU/pinned or device staging."""

    source = source.detach()
    indices = (
        indices_cpu
        if source.device.type == "cpu"
        else indices_cpu.to(source.device, non_blocking=True)
    )
    selected = source.index_select(0, indices)
    return selected.to(device=device, non_blocking=True)


def _expert_output(output) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(value):
        raise RuntimeError(
            "expert functional_call must return a Tensor or a tuple/list whose "
            f"first item is a Tensor, got {type(value).__name__}."
        )
    return value


def make_moe_expert_refresh_fn(
    *,
    expert: nn.Module,
    target_linear: nn.Linear,
    student_route: StudentExpertRoute | Mapping[str, torch.Tensor],
    mlp_inputs: torch.Tensor,
    y_running: torch.Tensor,
    old_expert_outputs: torch.Tensor,
    fp_out_for_this_layer: torch.Tensor,
    fisher: torch.Tensor | None,
    ctx: "RefreshContext",
    seq_len: int,
    global_sample_start: int | None = None,
    hit_sample_probability: float = 1.0,
    a_loss_ratio: float = 1.0,
    final_layer_analyzer: "ModelAnalyzer | None" = None,
    kl_topk: int = -1,
) -> Callable[..., torch.Tensor]:
    """Build one projection's contribution-decomposed Block-GD closure.

    The returned closure has the same calling convention as the legacy dense
    closure::

        update = refresh(stitched_weight_fp32, trailing_col_start, perm=None)

    It returns an FP32 update for ``[:, trailing_col_start:]``.  The caller
    applies that update to the working FP32 weight master.

    ``a_loss_ratio`` is deliberately restricted to ``1.0`` in the first MoE
    implementation.  A percentile population over conditionally selected,
    ragged expert hits needs its own distributed numerical contract and must
    not silently inherit dense chunk-local clipping.
    """

    if not isinstance(expert, nn.Module):
        raise TypeError(f"expert must be nn.Module, got {type(expert).__name__}.")
    if not isinstance(target_linear, nn.Linear):
        raise TypeError(
            "target_linear must be the underlying nn.Linear, got "
            f"{type(target_linear).__name__}."
        )
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
        raise ValueError(f"seq_len must be a positive integer, got {seq_len!r}.")
    if float(a_loss_ratio) != 1.0:
        raise ValueError(
            "MoE contribution refresh currently requires a_loss_ratio=1.0; "
            f"got {a_loss_ratio!r}."
        )
    hit_sample_probability = float(hit_sample_probability)
    if not 0.0 < hit_sample_probability <= 1.0:
        raise ValueError(
            "hit_sample_probability must be in (0, 1], got "
            f"{hit_sample_probability!r}."
        )
    if not isinstance(kl_topk, int) or isinstance(kl_topk, bool):
        raise ValueError(f"kl_topk must be an integer, got {kl_topk!r}.")
    if ctx.backward_bsz <= 0:
        raise ValueError(
            f"RefreshContext.backward_bsz must be positive, got {ctx.backward_bsz}."
        )

    for name, tensor, expected_dim in (
        ("mlp_inputs", mlp_inputs, 3),
        ("y_running", y_running, 3),
        ("fp_out_for_this_layer", fp_out_for_this_layer, 3),
        ("old_expert_outputs", old_expert_outputs, 2),
    ):
        if not torch.is_tensor(tensor) or tensor.dim() != expected_dim:
            shape = tuple(tensor.shape) if torch.is_tensor(tensor) else None
            raise ValueError(
                f"{name} must be a {expected_dim}D Tensor, got {shape}."
            )

    n_local = int(mlp_inputs.shape[0])
    if n_local <= 0:
        raise ValueError("mlp_inputs must contain at least one local sample.")
    if int(mlp_inputs.shape[1]) != seq_len:
        raise ValueError(
            f"mlp_inputs sequence length {mlp_inputs.shape[1]} != {seq_len}."
        )
    expected_leading = (n_local, seq_len)
    if tuple(y_running.shape[:2]) != expected_leading:
        raise ValueError(
            "y_running leading shape must match mlp_inputs: "
            f"{tuple(y_running.shape[:2])} != {expected_leading}."
        )
    if tuple(fp_out_for_this_layer.shape) != tuple(y_running.shape):
        raise ValueError(
            "fp_out_for_this_layer must match y_running exactly: "
            f"{tuple(fp_out_for_this_layer.shape)} != "
            f"{tuple(y_running.shape)}."
        )
    output_hidden = int(y_running.shape[-1])
    if tuple(old_expert_outputs.shape[1:]) != (output_hidden,):
        raise ValueError(
            "old_expert_outputs hidden width must match y_running: "
            f"{tuple(old_expert_outputs.shape)} vs H={output_hidden}."
        )
    if final_layer_analyzer is None and fisher is None:
        raise ValueError(
            "fisher is required unless final_layer_analyzer selects KL refresh."
        )

    flat_token_indices, route_weights = _route_tensors(student_route)
    if not torch.is_tensor(flat_token_indices):
        raise TypeError("student route flat token indices must be a Tensor.")
    if not torch.is_tensor(route_weights):
        raise TypeError("student route weights must be a Tensor.")
    if flat_token_indices.dim() != 1:
        raise ValueError(
            "student route flat token indices must be 1D, got "
            f"{tuple(flat_token_indices.shape)}."
        )
    if route_weights.dim() == 2 and route_weights.shape[1] == 1:
        route_weights = route_weights.reshape(-1)
    if route_weights.dim() != 1:
        raise ValueError(
            "student route weights must have shape (A,) or (A,1), got "
            f"{tuple(route_weights.shape)}."
        )
    assignment_count = int(flat_token_indices.numel())
    if int(route_weights.numel()) != assignment_count:
        raise ValueError(
            "student route ids/weights length mismatch: "
            f"{assignment_count} != {route_weights.numel()}."
        )
    if int(old_expert_outputs.shape[0]) != assignment_count:
        raise ValueError(
            "old_expert_outputs must align one-for-one with student route: "
            f"{old_expert_outputs.shape[0]} != {assignment_count}."
        )

    # Routing metadata is small compared with activations and is repeatedly
    # sliced in Python-sized backward batches. Keep one canonical CPU CSR view;
    # payload tensors and large activation/output staging remain where supplied.
    flat_cpu = flat_token_indices.detach().to(
        device="cpu", dtype=torch.int64
    ).contiguous()
    if assignment_count:
        min_flat = int(flat_cpu.min().item())
        max_flat = int(flat_cpu.max().item())
        if min_flat < 0 or max_flat >= n_local * seq_len:
            raise ValueError(
                "student route flat token index outside local calibration "
                f"range [0, {n_local * seq_len}): min={min_flat}, max={max_flat}."
            )
    route_sample_ids = torch.div(
        flat_cpu, seq_len, rounding_mode="floor"
    )
    route_order = torch.argsort(route_sample_ids, stable=True)
    counts = torch.bincount(route_sample_ids, minlength=n_local)
    sample_offsets = torch.empty(n_local + 1, dtype=torch.int64)
    sample_offsets[0] = 0
    torch.cumsum(counts, dim=0, out=sample_offsets[1:])

    weight_name = _underlying_weight_name(expert, target_linear)
    target_shape = tuple(target_linear.weight.shape)
    if tuple(ctx.exp_avg.shape) != target_shape or tuple(ctx.exp_avg_sq.shape) != target_shape:
        raise ValueError(
            "RefreshContext Adam state must match target_linear.weight: "
            f"weight={target_shape}, exp_avg={tuple(ctx.exp_avg.shape)}, "
            f"exp_avg_sq={tuple(ctx.exp_avg_sq.shape)}."
        )
    if global_sample_start is None:
        # This matches the existing equal contiguous DP shard convention. Runs
        # with uneven shards must pass the exact dist_utils.shard_slice start.
        global_sample_start = dist_utils.get_rank() * n_local
    if (
        not isinstance(global_sample_start, int)
        or isinstance(global_sample_start, bool)
        or global_sample_start < 0
    ):
        raise ValueError(
            "global_sample_start must be a non-negative integer, got "
            f"{global_sample_start!r}."
        )
    global_sample_end = global_sample_start + n_local
    mlp_inputs_flat = mlp_inputs.reshape(
        n_local * seq_len, mlp_inputs.shape[-1]
    )
    objective = "kl" if final_layer_analyzer is not None else "fisher_mse"

    def assignment_indices_for_batch(
        local_sample_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        route_position_parts: list[torch.Tensor] = []
        destination_parts: list[torch.Tensor] = []
        for batch_position, sample_idx in enumerate(local_sample_indices):
            start = int(sample_offsets[sample_idx].item())
            end = int(sample_offsets[sample_idx + 1].item())
            if start == end:
                continue
            positions = route_order[start:end]
            token_indices = flat_cpu.index_select(0, positions).remainder(
                seq_len
            )
            route_position_parts.append(positions)
            destination_parts.append(
                token_indices + batch_position * seq_len
            )
        if not route_position_parts:
            empty = torch.empty(0, dtype=torch.int64)
            return empty, empty
        return (
            torch.cat(route_position_parts, dim=0),
            torch.cat(destination_parts, dim=0),
        )

    def compute_loss(
        q_hidden: torch.Tensor,
        fp_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if final_layer_analyzer is not None:
            loss = kl_topk_loss(
                q_hidden,
                fp_hidden,
                final_layer_analyzer,
                kl_topk,
            )
        else:
            loss = fisher_mse_loss(
                q_hidden,
                fp_hidden,
                fisher,
                a_loss_ratio=1.0,
            )
        # Conditional expert-hit sampling estimates E[g | hit]. Multiplying
        # by P(hit) recovers E[g], because non-hit samples have exactly zero
        # derivative with respect to this expert.
        if hit_sample_probability != 1.0:
            loss = loss * hit_sample_probability
        return loss

    @torch.no_grad()
    def refresh(
        stitched_weight_fp32: torch.Tensor,
        trailing_col_start: int,
        perm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tuple(stitched_weight_fp32.shape) != target_shape:
            raise ValueError(
                "stitched_weight_fp32 must match target_linear.weight: "
                f"{tuple(stitched_weight_fp32.shape)} != {target_shape}."
            )
        if (
            not isinstance(trailing_col_start, int)
            or isinstance(trailing_col_start, bool)
            or not 0 <= trailing_col_start <= target_shape[1]
        ):
            raise ValueError(
                "trailing_col_start must be in [0, columns], got "
                f"{trailing_col_start!r} for columns={target_shape[1]}."
            )

        with nvtx.nvtx_range("moe_refresh.setup"):
            ctx.adam_step += 1
            selected_global = [
                int(index) for index in ctx.next_indices()
            ]
            if not selected_global:
                raise RuntimeError(
                    "MoE refresh scheduler returned an empty global sample set."
                )
            selected_local = [
                global_idx - global_sample_start
                for global_idx in selected_global
                if global_sample_start
                <= global_idx
                < global_sample_end
            ]
            partial_grad_sum = torch.zeros_like(stitched_weight_fp32)
            partial_count = 0
            partial_loss_sums = (
                torch.zeros(
                    1,
                    dtype=torch.float64,
                    device=partial_grad_sum.device,
                )
                if ctx.loss_observation_enabled
                else None
            )
            override_weight = stitched_weight_fp32.to(
                device=target_linear.weight.device,
                dtype=target_linear.weight.dtype,
            ).requires_grad_(True)

        iteration = 0
        for start in range(0, len(selected_local), ctx.backward_bsz):
            with nvtx.nvtx_range(f"moe_refresh.iter_{iteration}"):
                batch_local = selected_local[
                    start : start + ctx.backward_bsz
                ]
                batch_size = len(batch_local)
                batch_samples_cpu = torch.tensor(
                    batch_local, dtype=torch.int64
                )
                q_hidden = _index_select_to(
                    y_running,
                    batch_samples_cpu,
                    override_weight.device,
                )
                fp_hidden = _index_select_to(
                    fp_out_for_this_layer,
                    batch_samples_cpu,
                    override_weight.device,
                )
                route_positions_cpu, destinations_cpu = (
                    assignment_indices_for_batch(batch_local)
                )

                if route_positions_cpu.numel():
                    flat_input_indices_cpu = flat_cpu.index_select(
                        0, route_positions_cpu
                    )
                    expert_inputs = _index_select_to(
                        mlp_inputs_flat,
                        flat_input_indices_cpu,
                        override_weight.device,
                    )
                    old_outputs = _index_select_to(
                        old_expert_outputs,
                        route_positions_cpu,
                        override_weight.device,
                    )
                    weights = _index_select_to(
                        route_weights,
                        route_positions_cpu,
                        override_weight.device,
                    )
                    destinations = destinations_cpu.to(
                        override_weight.device, non_blocking=True
                    )
                    with torch.enable_grad():
                        with nvtx.nvtx_range("moe_refresh.expert_forward"):
                            candidate_outputs = _expert_output(
                                functional_call(
                                    expert,
                                    {weight_name: override_weight},
                                    (expert_inputs,),
                                    strict=False,
                                )
                            )
                        if tuple(candidate_outputs.shape) != tuple(
                            old_outputs.shape
                        ):
                            raise RuntimeError(
                                "candidate expert output shape changed under "
                                f"functional_call: {tuple(candidate_outputs.shape)} "
                                f"!= {tuple(old_outputs.shape)}."
                            )
                        contribution_delta = (
                            candidate_outputs
                            - old_outputs.to(candidate_outputs.dtype)
                        )
                        contribution_delta = contribution_delta * weights.to(
                            candidate_outputs.dtype
                        ).reshape(-1, 1)
                        q_flat = torch.index_add(
                            q_hidden.reshape(-1, output_hidden),
                            0,
                            destinations,
                            contribution_delta.to(q_hidden.dtype),
                        )
                        q_candidate = q_flat.reshape_as(q_hidden)
                        with nvtx.nvtx_range("moe_refresh.loss"):
                            loss = compute_loss(q_candidate, fp_hidden)
                        with nvtx.nvtx_range("moe_refresh.backward"):
                            (batch_grad,) = torch.autograd.grad(
                                loss,
                                override_weight,
                                retain_graph=False,
                            )
                    partial_grad_sum.add_(
                        batch_grad.detach().float(),
                        alpha=float(batch_size),
                    )
                    del (
                        expert_inputs,
                        old_outputs,
                        weights,
                        candidate_outputs,
                        contribution_delta,
                        q_flat,
                        q_candidate,
                        batch_grad,
                    )
                else:
                    # The selected sample does not use this expert. Its complete
                    # current-layer loss is useful for diagnostics but its
                    # derivative with respect to the override is exactly zero.
                    loss = compute_loss(q_hidden, fp_hidden)

                partial_count += batch_size
                if partial_loss_sums is not None:
                    partial_loss_sums[0].add_(
                        loss.detach().float(),
                        alpha=float(batch_size),
                    )
                del q_hidden, fp_hidden, loss
                iteration += 1

        # This collective is unconditional: a rank with no selected local
        # samples contributes a full-sized zero gradient and count zero.
        with nvtx.nvtx_range("moe_refresh.grad_allreduce"):
            global_count, global_loss_sums = _aggregate_refresh_sums(
                partial_grad_sum,
                partial_count,
                partial_loss_sums if ctx.trace_enabled else None,
            )
            if global_count != len(selected_global):
                raise RuntimeError(
                    "global refresh sample coverage does not match scheduler: "
                    f"all-reduced count={global_count}, "
                    f"scheduler count={len(selected_global)}. Check "
                    "global_sample_start and non-overlapping DP shards."
                )
            accum_grad = partial_grad_sum / float(global_count)

        if ctx.log_column_block_loss and not ctx.trace_enabled:
            if partial_loss_sums is None:
                raise RuntimeError(
                    "column-block loss logging enabled without loss sums."
                )
            global_loss_sums = _aggregate_loss_sums_for_logging(
                partial_loss_sums
            )
        if global_loss_sums is not None:
            ctx.record_loss_observation(
                global_loss_sums=global_loss_sums,
                global_count=global_count,
                sample_indices=selected_global,
                slide_alpha=None,
                has_next_loss=False,
                objective=objective,
            )

        with nvtx.nvtx_range("moe_refresh.adam_step"):
            # Keep the legacy coordinate convention: the Hessian act-order
            # permutation re-keys the natural-column gradient before moments.
            if perm is not None:
                accum_grad = accum_grad[:, perm]
            grad_slice = accum_grad[:, trailing_col_start:]
            if ctx.grad_clip > 0:
                grad_slice = grad_slice.clamp(
                    min=-ctx.grad_clip, max=ctx.grad_clip
                )
            exp_avg = ctx.exp_avg[:, trailing_col_start:]
            exp_avg_sq = ctx.exp_avg_sq[:, trailing_col_start:]
            exp_avg.mul_(ctx.beta1).add_(
                grad_slice, alpha=1.0 - ctx.beta1
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
            return step_size * (exp_avg / denominator)

    return refresh
