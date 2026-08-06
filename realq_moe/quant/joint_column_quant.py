"""GPU-resident GPTQ column-block state used only by sparse MoE experts.

The copied dense :class:`realq_moe.quant.realq_layer.RealQLayer` deliberately
keeps its historical monolithic ``quantize`` implementation.  Qwen3-MoE needs
a different scheduling primitive: all experts of one projection must advance
the same column block before one joint Block-GD forward/backward is run.

This module exposes that primitive without changing the dense execution path.
The first implementation is intentionally restricted to the accepted target
runtime (single CUDA rank, ``num_groups > 1``,
``group_parallel_quant="rank"``).  It reproduces the corresponding branch of
``RealQLayer.quantize`` block for block, but yields the stitched candidate
weight at every non-final boundary and waits for the caller to apply the
jointly-computed Adam update.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from realq.quant.triton_column_block import quantize_column_block
from realq_moe.quant.hessian import cholesky_inverse_batched_with_damp
from realq_moe.quant.realq_layer import RealQLayer
from realq_moe.utils import nvtx
from utils import dist_utils
from utils.quant_utils import (
    WeightQuantizer,
    _select_symmetric_union_scale,
    sym_quant_dequant,
)


def _joint_quantizer_signature(
    quantizer: WeightQuantizer,
) -> tuple[object, ...]:
    return (
        quantizer.bits,
        quantizer.perchannel,
        quantizer.sym,
        quantizer.mse,
        quantizer.norm,
        quantizer.grid,
        quantizer.maxshrink,
        quantizer.weight_groupsize,
        quantizer.w_clip_search_impl,
        quantizer.w_clip_update_impl,
        getattr(quantizer, "w_group_param_layout", "expanded"),
    )


def _validate_batched_groupwise_quantizers(
    quantizers: list[WeightQuantizer],
    weights: torch.Tensor,
    *,
    expected_mse: bool | None = None,
) -> str:
    """Validate the frozen MoE observer domain and return its lifecycle."""

    if not quantizers:
        raise ValueError("batched groupwise search requires quantizers.")
    if weights.dim() != 3:
        raise ValueError(
            "batched groupwise search expects weights shaped "
            f"(experts, rows, columns), got {tuple(weights.shape)}."
        )
    expert_count, rows, columns = map(int, weights.shape)
    if expert_count != len(quantizers):
        raise ValueError(
            f"weights contain {expert_count} experts but received "
            f"{len(quantizers)} quantizers."
        )
    if rows <= 0 or columns <= 0:
        raise ValueError(
            f"batched groupwise weights must be nonempty, got "
            f"{tuple(weights.shape)}."
        )
    if not weights.is_cuda or weights.dtype != torch.float32:
        raise ValueError(
            "the MoE batched observer requires CUDA float32 weights, got "
            f"device={weights.device}, dtype={weights.dtype}."
        )

    for expert_idx, quantizer in enumerate(quantizers):
        if not isinstance(quantizer, WeightQuantizer):
            raise TypeError(
                f"expert {expert_idx} quantizer must be WeightQuantizer, got "
                f"{type(quantizer).__name__}."
            )
    template = quantizers[0]
    signature = _joint_quantizer_signature(template)
    for expert_idx, quantizer in enumerate(quantizers):
        if _joint_quantizer_signature(quantizer) != signature:
            raise ValueError(
                "expert-batched parameter search requires identical "
                f"quantizer settings; expert {expert_idx} differs from "
                "expert 0."
            )
        if quantizer.bits >= 16:
            raise ValueError("W16 must bypass batched parameter search.")
        if not quantizer.sym:
            raise NotImplementedError(
                "MoE batched parameter search supports symmetric weights "
                "only."
            )
        if quantizer.weight_groupsize <= 0:
            raise NotImplementedError(
                "MoE batched parameter search requires weight_groupsize > 0."
            )
        if quantizer.w_clip_search_impl not in (
            "cartesian_legacy",
            "symmetric_union_exact",
        ):
            raise NotImplementedError(
                "MoE batched parameter search supports Cartesian legacy or "
                "the exact symmetric endpoint union."
            )
        if quantizer.w_clip_update_impl not in ("guarded", "where_out"):
            raise NotImplementedError(
                "MoE batched parameter search supports guarded or where_out "
                "winner updates."
            )
        if (
            getattr(quantizer, "w_group_param_layout", "expanded")
            not in ("expanded", "compact")
        ):
            raise NotImplementedError(
                "MoE batched parameter search supports expanded or compact "
                "group parameters."
            )
        if expected_mse is not None and bool(quantizer.mse) != expected_mse:
            raise ValueError(
                f"expert {expert_idx} quantizer mse={quantizer.mse!r} does "
                f"not match w_clip={expected_mse}."
            )

    expected_ready_columns = (
        (columns + template.weight_groupsize - 1)
        // template.weight_groupsize
        if template.w_group_param_layout == "compact"
        else columns
    )
    expected_ready_shape = (rows, expected_ready_columns)
    lifecycle = []
    for expert_idx, quantizer in enumerate(quantizers):
        scale_shape = tuple(quantizer.scale.shape)
        zero_shape = tuple(quantizer.zero.shape)
        if scale_shape == (1,) and zero_shape == (1,):
            lifecycle.append("fresh")
        elif (
            scale_shape == expected_ready_shape
            and zero_shape == expected_ready_shape
        ):
            lifecycle.append("ready")
        else:
            raise RuntimeError(
                f"expert {expert_idx} grouped quantizer has scale/zero shapes "
                f"{scale_shape}/{zero_shape}; expected fresh (1,)/(1,) or "
                f"ready {expected_ready_shape}/{expected_ready_shape}."
            )
    if len(set(lifecycle)) != 1:
        raise RuntimeError(
            "expert-batched parameter search does not accept a mixture of "
            "fresh and ready quantizers."
        )
    return lifecycle[0]


@torch.no_grad()
def _find_groupwise_params_batched_(
    quantizers: list[WeightQuantizer],
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run grouped parameter search across every expert in one batch.

    This is the MoE-only equivalent of calling
    ``WeightQuantizer.find_params`` once per expert.  The expert axis is a
    pure batch axis: extrema/error reductions stay on the original final
    group-width dimension, and candidate pairs retain their legacy
    lexicographic order and strict-``<`` winner rule.
    """

    lifecycle = _validate_batched_groupwise_quantizers(
        quantizers,
        weights,
    )
    if lifecycle != "fresh":
        raise RuntimeError(
            "_find_groupwise_params_batched_ requires fresh quantizers."
        )

    expert_count, rows, columns = map(int, weights.shape)
    template = quantizers[0]
    group_width = int(template.weight_groupsize)
    shared_maxq = template.maxq.to(device=weights.device)
    maxq = shared_maxq.reshape(1, 1, 1, 1)
    for quantizer in quantizers:
        quantizer.maxq = shared_maxq
    use_symmetric_union = (
        template.w_clip_search_impl == "symmetric_union_exact"
        and bool(torch.isfinite(weights).all())
    )

    def params_for_equal_width_groups(
        grouped_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xmax = torch.amax(grouped_weights, dim=-1, keepdim=True)
        xmin = torch.amin(grouped_weights, dim=-1, keepdim=True)
        scale = (
            torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            / maxq
        )
        zero = torch.zeros_like(scale)

        if (
            template.mse
            and use_symmetric_union
        ):
            scale = _select_symmetric_union_scale(
                grouped_weights,
                xmin.squeeze(-1),
                xmax.squeeze(-1),
                maxq=shared_maxq,
                norm=template.norm,
                grid=template.grid,
                candidate_count=int(
                    template.maxshrink * template.grid
                ),
                grouped_error_order=True,
            ).unsqueeze(-1)
            zero = torch.zeros_like(scale)
        elif template.mse:
            best = torch.full(
                grouped_weights.shape[:3],
                float("inf"),
                device=grouped_weights.device,
                dtype=grouped_weights.dtype,
            )
            candidate_count = int(
                template.maxshrink * template.grid
            )
            for i in range(candidate_count):
                for j in range(candidate_count):
                    xmin1 = (1 - i / template.grid) * xmin
                    xmax1 = (1 - j / template.grid) * xmax
                    scale1 = (
                        torch.maximum(torch.abs(xmin1), xmax1)
                        .clamp(min=1e-5)
                        / maxq
                    )
                    quantized = sym_quant_dequant(
                        grouped_weights,
                        scale1,
                        maxq,
                    )
                    err = (
                        (quantized - grouped_weights)
                        .abs()
                        .pow(template.norm)
                        .sum(-1)
                    )
                    improved = err < best
                    # Preserve the frozen strict-<, lexicographic winner
                    # semantics without converting a CUDA predicate to a
                    # Python bool for every Cartesian candidate.  The old
                    # branch imposed 625 device-to-host synchronisations per
                    # equal-width search on the accepted setting.
                    best.copy_(torch.where(improved, err, best))
                    scale.copy_(
                        torch.where(
                            improved.unsqueeze(-1),
                            scale1,
                            scale,
                        )
                    )
                    # Symmetric quantisation fixes zero at zero, so no
                    # candidate-dependent zero update is required.

        if template.w_group_param_layout == "compact":
            return scale.squeeze(-1), zero.squeeze(-1)

        width = int(grouped_weights.shape[-1])
        return (
            scale.expand(-1, -1, -1, width).reshape(
                expert_count, rows, -1
            ),
            zero.expand(-1, -1, -1, width).reshape(
                expert_count, rows, -1
            ),
        )

    full_columns = (columns // group_width) * group_width
    scale_parts: list[torch.Tensor] = []
    zero_parts: list[torch.Tensor] = []
    if full_columns:
        grouped = weights[:, :, :full_columns].reshape(
            expert_count,
            rows,
            full_columns // group_width,
            group_width,
        )
        scale, zero = params_for_equal_width_groups(grouped)
        scale_parts.append(scale)
        zero_parts.append(zero)
    if full_columns < columns:
        tail = weights[:, :, full_columns:].unsqueeze(2)
        scale, zero = params_for_equal_width_groups(tail)
        scale_parts.append(scale)
        zero_parts.append(zero)

    batched_scale = torch.cat(scale_parts, dim=2)
    batched_zero = torch.cat(zero_parts, dim=2)
    for expert_idx, quantizer in enumerate(quantizers):
        quantizer.scale = batched_scale[expert_idx]
        quantizer.zero = batched_zero[expert_idx]
        quantizer.weight_ncolumns = columns
    return batched_scale, batched_zero


@dataclass(frozen=True)
class JointBlockCandidate:
    """One expert's weight at a synchronized GPTQ block boundary.

    ``weight_fp32`` is always in the linear's natural column order.  The
    refresh implementation computes a gradient in that coordinate system,
    rekeys it with ``perm`` when act-order is enabled, and returns an update
    for the permuted trailing slice beginning at ``trailing_col_start``.
    """

    weight_fp32: torch.Tensor
    trailing_col_start: int
    perm: torch.Tensor | None
    block_index: int


class JointColumnQuantState:
    """Incremental, all-GPU GPTQ state for one expert projection."""

    def __init__(
        self,
        realq: RealQLayer,
        *,
        blocksize: int,
        percdamp: float,
        act_order: bool,
        w_clip: bool,
    ) -> None:
        if dist_utils.get_world_size() != 1:
            raise NotImplementedError(
                "JointColumnQuantState currently implements the accepted "
                "single-GPU MoE path only."
            )
        if realq.num_groups <= 1:
            raise NotImplementedError(
                "JointColumnQuantState currently requires num_groups > 1."
            )
        if not realq._finalized:
            raise RuntimeError("Call finalize_hessian() before joint prepare.")
        if realq.quantizer.bits >= 16:
            raise ValueError(
                "W16 is a no-op and must not create a joint quantization state."
            )
        if w_clip and not realq.quantizer.mse:
            raise ValueError(
                "w_clip=True requires a quantizer configured with mse=True."
            )
        if blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {blocksize}.")

        self.realq = realq
        self.linear = realq.linear
        self.quantizer = realq.quantizer
        self.rows = int(realq.rows)
        self.columns = int(realq.columns)
        self.num_groups = int(realq.num_groups)
        self.rows_per_group = int(realq.rows_per_group)
        self.blocksize = int(blocksize)
        self.block_index = 0
        self.next_col_start = 0
        self._finished = False

        self.dynamic_weight_groups = (
            self.quantizer.weight_groupsize > 0 and not act_order
        )
        if (
            self.dynamic_weight_groups
            and self.quantizer.weight_groupsize != self.blocksize
        ):
            raise ValueError(
                "Dynamic weight groups require weight_groupsize == blocksize: "
                f"{self.quantizer.weight_groupsize} != {self.blocksize}."
            )

        with nvtx.nvtx_range("moe_joint.prepare.weight"):
            self.W = self.linear.weight.data.clone().float()

        # Match the target RealQLayer branch: find static grouped parameters in
        # natural-column order before act-order permutation.
        with nvtx.nvtx_range("moe_joint.prepare.find_params"):
            if (
                not self.quantizer.ready()
                and not self.dynamic_weight_groups
            ):
                self.quantizer.find_params(self.W)

        with nvtx.nvtx_range("moe_joint.prepare.hessian"):
            hessian = self.realq.H.clone()
            dead = torch.diagonal(
                hessian, dim1=-2, dim2=-1
            ) == 0
            torch.diagonal(
                hessian, dim1=-2, dim2=-1
            )[dead] = 1
            # Single-rank target owns every output-row Hessian group.
            for group_id in range(self.num_groups):
                row_start = group_id * self.rows_per_group
                row_end = row_start + self.rows_per_group
                self.W[row_start:row_end, dead[group_id]] = 0

            self.perm = self.invperm = None
            if act_order:
                self.perm = torch.argsort(
                    self.realq.act_square, descending=True
                )
                self.W = self.W[:, self.perm]
                hessian = hessian[:, self.perm][:, :, self.perm]
                self.invperm = torch.argsort(self.perm)

            # ``group_parallel_quant="rank"`` uses the batched routine even at
            # world size one; retaining it here is required for the no-refresh
            # bit-equality oracle against the monolithic implementation.
            self.Hinv = cholesky_inverse_batched_with_damp(
                hessian, percdamp=percdamp
            )

        self.Q = torch.zeros_like(self.W)

    @property
    def finished(self) -> bool:
        return self._finished

    def _make_dynamic_group_quantizer(
        self, weight_block: torch.Tensor
    ):
        block_quantizer = type(self.quantizer)()
        block_quantizer.configure(
            bits=self.quantizer.bits,
            perchannel=self.quantizer.perchannel,
            sym=self.quantizer.sym,
            mse=self.quantizer.mse,
            norm=self.quantizer.norm,
            grid=self.quantizer.grid,
            maxshrink=self.quantizer.maxshrink,
            weight_groupsize=-1,
            w_clip_search_impl=self.quantizer.w_clip_search_impl,
            w_clip_update_impl=self.quantizer.w_clip_update_impl,
            w_group_param_layout=getattr(
                self.quantizer, "w_group_param_layout", "expanded"
            ),
        )
        block_quantizer.find_params(weight_block)
        return block_quantizer

    @torch.no_grad()
    def advance_block(self) -> JointBlockCandidate | None:
        """Quantize one column block and stop before the joint refresh.

        Returns ``None`` for the final block because there is no trailing
        suffix and REAL-Q therefore performs no Block-GD step at that boundary.
        """

        if self._finished:
            raise RuntimeError("advance_block called after final block.")

        i1 = self.next_col_start
        i2 = min(i1 + self.blocksize, self.columns)
        count = i2 - i1
        if count <= 0:
            raise RuntimeError(
                f"invalid joint block [{i1}, {i2}) for C={self.columns}."
            )

        with nvtx.nvtx_range(
            f"moe_joint.block_{self.block_index}.analytic"
        ):
            # This is the world=1, NUM_GROUPS>1 rank branch from
            # RealQLayer.quantize, with no inherited fastpath enabled.
            W1 = self.W[:, i1:i2].clone()
            block_quantizer = (
                self._make_dynamic_group_quantizer(W1)
                if self.dynamic_weight_groups
                else self.quantizer
            )
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            W1_g = W1.view(self.num_groups, self.rows_per_group, count)
            Err1_g = Err1.view(
                self.num_groups, self.rows_per_group, count
            )
            Hinv1_g = self.Hinv[:, i1:i2, i1:i2]

            for i in range(count):
                w_col = W1[:, i].unsqueeze(1)
                q_fake, _, _ = block_quantizer.fake_quantize(
                    w_col,
                    col_idx=(
                        self.perm[i1 + i]
                        if self.perm is not None
                        else i1 + i
                    ),
                )
                q_col = q_fake.flatten()
                Q1[:, i] = q_col
                d_per_row = Hinv1_g[:, i, i].repeat_interleave(
                    self.rows_per_group
                )
                err = (W1[:, i] - q_col) / d_per_row
                Err1[:, i] = err
                W1_g[:, :, i:].sub_(
                    err.view(
                        self.num_groups, self.rows_per_group
                    ).unsqueeze(-1)
                    * Hinv1_g[:, i, i:].unsqueeze(1)
                )

            self.Q[:, i1:i2] = Q1
            if i2 < self.columns:
                self.W.view(
                    self.num_groups,
                    self.rows_per_group,
                    self.columns,
                )[:, :, i2:].sub_(
                    torch.bmm(
                        Err1_g, self.Hinv[:, i1:i2, i2:]
                    )
                )

        completed_block = self.block_index
        self.next_col_start = i2
        self.block_index += 1
        if i2 == self.columns:
            self._finished = True
            return None

        with nvtx.nvtx_range(
            f"moe_joint.block_{completed_block}.stitch"
        ):
            if self.invperm is None:
                stitched = self.W.clone()
                stitched[:, :i2] = self.Q[:, :i2]
            else:
                q_natural = self.Q[:, self.invperm]
                w_natural = self.W[:, self.invperm]
                quantized_natural_cols = self.perm[:i2]
                stitched = w_natural.clone()
                stitched[:, quantized_natural_cols] = q_natural[
                    :, quantized_natural_cols
                ]
        return JointBlockCandidate(
            weight_fp32=stitched,
            trailing_col_start=i2,
            perm=self.perm,
            block_index=completed_block,
        )

    @torch.no_grad()
    def apply_update(self, update: torch.Tensor | None) -> None:
        """Apply one expert's joint Adam delta to its unquantized suffix."""

        if update is None:
            return
        trailing = self.columns - self.next_col_start
        expected = (self.rows, trailing)
        if tuple(update.shape) != expected:
            raise ValueError(
                "joint update shape must match the current trailing suffix: "
                f"{tuple(update.shape)} != {expected}."
            )
        self.W[:, self.next_col_start:].sub_(
            update.to(device=self.W.device, dtype=self.W.dtype)
        )

    @torch.no_grad()
    def writeback(self) -> None:
        """Commit the fully quantized weight to the expert module."""

        if not self._finished:
            raise RuntimeError("writeback requires every column block.")
        q_natural = (
            self.Q
            if self.invperm is None
            else self.Q[:, self.invperm]
        )
        self.linear.weight.data.copy_(
            q_natural.to(self.linear.weight.dtype)
        )

    def free(self) -> None:
        self.Hinv = None
        self.W = None
        self.Q = None


@dataclass(frozen=True)
class JointProjectionBoundary:
    """Batched boundary for every expert in one projection."""

    weights_fp32: torch.Tensor  # (E, rows, columns), natural order
    trailing_col_start: int
    perms: torch.Tensor  # (E, columns), permuted -> natural
    block_index: int


class JointMoeProjectionStepper:
    """Expert-batched analytic GPTQ for one Qwen3-MoE projection.

    Unlike :class:`JointColumnQuantState`, which is the bit-alignment oracle,
    this speed path batches Hessian factorization, inner-column arithmetic and
    outer compensation over the expert dimension.  It is intentionally
    MoE-only and leaves the copied dense implementation untouched.  The
    constructor consumes each source ``RealQLayer``'s projection-local
    Hessian/activation statistics after copying the batched state; callers
    must not attempt to reuse those accumulators.
    """

    @torch.no_grad()
    def __init__(
        self,
        realqs: list[RealQLayer],
        *,
        blocksize: int,
        percdamp: float,
        act_order: bool,
        w_clip: bool,
        triton_column_block: bool = False,
    ) -> None:
        if not realqs:
            raise ValueError("joint projection requires at least one expert.")
        if type(triton_column_block) is not bool:
            raise ValueError("triton_column_block must be bool.")
        if dist_utils.get_world_size() != 1:
            raise NotImplementedError(
                "JointMoeProjectionStepper currently requires world_size=1."
            )
        if not act_order:
            raise NotImplementedError(
                "The first expert-batched path implements the frozen "
                "act_order=True setting only; use JointColumnQuantState as "
                "the no-act-order oracle."
            )
        template = realqs[0]
        signature = (
            int(template.rows),
            int(template.columns),
            int(template.num_groups),
        )
        if signature[2] <= 1:
            raise NotImplementedError(
                "expert-batched path currently requires num_groups > 1."
            )
        for expert_idx, realq in enumerate(realqs):
            current = (
                int(realq.rows),
                int(realq.columns),
                int(realq.num_groups),
            )
            if current != signature:
                raise ValueError(
                    f"expert {expert_idx} projection shape/groups {current} "
                    f"!= {signature}."
                )
            if not realq._finalized:
                raise RuntimeError(
                    f"expert {expert_idx} Hessian is not finalized."
                )
            if realq.H.device != template.H.device:
                raise ValueError(
                    "all expert Hessians must already share one GPU."
                )
            if realq.quantizer.bits >= 16:
                raise ValueError("W16 must bypass the joint stepper.")
            if not realq.quantizer.sym:
                raise NotImplementedError(
                    "The accepted Qwen3-MoE weight setting is symmetric."
                )
            if w_clip and not realq.quantizer.mse:
                raise ValueError(
                    "w_clip=True requires every expert quantizer to use MSE."
                )
            if realq.quantizer.weight_groupsize <= 0:
                raise NotImplementedError(
                    "The first expert-batched path implements the frozen "
                    "grouped-weight setting only."
                )

        self.linears = [realq.linear for realq in realqs]
        self.expert_count = len(realqs)
        self.rtn_fallback_mask = torch.tensor(
            [
                bool(getattr(realq, "rtn_fallback", False))
                for realq in realqs
            ],
            dtype=torch.bool,
            device=template.H.device,
        )
        self.rows, self.columns, self.num_groups = signature
        self.rows_per_group = self.rows // self.num_groups
        self.blocksize = int(blocksize)
        self.triton_column_block = triton_column_block
        self.block_index = 0
        self.next_col_start = 0
        self._finished = False
        if self.blocksize <= 0:
            raise ValueError("blocksize must be positive.")

        with nvtx.nvtx_range("moe_joint_batched.prepare.weights"):
            natural_w = torch.stack(
                [
                    linear.weight.data.clone().float()
                    for linear in self.linears
                ],
                dim=0,
            )

        # Preserve the inherited Cartesian observer arithmetic, but execute
        # every expert lane in one batched search instead of issuing the same
        # candidate kernel sequence E times.
        with nvtx.nvtx_range("moe_joint_batched.prepare.find_params"):
            quantizers = [realq.quantizer for realq in realqs]
            self.weight_groupsize = int(
                quantizers[0].weight_groupsize
            )
            self.w_group_param_layout = getattr(
                quantizers[0], "w_group_param_layout", "expanded"
            )
            lifecycle = _validate_batched_groupwise_quantizers(
                quantizers,
                natural_w,
                expected_mse=bool(w_clip),
            )
            if lifecycle == "fresh":
                self.scales, _ = _find_groupwise_params_batched_(
                    quantizers,
                    natural_w,
                )
            else:
                self.scales = torch.stack(
                    [quantizer.scale for quantizer in quantizers],
                    dim=0,
                )
            self.rtn_weights = torch.zeros_like(natural_w)
            for expert_idx, is_fallback in enumerate(
                self.rtn_fallback_mask.tolist()
            ):
                if is_fallback:
                    # Materialize the accepted fallback through the actual
                    # grouped WeightQuantizer RTN implementation.  Identity
                    # Hessians below are placeholders that keep the hot and
                    # cold lanes shape-uniform in the joint kernels; writeback
                    # takes this direct RTN result for every cold lane.
                    self.rtn_weights[expert_idx].copy_(
                        quantizers[expert_idx].quantize(
                            natural_w[expert_idx]
                        )
                    )
            self.maxq = quantizers[0].maxq.to(
                device=natural_w.device,
                dtype=natural_w.dtype,
            ).reshape(1, 1).expand(
                self.expert_count, 1
            )

        with nvtx.nvtx_range("moe_joint_batched.prepare.hinv"):
            hessian = torch.stack(
                [realq.H for realq in realqs], dim=0
            )
            dead = torch.diagonal(
                hessian, dim1=-2, dim2=-1
            ) == 0
            torch.diagonal(
                hessian, dim1=-2, dim2=-1
            )[dead] = 1
            natural_w.view(
                self.expert_count,
                self.num_groups,
                self.rows_per_group,
                self.columns,
            ).masked_fill_(dead.unsqueeze(2), 0)

            act_square = torch.stack(
                [realq.act_square for realq in realqs], dim=0
            )
            self.perms = torch.argsort(
                act_square, dim=1, descending=True
            )
            self.invperms = torch.argsort(self.perms, dim=1)

            # From this point onward the joint stepper owns every statistic it
            # needs: the Hessians have been stacked into ``hessian`` and the
            # activation ordering into ``perms``.  Keeping the 128 source
            # RoutedRealQLayer Hessians alive would retain another 8 GiB for
            # Qwen3-30B-A3B up/gate while the batched Hinv and fixed-route
            # refresh cache are built.  Release only these projection-local
            # statistics; model weights and the Stage-0 GPU-resident stats
            # remain untouched.
            for realq in realqs:
                realq.free()
            del act_square

            w_index = self.perms.unsqueeze(1).expand(
                self.expert_count, self.rows, self.columns
            )
            self.W = torch.gather(natural_w, 2, w_index)

            row_index = self.perms[:, None, :, None].expand(
                self.expert_count,
                self.num_groups,
                self.columns,
                self.columns,
            )
            hessian = torch.gather(hessian, 2, row_index)
            col_index = self.perms[:, None, None, :].expand(
                self.expert_count,
                self.num_groups,
                self.columns,
                self.columns,
            )
            hessian = torch.gather(hessian, 3, col_index)
            self.Hinv = cholesky_inverse_batched_with_damp(
                hessian.reshape(
                    self.expert_count * self.num_groups,
                    self.columns,
                    self.columns,
                ),
                percdamp=percdamp,
            ).view(
                self.expert_count,
                self.num_groups,
                self.columns,
                self.columns,
            )

        self.Q = torch.zeros_like(self.W)

    @property
    def finished(self) -> bool:
        return self._finished

    def _scale_for_permuted_column(
        self, permuted_col: int
    ) -> torch.Tensor:
        if self.scales.shape[-1] == 1:
            return self.scales[:, :, 0]
        natural_cols = self.perms[:, permuted_col]
        if self.w_group_param_layout == "compact":
            natural_cols = torch.div(
                natural_cols,
                self.weight_groupsize,
                rounding_mode="floor",
            )
        return torch.gather(
            self.scales,
            2,
            natural_cols[:, None, None].expand(
                self.expert_count, self.rows, 1
            ),
        ).squeeze(2)

    def _scales_for_permuted_block(
        self, start: int, end: int
    ) -> torch.Tensor:
        natural_cols = self.perms[:, start:end]
        if self.w_group_param_layout == "compact":
            natural_cols = torch.div(
                natural_cols,
                self.weight_groupsize,
                rounding_mode="floor",
            )
        return torch.gather(
            self.scales,
            2,
            natural_cols[:, None, :].expand(
                self.expert_count, self.rows, end - start
            ),
        )

    @torch.no_grad()
    def advance_block(self) -> JointProjectionBoundary | None:
        if self._finished:
            raise RuntimeError("advance_block called after final block.")
        i1 = self.next_col_start
        i2 = min(i1 + self.blocksize, self.columns)
        count = i2 - i1

        with nvtx.nvtx_range(
            f"moe_joint_batched.block_{self.block_index}.analytic"
        ):
            W1 = self.W[:, :, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            W1_g = W1.view(
                self.expert_count,
                self.num_groups,
                self.rows_per_group,
                count,
            )
            Err1_g = Err1.view(
                self.expert_count,
                self.num_groups,
                self.rows_per_group,
                count,
            )
            Hinv1 = self.Hinv[:, :, i1:i2, i1:i2]

            if self.triton_column_block:
                q_flat, err_flat = quantize_column_block(
                    W1.reshape(self.expert_count * self.rows, count),
                    self._scales_for_permuted_block(
                        i1, i2
                    ).reshape(
                        self.expert_count * self.rows, count
                    ).contiguous(),
                    Hinv1.reshape(
                        self.expert_count * self.num_groups,
                        count,
                        count,
                    ),
                    self.maxq,
                    rows_per_group=self.rows_per_group,
                )
                Q1 = q_flat.view(
                    self.expert_count, self.rows, count
                )
                Err1 = err_flat.view(
                    self.expert_count, self.rows, count
                )
                Err1_g = Err1.view(
                    self.expert_count,
                    self.num_groups,
                    self.rows_per_group,
                    count,
                )
            else:
                # Reference path remains sequential along the column axis.
                for i in range(count):
                    w = W1[:, :, i]
                    scale = self._scale_for_permuted_column(i1 + i)
                    q_int = torch.clamp(
                        torch.round(w / scale),
                        min=-(self.maxq + 1),
                        max=self.maxq,
                    )
                    q = scale * q_int
                    Q1[:, :, i] = q
                    d = Hinv1[:, :, i, i]
                    d_per_row = d.unsqueeze(2).expand(
                        self.expert_count,
                        self.num_groups,
                        self.rows_per_group,
                    ).reshape(self.expert_count, self.rows)
                    err = (w - q) / d_per_row
                    Err1[:, :, i] = err
                    W1_g[:, :, :, i:].sub_(
                        err.view(
                            self.expert_count,
                            self.num_groups,
                            self.rows_per_group,
                            1,
                        )
                        * Hinv1[:, :, i, i:].unsqueeze(2)
                    )

            self.Q[:, :, i1:i2] = Q1
            if i2 < self.columns:
                lhs = Err1_g.reshape(
                    self.expert_count * self.num_groups,
                    self.rows_per_group,
                    count,
                )
                rhs = self.Hinv[:, :, i1:i2, i2:].reshape(
                    self.expert_count * self.num_groups,
                    count,
                    self.columns - i2,
                )
                self.W.view(
                    self.expert_count,
                    self.num_groups,
                    self.rows_per_group,
                    self.columns,
                )[:, :, :, i2:].sub_(
                    torch.bmm(lhs, rhs).view(
                        self.expert_count,
                        self.num_groups,
                        self.rows_per_group,
                        self.columns - i2,
                    )
                )

        completed_block = self.block_index
        self.next_col_start = i2
        self.block_index += 1
        if i2 == self.columns:
            self._finished = True
            return None

        with nvtx.nvtx_range(
            f"moe_joint_batched.block_{completed_block}.stitch"
        ):
            natural_index = self.invperms.unsqueeze(1).expand(
                self.expert_count, self.rows, self.columns
            )
            stitched = torch.gather(self.W, 2, natural_index)
            quantized_natural_cols = self.perms[:, :i2]
            scatter_index = quantized_natural_cols.unsqueeze(1).expand(
                self.expert_count, self.rows, i2
            )
            stitched.scatter_(
                2, scatter_index, self.Q[:, :, :i2]
            )
        return JointProjectionBoundary(
            weights_fp32=stitched,
            trailing_col_start=i2,
            perms=self.perms,
            block_index=completed_block,
        )

    @torch.no_grad()
    def apply_updates(
        self, updates: torch.Tensor | tuple[torch.Tensor | None, ...]
    ) -> None:
        trailing = self.columns - self.next_col_start
        if isinstance(updates, tuple):
            if len(updates) != self.expert_count:
                raise ValueError(
                    f"expected {self.expert_count} updates, got "
                    f"{len(updates)}."
                )
            normalized = []
            for update in updates:
                normalized.append(
                    torch.zeros(
                        (self.rows, trailing),
                        device=self.W.device,
                        dtype=self.W.dtype,
                    )
                    if update is None
                    else update.to(
                        device=self.W.device, dtype=self.W.dtype
                    )
                )
            updates = torch.stack(normalized, dim=0)
        expected = (self.expert_count, self.rows, trailing)
        if tuple(updates.shape) != expected:
            raise ValueError(
                f"joint update shape {tuple(updates.shape)} != {expected}."
            )
        if bool(self.rtn_fallback_mask.any().item()):
            # Zero-route experts are defined to be pure RTN.  They still
            # share the batched analytic kernels, but no BlockGD correction
            # may modify their trailing columns.
            updates = updates.masked_fill(
                self.rtn_fallback_mask[:, None, None], 0
            )
        self.W[:, :, self.next_col_start:].sub_(updates)

    @torch.no_grad()
    def writeback(self) -> None:
        if not self._finished:
            raise RuntimeError("writeback requires every block.")
        natural_index = self.invperms.unsqueeze(1).expand(
            self.expert_count, self.rows, self.columns
        )
        q_natural = torch.gather(self.Q, 2, natural_index)
        if bool(self.rtn_fallback_mask.any().item()):
            q_natural[self.rtn_fallback_mask] = self.rtn_weights[
                self.rtn_fallback_mask
            ]
        for expert_idx, linear in enumerate(self.linears):
            linear.weight.data.copy_(
                q_natural[expert_idx].to(linear.weight.dtype)
            )

    def free(self) -> None:
        self.Hinv = None
        self.W = None
        self.Q = None
        self.scales = None
        self.rtn_fallback_mask = None
        self.rtn_weights = None
