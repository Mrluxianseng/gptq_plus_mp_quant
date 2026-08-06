"""GPU-only batched Adam state for joint MoE projection refreshes.

The joint MoE refresh computes one gradient tensor with shape ``(E, R, C)``.
Keeping 128 independent :class:`RefreshContext` moment tensors would turn the
otherwise batched update back into 128 small gather/clip/Adam kernel streams.
This helper packs those moments once and performs the legacy trailing-suffix
update over the expert dimension in one set of tensor operations.

This is deliberately an MoE-only primitive.  It neither changes nor is used by
the inherited dense Block-GD implementation.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from realq_moe.refresh.block_gd import RefreshContext


def _context_signature(ctx: "RefreshContext") -> tuple[float, ...]:
    return (
        float(ctx.beta1),
        float(ctx.beta2),
        float(ctx.eps),
        float(ctx.layer_lr),
        float(ctx.grad_clip),
    )


class JointBatchedAdamState:
    """Contiguous Adam moments shared by one synchronized expert projection.

    ``RefreshContext.adam_step`` remains the source of truth for the optimizer
    step.  The surrounding refresh closure advances every context once before
    calling :meth:`step`, exactly as the existing joint implementation does.
    This helper implements only the old per-expert Adam body.
    """

    @torch.no_grad()
    def __init__(self, contexts: Sequence["RefreshContext"]) -> None:
        contexts = tuple(contexts)
        if not contexts:
            raise ValueError(
                "JointBatchedAdamState requires at least one RefreshContext."
            )

        first = contexts[0]
        first_shape = tuple(first.exp_avg.shape)
        first_device = first.exp_avg.device
        first_signature = _context_signature(first)
        first_step = int(first.adam_step)
        if first_device.type != "cuda":
            raise ValueError(
                "joint batched Adam moments must remain CUDA resident; "
                f"got {first_device}."
            )
        if first.exp_avg.dtype != torch.float32:
            raise ValueError(
                "joint batched Adam exp_avg must be FP32; "
                f"got {first.exp_avg.dtype}."
            )
        if first.exp_avg_sq.dtype != torch.float32:
            raise ValueError(
                "joint batched Adam exp_avg_sq must be FP32; "
                f"got {first.exp_avg_sq.dtype}."
            )
        if first.exp_avg_sq.device != first_device:
            raise ValueError(
                "joint batched Adam moments must share one CUDA device."
            )
        if tuple(first.exp_avg_sq.shape) != first_shape:
            raise ValueError(
                "joint batched Adam exp_avg/exp_avg_sq shapes differ for "
                f"expert 0: {first_shape} != "
                f"{tuple(first.exp_avg_sq.shape)}."
            )
        if len(first_shape) != 2:
            raise ValueError(
                "joint batched Adam expects 2-D linear-weight moments; "
                f"got {first_shape}."
            )

        for expert_idx, ctx in enumerate(contexts):
            if tuple(ctx.exp_avg.shape) != first_shape:
                raise ValueError(
                    f"context {expert_idx} exp_avg shape "
                    f"{tuple(ctx.exp_avg.shape)} != {first_shape}."
                )
            if tuple(ctx.exp_avg_sq.shape) != first_shape:
                raise ValueError(
                    f"context {expert_idx} exp_avg_sq shape "
                    f"{tuple(ctx.exp_avg_sq.shape)} != {first_shape}."
                )
            if (
                ctx.exp_avg.device != first_device
                or ctx.exp_avg_sq.device != first_device
            ):
                raise ValueError(
                    "all joint batched Adam moments must share CUDA device "
                    f"{first_device}; context {expert_idx} uses "
                    f"{ctx.exp_avg.device}/{ctx.exp_avg_sq.device}."
                )
            if (
                ctx.exp_avg.dtype != torch.float32
                or ctx.exp_avg_sq.dtype != torch.float32
            ):
                raise ValueError(
                    "all joint batched Adam moments must be FP32; context "
                    f"{expert_idx} uses "
                    f"{ctx.exp_avg.dtype}/{ctx.exp_avg_sq.dtype}."
                )
            if _context_signature(ctx) != first_signature:
                raise ValueError(
                    "joint batched Adam contexts must share beta1, beta2, "
                    "eps, layer_lr and grad_clip; context "
                    f"{expert_idx} differs."
                )
            if int(ctx.adam_step) != first_step:
                raise ValueError(
                    "joint batched Adam context steps must be synchronized; "
                    f"context {expert_idx} has {ctx.adam_step}, expected "
                    f"{first_step}."
                )

        self.contexts = contexts
        self.expert_count = len(contexts)
        self.rows, self.columns = first_shape
        self.device = first_device
        self.beta1, self.beta2, self.eps, self.layer_lr, self.grad_clip = (
            first_signature
        )

        # Stack once, then rebind every context to a zero-copy expert view.
        # Existing diagnostics and callers therefore continue to observe the
        # live optimizer state instead of stale pre-pack tensors.
        self.exp_avg = torch.stack(
            [ctx.exp_avg for ctx in contexts],
            dim=0,
        ).contiguous()
        self.exp_avg_sq = torch.stack(
            [ctx.exp_avg_sq for ctx in contexts],
            dim=0,
        ).contiguous()
        self._exp_avg_views = self.exp_avg.unbind(dim=0)
        self._exp_avg_sq_views = self.exp_avg_sq.unbind(dim=0)
        for ctx, exp_avg, exp_avg_sq in zip(
            contexts,
            self._exp_avg_views,
            self._exp_avg_sq_views,
        ):
            ctx.exp_avg = exp_avg
            ctx.exp_avg_sq = exp_avg_sq

    def _validated_adam_step(self) -> int:
        steps = {int(ctx.adam_step) for ctx in self.contexts}
        if len(steps) != 1:
            raise RuntimeError(
                "joint batched Adam context steps diverged: "
                f"{sorted(steps)}."
            )
        adam_step = next(iter(steps))
        if adam_step <= 0:
            raise RuntimeError(
                "joint batched Adam requires contexts to advance adam_step "
                "before step()."
            )
        for expert_idx, ctx in enumerate(self.contexts):
            if _context_signature(ctx) != (
                self.beta1,
                self.beta2,
                self.eps,
                self.layer_lr,
                self.grad_clip,
            ):
                raise RuntimeError(
                    "joint batched Adam hyperparameters changed after packing; "
                    f"context {expert_idx} differs."
                )
            if (
                ctx.exp_avg is not self._exp_avg_views[expert_idx]
                or ctx.exp_avg_sq is not self._exp_avg_sq_views[expert_idx]
            ):
                raise RuntimeError(
                    "joint batched Adam context moment view was rebound after "
                    f"packing for expert {expert_idx}."
                )
        return adam_step

    @torch.no_grad()
    def step(
        self,
        accumulated_grad_batch: torch.Tensor,
        trailing_start: int,
        perms: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the legacy Adam update to every expert suffix at once.

        Args:
            accumulated_grad_batch: FP32 natural-column gradients ``(E,R,C)``.
            trailing_start: First still-unquantized column in permuted order.
            perms: Per-expert natural-column permutation ``(E,C)``, or ``None``.

        Returns:
            FP32 update tensor with shape ``(E,R,C-trailing_start)``.
        """

        expected_shape = (
            self.expert_count,
            self.rows,
            self.columns,
        )
        if not torch.is_tensor(accumulated_grad_batch):
            raise TypeError(
                "accumulated_grad_batch must be a Tensor, got "
                f"{type(accumulated_grad_batch).__name__}."
            )
        if tuple(accumulated_grad_batch.shape) != expected_shape:
            raise ValueError(
                "joint accumulated gradient shape "
                f"{tuple(accumulated_grad_batch.shape)} != {expected_shape}."
            )
        if accumulated_grad_batch.device != self.device:
            raise ValueError(
                "joint accumulated gradient must remain on "
                f"{self.device}, got {accumulated_grad_batch.device}."
            )
        if accumulated_grad_batch.dtype != torch.float32:
            raise ValueError(
                "joint accumulated gradient must be FP32, got "
                f"{accumulated_grad_batch.dtype}."
            )
        if (
            not isinstance(trailing_start, int)
            or isinstance(trailing_start, bool)
            or not 0 <= trailing_start <= self.columns
        ):
            raise ValueError(
                f"trailing_start must be in [0, {self.columns}], got "
                f"{trailing_start!r}."
            )

        if perms is not None:
            if not torch.is_tensor(perms):
                raise TypeError(
                    f"perms must be a Tensor or None, got {type(perms).__name__}."
                )
            expected_perms = (self.expert_count, self.columns)
            if tuple(perms.shape) != expected_perms:
                raise ValueError(
                    f"joint perms shape {tuple(perms.shape)} != "
                    f"{expected_perms}."
                )
            if perms.device != self.device:
                raise ValueError(
                    f"joint perms must remain on {self.device}, got "
                    f"{perms.device}."
                )
            if perms.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "joint perms must be int32 or int64, got "
                    f"{perms.dtype}."
                )
            gather_index = perms.to(dtype=torch.int64).unsqueeze(1).expand(
                self.expert_count,
                self.rows,
                self.columns,
            )
            accumulated_grad_batch = torch.gather(
                accumulated_grad_batch,
                2,
                gather_index,
            )

        adam_step = self._validated_adam_step()
        grad_slice = accumulated_grad_batch[:, :, trailing_start:]
        if self.grad_clip > 0:
            grad_slice = grad_slice.clamp(
                min=-self.grad_clip,
                max=self.grad_clip,
            )

        exp_avg = self.exp_avg[:, :, trailing_start:]
        exp_avg_sq = self.exp_avg_sq[:, :, trailing_start:]
        exp_avg.mul_(self.beta1).add_(
            grad_slice,
            alpha=1.0 - self.beta1,
        )
        exp_avg_sq.mul_(self.beta2).addcmul_(
            grad_slice,
            grad_slice,
            value=1.0 - self.beta2,
        )
        bias_correction1 = 1.0 - self.beta1**adam_step
        bias_correction2 = 1.0 - self.beta2**adam_step
        denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
        denominator.add_(self.eps)
        step_size = self.layer_lr / bias_correction1
        return step_size * (exp_avg / denominator)
