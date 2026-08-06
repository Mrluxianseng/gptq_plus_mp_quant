"""Ragged expert Hessian accumulator for Qwen3-MoE.

The inherited :class:`RealQLayer` remains the copied dense implementation.
Only routed expert projections use this subclass, which changes Hessian
accumulation/finalisation while reusing the same Cholesky, GPTQ, Block-GD and
weight-write implementation.

Expert assignment counts are rank-ragged.  Consequently ``add_batch`` never
issues a collective.  Every rank instead accumulates all ``num_groups`` local
partials and performs one fixed-shape all-reduce in ``finalize_hessian``.  This
is the simple correctness baseline; a later reduce-scatter candidate must be
profiled as a MoE-only optimisation and may not alter the dense path.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from realq_moe.quant.realq_layer import LOSS_GRAD_SCALE, RealQLayer
from utils import dist_utils


class RoutedRealQLayer(RealQLayer):
    """One routed expert projection with globally normalised ragged statistics."""

    def __init__(
        self,
        linear: nn.Linear,
        saliency: torch.Tensor,
        quantizer,
        num_groups: int,
        dev: torch.device,
        *,
        normalization_token_count: int,
        group_parallel_quant: str = "none",
    ) -> None:
        if saliency.dim() != 3 or saliency.shape[1] != 1:
            raise ValueError(
                "routed saliency must have shape (assignments,1,num_groups); "
                f"got {tuple(saliency.shape)}."
            )
        if saliency.shape[-1] != num_groups:
            raise ValueError(
                "routed saliency group dimension does not match num_groups: "
                f"{saliency.shape[-1]} != {num_groups}."
            )
        resolved_dev = torch.device(dev)
        if (
            resolved_dev.type == "cuda"
            and resolved_dev.index is None
        ):
            resolved_dev = torch.device(
                "cuda", torch.cuda.current_device()
            )
        if saliency.device != resolved_dev:
            raise ValueError(
                "routed saliency must already be resident on the target "
                f"device; CPU/H2D staging is forbidden: "
                f"{saliency.device} != {resolved_dev}."
            )
        if (
            not isinstance(normalization_token_count, int)
            or isinstance(normalization_token_count, bool)
            or normalization_token_count <= 0
        ):
            raise ValueError(
                "normalization_token_count must be a positive global token "
                f"count, got {normalization_token_count!r}."
            )
        super().__init__(
            linear=linear,
            saliency=saliency,
            quantizer=quantizer,
            num_groups=num_groups,
            dev=dev,
            group_parallel_quant=group_parallel_quant,
        )
        self.normalization_token_count = int(normalization_token_count)

        # The dense rank-sharded constructor allocates only locally-owned
        # groups.  Ragged ranks cannot reduce per batch, so replace that buffer
        # with all groups and reduce exactly once after local accumulation.
        del self.H
        self.H = torch.zeros(
            (self.num_groups, self.columns, self.columns),
            device=self.dev,
        )
        self.rtn_fallback = False
        self.rtn_fallback_reason: str | None = None

    @torch.no_grad()
    def finalize_rtn_fallback(self, reason: str) -> None:
        """Finalize a globally unassigned expert as exact grouped RTN.

        An identity Hessian makes analytic GPTQ compensation diagonal-only,
        hence each column independently rounds the original weight.  The
        joint stepper additionally masks BlockGD updates for this lane.
        """

        if self._finalized:
            raise RuntimeError(
                "finalize_rtn_fallback called after Hessian finalization."
            )
        if self.index != 0 or self.token_count != 0:
            raise RuntimeError(
                "RTN fallback is valid only before routed Hessian "
                f"accumulation; index={self.index}, "
                f"token_count={self.token_count}."
            )
        if not isinstance(reason, str) or not reason:
            raise ValueError("RTN fallback requires a non-empty reason.")

        self.H.zero_()
        diagonal = torch.diagonal(self.H, dim1=-2, dim2=-1)
        diagonal.fill_(1.0)
        self.act_square.zero_()
        self._finalized = True
        self.rtn_fallback = True
        self.rtn_fallback_reason = reason

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        """Accumulate assignment inputs without any rank-dependent collective.

        ``inp`` may be ``(A,C)`` or ``(A,1,C)``.  Each row must follow the same
        order as the expert's Stage-0 saliency rows.
        """

        if self._finalized:
            raise RuntimeError("add_batch called after finalize_hessian.")
        if inp.dim() == 2:
            inp = inp.unsqueeze(1)
        if inp.dim() != 3 or inp.shape[1] != 1:
            raise ValueError(
                "routed inputs must have shape (A,C) or (A,1,C); got "
                f"{tuple(inp.shape)}."
            )
        assignment_count = int(inp.shape[0])
        sal = self.saliency[
            self.index : self.index + assignment_count
        ].to(self.dev)
        if int(sal.shape[0]) != assignment_count:
            raise RuntimeError(
                "routed Hessian received more inputs than saliency rows: "
                f"cursor={self.index}, requested={assignment_count}, "
                f"available={self.saliency.shape[0]}."
            )
        self.index += assignment_count

        x = inp.reshape(assignment_count, inp.shape[-1]).float()
        sal = sal.reshape(assignment_count, self.num_groups).float()
        if assignment_count:
            weighted = x.unsqueeze(0).mul(
                sal.transpose(0, 1).unsqueeze(-1)
            )
            x_t = x.transpose(0, 1).unsqueeze(0).expand(
                self.num_groups, -1, -1
            )
            self.H.add_(torch.bmm(x_t, weighted))
            self.act_square.add_((x ** 2).sum(0))
        self.token_count += assignment_count

    @torch.no_grad()
    def finalize_hessian(self) -> None:
        if self._finalized:
            return
        if self.index != int(self.saliency.shape[0]):
            raise RuntimeError(
                "routed Hessian consumed "
                f"{self.index}/{self.saliency.shape[0]} saliency rows."
            )

        # All ranks execute this same pair of collectives once per expert
        # projection, including ranks with no local assignments.
        dist_utils.allreduce_sum_(self.H)
        dist_utils.allreduce_sum_(self.act_square)
        total_assignments = dist_utils.allreduce_sum_scalar(self.index)
        total_input_rows = dist_utils.allreduce_sum_scalar(self.token_count)
        if total_assignments <= 0 or total_input_rows <= 0:
            raise RuntimeError(
                "routed expert is globally cold; teacher coverage must be "
                "validated before Stage 1."
            )
        if total_assignments != total_input_rows:
            raise RuntimeError(
                "routed Hessian assignment/input count mismatch after reduce: "
                f"{total_assignments} != {total_input_rows}."
            )

        self.H.div_(float(self.normalization_token_count))
        self.H.div_(LOSS_GRAD_SCALE * LOSS_GRAD_SCALE)
        self.act_square.div_(float(total_input_rows))

        # Quantisation owns only the output-row groups assigned to this rank.
        if self.hessian_group_sharded:
            group_ids = self.hessian_group_ids.to(
                device=self.H.device, dtype=torch.long
            )
            self.H = self.H.index_select(0, group_ids).contiguous()

        self.H = 0.5 * (self.H + self.H.transpose(-1, -2))
        if not torch.isfinite(self.H).all():
            logging.warning(
                "[realq_moe.routed_realq_layer] non-finite entries in H for "
                "expert projection shape (%d, %d); replacing with zeros.",
                self.rows,
                self.columns,
            )
            self.H = torch.nan_to_num(
                self.H, nan=0.0, posinf=0.0, neginf=0.0
            )
        self._finalized = True
