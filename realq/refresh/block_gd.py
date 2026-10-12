"""Block-boundary full-Transformer-block refresh for REAL-Q.

The closure still fires at every non-final GPTQ column-block boundary, so its
backward scope and scheduling match REAL-Q. Unlike the source implementation,
one backward updates every quantisable weight that has not yet been committed:
the current linear's trailing columns, all later linears in the current
Transformer block, and—while the sliding loss arm participates—all linears in
the next block. Already-quantized weights remain locked.

Each future linear owns an FP32 master and persistent Adam moments before its
own GPTQ sweep begins. The functional student forward derives low-precision
autograd leaves from those masters without mutating the module's FP teacher
weights. When that linear becomes current, its accumulated FP32 master is
handed to :meth:`realq.quant.RealQLayer.quantize`.
"""
from __future__ import annotations

import logging
import hashlib
import math
import os
import random as _random
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from torch.func import functional_call

from realq.alignment import (
    ActiveWeightAudit,
    RefreshStep,
    RefreshTraceWriter,
)
from realq.refresh.fisher_loss import fisher_mse_loss
from realq.refresh import triton_block_adam
from realq.utils import nvtx
from utils import dist_utils
from utils.saliency_utils import global_percentile

if TYPE_CHECKING:
    from realq.runner.streams import LayerInputs


def layer_lr_for_schedule(
    base_lr: float,
    layer_idx: int,
    num_layers: int,
    base_ratio: float,
    schedule: str,
    *,
    activation_aware: bool = False,
) -> float:
    """Per-layer lr ramp matching old ``compute_layer_lr_scale`` +
    ``compute_scheduled_layer_lr`` (lines 647-676).

    ``activation_aware=True``: every non-final transformer layer uses
    ``base_lr`` as a constant.  In the paper's learning-rate table, aware rows
    report this constant directly (whereas scheduled rows report the final
    value); therefore the schedule's ``base_ratio`` must not be applied again.

    schedule="none": every layer uses ``base_lr`` directly. ``base_ratio`` is
    ignored. Mirrors legacy ``compute_layer_lr_scale``'s ``schedule in
    (None, "", "none")`` short-circuit returning 1.0, then
    ``compute_scheduled_layer_lr`` collapsing to ``target_lr`` when scale=1.0.

    schedule="cosine": layer 0 gets ``base_lr * base_ratio`` and indices use
    the paper's literal all-``L`` ``sin(π·x/2)`` ramp. Index ``L-1`` would
    reach ``base_lr``, but the actual final transformer block has a separate
    true-KL LR override; therefore the deepest scheduled non-final block
    (index ``L-2``) does not reach the endpoint. See
    ``docs/REALQ_PAPER_PROTOCOL.md`` for this paper ambiguity.
    """
    if activation_aware:
        return base_lr
    if schedule == "none":
        return base_lr
    if schedule != "cosine":
        raise ValueError(
            f"Unknown grad_lr_layer_schedule={schedule!r}; expected 'none' or 'cosine'."
        )
    if num_layers <= 1:
        scale = 1.0
    else:
        x = layer_idx / (num_layers - 1)
        scale = math.sin(math.pi * x / 2.0)
    return base_lr * (base_ratio + (1.0 - base_ratio) * scale)


class _SharedSampleScheduler:
    """Round-robin sample scheduler shared across ALL block_gd refreshes
    in a run. Mirrors old ``BackwardSampleScheduler`` (lines 590-612)
    in the ``--dp_global_shuffle=True`` branch (lines 7885-7895): a
    single ``random.Random`` seeded with ``cfg.refresh_seed`` (NO per-rank
    offset) over the GLOBAL ``[0, nsamples)`` index range. Every rank
    constructs the scheduler with the same ``(n_total, chunk_size,
    seed)`` so ``next_indices()`` returns the IDENTICAL global id list
    on every rank; the refresh closure then filters those global ids to
    its own contiguous shard ``[rank * n_local, (rank+1) * n_local)``.

    chunk_size here = ``backward_samples`` (the GLOBAL per-refresh sample
    count, NOT divided by world). After per-rank filtering, the local
    sub-list lengths sum to ``backward_samples`` across the world but
    are not individually balanced. The first epoch is deliberately ordered,
    so contiguous DP shards can make that imbalance extreme (including empty
    ranks); later globally shuffled chunks remain variably imbalanced. This
    reproduces old GPTQ+ ``--dp_global_shuffle=True`` exactly and makes the
    consumed global sample sequence match a one-rank run.

    Shared state means consecutive ``next_indices()`` calls — across
    different layers and different modules within a layer — return
    DIFFERENT chunks rather than every module replaying the same prefix.
    """

    def __init__(self, n_total: int, chunk_size: int, seed: int) -> None:
        if n_total <= 0:
            raise ValueError(f"n_total must be positive, got {n_total}.")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        if n_total % chunk_size != 0:
            raise ValueError(
                f"n_total ({n_total}) must be divisible by chunk_size "
                f"({chunk_size}) so the round-robin scheduler exhausts "
                f"each chunk cleanly (matches old BackwardSampleScheduler "
                f"check at gptq_plus_utils.py:594)."
            )
        self.n_total = int(n_total)
        self.chunk_size = int(chunk_size)
        self._order = list(range(self.n_total))
        self._rng = _random.Random(int(seed))
        self._cursor = 0

    def next_indices(self) -> list[int]:
        """Return ``chunk_size`` GLOBAL sample indices (Python list).
        Same on every rank when constructed with the same seed."""
        if self._cursor >= self.n_total:
            self._rng.shuffle(self._order)
            self._cursor = 0
        idx = self._order[self._cursor : self._cursor + self.chunk_size]
        self._cursor += self.chunk_size
        return idx

    def audit_state(self) -> dict[str, object]:
        """Return a read-only snapshot proving whether a caller consumed it."""

        return {
            "n_total": self.n_total,
            "chunk_size": self.chunk_size,
            "cursor": self._cursor,
            "order": list(self._order),
        }


class RefreshContext:
    """Per-current-module sampling and diagnostic context.

    REALQ-Plus keeps Adam moments in :class:`BlockRefreshState`, because a
    weight can receive updates before its own linear quantisation begins.
    This lightweight context retains the run-wide sample scheduler and the
    existing per-column-block diagnostics.
    """

    def __init__(
        self,
        module: "nn.Module",
        layer_lr: float,
        grad_clip: float,
        backward_bsz: int,
        scheduler: "_SharedSampleScheduler",
        trace_writer: "RefreshTraceWriter | None" = None,
        trace_layer: int | None = None,
        trace_module: str | None = None,
        blocksize: int | None = None,
        log_column_block_loss: bool = False,
        fused_block_adam: bool = False,
    ) -> None:
        self.module = module
        self.layer_lr = float(layer_lr)
        self.grad_clip = float(grad_clip)
        self.backward_bsz = int(backward_bsz)
        self.scheduler = scheduler
        self.trace_writer = trace_writer
        self.trace_layer = trace_layer
        self.trace_module = trace_module
        self.blocksize = blocksize
        self.log_column_block_loss = log_column_block_loss
        self.fused_block_adam = fused_block_adam
        if type(self.log_column_block_loss) is not bool:
            raise ValueError(
                "log_column_block_loss must be bool, got "
                f"{self.log_column_block_loss!r}"
            )
        if type(self.fused_block_adam) is not bool:
            raise ValueError(
                "fused_block_adam must be bool, got "
                f"{self.fused_block_adam!r}"
            )
        if self.loss_observation_enabled and (
            trace_layer is None or trace_module is None or blocksize is None
        ):
            raise ValueError(
                "enabled refresh loss observation requires trace_layer, "
                "trace_module, and blocksize"
            )
        self.adam_step = 0
        # The experimental cross-module prior may initialize only v from a
        # donor. Keep its bias-correction age separate from m's true local age.
        self.adam_v_step = 0
        self._adam_v_transfer_applied = False
        # Per-current-linear column-boundary ordinal. REALQ-Plus Adam steps
        # can be larger because the weight may already have been updated while
        # it was a future/sliding weight; trace column ranges must not use that
        # persistent optimizer step as their block index.
        self.refresh_step = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        # Retained for the public single-linear refresh compatibility path.
        # Full-block REAL-Q stores moments in ``BlockRefreshState`` instead.
        weight = module.weight
        self.exp_avg = torch.zeros_like(weight, dtype=torch.float32)
        self.exp_avg_sq = torch.zeros_like(weight, dtype=torch.float32)

    def next_indices(self) -> list[int]:
        return self.scheduler.next_indices()

    def allocate_backward_invocation_id(self) -> int | None:
        if not self.trace_enabled:
            return None
        assert self.trace_writer is not None
        return self.trace_writer.allocate_backward_invocation_id()

    @property
    def trace_enabled(self) -> bool:
        return self.trace_writer is not None and self.trace_writer.enabled

    @property
    def loss_observation_enabled(self) -> bool:
        return self.trace_enabled or self.log_column_block_loss

    def record_loss_observation(
        self,
        *,
        global_loss_sums: torch.Tensor,
        global_count: int,
        sample_indices: list[int],
        slide_alpha: float | None,
        has_next_loss: bool,
        objective: str,
        backward_invocation_id: int | None = None,
        backward_chunk_sizes: tuple[int, ...] = (),
        active_weight_audits: tuple[ActiveWeightAudit, ...] = (),
    ) -> None:
        """Print and/or trace one globally reduced backward objective."""

        if not self.loss_observation_enabled or not dist_utils.is_main():
            return
        if global_loss_sums.numel() not in (1, 3):
            raise ValueError(
                "refresh loss observation expects [total] or "
                "[total, current, next] global loss sums"
            )
        if objective.endswith("_full_block") and self.trace_enabled:
            if (
                backward_invocation_id is None
                or not backward_chunk_sizes
                or not active_weight_audits
            ):
                raise RuntimeError(
                    "full-block trace requires runtime backward and "
                    "active-weight audit evidence"
                )
        block = self.refresh_step - 1
        col_start = block * int(self.blocksize)
        col_end = min(
            col_start + int(self.blocksize),
            int(self.module.weight.shape[1]),
        )
        denom = float(global_count)
        mean_loss = float(global_loss_sums[0].item()) / denom
        mean_loss_current = (
            float(global_loss_sums[1].item()) / denom
            if global_loss_sums.numel() == 3
            else mean_loss
        )
        mean_loss_next = (
            float(global_loss_sums[2].item()) / denom
            if has_next_loss and global_loss_sums.numel() == 3
            else None
        )
        if self.log_column_block_loss:
            logging.info(
                "[realq.column_block_loss] layer=%d module=%s block=%d "
                "columns=[%d,%d) column_space=quant_order adam_step=%d "
                "objective=%s loss=%.12g loss_current=%.12g loss_next=%s "
                "slide_alpha=%s lr=%.12g adam_step_size=%.12g "
                "global_samples=%d",
                int(self.trace_layer),
                str(self.trace_module),
                block,
                col_start,
                col_end,
                self.adam_step,
                objective,
                mean_loss,
                mean_loss_current,
                (
                    f"{mean_loss_next:.12g}"
                    if mean_loss_next is not None
                    else "none"
                ),
                (
                    f"{slide_alpha:.12g}"
                    if slide_alpha is not None
                    else "none"
                ),
                self.layer_lr,
                self.layer_lr / (1.0 - self.beta1 ** self.adam_step),
                global_count,
            )
        if self.trace_enabled:
            self.trace_writer.record(
                RefreshStep(
                    layer=int(self.trace_layer),
                    module=str(self.trace_module),
                    block=block,
                    col_start=col_start,
                    col_end=col_end,
                    adam_step=self.adam_step,
                    loss=mean_loss,
                    loss_current=mean_loss_current,
                    loss_next=mean_loss_next,
                    slide_alpha=slide_alpha,
                    sample_indices=tuple(
                        int(index) for index in sample_indices
                    ),
                    objective=objective,
                    backward_invocation_id=backward_invocation_id,
                    backward_chunk_sizes=backward_chunk_sizes,
                    backward_bsz=self.backward_bsz,
                    global_count=int(global_count),
                    active_weights=active_weight_audits,
                )
            )


def _aggregate_loss_sums_for_logging(
    partial_loss_sums: torch.Tensor,
) -> torch.Tensor:
    """All-reduce diagnostic loss sums without touching gradient reduction.

    Keeping this collective separate from ``_aggregate_refresh_sums`` means
    enabling console/file logging cannot change the size (and therefore the
    collective algorithm) of the optimizer's ``[gradient | count]`` buffer.
    The loss values are diagnostics only and never feed the Adam update.
    """

    global_loss_sums = partial_loss_sums.clone()
    dist_utils.allreduce_sum_(global_loss_sums)
    return global_loss_sums


def _aggregate_refresh_sums(
    partial_grad_sum: torch.Tensor,
    partial_count: int,
    partial_loss_sums: torch.Tensor | None = None,
) -> tuple[int, torch.Tensor | None]:
    """All-reduce refresh gradient/count and optional loss sums in one pack.

    With tracing disabled, the packed layout remains exactly the historical
    ``[grad | count]`` layout.  Tracing appends three diagnostics
    ``[loss]`` and, only for an evaluated slide arm,
    ``[loss_current | loss_next]``. This exactly matches the legacy pack width
    at the corresponding step. The branch is opt-in and every rank
    participates so rank zero writes global sample sums.
    """

    if dist_utils.get_world_size() > 1:
        count_t = torch.tensor(
            [float(partial_count)],
            dtype=partial_grad_sum.dtype,
            device=partial_grad_sum.device,
        )
        grad_flat = partial_grad_sum.reshape(-1)
        parts = [grad_flat, count_t]
        if partial_loss_sums is not None:
            parts.append(
                partial_loss_sums.to(
                    device=partial_grad_sum.device,
                    dtype=partial_grad_sum.dtype,
                )
            )
        packed = torch.cat(parts)
        dist_utils.allreduce_sum_(packed)
        partial_grad_sum.copy_(packed[: grad_flat.numel()].view_as(partial_grad_sum))
        scalar_out = packed[grad_flat.numel() :]
        global_count = int(scalar_out[0].item())
        global_loss_sums = (
            scalar_out[1:].clone() if partial_loss_sums is not None else None
        )
    else:
        global_count = int(partial_count)
        global_loss_sums = partial_loss_sums
    if global_count <= 0:
        raise RuntimeError("refresh produced zero samples across all ranks")
    return global_count, global_loss_sums


def _functional_weight_name(layer: nn.Module, module: nn.Module) -> str:
    """Resolve the dotted parameter name of ``module.weight`` inside ``layer``.

    Needed for ``functional_call``: it expects the full path key (e.g.
    ``self_attn.q_proj.weight`` or ``self_attn.q_proj.module.weight`` when
    wrapped by ActQuantWrapper) so the override slots into the right
    parameter during forward.
    """
    target_id = id(module.weight)
    for name, p in layer.named_parameters():
        if id(p) == target_id:
            return name
    raise RuntimeError(
        "Could not find module.weight inside layer.named_parameters() — "
        "did the linear get re-wrapped after RealQLayer was constructed?"
    )


@dataclass
class _WeightRefreshState:
    """FP32 master and persistent Adam state for one quantisable linear."""

    name: str
    module: nn.Module
    parameter_name: str
    shape: tuple[int, int]
    master: torch.Tensor | None
    exp_avg: torch.Tensor | None
    exp_avg_sq: torch.Tensor | None
    step: int = 0
    # Normally tracks ``step`` exactly; separate age permits a mathematically
    # coherent v-only restart without changing m's bias correction.
    v_step: int = 0
    quantizing: bool = False
    quantized: bool = False


@dataclass
class _ActiveWeight:
    state: _WeightRefreshState
    leaf: torch.Tensor
    source: torch.Tensor
    scope: str
    source_storage_id: str
    active_columns: slice | torch.Tensor | None
    is_current: bool


def tensor_storage_identity(tensor: torch.Tensor) -> str:
    """Stable-within-process identity for proving shared live storage."""

    storage = tensor.untyped_storage()
    return (
        f"{tensor.device.type}:{tensor.device.index}:"
        f"{storage.data_ptr()}:{storage.nbytes()}"
    )


class BlockRefreshState:
    """Persistent update state for every quantisable weight in one block.

    A full FP32 master is retained until the corresponding linear begins its
    GPTQ sweep.  While another linear is current, that master participates in
    the functional block forward and receives the same backward-driven Adam
    step.  Once the linear becomes current, :meth:`begin_quantization` hands
    ownership of its master to :class:`RealQLayer`; the stitched current
    weight supplied by the column loop then replaces it in functional calls.
    """

    def __init__(
        self,
        layer: nn.Module,
        named_modules: list[tuple[str, nn.Module]],
    ) -> None:
        if not named_modules:
            raise ValueError("BlockRefreshState requires at least one linear.")
        self.layer = layer
        self._states: dict[str, _WeightRefreshState] = {}
        for name, module in named_modules:
            if name in self._states:
                raise ValueError(f"duplicate quantisable module name {name!r}")
            weight = module.weight.detach()
            if weight.dim() != 2:
                raise ValueError(
                    f"{name}.weight must be 2-D, got {tuple(weight.shape)}"
                )
            master = weight.float().clone()
            self._states[name] = _WeightRefreshState(
                name=name,
                module=module,
                parameter_name=_functional_weight_name(layer, module),
                shape=tuple(weight.shape),
                master=master,
                exp_avg=torch.zeros_like(master),
                exp_avg_sq=torch.zeros_like(master),
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._states)

    def unquantized_master_mapping(
        self,
    ) -> "OrderedDict[str, torch.Tensor]":
        """Return live FP32 masters in block quantization order.

        The mapping container is new, but its tensors are the actual masters
        owned by this state.  A block-level second-order compensator and
        Block-GD therefore update the same future-weight storage instead of
        maintaining competing copies.

        This view is acquired only at a pristine block boundary.  The current
        tensor reference remains valid after ``begin_quantization`` transfers
        ownership to ``RealQLayer``; future references remain state-owned
        until their respective handoffs.
        """

        masters: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, state in self._states.items():
            if state.quantized or state.quantizing or state.master is None:
                raise RuntimeError(
                    "unquantized master view requires a pristine block "
                    f"boundary; {name}: quantized={state.quantized}, "
                    f"quantizing={state.quantizing}, "
                    f"has_master={state.master is not None}"
                )
            if state.master.dtype != torch.float32:
                raise RuntimeError(
                    f"{name} master must be FP32, got {state.master.dtype}"
                )
            masters[name] = state.master
        return masters

    def rebind(
        self,
        layer: nn.Module,
        named_modules: list[tuple[str, nn.Module]],
    ) -> None:
        """Rebind module objects after CPU-master materialisation."""

        rebound = dict(named_modules)
        if tuple(rebound) != self.names:
            raise RuntimeError(
                "quantisable module order changed while reusing block state: "
                f"expected {self.names}, got {tuple(rebound)}"
            )
        self.layer = layer
        for name, state in self._states.items():
            module = rebound[name]
            if tuple(module.weight.shape) != state.shape:
                raise RuntimeError(
                    f"{name}.weight shape changed from "
                    f"{state.shape} to {tuple(module.weight.shape)}"
                )
            state.module = module
            state.parameter_name = _functional_weight_name(layer, module)
            device = module.weight.device
            if state.master is not None and state.master.device != device:
                state.master = state.master.to(device)
            if state.exp_avg is not None and state.exp_avg.device != device:
                state.exp_avg = state.exp_avg.to(device)
                state.exp_avg_sq = state.exp_avg_sq.to(device)

    def begin_quantization(self, name: str) -> torch.Tensor:
        state = self._states[name]
        if state.quantized or state.quantizing or state.master is None:
            raise RuntimeError(
                f"cannot begin quantization for {name}: "
                f"quantized={state.quantized}, quantizing={state.quantizing}, "
                f"has_master={state.master is not None}"
            )
        master = state.master
        state.master = None
        state.quantizing = True
        return master

    def finish_quantization(self, name: str) -> None:
        state = self._states[name]
        if not state.quantizing or state.quantized:
            raise RuntimeError(
                f"cannot finish quantization for inactive module {name!r}"
            )
        state.quantizing = False
        state.quantized = True
        # The deployed quantized weight is now stored in module.weight and can
        # never be updated again.  Its optimizer state is no longer needed.
        state.exp_avg = None
        state.exp_avg_sq = None

    def expected_next_step(self, name: str) -> int:
        return self._states[name].step + 1

    def make_overrides(
        self,
        *,
        current_name: str | None = None,
        current_weight_fp32: torch.Tensor | None = None,
        trailing_col_start: int = 0,
        perm: torch.Tensor | None = None,
        trace_scope: str = "current_block",
    ) -> tuple[dict[str, torch.Tensor], list[_ActiveWeight]]:
        """Create functional-call leaves for all not-yet-quantized weights."""

        if (current_name is None) != (current_weight_fp32 is None):
            raise ValueError(
                "current_name and current_weight_fp32 must be provided together"
            )
        if trace_scope not in ("current_block", "next_block"):
            raise ValueError(f"invalid trace scope {trace_scope!r}")
        overrides: dict[str, torch.Tensor] = {}
        active: list[_ActiveWeight] = []
        for name, state in self._states.items():
            if state.quantized:
                continue
            is_current = name == current_name
            if is_current:
                if not state.quantizing:
                    raise RuntimeError(
                        f"current module {name!r} is not marked quantizing"
                )
                source = current_weight_fp32
                if perm is None:
                    active_columns: slice | torch.Tensor | None = slice(
                        trailing_col_start, None
                    )
                else:
                    active_columns = perm[trailing_col_start:]
            else:
                if state.quantizing:
                    raise RuntimeError(
                        f"unexpected second quantizing module {name!r}"
                    )
                if state.master is None:
                    raise RuntimeError(
                        f"active future module {name!r} has no FP32 master"
                    )
                source = state.master
                active_columns = None
            if tuple(source.shape) != tuple(state.module.weight.shape):
                raise RuntimeError(
                    f"{name} override shape {tuple(source.shape)} does not "
                    f"match module weight {tuple(state.module.weight.shape)}"
                )
            # BF16/FP16 leaves reproduce the deployed forward precision while
            # the source/master and optimizer state remain FP32.
            leaf = source.to(dtype=state.module.weight.dtype)
            # FP32 master -> BF16/FP16 conversion already allocates distinct
            # storage. Cloning it again was one full weight copy per active
            # linear and refresh.
            if leaf.data_ptr() == source.data_ptr():
                leaf = leaf.clone()
            leaf = leaf.detach().requires_grad_(True)
            overrides[state.parameter_name] = leaf
            active.append(
                _ActiveWeight(
                    state=state,
                    leaf=leaf,
                    source=source,
                    scope=trace_scope,
                    source_storage_id=tensor_storage_identity(source),
                    active_columns=active_columns,
                    is_current=is_current,
                )
            )
        if current_name is not None and not any(x.is_current for x in active):
            raise RuntimeError(
                f"current module {current_name!r} was absent from active weights"
            )
        return overrides, active

    def assert_complete(self) -> None:
        incomplete = [
            name for name, state in self._states.items()
            if not state.quantized
        ]
        if incomplete:
            raise RuntimeError(
                f"block quantization ended with active weights: {incomplete}"
            )

    def release(self) -> None:
        for state in self._states.values():
            state.master = None
            state.exp_avg = None
            state.exp_avg_sq = None


def _aggregate_block_refresh_sums(
    partial_grad_sums: list[torch.Tensor],
    partial_used: list[bool],
    partial_count: int,
    partial_loss_sums: torch.Tensor | None,
) -> tuple[int, list[bool], torch.Tensor | None]:
    """All-reduce all active-weight gradients plus scalar diagnostics."""

    if len(partial_grad_sums) != len(partial_used):
        raise ValueError("gradient and used-flag lengths differ")
    if not partial_grad_sums:
        raise RuntimeError("block refresh has no active gradients")
    if dist_utils.get_world_size() > 1:
        numels = [tensor.numel() for tensor in partial_grad_sums]
        dtype = partial_grad_sums[0].dtype
        device = partial_grad_sums[0].device
        scalar_parts = [
            torch.tensor(
                [float(partial_count), *[float(v) for v in partial_used]],
                dtype=dtype,
                device=device,
            )
        ]
        if partial_loss_sums is not None:
            scalar_parts.append(
                partial_loss_sums.to(device=device, dtype=dtype)
            )
        packed = torch.cat(
            [tensor.reshape(-1) for tensor in partial_grad_sums]
            + scalar_parts
        )
        dist_utils.allreduce_sum_(packed)
        cursor = 0
        for tensor, numel in zip(partial_grad_sums, numels):
            tensor.copy_(packed[cursor : cursor + numel].view_as(tensor))
            cursor += numel
        global_count = int(packed[cursor].item())
        cursor += 1
        global_used = [
            bool(packed[cursor + index].item() > 0)
            for index in range(len(partial_used))
        ]
        cursor += len(partial_used)
        global_loss_sums = (
            packed[cursor:].clone()
            if partial_loss_sums is not None
            else None
        )
    else:
        global_count = int(partial_count)
        global_used = list(partial_used)
        global_loss_sums = partial_loss_sums
    if global_count <= 0:
        raise RuntimeError("block refresh produced zero samples across all ranks")
    return global_count, global_used, global_loss_sums


_ADAM_HISTORY_POLICIES = {
    "keep",
    "reset_each_refresh",
    "reset_m_each_refresh",
    "reset_m_and_m_age_each_refresh",
    "reset_v_each_refresh",
    "reset_v_and_v_age_each_refresh",
    "reset_bias_age_each_refresh",
    "radam",
    "adabelief",
    "adopt",
}


def _adam_history_policy() -> str:
    policy = os.environ.get("REALQ_ADAM_HISTORY_POLICY", "keep")
    if policy not in _ADAM_HISTORY_POLICIES:
        raise ValueError(
            "REALQ_ADAM_HISTORY_POLICY must be one of "
            f"{sorted(_ADAM_HISTORY_POLICIES)}, got {policy!r}"
        )
    return policy


def _adam_history_policy_for_layer(layer_idx: int | None) -> str:
    """Apply the selected research policy only to an optional layer set.

    ``REALQ_ADAM_HISTORY_LAYERS`` accepts ``all`` (the default), comma-
    separated layer indices, and inclusive ranges such as ``0-26``. The
    explicit range makes early/final-layer scope experiments reproducible and
    avoids baking a particular model depth into the optimizer.
    """
    policy = _adam_history_policy()
    scope = os.environ.get("REALQ_ADAM_HISTORY_LAYERS", "all").strip()
    if not scope or scope.lower() == "all":
        return policy
    if layer_idx is None:
        raise ValueError(
            "REALQ_ADAM_HISTORY_LAYERS requires a known layer index; "
            f"got scope={scope!r}"
        )
    selected = False
    for part in scope.split(","):
        item = part.strip()
        if not item:
            raise ValueError(
                "REALQ_ADAM_HISTORY_LAYERS contains an empty item: "
                f"{scope!r}"
            )
        if "-" in item:
            bounds = item.split("-")
            if len(bounds) != 2 or not all(x.strip().isdigit() for x in bounds):
                raise ValueError(
                    "REALQ_ADAM_HISTORY_LAYERS ranges must be inclusive "
                    f"nonnegative integers, got {item!r}"
                )
            low, high = (int(x.strip()) for x in bounds)
            if high < low:
                raise ValueError(
                    "REALQ_ADAM_HISTORY_LAYERS range end precedes start: "
                    f"{item!r}"
                )
        else:
            if not item.isdigit():
                raise ValueError(
                    "REALQ_ADAM_HISTORY_LAYERS entries must be layer indices "
                    f"or inclusive ranges, got {item!r}"
                )
            low = high = int(item)
        selected = selected or low <= int(layer_idx) <= high
    return policy if selected else "keep"


def _zero_adam_buffer_selected(
    buffer: torch.Tensor, columns: slice | torch.Tensor | None
) -> None:
    if columns is None:
        buffer.zero_()
    elif isinstance(columns, slice):
        buffer[:, columns].zero_()
    else:
        reset_index = columns.to(device=buffer.device, dtype=torch.long)
        buffer.index_fill_(1, reset_index, 0.0)


def _adam_probe_target_rule(ctx: "RefreshContext") -> str:
    """Return the one-step diagnostic rule at a selected REAL-Q refresh."""

    rule = os.environ.get("REALQ_ADAM_PROBE_RULE", "adam")
    if rule not in (
        "adam",
        "reset_m",
        "reset_m_direction_norm_matched_adam",
        "adam_direction_norm_matched_reset_m",
        "grams_direction",
        "norm_matched_gradient",
        "cautious",
        "cautious_mask",
        "cautious_scale",
        "no_parameter_update",
    ):
        raise ValueError(
            "REALQ_ADAM_PROBE_RULE must be one of 'adam', "
            "'reset_m', 'reset_m_direction_norm_matched_adam', "
            "'adam_direction_norm_matched_reset_m', "
            "'grams_direction', 'norm_matched_gradient', 'cautious', "
            "'cautious_mask', 'cautious_scale', or 'no_parameter_update', "
            f"got {rule!r}"
        )
    target_vars = (
        "REALQ_ADAM_PROBE_LAYER",
        "REALQ_ADAM_PROBE_MODULE",
        "REALQ_ADAM_PROBE_BLOCK",
    )
    target_values = [os.environ.get(name) for name in target_vars]
    if all(value is None for value in target_values):
        if rule != "adam":
            raise ValueError(
                "non-default REALQ_ADAM_PROBE_RULE requires a target layer, "
                "module, and block"
            )
        return "adam"
    if any(value is None for value in target_values):
        raise ValueError(
            "set all of REALQ_ADAM_PROBE_LAYER, "
            "REALQ_ADAM_PROBE_MODULE, and REALQ_ADAM_PROBE_BLOCK"
        )
    target_layer, target_module, target_block = target_values
    matches = _is_adam_probe_target(ctx)
    return rule if matches else "adam"


def _is_adam_probe_target(ctx: "RefreshContext") -> bool:
    target_layer = os.environ.get("REALQ_ADAM_PROBE_LAYER")
    target_module = os.environ.get("REALQ_ADAM_PROBE_MODULE")
    target_block = os.environ.get("REALQ_ADAM_PROBE_BLOCK")
    if target_layer is None or target_module is None or target_block is None:
        return False
    return (
        ctx.trace_layer is not None
        and int(ctx.trace_layer) == int(target_layer)
        and ctx.trace_module == target_module
        and ctx.refresh_step - 1 == int(target_block)
    )


def _probe_tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _norm_matched_gradient_update(
    adam_update: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    """Use the clipped gradient direction at Adam's active-slice L2 norm."""

    update_norm = torch.linalg.vector_norm(adam_update)
    gradient_norm = torch.linalg.vector_norm(gradient)
    if gradient_norm.item() == 0.0:
        return torch.zeros_like(gradient)
    return gradient * (update_norm / gradient_norm)


def _match_update_l2_norm(
    update_to_scale: torch.Tensor, reference_update: torch.Tensor
) -> torch.Tensor:
    """Scale one update to another update's global L2 norm."""

    update_norm = torch.linalg.vector_norm(update_to_scale)
    reference_norm = torch.linalg.vector_norm(reference_update)
    if update_norm.item() == 0.0:
        return torch.zeros_like(update_to_scale)
    return update_to_scale * (reference_norm / update_norm)


def _log_adam_probe_snapshot(
    *,
    ctx: "RefreshContext",
    rule: str,
    weight: torch.Tensor,
    grad: torch.Tensor,
    exp_avg_before: torch.Tensor,
    exp_avg_sq_before: torch.Tensor,
) -> None:
    if not _is_adam_probe_target(ctx):
        return
    logging.info(
        "[adam_probe_snapshot] layer=%s module=%s block=%d rule=%s "
        "adam_step=%d w_sha256=%s g_sha256=%s m_sha256=%s v_sha256=%s "
        "w_norm=%.9g g_norm=%.9g m_norm=%.9g v_norm=%.9g",
        ctx.trace_layer,
        ctx.trace_module,
        ctx.refresh_step - 1,
        rule,
        ctx.adam_step,
        _probe_tensor_sha256(weight),
        _probe_tensor_sha256(grad),
        _probe_tensor_sha256(exp_avg_before),
        _probe_tensor_sha256(exp_avg_sq_before),
        float(torch.linalg.vector_norm(weight.detach().float()).item()),
        float(torch.linalg.vector_norm(grad.detach().float()).item()),
        float(torch.linalg.vector_norm(exp_avg_before.detach().float()).item()),
        float(torch.linalg.vector_norm(exp_avg_sq_before.detach().float()).item()),
    )


def _adam_update_selected(
    state: _WeightRefreshState,
    grad: torch.Tensor,
    columns: slice | torch.Tensor | None,
    *,
    lr: float,
    grad_clip: float,
    compact_update: bool = False,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """Advance one Adam state and return a full or active-only update."""

    if state.exp_avg is None or state.exp_avg_sq is None:
        raise RuntimeError(f"optimizer state for {state.name!r} was released")
    # Research-only ablations for Adam state across REAL-Q's discrete
    # quantize/compensate refresh boundaries. The default `keep` path leaves
    # the original optimizer transition untouched.
    history_policy = _adam_history_policy_for_layer(layer_idx)
    reset_m = history_policy in (
        "reset_each_refresh",
        "reset_m_each_refresh",
        "reset_m_and_m_age_each_refresh",
    )
    reset_v = history_policy in (
        "reset_each_refresh",
        "reset_v_each_refresh",
        "reset_v_and_v_age_each_refresh",
    )
    reset_v_age = history_policy == "reset_v_and_v_age_each_refresh"
    reset_m_age = history_policy == "reset_m_and_m_age_each_refresh"
    reset_age = history_policy in (
        "reset_each_refresh", "reset_bias_age_each_refresh"
    )
    if history_policy != "keep":
        if not getattr(state, "_history_ablation_logged", False):
            logging.info(
                "[adam_history_ablation] policy=%s layer=%s module=%s",
                history_policy,
                layer_idx,
                state.name,
            )
            state._history_ablation_logged = True
        if reset_m:
            _zero_adam_buffer_selected(state.exp_avg, columns)
        if reset_v:
            _zero_adam_buffer_selected(state.exp_avg_sq, columns)
        if reset_v_age:
            state.v_step = 0
        if reset_m_age:
            # Restart only first-moment history and its bias-correction age;
            # preserve the second-moment trajectory and its independent age.
            state.step = 0
        if reset_age:
            state.step = 0
            state.v_step = 0
    state.step += 1
    state.v_step += 1
    if columns is None:
        grad_sel = grad
        ea = state.exp_avg
        ev = state.exp_avg_sq
        index = None
    elif isinstance(columns, slice):
        grad_sel = grad[:, columns]
        ea = state.exp_avg[:, columns]
        ev = state.exp_avg_sq[:, columns]
        index = None
    else:
        index = columns.to(device=grad.device, dtype=torch.long)
        grad_sel = grad.index_select(1, index)
        ea = state.exp_avg.index_select(1, index)
        ev = state.exp_avg_sq.index_select(1, index)
    if grad_clip > 0:
        grad_sel = grad_sel.clamp(min=-grad_clip, max=grad_clip)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    adopt_first_step = history_policy == "adopt" and state.step == 1
    if history_policy == "adopt":
        if adopt_first_step:
            # ADOPT initializes v_0 from a gradient, then starts its first
            # normalized-momentum update. Reuse this refresh's already
            # available gradient for that initialization, preserving REAL-Q's
            # no-extra-backward constraint and Adam's sign-like first step.
            normalized_grad = grad_sel / grad_sel.abs().clamp_min(eps)
            ea.copy_(normalized_grad)
            ev.copy_(grad_sel.square())
        else:
            normalized_grad = grad_sel / ev.sqrt().clamp_min(eps)
            ea.mul_(beta1).add_(normalized_grad, alpha=1.0 - beta1)
            ev.mul_(beta2).addcmul_(
                grad_sel, grad_sel, value=1.0 - beta2
            )
    else:
        ea.mul_(beta1).add_(grad_sel, alpha=1.0 - beta1)
        second_moment_input = (
            grad_sel - ea if history_policy == "adabelief" else grad_sel
        )
        ev.mul_(beta2).addcmul_(
            second_moment_input,
            second_moment_input,
            value=1.0 - beta2,
        )
    if index is not None:
        state.exp_avg.index_copy_(1, index, ea)
        state.exp_avg_sq.index_copy_(1, index, ev)
    bc1 = 1.0 - beta1 ** state.step
    bc2 = 1.0 - beta2 ** state.v_step
    denom = ev.sqrt() / math.sqrt(bc2)
    denom.add_(eps)
    if history_policy == "adopt":
        selected_update = lr * ea
    elif history_policy == "radam":
        rho_inf = 2.0 / (1.0 - beta2) - 1.0
        beta2_t = beta2 ** state.v_step
        rho_t = rho_inf - 2.0 * state.v_step * beta2_t / (1.0 - beta2_t)
        m_hat = ea / bc1
        if rho_t > 5.0:
            rect = math.sqrt(
                (rho_t - 4.0)
                * (rho_t - 2.0)
                * rho_inf
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
            )
            selected_update = lr * rect * (m_hat / denom)
        else:
            selected_update = lr * m_hat
    else:
        selected_update = (lr / bc1) * (ea / denom)
    if compact_update:
        return selected_update
    update = torch.zeros_like(grad)
    if columns is None:
        update.copy_(selected_update)
    elif isinstance(columns, slice):
        update[:, columns].copy_(selected_update)
    else:
        update.index_copy_(1, index, selected_update)
    return update


def _apply_block_adam(
    active: list[_ActiveWeight],
    grad_sums: list[torch.Tensor | None],
    global_used: list[bool],
    global_count: int,
    *,
    lr: float,
    grad_clip: float,
    compact_update: bool = False,
    collect_audits: bool = True,
    gradients_are_means: bool = False,
    layer_idx: int | None = None,
) -> tuple[torch.Tensor, tuple[ActiveWeightAudit, ...]]:
    """Update every active FP32 master and return auditable runtime evidence."""

    current_update = None
    pending_audits: list[dict[str, object]] = []
    scalar_tensors: list[torch.Tensor] = []

    def scalar_index(value: torch.Tensor) -> int:
        # All callers pass norms of FP32 masters/updates.  The old ``float``
        # calls were dtype no-ops and obscured that invariant.
        scalar_tensors.append(value.detach())
        return len(scalar_tensors) - 1

    for entry, grad_sum, used in zip(active, grad_sums, global_used):
        storage_before = tensor_storage_identity(entry.source)
        if storage_before != entry.source_storage_id:
            raise RuntimeError(
                f"{entry.scope}:{entry.state.name} source storage changed "
                "during one backward invocation"
            )
        step_before = int(entry.state.step)
        source_l2_before_index = (
            scalar_index(torch.linalg.vector_norm(entry.source))
            if collect_audits
            else None
        )
        source_l2_after_index: int | None = None
        storage_after: str | None = None
        if not used:
            if grad_sum is not None:
                rows, columns = grad_sum.shape
                device = grad_sum.device
            else:
                rows, columns = entry.state.shape
                device = entry.source.device
            if compact_update:
                if entry.active_columns is None:
                    active_columns = columns
                elif isinstance(entry.active_columns, slice):
                    active_columns = len(
                        range(*entry.active_columns.indices(columns))
                    )
                else:
                    active_columns = int(entry.active_columns.numel())
                update = torch.zeros(
                    (rows, active_columns),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                update = torch.zeros(
                    (rows, columns),
                    dtype=torch.float32,
                    device=device,
                )
            if entry.is_current:
                current_update = update
            else:
                if entry.state.master is None:
                    raise RuntimeError(
                        f"future weight {entry.state.name!r} lost its "
                        "FP32 master"
                    )
                storage_after = tensor_storage_identity(
                    entry.state.master
                )
                if collect_audits:
                    source_l2_after_index = scalar_index(
                        torch.linalg.vector_norm(entry.state.master)
                    )
        else:
            if grad_sum is None:
                raise RuntimeError(
                    f"active gradient for {entry.state.name!r} is missing"
                )
            fused = False
            update: torch.Tensor | None = None
            if (
                compact_update
                and not collect_audits
                and _adam_history_policy_for_layer(layer_idx) == "keep"
                and not isinstance(entry.active_columns, slice)
                and entry.state.exp_avg is not None
                and entry.state.exp_avg_sq is not None
                and triton_block_adam.can_fuse(
                    grad_sum,
                    entry.state.exp_avg,
                    entry.state.exp_avg_sq,
                    entry.source,
                    entry.active_columns,
                )
            ):
                entry.state.step += 1
                entry.state.v_step += 1
                update = triton_block_adam.fused_adam_step(
                    grad_sum,
                    entry.state.exp_avg,
                    entry.state.exp_avg_sq,
                    entry.source,
                    entry.active_columns,
                    step=entry.state.step,
                    lr=lr,
                    grad_clip=grad_clip,
                    grad_scale=(
                        1.0
                        if gradients_are_means
                        else 1.0 / float(global_count)
                    ),
                    update_source=not entry.is_current,
                )
                fused = True
            else:
                grad = grad_sum.detach().float()
                if not gradients_are_means:
                    grad = grad / float(global_count)
                update = _adam_update_selected(
                    entry.state,
                    grad,
                    entry.active_columns,
                    lr=lr,
                    grad_clip=grad_clip,
                    compact_update=compact_update,
                    layer_idx=layer_idx,
                )
            if entry.is_current:
                if update is None:
                    raise RuntimeError("current fused Adam returned no update")
                current_update = update
            else:
                if entry.state.master is None:
                    raise RuntimeError(
                        f"future weight {entry.state.name!r} lost its "
                        "FP32 master"
                    )
                if tensor_storage_identity(
                    entry.state.master
                ) != entry.source_storage_id:
                    raise RuntimeError(
                        f"{entry.scope}:{entry.state.name} no longer "
                        "updates the backward source master"
                    )
                if not fused:
                    if update is None:
                        raise RuntimeError("future Adam returned no update")
                    entry.state.master.sub_(update)
                storage_after = tensor_storage_identity(
                    entry.state.master
                )
                if collect_audits:
                    source_l2_after_index = scalar_index(
                        torch.linalg.vector_norm(entry.state.master)
                    )
        if entry.active_columns is None:
            active_column_count = entry.state.shape[1]
        elif isinstance(entry.active_columns, slice):
            active_column_count = len(
                range(*entry.active_columns.indices(entry.state.shape[1]))
            )
        else:
            active_column_count = int(entry.active_columns.numel())
        if collect_audits:
            assert update is not None
            pending_audits.append(
            {
                "entry": entry,
                "used": bool(used),
                "active_column_count": active_column_count,
                "storage_after": storage_after,
                "step_before": step_before,
                "step_after": int(entry.state.step),
                "update_l2_index": scalar_index(
                    torch.linalg.vector_norm(update)
                ),
                "source_l2_before_index": source_l2_before_index,
                "source_l2_after_index": source_l2_after_index,
            }
            )
    if current_update is None:
        raise RuntimeError("block refresh did not produce a current-weight update")
    if not collect_audits:
        return current_update, ()
    scalar_values = (
        torch.stack(scalar_tensors).cpu().tolist()
        if scalar_tensors
        else []
    )
    audits: list[ActiveWeightAudit] = []
    for pending in pending_audits:
        entry = pending["entry"]
        assert isinstance(entry, _ActiveWeight)
        used = bool(pending["used"])
        update_l2 = float(
            scalar_values[int(pending["update_l2_index"])]
        )
        source_l2_before = float(
            scalar_values[int(pending["source_l2_before_index"])]
        )
        after_index = pending["source_l2_after_index"]
        source_l2_after = (
            None
            if after_index is None
            else float(scalar_values[int(after_index)])
        )
        if not math.isfinite(update_l2) or not math.isfinite(
            source_l2_before
        ) or (
            source_l2_after is not None
            and not math.isfinite(source_l2_after)
        ):
            raise RuntimeError(
                f"non-finite Block-GD update audit for "
                f"{entry.scope}:{entry.state.name}"
            )
        audits.append(
            ActiveWeightAudit(
                scope=entry.scope,
                name=entry.state.name,
                parameter_name=entry.state.parameter_name,
                is_current=entry.is_current,
                used=used,
                active_column_count=int(
                    pending["active_column_count"]
                ),
                source_storage_id=entry.source_storage_id,
                storage_id_after=pending["storage_after"],
                optimizer_step_before=int(pending["step_before"]),
                optimizer_step_after=int(pending["step_after"]),
                update_applied=bool(used and not entry.is_current),
                update_l2=update_l2,
                source_l2_before=source_l2_before,
                source_l2_after=source_l2_after,
            )
        )
    return current_update, tuple(audits)


def _make_single_linear_grad_refresh_fn_legacy(
    *,
    layer: "nn.Module",
    module: "nn.Module",
    layer_state: "LayerInputs",
    fp_out_for_this_layer: torch.Tensor,
    fisher: torch.Tensor,
    ctx: RefreshContext,
    # loss_slide_window kwargs (Stage 2). When ``next_layer`` is None the
    # closure uses ONLY the current-layer fisher_mse; when it's provided
    # the closure linearly blends current-layer fisher_mse with next-layer
    # fisher_mse via ``slide_alpha_fn()``. The slide α schedule spans the
    # WHOLE transformer block (sum of refreshes across all four module
    # groups in the layer), so the caller passes a closure that returns
    # the CURRENT α and advances its own cumulative refresh counter.
    next_layer: "nn.Module | None" = None,
    next_fp_out: "torch.Tensor | None" = None,
    next_fisher: "torch.Tensor | None" = None,
    slide_alpha_fn: "Callable[[], float] | None" = None,
    # Outlier clip on the refresh-loss delta. ``1.0`` (default) disables it
    # — matches old GPTQ+ when ``--a_loss_ratio 1.0`` (the default).
    # ratio < 1.0 caps the top (1 - ratio) fraction of |delta|; see
    # :func:`realq.refresh.fisher_loss._scale_delta_by_abs_quantile`.
    a_loss_ratio: float = 1.0,
    # ``global_refresh`` is world/chunk invariant but requires a graph-free
    # prepass and distributed exact percentile. ``local_backward_chunk`` is
    # the historical paper-code behavior and computes P95 in each loss call.
    a_loss_clip_scope: str = "local_backward_chunk",
) -> Callable[..., torch.Tensor]:
    """Build the per-block refresh closure for one linear.

    Closure signature::

        update = refresh(stitched_weight_fp32, trailing_col_start)

    where ``stitched_weight_fp32`` is the FULL (rows, columns) fp32 weight
    that the autograd forward should see — the caller stitches Q for
    already-quantised columns and the working fp32 W for the rest. The
    closure casts it to module dtype for forward via ``functional_call``,
    runs autograd over ``backward_samples`` total samples partitioned into
    gradient-accumulation chunks of at most ``backward_bsz``,
    runs Adam on the trailing column slice of the gradient, and RETURNS
    the fp32 update tensor (rows, trailing_cols).

    The closure does NOT touch ``module.weight`` and does NOT mutate any
    tensor on the caller's side — by handing back the fp32 update we let
    the caller apply it to its fp32 working W master, sidestepping the
    bf16 round-trip that was killing tiny per-step Adam updates (lr~1e-8
    falls below bf16 ULP at typical Qwen3 weight magnitudes ~1e-2).

    Returns ``None`` when no samples were produced for this rank (degenerate).
    """
    inps = layer_state.inps
    am = layer_state.attention_mask
    pi = layer_state.position_ids
    pe = layer_state.position_embeddings
    weight_name = _functional_weight_name(layer, module)
    if a_loss_clip_scope not in (
        "global_refresh",
        "local_backward_chunk",
    ):
        raise ValueError(
            "a_loss_clip_scope must be 'global_refresh' or "
            "'local_backward_chunk', "
            f"got {a_loss_clip_scope!r}."
        )

    def refresh(
        stitched_weight_fp32: torch.Tensor,
        trailing_col_start: int,
        perm: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        transfer_target_layer = os.environ.get(
            "REALQ_ADAM_V_TRANSFER_TARGET_LAYER"
        )
        transfer_target_module = os.environ.get(
            "REALQ_ADAM_V_TRANSFER_TARGET_MODULE"
        )
        if (
            os.environ.get("REALQ_ADAM_V_TRANSFER_MODE")
            and transfer_target_layer is not None
            and transfer_target_module == str(ctx.trace_module)
            and transfer_target_layer == str(ctx.trace_layer)
            and ctx.refresh_step == 0
        ):
            logging.info(
                "[adam_v_transfer_entry] layer=%s module=%s "
                "refresh=0 mode=%s env_dir=%s m_step=%d v_step=%d "
                "active_suffix_start=%d active_suffix_width=%d",
                ctx.trace_layer,
                ctx.trace_module,
                os.environ.get("REALQ_ADAM_V_TRANSFER_MODE"),
                os.environ.get("REALQ_ADAM_TRANSFER_DIAG_DIR"),
                ctx.adam_step,
                ctx.adam_v_step,
                trailing_col_start,
                stitched_weight_fp32.shape[1] - trailing_col_start,
            )
        # Match old GPTQ+ ``collect_true_weight_gradient`` (lines 6500-6726)
        # under ``--dp_global_shuffle=True``:
        #   selected_global = scheduler.next_indices()  # backward_samples global ids
        #   selected_local = [gi - rank_start for gi in selected_global
        #                     if rank_start <= gi < rank_end]   # filter to shard
        #   for start in range(0, len(selected_local), backward_bsz):
        #       batch = selected_local[start:start+backward_bsz]
        #       grad_i = autograd.grad(loss(batch), override)
        #       partial_grad_sum += batch_size * grad_i      # SUM over samples
        #       partial_count += batch_size
        #   allreduce(partial_grad_sum, partial_count)       # ALWAYS, even if 0
        #   grad = partial_grad_sum / global_count           # mean over global samples
        # Multiplying by batch_size before adding (and dividing by total
        # count after) reproduces a sample-wise mean WITHOUT relying on a
        # per-batch ``mean`` op being commutative across batches in fp32.
        #
        # Adam step counter advances on EVERY rank for every refresh,
        # regardless of local sample count, so per-rank Adam bias-correction
        # stays in sync — required because ``allreduce`` makes ``accum_grad``
        # identical on all ranks, and we want the resulting Adam update to
        # be identical too.
        with nvtx.nvtx_range("refresh.setup"):
            ctx.adam_step += 1
            ctx.adam_v_step += 1
            ctx.refresh_step += 1
            if (
                ctx.refresh_step == 2
                and ctx._adam_v_transfer_applied
                and os.environ.get("REALQ_ADAM_V_TRANSFER_MODE")
                and os.environ.get("REALQ_ADAM_V_TRANSFER_TARGET_LAYER")
                == str(ctx.trace_layer)
                and os.environ.get("REALQ_ADAM_V_TRANSFER_TARGET_MODULE")
                == str(ctx.trace_module)
            ):
                logging.info(
                    "[adam_v_transfer_age] layer=%s module=%s refresh=1 "
                    "m_step=%d v_step=%d",
                    ctx.trace_layer,
                    ctx.trace_module,
                    ctx.adam_step,
                    ctx.adam_v_step,
                )
            selected_global = ctx.next_indices()
            # GLOBAL → LOCAL filter. ``inps`` is this rank's contiguous shard
            # of size ``n_local`` starting at ``rank * n_local`` in the global
            # ``[0, nsamples)`` index space (matches dp_shard slicing at
            # gptq_plus_utils.py:7802 and realq layer-0 capture upstream).
            rank = dist_utils.get_rank()
            n_local = inps.shape[0]
            rank_start = rank * n_local
            rank_end = rank_start + n_local
            selected = [gi - rank_start for gi in selected_global if rank_start <= gi < rank_end]
            # slide_alpha is computed ONCE per refresh CALL (matching old GPTQ+
            # which passes a single ``slide_alpha`` from fasterquant into
            # ``gradient_refresh_fn``). Per mini-batch ``slide_alpha_fn()``
            # would advance the cursor multiple times per refresh and produce
            # the wrong α schedule across modules. Computed BEFORE the empty-
            # local-shard short-circuit so the α schedule advances at the same
            # cadence on every rank — ``slide_alpha_fn()`` carries cumulative
            # state and must tick once per refresh CALL on each rank.
            slide_alpha: float | None = None
            if next_layer is not None and slide_alpha_fn is not None:
                slide_alpha = float(slide_alpha_fn())
            # Pre-allocate the gradient sum so the empty-local-shard case can
            # still participate in the cross-rank allreduce with a valid zeros
            # tensor (matches old ``partial_grad_sum = torch.zeros_like(
            # override_weight, dtype=fp32)`` at gptq_plus_utils.py:6434, which
            # is constructed BEFORE the ``if len(selected_indices) > 0`` guard).
            partial_grad_sum = torch.zeros_like(stitched_weight_fp32)
            partial_count = 0
            partial_loss_sums = (
                torch.zeros(
                    (
                        3
                        if slide_alpha is not None and slide_alpha < 1.0
                        else 1
                    ),
                    # Legacy accumulates local loss sums as Python floats
                    # (binary64), then casts only when a distributed packed
                    # all-reduce is needed. Mirror that precision here.
                    dtype=torch.float64,
                    device=partial_grad_sum.device,
                )
                if ctx.loss_observation_enabled else None
            )
            # Cast the stitched fp32 weight to module dtype ONCE per refresh
            # and pass it to functional_call for every backward batch. Old
            # GPTQ+ does the same downcast inside ``collect_true_weight_gradient``
            # line 6422-6432 (`.to(target_dev, dtype=target_dtype)`) and reuses
            # the resulting bf16 leaf tensor across all per-batch backwards.
            # Hoisting the cast saves a fresh (rows, columns) bf16 alloc + cast
            # per backward batch (4-8 batches per refresh × 6 modules ×
            # n_layers refreshes adds up). ``autograd.grad`` doesn't touch
            # ``.grad``, so reusing one leaf across multiple backwards is safe.
            override_dtype = module.weight.data.dtype
            override_weight = stitched_weight_fp32.to(override_dtype).requires_grad_(True)

        # In ``global_refresh`` mode, ``a_loss_ratio`` is defined over the
        # complete GLOBAL sample set for this refresh. Run a graph-free
        # prepass, gather |delta| values, and broadcast one exact cap used by
        # every backward accumulation chunk. ``local_backward_chunk``
        # deliberately skips this block: fisher_mse_loss receives
        # threshold=None below and reproduces the historical rank-/chunk-local
        # percentile in-line.
        a_loss_threshold = None
        next_a_loss_threshold = None
        if (
            a_loss_ratio < 1.0
            and a_loss_clip_scope == "global_refresh"
        ):
            local_abs_delta: list[torch.Tensor] = []
            local_abs_next_delta: list[torch.Tensor] = []
            with torch.no_grad(), nvtx.nvtx_range(
                "refresh.activation_clip_prepass"
            ):
                for start in range(0, len(selected), ctx.backward_bsz):
                    batch_idx = selected[
                        start : start + ctx.backward_bsz
                    ]
                    batch_size = len(batch_idx)
                    sample_idx = torch.tensor(
                        batch_idx,
                        dtype=torch.long,
                        device=inps.device,
                    )
                    x = inps.index_select(0, sample_idx)
                    fp_target = fp_out_for_this_layer.index_select(
                        0, sample_idx
                    ).to(x.device)
                    kw = {}
                    if am is not None:
                        kw["attention_mask"] = (
                            am.expand(batch_size, *am.shape[1:])
                            if am.shape[0] != batch_size
                            else am
                        )
                    if pi is not None:
                        kw["position_ids"] = (
                            pi.expand(batch_size, -1)
                            if pi.shape[0] != batch_size
                            else pi
                        )
                    if pe is not None:
                        kw["position_embeddings"] = (
                            (
                                pe[0].expand(
                                    batch_size, *pe[0].shape[1:]
                                )
                                if pe[0].shape[0] != batch_size
                                else pe[0]
                            ),
                            (
                                pe[1].expand(
                                    batch_size, *pe[1].shape[1:]
                                )
                                if pe[1].shape[0] != batch_size
                                else pe[1]
                            ),
                        )
                    out = functional_call(
                        layer,
                        {weight_name: override_weight},
                        (x,),
                        kw,
                        strict=False,
                    )
                    q_out = out[0] if isinstance(out, tuple) else out
                    local_abs_delta.append(
                        (q_out - fp_target).float().abs().reshape(-1)
                    )
                    if (
                        slide_alpha is not None
                        and slide_alpha < 1.0
                    ):
                        next_pkg = next_layer(q_out, **kw)
                        next_q_out = (
                            next_pkg[0]
                            if isinstance(next_pkg, tuple)
                            else next_pkg
                        )
                        fp_target_next = next_fp_out.index_select(
                            0, sample_idx
                        ).to(next_q_out.device)
                        local_abs_next_delta.append(
                            (next_q_out - fp_target_next)
                            .float()
                            .abs()
                            .reshape(-1)
                        )
            empty = torch.empty(
                0,
                dtype=torch.float32,
                device=override_weight.device,
            )
            local_values = (
                torch.cat(local_abs_delta)
                if local_abs_delta
                else empty
            )
            a_loss_threshold = global_percentile(
                local_values, float(a_loss_ratio)
            ).detach()
            if slide_alpha is not None and slide_alpha < 1.0:
                local_next_values = (
                    torch.cat(local_abs_next_delta)
                    if local_abs_next_delta
                    else empty
                )
                next_a_loss_threshold = global_percentile(
                    local_next_values, float(a_loss_ratio)
                ).detach()
            del local_abs_delta, local_abs_next_delta, local_values
        iter_idx = 0
        for start in range(0, len(selected), ctx.backward_bsz):
            with nvtx.nvtx_range(f"refresh.iter_{iter_idx}"):
                batch_idx = selected[start : start + ctx.backward_bsz]
                batch_size = len(batch_idx)
                sample_idx = torch.tensor(batch_idx, dtype=torch.long, device=inps.device)
                x = inps.index_select(0, sample_idx)
                fp_target = fp_out_for_this_layer.index_select(0, sample_idx).to(x.device)
                kw = {}
                if am is not None:
                    kw["attention_mask"] = am.expand(batch_size, *am.shape[1:]) if am.shape[0] != batch_size else am
                if pi is not None:
                    kw["position_ids"] = pi.expand(batch_size, -1) if pi.shape[0] != batch_size else pi
                if pe is not None:
                    kw["position_embeddings"] = (
                        pe[0].expand(batch_size, *pe[0].shape[1:]) if pe[0].shape[0] != batch_size else pe[0],
                        pe[1].expand(batch_size, *pe[1].shape[1:]) if pe[1].shape[0] != batch_size else pe[1],
                    )
                with torch.enable_grad():
                    with nvtx.nvtx_range("refresh.forward"):
                        out = functional_call(
                            layer,
                            {weight_name: override_weight},
                            (x,),
                            kw,
                            strict=False,
                        )
                        q_out = out[0] if isinstance(out, tuple) else out
                    with nvtx.nvtx_range("refresh.loss"):
                        loss_curr = fisher_mse_loss(
                            q_out,
                            fp_target,
                            fisher,
                            a_loss_ratio=a_loss_ratio,
                            a_loss_threshold=a_loss_threshold,
                        )
                        loss_next = None
                        # Old GPTQ+ ``collect_true_weight_gradient`` only triggers the
                        # next-layer arm when ``slide_alpha < 1.0`` (line 6450).
                        # At α=1.0 it skips the blend entirely:
                        #     refresh_loss = refresh_loss_current
                        # Doing the explicit ``1.0*curr + 0.0*next`` blend would
                        # introduce fp32 rounding diffs and break bit-exactness.
                        if slide_alpha is not None and slide_alpha < 1.0:
                            with nvtx.nvtx_range("refresh.next_layer_forward"):
                                next_q_out_pkg = next_layer(q_out, **kw)
                                next_q_out = next_q_out_pkg[0] if isinstance(next_q_out_pkg, tuple) else next_q_out_pkg
                            fp_target_next = next_fp_out.index_select(0, sample_idx).to(next_q_out.device)
                            loss_next = fisher_mse_loss(
                                next_q_out,
                                fp_target_next,
                                next_fisher,
                                a_loss_ratio=a_loss_ratio,
                                a_loss_threshold=next_a_loss_threshold,
                            )
                            loss = slide_alpha * loss_curr + (1.0 - slide_alpha) * loss_next
                        else:
                            loss = loss_curr
                    with nvtx.nvtx_range("refresh.backward"):
                        (batch_grad,) = torch.autograd.grad(loss, override_weight, retain_graph=False)
                with nvtx.nvtx_range("refresh.accumulate"):
                    batch_grad_fp32 = batch_grad.detach().float()
                    partial_grad_sum.add_(batch_grad_fp32, alpha=float(batch_size))
                    partial_count += batch_size
                    if partial_loss_sums is not None:
                        partial_loss_sums[0].add_(
                            loss.detach(), alpha=float(batch_size),
                        )
                        if partial_loss_sums.numel() == 3:
                            partial_loss_sums[1].add_(
                                loss_curr.detach(),
                                alpha=float(batch_size),
                            )
                        if loss_next is not None and partial_loss_sums.numel() == 3:
                            partial_loss_sums[2].add_(
                                loss_next.detach(),
                                alpha=float(batch_size),
                            )
                iter_idx += 1
        # All-reduce per-rank partial sum + count, then divide. ALWAYS
        # runs on every rank (even with partial_count == 0) so NCCL stays
        # in lock-step. Matches old GPTQ+ ``make_gradient_refresh_fn``
        # (lines 9026-9057) — packed all-reduce of (grad_flat, count) so
        # summation order is identical across runs.
        with nvtx.nvtx_range("refresh.grad_allreduce"):
            global_count, global_loss_sums = _aggregate_refresh_sums(
                partial_grad_sum,
                partial_count,
                partial_loss_sums if ctx.trace_enabled else None,
            )
            accum_grad = partial_grad_sum / float(global_count)
        if ctx.log_column_block_loss and not ctx.trace_enabled:
            if partial_loss_sums is None:
                raise RuntimeError(
                    "column-block loss logging enabled without loss sums"
                )
            global_loss_sums = _aggregate_loss_sums_for_logging(
                partial_loss_sums
            )
        if global_loss_sums is not None:
            ctx.record_loss_observation(
                global_loss_sums=global_loss_sums,
                global_count=global_count,
                sample_indices=selected_global,
                slide_alpha=slide_alpha,
                has_next_loss=(
                    slide_alpha is not None and slide_alpha < 1.0
                ),
                objective="fisher_mse",
            )
        with nvtx.nvtx_range("refresh.adam_step"):
            # act_order: re-key the natural-order grad into PERMUTED column order
            # so the Adam state slice [:, trailing_col_start:] sees only the
            # not-yet-quantised columns. Old GPTQ+ does the equivalent at
            # gptq_plus_utils.py:3038-3050 (``refreshed_grad_sub[:, state["perm"]]``
            # then sliced from i2 inside ``_compute_grad_optimizer_update``).
            # Without this re-keying, Adam state evolves for every natural-order
            # column on every refresh — including columns that map to ALREADY-
            # quantised permuted positions [0..i2) — and the resulting trailing
            # update diverges from the legacy reference by ~1-7e-2 per quant
            # bin after a few refreshes. ``ctx.exp_avg`` and ``ctx.exp_avg_sq``
            # are interpreted in the SAME (permuted) coordinate frame as the
            # incoming grad: at init they are zeros so the frame choice doesn't
            # matter; after the first refresh, every access uses permuted
            # indexing, mirroring old GPTQPlus subgroup state which was created
            # AFTER ``W_sub = W_sub[:, perm]``.
            if perm is not None:
                accum_grad = accum_grad[:, perm]
            history_policy = _adam_history_policy_for_layer(ctx.trace_layer)
            reset_m = history_policy in (
                "reset_each_refresh",
                "reset_m_each_refresh",
                "reset_m_and_m_age_each_refresh",
            )
            reset_v = history_policy in (
                "reset_each_refresh",
                "reset_v_each_refresh",
                "reset_v_and_v_age_each_refresh",
            )
            reset_v_age = history_policy == "reset_v_and_v_age_each_refresh"
            reset_m_age = history_policy == "reset_m_and_m_age_each_refresh"
            reset_age = history_policy in (
                "reset_each_refresh", "reset_bias_age_each_refresh"
            )
            if history_policy != "keep":
                if reset_m:
                    ctx.exp_avg[:, trailing_col_start:].zero_()
                if reset_v:
                    ctx.exp_avg_sq[:, trailing_col_start:].zero_()
                if reset_v_age:
                    # m keeps its real module age; only v is freshly
                    # bias-corrected for this refresh (v_hat = clipped g^2).
                    ctx.adam_v_step = 1
                if reset_m_age:
                    # m is zeroed only on the active floating suffix. The
                    # current gradient is its fresh first observation; retain
                    # the historical v estimate and its independent age.
                    ctx.adam_step = 1
                if reset_age:
                    # refresh() increments the counter at entry, so initialize
                    # its diagnostic/bias-correction age to zero here.
                    ctx.adam_step = 1
                    ctx.adam_v_step = 1
                if not getattr(ctx, "_history_ablation_logged", False):
                    logging.info(
                        "[adam_history_ablation] policy=%s layer=%s module=%s",
                        history_policy,
                        ctx.trace_layer,
                        ctx.trace_module,
                    )
                    ctx._history_ablation_logged = True
            # Slice trailing columns and grad-clip (per-element clamp; matches
            # old ``_compute_grad_optimizer_update_batched`` line 1105-1106).
            grad_slice = accum_grad[:, trailing_col_start:]
            if ctx.grad_clip > 0:
                grad_slice = grad_slice.clamp(min=-ctx.grad_clip, max=ctx.grad_clip)
            # Adam moments on the trailing slice. Match old order:
            #   denom = sqrt(ev) / sqrt(bc2) + eps  (NOT sqrt(ev / bc2))
            ea = ctx.exp_avg[:, trailing_col_start:]
            ev = ctx.exp_avg_sq[:, trailing_col_start:]
            transfer_info = None
            if os.environ.get("REALQ_ADAM_V_TRANSFER_MODE"):
                from realq.refresh.adam_transfer_diagnostic import (
                    initialize_v_from_previous_row_profile,
                )

                transfer_info = initialize_v_from_previous_row_profile(
                    layer=int(ctx.trace_layer),
                    module=str(ctx.trace_module),
                    refresh_index=ctx.refresh_step - 1,
                    grad=grad_slice,
                    exp_avg_sq=ev,
                    beta2=ctx.beta2,
                )
                if transfer_info is not None:
                    if ctx.adam_step != 1:
                        raise RuntimeError(
                            "normalized row-prior transfer is only valid at "
                            "the target module's first local Adam step; got "
                            f"{ctx.adam_step} for layer={ctx.trace_layer} "
                            f"module={ctx.trace_module}"
                        )
                    logging.info(
                        "[adam_v_transfer] layer=%s module=%s "
                        "donor_layer=%d refresh=%d corr=%.5f "
                        "donor_to_target_scale=%.6g shrinkage=%.4f "
                        "m_step=%d virtual_v_step=%d",
                        ctx.trace_layer,
                        ctx.trace_module,
                        int(ctx.trace_layer) - 1,
                        ctx.refresh_step - 1,
                        float(transfer_info["correlation"]),
                        float(transfer_info["donor_to_target_scale"]),
                        float(transfer_info["shrinkage"]),
                        ctx.adam_step,
                        int(transfer_info["target_v_step"]),
                    )
                    ctx.adam_v_step = int(transfer_info["target_v_step"])
                    ctx._adam_v_transfer_applied = True
            probe_rule = _adam_probe_target_rule(ctx)
            _log_adam_probe_snapshot(
                ctx=ctx,
                rule=probe_rule,
                weight=stitched_weight_fp32[:, trailing_col_start:],
                grad=grad_slice,
                exp_avg_before=ea,
                exp_avg_sq_before=ev,
            )
            adopt_first_step = history_policy == "adopt" and ctx.adam_step == 1
            if history_policy == "adopt":
                if adopt_first_step:
                    normalized_grad = grad_slice / grad_slice.abs().clamp_min(
                        ctx.eps
                    )
                    ea.copy_(normalized_grad)
                    ev.copy_(grad_slice.square())
                else:
                    normalized_grad = grad_slice / ev.sqrt().clamp_min(ctx.eps)
                    ea.mul_(ctx.beta1).add_(
                        normalized_grad, alpha=1.0 - ctx.beta1
                    )
                    ev.mul_(ctx.beta2).addcmul_(
                        grad_slice, grad_slice, value=1.0 - ctx.beta2
                    )
            else:
                ea.mul_(ctx.beta1).add_(grad_slice, alpha=1.0 - ctx.beta1)
                second_moment_input = (
                    grad_slice - ea
                    if history_policy == "adabelief"
                    else grad_slice
                )
                ev.mul_(ctx.beta2).addcmul_(
                    second_moment_input,
                    second_moment_input,
                    value=1.0 - ctx.beta2,
                )
            bc1 = 1.0 - ctx.beta1 ** ctx.adam_step
            v_step = (
                int(transfer_info["target_v_step"])
                if transfer_info is not None
                else ctx.adam_v_step
            )
            bc2 = 1.0 - ctx.beta2 ** v_step
            denom = ev.sqrt() / math.sqrt(bc2)
            denom.add_(ctx.eps)
            step_size = ctx.layer_lr / bc1
            if history_policy == "adopt":
                update = ctx.layer_lr * ea
            elif history_policy == "radam":
                rho_inf = 2.0 / (1.0 - ctx.beta2) - 1.0
                beta2_t = ctx.beta2 ** v_step
                rho_t = (
                    rho_inf
                    - 2.0 * v_step * beta2_t / (1.0 - beta2_t)
                )
                if rho_t > 5.0:
                    rect = math.sqrt(
                        (rho_t - 4.0)
                        * (rho_t - 2.0)
                        * rho_inf
                        / (
                            (rho_inf - 4.0)
                            * (rho_inf - 2.0)
                            * rho_t
                        )
                    )
                    update = step_size * rect * (ea / denom)
                else:
                    update = step_size * ea
            else:
                update = step_size * (ea / denom)
            if transfer_info is not None:
                # Compare the candidate preconditioner with the exact base
                # Adam first-step estimator vhat=g^2. This is diagnostic-only
                # and runs only at the explicitly selected transfer target.
                with torch.no_grad():
                    grad32 = grad_slice.detach().float()
                    base_vhat = grad32.square()
                    candidate_vhat = ev.detach().float() / max(bc2, 1e-30)
                    base_update = ctx.layer_lr * grad32 / (
                        grad32.abs() + ctx.eps
                    )
                    candidate_update = update.detach().float()
                    base_norm = torch.linalg.vector_norm(base_update).clamp_min(1e-30)
                    candidate_norm = torch.linalg.vector_norm(candidate_update).clamp_min(1e-30)
                    update_cosine = torch.sum(base_update * candidate_update) / (
                        base_norm * candidate_norm
                    )
                    relative_update_change = torch.linalg.vector_norm(
                        candidate_update - base_update
                    ) / base_norm
                    vhat_ratio = candidate_vhat / base_vhat.clamp_min(1e-30)
                    within_row = base_vhat - base_vhat.mean(dim=1, keepdim=True)
                    within_row_cv = torch.sqrt(
                        within_row.square().mean()
                        / base_vhat.mean().square().clamp_min(1e-30)
                    )
                    logging.info(
                        "[adam_v_transfer_effect] layer=%s module=%s "
                        "refresh=%d update_cosine=%.8g "
                        "relative_update_l2_change=%.8g "
                        "coordinate_vhat_ratio_p05=%.8g "
                        "coordinate_vhat_ratio_median=%.8g "
                        "coordinate_vhat_ratio_p95=%.8g "
                        "target_within_row_g2_cv=%.8g",
                        ctx.trace_layer,
                        ctx.trace_module,
                        ctx.refresh_step - 1,
                        float(update_cosine.item()),
                        float(relative_update_change.item()),
                        float(torch.quantile(vhat_ratio, 0.05).item()),
                        float(torch.quantile(vhat_ratio, 0.50).item()),
                        float(torch.quantile(vhat_ratio, 0.95).item()),
                        float(within_row_cv.item()),
                    )
            if os.environ.get("REALQ_ADAM_TRANSFER_DIAG_DIR"):
                from realq.refresh.adam_transfer_diagnostic import record_refresh

                record_refresh(
                    layer=int(ctx.trace_layer),
                    module=str(ctx.trace_module),
                    refresh_index=ctx.refresh_step - 1,
                    adam_step=ctx.adam_v_step,
                    grad=grad_slice,
                    exp_avg_sq=ev,
                    beta2=ctx.beta2,
                )
            if os.environ.get("REALQ_ADAM_MOMENT_DRIFT_DIAG_DIR"):
                from realq.refresh.adam_transfer_diagnostic import record_moment_drift

                record_moment_drift(
                    layer=int(ctx.trace_layer),
                    module=str(ctx.trace_module),
                    refresh_index=ctx.refresh_step - 1,
                    adam_step=ctx.adam_v_step,
                    grad=grad_slice,
                    exp_avg_sq=ev,
                    beta2=ctx.beta2,
                )
            if _is_adam_probe_target(ctx):
                probe_reference_update = update.detach().float().clone()
                product = update.detach().float() * grad_slice.detach().float()
                product_abs = product.abs()
                weight_sum = product_abs.sum().clamp_min(1e-30)
                cautious_mask = product > 0
                cautious_scale = cautious_mask.numel() / (cautious_mask.sum().item() + 1.0)
                cos_d_g = torch.sum(
                    update.detach().float() * grad_slice.detach().float()
                ) / (
                    torch.linalg.vector_norm(update.detach().float())
                    * torch.linalg.vector_norm(grad_slice.detach().float())
                ).clamp_min(1e-30)
                logging.info(
                    "[adam_probe_direction] layer=%s module=%s block=%d "
                    "cos_descent_update_grad=%.9g conflict_fraction=%.9g "
                    "conflict_update_mass_fraction=%.9g adam_update_l2=%.9g "
                    "grams_update_l2=%.9g zero_gradient_fraction=%.9g "
                    "cautious_retained_fraction=%.9g cautious_scale=%.9g",
                    ctx.trace_layer,
                    ctx.trace_module,
                    ctx.refresh_step - 1,
                    float(cos_d_g.item()),
                    float((product < 0).float().mean().item()),
                    float(product_abs[product < 0].sum().div(weight_sum).item()),
                    float(torch.linalg.vector_norm(update.detach().float()).item()),
                    float(
                        torch.linalg.vector_norm(
                            torch.sign(grad_slice.detach().float())
                            * update.detach().float().abs()
                        ).item()
                    ),
                    float((grad_slice == 0).float().mean().item()),
                    float(cautious_mask.float().mean().item()),
                    float(cautious_scale),
                )
            transition_observer = getattr(refresh, "_adam_transition_observer", None)
            if transition_observer is not None and _is_adam_probe_target(ctx):
                transition_observer.capture_refresh(
                    inps=inps, fp_target=fp_out_for_this_layer,
                    fisher=fisher, next_fp_target=next_fp_out,
                    next_fisher=next_fisher, selected=selected,
                    kw=kw, weight_name=weight_name, slide_alpha=slide_alpha,
                    a_loss_ratio=a_loss_ratio, a_loss_threshold=a_loss_threshold,
                    next_a_loss_threshold=next_a_loss_threshold,
                    raw_gradient=accum_grad[:, trailing_col_start:],
                    clipped_gradient=grad_slice, adam_update=update,
                    adam_step=ctx.adam_step, grad_clip=ctx.grad_clip,
                    actual_loss=float(global_loss_sums[0].item() / global_count),
                    module_dtype=str(module.weight.dtype),
                    layer_reference_state=layer.state_dict(),
                )
            if probe_rule == "grams_direction":
                update = torch.sign(grad_slice) * update.abs()
            elif probe_rule == "norm_matched_gradient":
                update = _norm_matched_gradient_update(update, grad_slice)
            elif probe_rule in (
                "reset_m",
                "reset_m_direction_norm_matched_adam",
                "adam_direction_norm_matched_reset_m",
            ):
                # At this refresh, a correctly reset m and m-age produce
                # m_hat=g while retaining the same post-EMA v and v-age.
                reset_m_update = ctx.layer_lr * grad_slice / denom
                if probe_rule == "reset_m":
                    update = reset_m_update
                elif probe_rule == "reset_m_direction_norm_matched_adam":
                    update = _match_update_l2_norm(reset_m_update, update)
                else:
                    update = _match_update_l2_norm(update, reset_m_update)
            elif probe_rule in ("cautious", "cautious_mask", "cautious_scale"):
                mask = (update * grad_slice) > 0
                if probe_rule != "cautious_scale":
                    update = update * mask.to(dtype=update.dtype)
                if probe_rule != "cautious_mask":
                    update.mul_(mask.numel() / (mask.sum().item() + 1.0))
            elif probe_rule == "no_parameter_update":
                update.zero_()
            if _is_adam_probe_target(ctx):
                applied = update.detach().float()
                reference = probe_reference_update
                applied_norm = torch.linalg.vector_norm(applied)
                reference_norm = torch.linalg.vector_norm(reference)
                cosine = torch.sum(applied * reference) / (
                    applied_norm * reference_norm
                ).clamp_min(1e-30)
                logging.info(
                    "[adam_probe_applied] layer=%s module=%s block=%d rule=%s "
                    "update_l2=%.9g adam_update_l2=%.9g "
                    "norm_ratio_to_adam=%.9g cosine_to_adam=%.9g "
                    "relative_delta_l2=%.9g",
                    ctx.trace_layer,
                    ctx.trace_module,
                    ctx.refresh_step - 1,
                    probe_rule,
                    float(applied_norm.item()),
                    float(reference_norm.item()),
                    float((applied_norm / reference_norm.clamp_min(1e-30)).item()),
                    float(cosine.item()),
                    float(
                        torch.linalg.vector_norm(applied - reference)
                        / reference_norm.clamp_min(1e-30)
                    ),
                )
            # Return fp32 update — caller subtracts from its fp32 working W
            # master so the precision of the per-step delta survives even when
            # lr is below module-dtype ULP.
            return update

    refresh._realq_update_layout = "trailing_quant_order"
    return refresh


def _refresh_batch_kwargs(
    *,
    batch_size: int,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    position_embeddings: tuple | None,
) -> dict[str, object]:
    kw: dict[str, object] = {}
    if attention_mask is not None:
        kw["attention_mask"] = (
            attention_mask.expand(
                batch_size, *attention_mask.shape[1:]
            )
            if attention_mask.shape[0] != batch_size
            else attention_mask
        )
    if position_ids is not None:
        kw["position_ids"] = (
            position_ids.expand(batch_size, -1)
            if position_ids.shape[0] != batch_size
            else position_ids
        )
    if position_embeddings is not None:
        kw["position_embeddings"] = (
            (
                position_embeddings[0].expand(
                    batch_size, *position_embeddings[0].shape[1:]
                )
                if position_embeddings[0].shape[0] != batch_size
                else position_embeddings[0]
            ),
            (
                position_embeddings[1].expand(
                    batch_size, *position_embeddings[1].shape[1:]
                )
                if position_embeddings[1].shape[0] != batch_size
                else position_embeddings[1]
            ),
        )
    return kw


def make_grad_refresh_fn(
    *,
    layer: nn.Module,
    module: nn.Module,
    module_name: str | None = None,
    block_state: BlockRefreshState | None = None,
    layer_state: "LayerInputs",
    fp_out_for_this_layer: torch.Tensor,
    fisher: torch.Tensor,
    ctx: RefreshContext,
    next_layer: nn.Module | None = None,
    next_block_state: BlockRefreshState | None = None,
    next_fp_out: torch.Tensor | None = None,
    next_fisher: torch.Tensor | None = None,
    slide_alpha_fn: Callable[[], float] | None = None,
    a_loss_ratio: float = 1.0,
    a_loss_clip_scope: str = "local_backward_chunk",
) -> Callable[..., torch.Tensor]:
    """Build full-block Block-GD, retaining the legacy public call contract.

    The returned closure still fires at the current linear's column-block
    boundaries.  Its forward overrides every not-yet-quantized linear weight
    in the current transformer block with an FP32-master-derived leaf.  When
    the sliding arm is active, it does the same for the next transformer
    block.  One backward therefore produces and applies Adam updates to every
    quantisable weight reached by the graph, while already-quantized weights
    remain locked in ``module.weight``.

    With ``ctx.fused_block_adam`` the closure returns only the active suffix in
    GPTQ quant order.  The compatibility path returns the historical full,
    natural-column-order FP32 update.  ``RealQLayer`` understands both layouts.
    """

    if module_name is None or block_state is None:
        if (
            module_name is not None
            or block_state is not None
            or next_block_state is not None
        ):
            raise ValueError(
                "module_name and block_state must be supplied together; "
                "next_block_state is valid only for full-block refresh."
            )
        return _make_single_linear_grad_refresh_fn_legacy(
            layer=layer,
            module=module,
            layer_state=layer_state,
            fp_out_for_this_layer=fp_out_for_this_layer,
            fisher=fisher,
            ctx=ctx,
            next_layer=next_layer,
            next_fp_out=next_fp_out,
            next_fisher=next_fisher,
            slide_alpha_fn=slide_alpha_fn,
            a_loss_ratio=a_loss_ratio,
            a_loss_clip_scope=a_loss_clip_scope,
        )

    inps = layer_state.inps
    am = layer_state.attention_mask
    pi = layer_state.position_ids
    pe = layer_state.position_embeddings
    if a_loss_clip_scope not in (
        "global_refresh",
        "local_backward_chunk",
    ):
        raise ValueError(
            "a_loss_clip_scope must be 'global_refresh' or "
            f"'local_backward_chunk', got {a_loss_clip_scope!r}."
        )
    if next_layer is None:
        if next_block_state is not None:
            raise ValueError("next_block_state requires next_layer")
    elif next_block_state is None:
        raise ValueError("sliding next_layer requires next_block_state")

    def refresh(
        stitched_weight_fp32: torch.Tensor,
        trailing_col_start: int,
        perm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with nvtx.nvtx_range("refresh.setup"):
            backward_invocation_id = (
                ctx.allocate_backward_invocation_id()
            )
            selected_global = ctx.next_indices()
            rank = dist_utils.get_rank()
            n_local = inps.shape[0]
            rank_start = rank * n_local
            rank_end = rank_start + n_local
            selected = [
                global_index - rank_start
                for global_index in selected_global
                if rank_start <= global_index < rank_end
            ]
            slide_alpha = None
            if next_layer is not None and slide_alpha_fn is not None:
                slide_alpha = float(slide_alpha_fn())

            current_overrides, active = block_state.make_overrides(
                current_name=module_name,
                current_weight_fp32=stitched_weight_fp32,
                trailing_col_start=trailing_col_start,
                perm=perm,
                trace_scope="current_block",
            )
            next_overrides: dict[str, torch.Tensor] = {}
            if slide_alpha is not None and slide_alpha < 1.0:
                next_overrides, next_active = (
                    next_block_state.make_overrides(
                        trace_scope="next_block"
                    )
                )
                active.extend(next_active)
            leaves = [entry.leaf for entry in active]
            # The normal one-GPU protocol is exactly one 32-sample backward.
            # Its loss is already a batch mean, so the historical
            # ``bf16_grad.float() * 32 / 32`` accumulation is algebraically
            # redundant.  Keep that raw autograd tensor alive until the fused
            # Adam kernel consumes it.  Multi-rank and multi-chunk runs retain
            # the accumulation/all-reduce oracle below.
            direct_single_chunk = bool(
                ctx.fused_block_adam
                and dist_utils.get_world_size() == 1
                and 0 < len(selected) <= ctx.backward_bsz
            )
            partial_grad_sums: list[torch.Tensor | None] = (
                [None] * len(active)
                if direct_single_chunk
                else [
                    torch.zeros_like(entry.state.exp_avg)
                    for entry in active
                ]
            )
            partial_used = [False] * len(active)
            partial_count = 0
            backward_chunk_sizes: list[int] = []
            partial_loss_sums = (
                torch.zeros(
                    (
                        3
                        if slide_alpha is not None and slide_alpha < 1.0
                        else 1
                    ),
                    dtype=torch.float64,
                    device=stitched_weight_fp32.device,
                )
                if ctx.loss_observation_enabled
                else None
            )
            # The current module may already have received many updates while
            # it was a future weight. Diagnostics must expose that persistent
            # Adam step rather than restarting at one per linear.
            ctx.refresh_step += 1
            ctx.adam_step = block_state.expected_next_step(module_name)
            if (
                _adam_history_policy_for_layer(ctx.trace_layer)
                in (
                    "reset_each_refresh",
                    "reset_m_and_m_age_each_refresh",
                )
            ):
                # The experimental policy resets the state inside
                # _adam_update_selected. Keep loss/step diagnostics aligned
                # with the actual bias-correction age used by that update.
                ctx.adam_step = 1

        a_loss_threshold = None
        next_a_loss_threshold = None
        if (
            a_loss_ratio < 1.0
            and a_loss_clip_scope == "global_refresh"
        ):
            local_abs_delta: list[torch.Tensor] = []
            local_abs_next_delta: list[torch.Tensor] = []
            with torch.no_grad(), nvtx.nvtx_range(
                "refresh.activation_clip_prepass"
            ):
                for start in range(0, len(selected), ctx.backward_bsz):
                    batch_idx = selected[
                        start : start + ctx.backward_bsz
                    ]
                    batch_size = len(batch_idx)
                    sample_idx = torch.tensor(
                        batch_idx,
                        dtype=torch.long,
                        device=inps.device,
                    )
                    x = inps.index_select(0, sample_idx)
                    fp_target = fp_out_for_this_layer.index_select(
                        0, sample_idx
                    ).to(x.device)
                    kw = _refresh_batch_kwargs(
                        batch_size=batch_size,
                        attention_mask=am,
                        position_ids=pi,
                        position_embeddings=pe,
                    )
                    out = functional_call(
                        layer,
                        current_overrides,
                        (x,),
                        kw,
                        strict=False,
                    )
                    q_out = out[0] if isinstance(out, tuple) else out
                    local_abs_delta.append(
                        (q_out - fp_target).float().abs().reshape(-1)
                    )
                    if slide_alpha is not None and slide_alpha < 1.0:
                        next_pkg = functional_call(
                            next_layer,
                            next_overrides,
                            (q_out,),
                            kw,
                            strict=False,
                        )
                        next_q_out = (
                            next_pkg[0]
                            if isinstance(next_pkg, tuple)
                            else next_pkg
                        )
                        fp_target_next = next_fp_out.index_select(
                            0, sample_idx
                        ).to(next_q_out.device)
                        local_abs_next_delta.append(
                            (next_q_out - fp_target_next)
                            .float()
                            .abs()
                            .reshape(-1)
                        )
            empty = torch.empty(
                0,
                dtype=torch.float32,
                device=stitched_weight_fp32.device,
            )
            local_values = (
                torch.cat(local_abs_delta) if local_abs_delta else empty
            )
            a_loss_threshold = global_percentile(
                local_values, float(a_loss_ratio)
            ).detach()
            if slide_alpha is not None and slide_alpha < 1.0:
                local_next_values = (
                    torch.cat(local_abs_next_delta)
                    if local_abs_next_delta
                    else empty
                )
                next_a_loss_threshold = global_percentile(
                    local_next_values, float(a_loss_ratio)
                ).detach()

        for iter_idx, start in enumerate(
            range(0, len(selected), ctx.backward_bsz)
        ):
            with nvtx.nvtx_range(f"refresh.iter_{iter_idx}"):
                batch_idx = selected[start : start + ctx.backward_bsz]
                batch_size = len(batch_idx)
                backward_chunk_sizes.append(batch_size)
                sample_idx = torch.tensor(
                    batch_idx, dtype=torch.long, device=inps.device
                )
                x = inps.index_select(0, sample_idx)
                fp_target = fp_out_for_this_layer.index_select(
                    0, sample_idx
                ).to(x.device)
                kw = _refresh_batch_kwargs(
                    batch_size=batch_size,
                    attention_mask=am,
                    position_ids=pi,
                    position_embeddings=pe,
                )
                with torch.enable_grad():
                    with nvtx.nvtx_range("refresh.forward"):
                        out = functional_call(
                            layer,
                            current_overrides,
                            (x,),
                            kw,
                            strict=False,
                        )
                        q_out = out[0] if isinstance(out, tuple) else out
                    with nvtx.nvtx_range("refresh.loss"):
                        loss_curr = fisher_mse_loss(
                            q_out,
                            fp_target,
                            fisher,
                            a_loss_ratio=a_loss_ratio,
                            a_loss_threshold=a_loss_threshold,
                        )
                        loss_next = None
                        if slide_alpha is not None and slide_alpha < 1.0:
                            with nvtx.nvtx_range(
                                "refresh.next_layer_forward"
                            ):
                                next_pkg = functional_call(
                                    next_layer,
                                    next_overrides,
                                    (q_out,),
                                    kw,
                                    strict=False,
                                )
                                next_q_out = (
                                    next_pkg[0]
                                    if isinstance(next_pkg, tuple)
                                    else next_pkg
                                )
                            fp_target_next = next_fp_out.index_select(
                                0, sample_idx
                            ).to(next_q_out.device)
                            loss_next = fisher_mse_loss(
                                next_q_out,
                                fp_target_next,
                                next_fisher,
                                a_loss_ratio=a_loss_ratio,
                                a_loss_threshold=next_a_loss_threshold,
                            )
                            loss = (
                                slide_alpha * loss_curr
                                + (1.0 - slide_alpha) * loss_next
                            )
                        else:
                            loss = loss_curr
                    with nvtx.nvtx_range("refresh.backward"):
                        batch_grads = torch.autograd.grad(
                            loss,
                            leaves,
                            retain_graph=False,
                            allow_unused=True,
                        )
                with nvtx.nvtx_range("refresh.accumulate"):
                    for index, batch_grad in enumerate(batch_grads):
                        if batch_grad is None:
                            continue
                        if direct_single_chunk:
                            if partial_grad_sums[index] is not None:
                                raise RuntimeError(
                                    "single-chunk refresh received a second "
                                    "gradient contribution"
                                )
                            partial_grad_sums[index] = batch_grad.detach()
                        else:
                            grad_sum = partial_grad_sums[index]
                            if grad_sum is None:
                                raise RuntimeError(
                                    "gradient accumulation buffer is missing"
                                )
                            grad_sum.add_(
                                batch_grad.detach().float(),
                                alpha=float(batch_size),
                            )
                        partial_used[index] = True
                    partial_count += batch_size
                    if partial_loss_sums is not None:
                        partial_loss_sums[0].add_(
                            loss.detach(),
                            alpha=float(batch_size),
                        )
                        if partial_loss_sums.numel() == 3:
                            partial_loss_sums[1].add_(
                                loss_curr.detach(),
                                alpha=float(batch_size),
                            )
                            partial_loss_sums[2].add_(
                                loss_next.detach(),
                                alpha=float(batch_size),
                            )

        with nvtx.nvtx_range("refresh.grad_allreduce"):
            if direct_single_chunk:
                if partial_count <= 0:
                    raise RuntimeError("block refresh produced zero samples")
                global_count = partial_count
                global_used = list(partial_used)
                global_loss_sums = (
                    partial_loss_sums if ctx.trace_enabled else None
                )
            else:
                dense_grad_sums = [
                    grad_sum
                    for grad_sum in partial_grad_sums
                    if grad_sum is not None
                ]
                if len(dense_grad_sums) != len(partial_grad_sums):
                    raise RuntimeError(
                        "dense refresh accumulation unexpectedly lost a buffer"
                    )
                (
                    global_count,
                    global_used,
                    global_loss_sums,
                ) = _aggregate_block_refresh_sums(
                    dense_grad_sums,
                    partial_used,
                    partial_count,
                    partial_loss_sums if ctx.trace_enabled else None,
                )
        if ctx.log_column_block_loss and not ctx.trace_enabled:
            if partial_loss_sums is None:
                raise RuntimeError(
                    "column-block loss logging enabled without loss sums"
                )
            global_loss_sums = _aggregate_loss_sums_for_logging(
                partial_loss_sums
            )
        with nvtx.nvtx_range("refresh.adam_step"):
            update, active_weight_audits = _apply_block_adam(
                active,
                partial_grad_sums,
                global_used,
                global_count,
                lr=ctx.layer_lr,
                grad_clip=ctx.grad_clip,
                compact_update=ctx.fused_block_adam,
                collect_audits=ctx.trace_enabled,
                gradients_are_means=direct_single_chunk,
                layer_idx=ctx.trace_layer,
            )
        if global_loss_sums is not None:
            ctx.record_loss_observation(
                global_loss_sums=global_loss_sums,
                global_count=global_count,
                sample_indices=selected_global,
                slide_alpha=slide_alpha,
                has_next_loss=(
                    slide_alpha is not None and slide_alpha < 1.0
                ),
                objective="fisher_mse_full_block",
                backward_invocation_id=backward_invocation_id,
                backward_chunk_sizes=tuple(backward_chunk_sizes),
                active_weight_audits=active_weight_audits,
            )
        if block_state._states[module_name].step != ctx.adam_step:
            raise RuntimeError(
                f"current Adam step drift for {module_name}: "
                f"state={block_state._states[module_name].step}, "
                f"diagnostic={ctx.adam_step}"
            )
        return update

    refresh._realq_update_layout = (
        "trailing_quant_order_full_block"
        if ctx.fused_block_adam
        else "full_natural"
    )
    return refresh
