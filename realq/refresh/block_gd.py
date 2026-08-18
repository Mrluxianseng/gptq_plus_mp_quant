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
import math
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


def _adam_update_selected(
    state: _WeightRefreshState,
    grad: torch.Tensor,
    columns: slice | torch.Tensor | None,
    *,
    lr: float,
    grad_clip: float,
    compact_update: bool = False,
) -> torch.Tensor:
    """Advance one Adam state and return a full or active-only update."""

    if state.exp_avg is None or state.exp_avg_sq is None:
        raise RuntimeError(f"optimizer state for {state.name!r} was released")
    state.step += 1
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
    ea.mul_(beta1).add_(grad_sel, alpha=1.0 - beta1)
    ev.mul_(beta2).addcmul_(grad_sel, grad_sel, value=1.0 - beta2)
    if index is not None:
        state.exp_avg.index_copy_(1, index, ea)
        state.exp_avg_sq.index_copy_(1, index, ev)
    bc1 = 1.0 - beta1 ** state.step
    bc2 = 1.0 - beta2 ** state.step
    denom = ev.sqrt() / math.sqrt(bc2)
    denom.add_(eps)
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
            ctx.refresh_step += 1
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
            # Slice trailing columns and grad-clip (per-element clamp; matches
            # old ``_compute_grad_optimizer_update_batched`` line 1105-1106).
            grad_slice = accum_grad[:, trailing_col_start:]
            if ctx.grad_clip > 0:
                grad_slice = grad_slice.clamp(min=-ctx.grad_clip, max=ctx.grad_clip)
            # Adam moments on the trailing slice. Match old order:
            #   denom = sqrt(ev) / sqrt(bc2) + eps  (NOT sqrt(ev / bc2))
            ea = ctx.exp_avg[:, trailing_col_start:]
            ev = ctx.exp_avg_sq[:, trailing_col_start:]
            ea.mul_(ctx.beta1).add_(grad_slice, alpha=1.0 - ctx.beta1)
            ev.mul_(ctx.beta2).addcmul_(grad_slice, grad_slice, value=1.0 - ctx.beta2)
            bc1 = 1.0 - ctx.beta1 ** ctx.adam_step
            bc2 = 1.0 - ctx.beta2 ** ctx.adam_step
            denom = ev.sqrt() / math.sqrt(bc2)
            denom.add_(ctx.eps)
            step_size = ctx.layer_lr / bc1
            update = step_size * (ea / denom)
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
