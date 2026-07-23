"""Block-boundary refresh: Adam optimiser + grad_clip + reverse-cosine schedule.

Per-linear closure that fires at every GPTQ block boundary (except the last
block of the linear, since there are no trailing columns left to update).
Workflow:

    1. Sample ``backward_bsz`` calibration items from this rank's shard.
    2. Forward the current transformer layer with all upstream linears already
       quantised AND the current linear's already-quantised columns frozen
       at their Q values; the not-yet-quantised columns participate in
       autograd via ``module.weight.grad``.
    3. Compute :func:`fisher_mse_loss` against the precomputed FP reference
       output (``fp_out_for_this_layer``).
    4. ``loss.backward()``, all-reduce the per-column grads across DP ranks,
       grad-clip, then Adam-step the unquantised columns IN PLACE on
       ``module.weight.data``.

The closure is invoked by :meth:`realq.quant.RealQLayer.quantize` after each
block's outer compensation; ``i2`` is the column index where the next block
begins.
"""
from __future__ import annotations

import math
import random as _random
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from torch.func import functional_call

from realq.alignment import RefreshStep, RefreshTraceWriter
from realq.refresh.fisher_loss import fisher_mse_loss
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

    schedule="cosine": layer 0 gets ``base_lr * base_ratio``, the deepest
    layer gets ``base_lr``, intermediate layers use the paper's
    ``sin(π·x/2)`` reverse-cosine ramp. With ``base_ratio = 0.01`` and
    ``base_lr = 1e-4``,
    layer 0 = 1e-6, last layer = 1e-4.
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
    are not individually balanced — that is the deliberate trade-off of
    global shuffle (some refreshes have a slight load imbalance, but
    the consumed sample distribution matches the old gptq_plus_utils.py
    ``--dp_global_shuffle=True`` reference exactly for bit-exactness).

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


class RefreshContext:
    """State carried across all block_gd refreshes for a SINGLE linear module.

    Holds the Adam moments per-module and a REFERENCE to the run-wide
    sample scheduler (the scheduler itself is owned by the runner so its
    cursor advances across modules — see ``_SharedSampleScheduler``).
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
        if self.trace_enabled and (
            trace_layer is None or trace_module is None or blocksize is None
        ):
            raise ValueError(
                "enabled refresh tracing requires trace_layer, trace_module, "
                "and blocksize"
            )
        # Adam state, full-tensor shape; only the trailing column slice is
        # touched per call but keeping the full shape simplifies indexing.
        W = module.weight
        self.exp_avg = torch.zeros_like(W, dtype=torch.float32)
        self.exp_avg_sq = torch.zeros_like(W, dtype=torch.float32)
        self.adam_step = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def next_indices(self) -> list[int]:
        return self.scheduler.next_indices()

    @property
    def trace_enabled(self) -> bool:
        return self.trace_writer is not None and self.trace_writer.enabled

    def record_trace(
        self,
        *,
        global_loss_sums: torch.Tensor,
        global_count: int,
        sample_indices: list[int],
        slide_alpha: float | None,
        has_next_loss: bool,
    ) -> None:
        """Record globally reduced loss statistics for the current Adam step."""

        if not self.trace_enabled or not dist_utils.is_main():
            return
        if global_loss_sums.numel() not in (1, 3):
            raise ValueError(
                "refresh trace expects [total] or "
                "[total, current, next] global loss sums"
            )
        block = self.adam_step - 1
        col_start = block * int(self.blocksize)
        col_end = min(
            col_start + int(self.blocksize),
            int(self.module.weight.shape[1]),
        )
        denom = float(global_count)
        self.trace_writer.record(
            RefreshStep(
                layer=int(self.trace_layer),
                module=str(self.trace_module),
                block=block,
                col_start=col_start,
                col_end=col_end,
                adam_step=self.adam_step,
                loss=float(global_loss_sums[0].item()) / denom,
                loss_current=(
                    float(global_loss_sums[1].item()) / denom
                    if global_loss_sums.numel() == 3
                    else float(global_loss_sums[0].item()) / denom
                ),
                loss_next=(
                    float(global_loss_sums[2].item()) / denom
                    if has_next_loss and global_loss_sums.numel() == 3
                    else None
                ),
                slide_alpha=slide_alpha,
                sample_indices=tuple(int(index) for index in sample_indices),
            )
        )


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


def make_grad_refresh_fn(
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
) -> Callable[..., torch.Tensor]:
    """Build the per-block refresh closure for one linear.

    Closure signature::

        update = refresh(stitched_weight_fp32, trailing_col_start)

    where ``stitched_weight_fp32`` is the FULL (rows, columns) fp32 weight
    that the autograd forward should see — the caller stitches Q for
    already-quantised columns and the working fp32 W for the rest. The
    closure casts it to module dtype for forward via ``functional_call``,
    runs autograd over ``backward_samples`` mini-batches of ``backward_bsz``,
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
                if ctx.trace_enabled else None
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

        # ``a_loss_ratio`` is defined over the GLOBAL refresh mini-batch.  A
        # rank-/microbatch-local P95 changes the objective with world size and
        # partitioning.  Run a graph-free prepass, gather only |delta| values,
        # and broadcast one exact cap used by every backward microbatch.
        a_loss_threshold = None
        next_a_loss_threshold = None
        if a_loss_ratio < 1.0:
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
                            loss.detach().float(), alpha=float(batch_size),
                        )
                        if partial_loss_sums.numel() == 3:
                            partial_loss_sums[1].add_(
                                loss_curr.detach().float(),
                                alpha=float(batch_size),
                            )
                        if loss_next is not None and partial_loss_sums.numel() == 3:
                            partial_loss_sums[2].add_(
                                loss_next.detach().float(),
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
                partial_loss_sums,
            )
            accum_grad = partial_grad_sum / float(global_count)
        if global_loss_sums is not None:
            ctx.record_trace(
                global_loss_sums=global_loss_sums,
                global_count=global_count,
                sample_indices=selected_global,
                slide_alpha=slide_alpha,
                has_next_loss=(
                    slide_alpha is not None and slide_alpha < 1.0
                ),
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

    return refresh
