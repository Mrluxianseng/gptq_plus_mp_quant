"""Final-layer KL refresh: drives the LAST transformer block's block_gd
against the model's own ``norm + lm_head`` output instead of fisher_mse.

Old GPTQ+ ``compute_refresh_loss(refresh_loss_type='kl')`` (lines 5641-5657)
computes the loss real-time — no precomputed cache. The teacher logits are
``lm_head(norm(fp_hidden))`` for the same calibration sample whose student
logits we just produced via ``lm_head(norm(q_hidden))``. We mirror that
exactly so the final layer doesn't need any extra precompute pipeline.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from realq.refresh.block_gd import (
    _aggregate_loss_sums_for_logging,
    _aggregate_refresh_sums,
    _functional_weight_name,
)
from realq.utils import nvtx
from utils import dist_utils
from utils.loss_utils import tokenwise_kl_from_logits

if TYPE_CHECKING:
    from realq.refresh.block_gd import RefreshContext
    from realq.runner.streams import LayerInputs
    from utils.model_utils import ModelAnalyzer


def kl_topk_loss(
    q_hidden: torch.Tensor,
    fp_hidden: torch.Tensor,
    analyzer: "ModelAnalyzer",
    kl_topk: int,
) -> torch.Tensor:
    """Match old ``compute_refresh_loss(kl)`` line-for-line.

    Reduction: ``sum`` over vocab (after the kl_topk slice), ``mean`` over
    the flattened ``(B, T)`` axis. ``F.kl_div(log_q, p)`` returns the
    per-element Σ p · (log p − log q) (with ``reduction='none'``); summing
    over vocab gives the per-token KL, then we mean over (B*T).
    """
    norm = analyzer.get_layernorm_before_head()
    lm_head = analyzer.get_lm_head()
    # The LM head itself follows the model dtype, but KL distribution math
    # must be fp32.  Keeping softmax/log_softmax in bf16 can round away the
    # small student/teacher differences this "true KL" final-layer objective
    # is supposed to optimize.
    logits = lm_head(norm(q_hidden)).float()
    logits_fp = lm_head(norm(fp_hidden)).float()
    if kl_topk > 0:
        logits_fp, indices = logits_fp.topk(kl_topk, dim=-1, sorted=False)
        logits = logits.gather(-1, indices)
    return tokenwise_kl_from_logits(logits, logits_fp).mean()


def make_kl_refresh_fn(
    *,
    layer: "nn.Module",
    module: "nn.Module",
    layer_state: "LayerInputs",
    fp_out_for_this_layer: torch.Tensor,
    analyzer: "ModelAnalyzer",
    kl_topk: int,
    ctx: "RefreshContext",
) -> Callable[..., torch.Tensor]:
    """Drop-in replacement for ``make_grad_refresh_fn`` whose loss is
    KL-vs-real-time-FP-logits instead of fisher_mse. Closure signature
    matches ``make_grad_refresh_fn``: takes a stitched fp32 weight + the
    trailing column index, returns the fp32 update tensor for the
    trailing slice. Caller is responsible for applying the update to its
    fp32 working W master (RealQLayer.quantize) so per-step Adam deltas
    don't get truncated by bf16 ULPs.
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
        # Mirror block_gd.refresh under ``--dp_global_shuffle=True``: a
        # GLOBAL scheduler returns the same id list on every rank; each
        # rank filters to its own ``[rank * n_local, (rank+1) * n_local)``
        # shard, runs the local backward, then participates in the
        # cross-rank packed allreduce REGARDLESS of whether its local
        # shard came up empty (so NCCL stays in sync and Adam step
        # counter advances uniformly across ranks).
        with nvtx.nvtx_range("kl_refresh.setup"):
            ctx.adam_step += 1
            selected_global = ctx.next_indices()
            rank = dist_utils.get_rank()
            n_local = inps.shape[0]
            rank_start = rank * n_local
            rank_end = rank_start + n_local
            selected = [gi - rank_start for gi in selected_global if rank_start <= gi < rank_end]
            # Pre-allocate so the empty-local-shard path still allreduces a
            # valid zeros tensor. Matches old GPTQ+ ``partial_grad_sum =
            # torch.zeros_like(override_weight, dtype=fp32)`` at
            # gptq_plus_utils.py:6434, hoisted ABOVE the
            # ``if len(selected_indices) > 0`` guard.
            partial_grad_sum = torch.zeros_like(stitched_weight_fp32)
            partial_count = 0
            partial_loss_sums = (
                torch.zeros(
                    1,
                    # Match the legacy Python-float (binary64) local
                    # accumulation before any distributed fp32 pack.
                    dtype=torch.float64,
                    device=partial_grad_sum.device,
                )
                if ctx.loss_observation_enabled else None
            )
            # Cast stitched fp32 weight to module dtype ONCE per refresh and
            # reuse the bf16 leaf across all backward batches. ``autograd.grad``
            # doesn't touch ``.grad``, so reusing one leaf across multiple
            # backwards is safe. Mirrors block_gd.refresh's hoist; matches old
            # GPTQ+ ``collect_true_weight_gradient`` line 6420-6433.
            override_dtype = module.weight.data.dtype
            override_weight = stitched_weight_fp32.to(override_dtype).requires_grad_(True)
        iter_idx = 0
        for start in range(0, len(selected), ctx.backward_bsz):
            with nvtx.nvtx_range(f"kl_refresh.iter_{iter_idx}"):
                batch_idx = selected[start : start + ctx.backward_bsz]
                batch_size = len(batch_idx)
                sample_idx = torch.tensor(batch_idx, dtype=torch.long, device=inps.device)
                x = inps.index_select(0, sample_idx)
                fp_target_hidden = fp_out_for_this_layer.index_select(0, sample_idx).to(x.device)
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
                    with nvtx.nvtx_range("kl_refresh.forward"):
                        out = functional_call(
                            layer,
                            {weight_name: override_weight},
                            (x,),
                            kw,
                            strict=False,
                        )
                        q_hidden = out[0] if isinstance(out, tuple) else out
                    with nvtx.nvtx_range("kl_refresh.loss"):
                        loss = kl_topk_loss(q_hidden, fp_target_hidden, analyzer, kl_topk)
                    with nvtx.nvtx_range("kl_refresh.backward"):
                        (batch_grad,) = torch.autograd.grad(loss, override_weight, retain_graph=False)
                with nvtx.nvtx_range("kl_refresh.accumulate"):
                    batch_grad_fp32 = batch_grad.detach().float()
                    partial_grad_sum.add_(batch_grad_fp32, alpha=float(batch_size))
                    partial_count += batch_size
                    if partial_loss_sums is not None:
                        weighted_loss = loss.detach().float()
                        partial_loss_sums[0].add_(
                            weighted_loss, alpha=float(batch_size),
                        )
                iter_idx += 1
        # All-reduce ALWAYS (even with partial_count == 0) so NCCL stays
        # synchronised across ranks under global-shuffle load imbalance.
        with nvtx.nvtx_range("kl_refresh.grad_allreduce"):
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
                slide_alpha=None,
                has_next_loss=False,
                objective="kl",
            )
        with nvtx.nvtx_range("kl_refresh.adam_step"):
            # act_order: re-key natural-order grad → permuted column order so the
            # Adam state slice matches old GPTQ+ subgroup state. See block_gd.py
            # for the full rationale.
            if perm is not None:
                accum_grad = accum_grad[:, perm]
            grad_slice = accum_grad[:, trailing_col_start:]
            if ctx.grad_clip > 0:
                grad_slice = grad_slice.clamp(min=-ctx.grad_clip, max=ctx.grad_clip)
            ea = ctx.exp_avg[:, trailing_col_start:]
            ev = ctx.exp_avg_sq[:, trailing_col_start:]
            ea.mul_(ctx.beta1).add_(grad_slice, alpha=1.0 - ctx.beta1)
            ev.mul_(ctx.beta2).addcmul_(grad_slice, grad_slice, value=1.0 - ctx.beta2)
            bc1 = 1.0 - ctx.beta1 ** ctx.adam_step
            bc2 = 1.0 - ctx.beta2 ** ctx.adam_step
            # Match old order: sqrt(ev) / sqrt(bc2) + eps  (NOT sqrt(ev / bc2)).
            denom = ev.sqrt() / math.sqrt(bc2)
            denom.add_(ctx.eps)
            step_size = ctx.layer_lr / bc1
            update = step_size * (ea / denom)
            return update

    return refresh
