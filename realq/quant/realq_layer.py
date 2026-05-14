"""Per-module GPTQ accumulator and per-column quantisation.

One ``RealQLayer`` instance owns the Hessian + quantizer for one nn.Linear.
The pipeline drives it like:

    realq = RealQLayer(linear, saliency, num_groups=1, dev)
    for x_batch in inputs:
        realq.add_batch(x_batch)
    realq.finalize_hessian()
    realq.quantize(blocksize=128, percdamp=0.01,
                   act_order=False, w_clip=False, grad_refresh_fn=None)
    # linear.weight is now the dequantised quantised weight in-place.

Sub-task 3 hard-codes ``num_groups=1`` (one Hessian per linear, all rows share
it) and skips act_order / w_clip / block-gd refresh. Those land in
sub-tasks 4–6.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from realq.quant.hessian import cholesky_inverse_with_damp
from realq.utils import nvtx
from utils import dist_utils

LOSS_GRAD_SCALE = 1000.0  # must match precompute.static_e2e.LOSS_GRAD_SCALE.

if TYPE_CHECKING:
    from utils.quant_utils import WeightQuantizer  # noqa: F401


class RealQLayer:
    def __init__(
        self,
        linear: nn.Linear,
        saliency: torch.Tensor,    # (N_local, T, num_groups) cpu fp32
        quantizer,                  # utils.quant_utils.WeightQuantizer instance
        num_groups: int,
        dev: torch.device,
        group_parallel_quant: str = "none",
    ) -> None:
        self.linear = linear
        self.dev = dev
        self.num_groups = num_groups
        self.rows, self.columns = linear.weight.shape
        if self.rows % num_groups != 0:
            raise ValueError(
                f"out_features ({self.rows}) must be divisible by num_groups ({num_groups})."
            )
        self.rows_per_group = self.rows // num_groups
        # Saliency stays on CPU; sliced + moved to GPU in add_batch on demand.
        self.saliency_cpu = saliency.float()
        self.quantizer = quantizer
        # Rank shard setup. Mirrors gptq_plus's `hessian_group_shard` path:
        # under group_parallel_quant="rank" with NUM_GROUPS>1 each rank owns a
        # contiguous slice of OUTPUT ROWS; H is allocated only for the row
        # groups containing those rows. NUM_GROUPS=1 takes the legacy
        # all-reduce path inside `finalize_hessian` (single group, every rank
        # owns it — no per-group sharding needed).
        world = dist_utils.get_world_size()
        rank = dist_utils.get_rank()
        self.hessian_group_sharded = (
            group_parallel_quant == "rank"
            and world > 1
            and num_groups > 1
        )
        # Defaults for the non-sharded path: every rank owns every group.
        self.hessian_group_ids = torch.arange(num_groups, dtype=torch.long)
        self.hessian_group_to_pos = torch.arange(num_groups, dtype=torch.long)
        self.hessian_group_owner_slots = None
        self.hessian_group_zero = None
        self.local_row_start = 0
        self.local_row_end = self.rows
        if self.hessian_group_sharded:
            if self.rows % world != 0:
                raise ValueError(
                    "group_parallel_quant=rank requires out_features "
                    f"({self.rows}) divisible by world_size ({world})."
                )
            # Per-group bmm in `quantize` requires that each owned local group
            # contributes the same number of rows (R_per_group_local). That
            # holds iff world divides num_groups OR num_groups divides world
            # (Cases A/B/C in GROUP_PARALLEL_RANK_ANALYSIS.md). Case D
            # (irregular) breaks the (G_l, R_l, ...) layout and is rejected
            # here so the fast path doesn't need a per-row fallback.
            if num_groups % world != 0 and world % num_groups != 0:
                raise ValueError(
                    "group_parallel_quant=rank requires world_size and num_groups "
                    "to have a divisibility relationship: "
                    f"world ({world}) % num_groups ({num_groups}) == 0 OR "
                    f"num_groups ({num_groups}) % world ({world}) == 0. "
                    "Irregular configurations (Case D in "
                    "GROUP_PARALLEL_RANK_ANALYSIS.md) are not supported."
                )
            self.local_row_start = rank * self.rows // world
            self.local_row_end = (rank + 1) * self.rows // world
            local_rows = torch.arange(
                self.local_row_start, self.local_row_end, dtype=torch.long
            )
            local_group_ids = torch.unique(
                torch.div(local_rows, self.rows_per_group, rounding_mode="floor")
            )
            if local_group_ids.numel() <= 0:
                raise RuntimeError("rank shard resolved an empty local group set.")
            group_to_pos = torch.full((num_groups,), -1, dtype=torch.long)
            group_to_pos[local_group_ids] = torch.arange(
                local_group_ids.numel(), dtype=torch.long
            )
            # Per-group owner mask: ranks whose local row range intersects
            # group g. Drives _reduce_scatter_hessian_group_'s input_list.
            group_owner_mask = torch.zeros((num_groups, world), dtype=torch.bool)
            for owner_rank in range(world):
                owner_row_start = owner_rank * self.rows // world
                owner_row_end = (owner_rank + 1) * self.rows // world
                owner_rows = torch.arange(
                    owner_row_start, owner_row_end, dtype=torch.long
                )
                owner_group_ids = torch.unique(
                    torch.div(owner_rows, self.rows_per_group, rounding_mode="floor")
                )
                group_owner_mask[owner_group_ids, owner_rank] = True
            self.hessian_group_ids = local_group_ids
            self.hessian_group_to_pos = group_to_pos
            self.hessian_group_owner_slots = [
                group_owner_mask[g].nonzero(as_tuple=False).flatten().tolist()
                for g in range(num_groups)
            ]
        # Per-rank running sums for H and act_square. Both finalised in
        # `finalize_hessian()`. H is (num_local_groups, C, C); under rank
        # sharding num_local_groups < num_groups so the H footprint shrinks.
        self.H = torch.zeros(
            (int(self.hessian_group_ids.numel()), self.columns, self.columns),
            device=dev,
        )
        if self.hessian_group_sharded:
            # Stride-0 zeros buffer reused as the non-owner input slot in
            # every reduce_scatter call (avoids per-call allocation).
            self.hessian_group_zero = torch.zeros(
                (), device=dev, dtype=self.H.dtype,
            ).expand(self.columns, self.columns)
        self.act_square = torch.zeros((self.columns,), device=dev)
        self.index = 0          # local sample cursor into saliency_cpu
        self.token_count = 0    # local token count (sum of B*T per add_batch)
        self._finalized = False

    @torch.no_grad()
    def _reduce_scatter_hessian_group_(
        self, global_group_id: int, block: torch.Tensor,
    ) -> torch.Tensor:
        """SUM-reduce a single global Hessian group to its owner ranks.

        Mirrors ``GPTQPlus._reduce_scatter_hessian_group_``. When
        ``world == k * num_groups`` a group has multiple owners; we satisfy
        ``reduce_scatter`` by placing the same local partial block into every
        owner slot of ``input_list`` and stride-0 zeros elsewhere. After the
        call ``block`` on each owner holds ``sum_r block_r`` (the global
        partial sum); on non-owners it holds zero (and the caller skips the
        write via ``hessian_group_to_pos[g] >= 0``).
        """
        if not self.hessian_group_sharded or dist_utils.get_world_size() <= 1:
            return block
        import torch.distributed as dist
        owners = self.hessian_group_owner_slots[int(global_group_id)]
        if not owners:
            raise RuntimeError(f"Hessian group {global_group_id} has no owner ranks.")
        if (
            self.hessian_group_zero is None
            or self.hessian_group_zero.shape != block.shape
        ):
            self.hessian_group_zero = torch.zeros(
                (), device=block.device, dtype=block.dtype,
            ).expand_as(block)
        input_list = [
            block if r in owners else self.hessian_group_zero
            for r in range(dist_utils.get_world_size())
        ]
        dist.reduce_scatter(block, input_list, op=dist.ReduceOp.SUM)
        return block

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        """Accumulate one batch of inputs into the per-rank Hessian.

        ``inp`` shape: ``(B, T, in_features)`` or ``(B*T, in_features)``.
        Saliency is sliced from ``saliency_cpu[index:index+B]`` so callers must
        feed batches in the same order they were laid down during precompute.
        """
        if self._finalized:
            raise RuntimeError("add_batch called after finalize_hessian.")
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        assert inp.dim() == 3, f"expected 2/3D input, got {inp.dim()}"
        B = inp.shape[0]
        sal = self.saliency_cpu[self.index : self.index + B].to(self.dev)
        self.index += B
        # Match old code's sink-strip path: if saliency was clipped (sink
        # tokens dropped), trim the activation prefix to align T. Sub-task 3
        # never sets sink_size>0 so this is a no-op.
        if inp.shape[1] > sal.shape[1]:
            inp = inp[:, inp.shape[1] - sal.shape[1] :]
        inp = inp.reshape(-1, inp.shape[-1]).float()
        sal = sal.reshape(-1, sal.shape[-1]).float()  # (B*T, num_groups)
        if self.hessian_group_sharded:
            # Per-batch + per-group reduce_scatter. Each rank computes the
            # partial X^T diag(s_g) X for ALL groups (collective requires
            # full participation), but only the owners of group g write the
            # reduce-scattered sum into their local H buffer.
            inp_T = inp.transpose(0, 1).contiguous()
            for group_id in range(self.num_groups):
                weighted = inp.mul(sal[:, group_id].unsqueeze(1))   # (BT, C)
                block = inp_T.matmul(weighted)                       # (C, C)
                self._reduce_scatter_hessian_group_(group_id, block)
                local_pos = int(self.hessian_group_to_pos[group_id].item())
                if local_pos >= 0:
                    self.H[local_pos].add_(block)
        else:
            # Per-group weighted X^T diag(s_g) X, shape (G, C, C).
            weighted = inp.unsqueeze(0).mul(sal.transpose(0, 1).unsqueeze(-1))  # (G, BT, C)
            inp_T = inp.transpose(0, 1).unsqueeze(0).expand(self.num_groups, -1, -1)
            block = torch.bmm(inp_T, weighted)  # (G, C, C)
            self.H.add_(block)
        self.act_square.add_((inp ** 2).sum(0))
        self.token_count += inp.shape[0]

    @torch.no_grad()
    def finalize_hessian(self) -> None:
        if self._finalized:
            return
        if self.hessian_group_sharded:
            # add_batch already reduce-scattered each global group to its
            # owners, so self.H carries the global unnormalised sum for the
            # owned groups only. Skip the (G, C, C) all-reduce.
            pass
        else:
            dist_utils.allreduce_sum_(self.H)
        dist_utils.allreduce_sum_(self.act_square)
        total_samples = dist_utils.allreduce_sum_scalar(self.index)
        total_tokens = dist_utils.allreduce_sum_scalar(self.token_count)
        if total_samples <= 0 or total_tokens <= 0:
            raise RuntimeError("finalize_hessian called before any add_batch ran.")
        seq_len = total_tokens / total_samples
        denom = total_samples * seq_len
        self.H.div_(denom)
        # Cancel the loss_grad_scale^2 factor baked into saliency by precompute.
        # Saliency = grad^2 with grad coming from loss * LOSS_GRAD_SCALE backward,
        # so it carries a (LOSS_GRAD_SCALE^2 = 1e6) multiplier. Saliency uses
        # only relative values, but H = X^T diag(s) X feeds into Cholesky+block
        # update where the absolute scale matters: without this division Hinv is
        # 1e-6× too small, the inner GPTQ compensation is wildly under-applied,
        # and downstream layers see drifted X. Old code applies the same divide
        # via `hessian_saliency_scale=_E2E_PRECOMPUTE_QUADRATIC_SCALE`.
        self.H.div_(LOSS_GRAD_SCALE * LOSS_GRAD_SCALE)
        self.act_square.div_(seq_len)
        # Symmetrise + sanitise (matches old GPTQPlus behaviour).
        self.H = 0.5 * (self.H + self.H.transpose(-1, -2))
        if not torch.isfinite(self.H).all():
            logging.warning(
                "[realq.realq_layer] non-finite entries in H for layer of shape "
                "(%d, %d); replacing with zeros.", self.rows, self.columns,
            )
            self.H = torch.nan_to_num(self.H, nan=0.0, posinf=0.0, neginf=0.0)
        self._finalized = True

    @torch.no_grad()
    def quantize(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        act_order: bool = False,
        w_clip: bool = False,
        grad_refresh_fn=None,
        group_parallel_quant: str = "none",
    ) -> None:
        """Run GPTQ on this linear's weight. Mutates ``self.linear.weight``.

        ``act_order``: sort columns by H diagonal magnitude (descending). The
        quantised result is permuted back at the end so the linear sees the
        original column order.

        ``w_clip``: handled at quantizer construction time (``mse=True`` in
        :func:`realq.runner.layer_loop._make_quantizer`); the flag is here only
        to refuse misuse when the quantizer wasn't configured for it.

        ``grad_refresh_fn``: see ``realq.refresh.block_gd`` docstring.

        ``group_parallel_quant``: ``"rank"`` shards the per-row find_params
        + per-row inner block update across DP ranks, with an all-gather at
        the end to replicate Q. ``"none"`` runs the full quantize on every
        rank (correct but redundant). For act_order paths we currently
        force ``none`` because the rank mode would need extra synchronisation
        to keep the per-rank permutation consistent.
        """
        if not self._finalized:
            raise RuntimeError("Call finalize_hessian() before quantize().")
        if w_clip and not self.quantizer.mse:
            raise ValueError(
                "w_clip=True requires the quantizer to be configured with mse=True."
            )
        # rank_mode is the unified "shard rows across ranks + flat (rows, count)
        # inner layout" path. world=1 also takes it so that single-GPU produces
        # bit-equal results to multi-GPU (and to the legacy code's
        # ``_fasterquant_group_parallel`` single-GPU branch). In single-GPU the
        # ``world > 1`` gates inside the branch turn the all-gathers into no-ops
        # and ``row_slice_for_rank(0, 1, rows) == slice(0, rows)``.
        rank_mode = group_parallel_quant == "rank"
        # NUM_GROUPS>1 rank mode requires the H buffer to have been
        # constructed in shard form via __init__ (so add_batch knew to
        # reduce_scatter per group instead of all-reducing the full tensor).
        # Only relevant under DP — at world=1 ``hessian_group_sharded`` stays
        # False (nothing to shard) and the rank-mode flat layout works on top
        # of the full ``(num_groups, C, C)`` H buffer.
        _world = dist_utils.get_world_size()
        if (
            rank_mode
            and self.num_groups > 1
            and _world > 1
            and not self.hessian_group_sharded
        ):
            raise RuntimeError(
                "rank mode + NUM_GROUPS>1 requires hessian_group_sharded=True; "
                "pass group_parallel_quant='rank' to RealQLayer.__init__."
            )
        W = self.linear.weight.data.clone().float()
        # H is (num_local_groups, C, C). For NUM_GROUPS=1 this collapses to
        # (1, C, C) and downstream uses the legacy H_per_group[0] shortcut.
        # For NUM_GROUPS>1 rank mode num_local_groups is the rank's owned
        # group count (always >= 1).
        H_per_group = self.H

        # ----------- find_params (row-parallel under rank mode) ------------
        # Per-row scale/zero from the un-permuted W. This MUST run BEFORE
        # act_order: old code calls find_params on the original column layout,
        # then permutes both W and H. Even though minmax is permutation-
        # invariant per row, w_clip's MSE search loops in a way whose tensor
        # strides change with permutation and produce non-bit-exact results.
        with nvtx.nvtx_range("quant.find_params"):
            world = dist_utils.get_world_size()
            rank = dist_utils.get_rank()
            if rank_mode:
                from realq.parallel.group_quant import row_slice_for_rank
                row_sl = row_slice_for_rank(rank, world, self.rows)
            else:
                row_sl = slice(0, self.rows)
            if not self.quantizer.ready():
                if rank_mode and world > 1:
                    # Each rank computes find_params for its own row slice, then
                    # all-gathers (rows, 1) scale/zero so the full per-row params
                    # are visible on every rank for fake_quantize's st_idx/end_idx
                    # slicing during the inner block.
                    self.quantizer.find_params(W[row_sl])
                    import torch.distributed as _dist
                    full_scale = torch.empty(
                        (self.rows, 1),
                        dtype=self.quantizer.scale.dtype,
                        device=self.quantizer.scale.device,
                    )
                    _dist.all_gather_into_tensor(full_scale, self.quantizer.scale.contiguous())
                    self.quantizer.scale = full_scale
                    full_zero = torch.empty(
                        (self.rows, 1),
                        dtype=self.quantizer.zero.dtype,
                        device=self.quantizer.zero.device,
                    )
                    _dist.all_gather_into_tensor(full_zero, self.quantizer.zero.contiguous())
                    self.quantizer.zero = full_zero
                else:
                    self.quantizer.find_params(W)

        # ----------- act_order ------------
        # act_square is all-reduced in finalize_hessian, so perm is identical
        # on every rank. Apply to columns of W and to the (C, C) dims of the
        # local H buffer; rows are NOT permuted.
        perm = invperm = None
        if act_order:
            with nvtx.nvtx_range("quant.act_order"):
                perm = torch.argsort(self.act_square, descending=True)
                W = W[:, perm]
                H_per_group = H_per_group[:, perm][:, :, perm]
                invperm = torch.argsort(perm)

        # ----------- Hinv ------------
        # Per-(local) group Hinv. NUM_GROUPS=1 takes a separate path that
        # calls ``cholesky_inverse_with_damp`` on the (C, C) tensor directly
        # so the result is bit-equal to sub-task 4 (no extra .empty_like +
        # indexed write step that caused observable last-bit drift on Qwen3).
        with nvtx.nvtx_range("quant.hinv"):
            if self.num_groups == 1:
                Hinv_single = cholesky_inverse_with_damp(H_per_group[0], percdamp=percdamp)
            else:
                Hinv_per_group = torch.empty_like(H_per_group)
                for g in range(H_per_group.shape[0]):
                    Hinv_per_group[g] = cholesky_inverse_with_damp(H_per_group[g], percdamp=percdamp)

        Q = torch.zeros_like(W)
        # NUM_GROUPS=1 fast path: keeps the original (R, 1) @ (1, count)
        # matmul which lands on a different cuBLAS kernel than the per-group
        # broadcast-mul fallback below — and the kernel choice changes the
        # last-bit output for fp32 even with bit-equal inputs. Without this
        # branch the NUM_GROUPS=1 case loses bit-exactness with the old
        # GPTQPlus by a few ulps.
        if self.num_groups == 1:
            # Rank-mode row sharding: each rank only does find_params + inner
            # block + outer compensation on its OWN slice of output rows.
            # Q is gathered at the end so module.weight ends up replicated.
            # block_gd refresh is unaffected — module.weight is synced before
            # each refresh, the refresh's backward sees the full weight, and
            # Adam runs on every rank redundantly so the post-refresh
            # trailing slice stays consistent across ranks.
            W_local = W[row_sl]
            Q_local = torch.zeros_like(W_local)
            # Full-rows Q replica updated incrementally per block via
            # all_gather (sync_weight_rows_from_col_ optimization). Stays None
            # when refresh isn't enabled (we sync Q once at end of all blocks
            # instead). Mirrors the NUM_GROUPS>1 path's per-block chunk gather
            # so refresh only pays bandwidth for trailing W + this block's Q,
            # not 2× full weight per refresh as the previous full-W gather did.
            full_Q_running = None
            if grad_refresh_fn is not None and rank_mode and world > 1:
                full_Q_running = torch.zeros_like(W)

            block_idx = 0
            for i1 in range(0, self.columns, blocksize):
                with nvtx.nvtx_range(f"block_{block_idx}"):
                    i2 = min(i1 + blocksize, self.columns)
                    count = i2 - i1
                    W1_local = W_local[:, i1:i2].clone()
                    Q1_local = torch.zeros_like(W1_local)
                    Err1_local = torch.zeros_like(W1_local)
                    Hinv1 = Hinv_single[i1:i2, i1:i2]
                    with nvtx.nvtx_range("block.inner_cols"):
                        for i in range(count):
                            w = W1_local[:, i]
                            d = Hinv1[i, i]
                            w_col = w.unsqueeze(1)
                            # Slice scale/zero to local rows so fake_quantize sees
                            # the per-row params for the rows we own.
                            q_fake, _, _ = self.quantizer.fake_quantize(
                                w_col,
                                st_idx=row_sl.start,
                                end_idx=row_sl.stop,
                            )
                            q = q_fake.flatten()
                            Q1_local[:, i] = q
                            err = (w - q) / d
                            W1_local[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                            Err1_local[:, i] = err
                    Q_local[:, i1:i2] = Q1_local
                    with nvtx.nvtx_range("block.outer_compensate"):
                        W_local[:, i2:] -= Err1_local.matmul(Hinv_single[i1:i2, i2:])

                    if grad_refresh_fn is not None and i2 < self.columns:
                        with nvtx.nvtx_range("block.refresh"):
                            # Sync the local rank's slice into a full-rows Q + W
                            # before refresh, so the autograd backward sees the
                            # complete weight. The refresh closure returns an fp32
                            # update tensor; we apply it directly to the fp32
                            # working ``W_local`` so per-step Adam deltas don't get
                            # truncated by bf16 ULPs (CRITICAL for tiny effective
                            # learning rates — see REFACTOR_NOTES "bf16 round-trip"
                            # entry). ``module.weight.data`` stays untouched
                            # outside of being read once for the autograd forward.
                            #
                            # Bandwidth: gather only THIS block's Q chunk (so
                            # ``full_Q_running`` carries all already-quantised
                            # cols) + trailing W (cols still in fp32 working
                            # state). Naive 2× full-weight gather wastes
                            # ``2 × rows × columns`` per refresh; the chunked
                            # form is ``rows × (count + cols - i2)`` ≈
                            # ``rows × cols`` over a full layer, so 2× cheaper
                            # in NCCL bytes.
                            if rank_mode and world > 1:
                                import torch.distributed as _dist
                                chunk_full = torch.empty(
                                    (self.rows, count),
                                    device=self.dev, dtype=Q_local.dtype,
                                )
                                _dist.all_gather_into_tensor(
                                    chunk_full, Q1_local.contiguous(),
                                )
                                full_Q_running[:, i1:i2] = chunk_full
                                W_trailing = torch.empty(
                                    (self.rows, self.columns - i2),
                                    device=self.dev, dtype=W_local.dtype,
                                )
                                _dist.all_gather_into_tensor(
                                    W_trailing, W_local[:, i2:].contiguous(),
                                )
                                full_Q = full_Q_running
                            else:
                                full_Q = Q_local
                                W_trailing = W_local[:, i2:]
                            if invperm is None:
                                # No act_order: stitched weight is straightforward.
                                # ``trailing_col_start`` = i2; update applies to
                                # ``W_local[:, i2:]`` (rank-local rows × trailing
                                # cols). For rank mode the update spans all rows
                                # but we only slice the rank's row range.
                                # ``empty_like + 2 slice assigns`` skips the
                                # full-W clone the previous version did before
                                # overwriting the leading half with Q.
                                stitched_fp32 = torch.empty_like(W)
                                stitched_fp32[:, :i2] = full_Q[:, :i2]
                                stitched_fp32[:, i2:] = W_trailing
                                update = grad_refresh_fn(stitched_fp32, i2)
                                if update is not None:
                                    W_local[:, i2:].sub_(update[row_sl])
                            else:
                                # act_order: the autograd forward sees a
                                # NATURAL-order weight (columns in original
                                # layout); we stitch Q and W back to natural order
                                # for the forward, and let the closure return a
                                # NATURAL-order update covering all columns. We
                                # then permute it back to PERMUTED order, take the
                                # trailing slice, and apply to W_local. Old code's
                                # equivalent path (lines 1989-2015) does the same
                                # in PERMUTED coord throughout — algebraically
                                # equivalent for the trailing slice when the only
                                # adam state we read is exp_avg[:, i2:].
                                #
                                # The trailing slice in PERMUTED order maps to
                                # scattered NATURAL cols, so we can't avoid a
                                # full W gather here. Fall back to full-W
                                # gather only on the act_order branch.
                                if rank_mode and world > 1:
                                    full_W = torch.empty_like(W)
                                    _dist.all_gather_into_tensor(
                                        full_W, W_local.contiguous(),
                                    )
                                else:
                                    full_W = W_local
                                Q_nat = full_Q[:, invperm]
                                W_nat = full_W[:, invperm]
                                quant_nat_cols = perm[:i2]
                                stitched_nat = W_nat.clone()
                                stitched_nat[:, quant_nat_cols] = Q_nat[:, quant_nat_cols]
                                # Closure operates in NATURAL coord (i.e.
                                # trailing_col_start=0 means update covers ALL
                                # natural columns). We then permute back to
                                # PERMUTED, slice trailing, apply to W_local.
                                full_update = grad_refresh_fn(stitched_nat, 0)
                                if full_update is not None:
                                    update_permuted = full_update[:, perm]
                                    W_local[:, i2:].sub_(update_permuted[row_sl, i2:])
                    block_idx += 1

            # All-gather the per-rank Q_local into the full-rows Q so the
            # final ``module.weight`` write below replicates correctly.
            if rank_mode and world > 1:
                import torch.distributed as _dist
                _dist.all_gather_into_tensor(Q, Q_local.contiguous())
            else:
                Q = Q_local
        elif rank_mode:
            # NUM_GROUPS > 1 + rank mode, per-group bmm layout:
            # ``W_local`` is a (G_l, R_l, columns) view of the rank's
            # contiguous row shard, where
            #   G_l = self.hessian_group_ids.numel() = local unique groups
            #   R_l = n_local // G_l                = rows per local group
            # Uniformity (R_l independent of group) is enforced in __init__
            # by requiring world % num_groups == 0 OR num_groups % world == 0.
            # Inner block compensation is broadcast across R_l per group;
            # outer compensation is a SINGLE batched bmm with batch=G_l
            # (typically 1 when world=num_groups), reducing 256 small per-row
            # bmms to one big GEMM. Mirrors old GPTQ+
            # ``_fasterquant_group_parallel`` (gptq_plus_utils.py:1879-1894).
            import torch.distributed as _dist
            n_local = self.local_row_end - self.local_row_start
            G_l = int(self.hessian_group_ids.numel())
            if n_local % G_l != 0:
                raise RuntimeError(
                    f"rank shard not uniform across local groups: "
                    f"n_local={n_local} G_l={G_l}. This should have been "
                    f"caught by the world/num_groups divisibility check in "
                    f"__init__."
                )
            R_l = n_local // G_l

            # 2D view for find_params slicing + refresh trailing gather;
            # 3D view (same storage) for inner block / outer compensation.
            W_local_2d = W[row_sl].clone()                          # (n_local, columns)
            W_local = W_local_2d.view(G_l, R_l, self.columns)
            Q_local_2d = torch.zeros_like(W_local_2d)
            Q_local = Q_local_2d.view(G_l, R_l, self.columns)

            # Full-rows Q replica updated incrementally per block via
            # all_gather. Used at refresh time to stitch together the
            # leading (already-quantised) columns. Stays None when refresh
            # is not enabled (we sync Q once at end of all blocks instead).
            full_Q_running = None
            if grad_refresh_fn is not None and world > 1:
                full_Q_running = torch.zeros_like(W)

            for i1 in range(0, self.columns, blocksize):
                with nvtx.nvtx_range(f"block_{i1 // blocksize}"):
                    i2 = min(i1 + blocksize, self.columns)
                    count = i2 - i1
                    W1 = W_local[:, :, i1:i2].clone()              # (G_l, R_l, count)
                    Q1 = torch.zeros_like(W1)
                    Err1 = torch.zeros_like(W1)
                    Hinv1 = Hinv_per_group[:, i1:i2, i1:i2]        # (G_l, count, count)

                    with nvtx.nvtx_range("block.inner_cols"):
                        for i in range(count):
                            w_col = W1[:, :, i]                    # (G_l, R_l)
                            # fake_quantize expects (n_local, 1) in natural
                            # row order; reshape (G_l, R_l) → (G_l*R_l, 1)
                            # preserves the [local_row_start, local_row_end)
                            # order because local groups are contiguous in
                            # row space.
                            w_col_flat = w_col.reshape(-1, 1)
                            q_fake, _, _ = self.quantizer.fake_quantize(
                                w_col_flat,
                                st_idx=row_sl.start,
                                end_idx=row_sl.stop,
                            )
                            q_col = q_fake.reshape(G_l, R_l)
                            Q1[:, :, i] = q_col
                            d = Hinv1[:, i, i].unsqueeze(1)        # (G_l, 1) — broadcast across R_l
                            err = (w_col - q_col) / d              # (G_l, R_l)
                            Err1[:, :, i] = err
                            # In-block compensation: every remaining col
                            # [i, count) gets err * Hinv_row[:, i, i:]
                            # subtracted, broadcast across R_l rows of each
                            # group.
                            #   err.unsqueeze(-1):              (G_l, R_l, 1)
                            #   Hinv1[:, i, i:].unsqueeze(1):   (G_l, 1, count - i)
                            # → broadcast to                     (G_l, R_l, count - i)
                            W1[:, :, i:].sub_(
                                err.unsqueeze(-1) * Hinv1[:, i, i:].unsqueeze(1)
                            )

                    Q_local[:, :, i1:i2] = Q1

                    # Outer (cross-block) compensation as a single batched
                    # bmm with batch=G_l. For W=num_groups (the common 4-card
                    # num_groups=4 setup) G_l=1 and this degenerates to one
                    # big GEMM — vs. the previous per-row layout that
                    # launched n_local small bmms.
                    if i2 < self.columns:
                        with nvtx.nvtx_range("block.outer_compensate"):
                            Hinv_outer = Hinv_per_group[:, i1:i2, i2:]   # (G_l, count, columns - i2)
                            outer_update = torch.bmm(Err1, Hinv_outer)    # (G_l, R_l, columns - i2)
                            W_local[:, :, i2:].sub_(outer_update)

                    # block_gd refresh
                    if grad_refresh_fn is not None and i2 < self.columns:
                        with nvtx.nvtx_range("block.refresh"):
                            if world > 1:
                                # sync_weight_rows_from_col_ optimization: only
                                # gather Q for THIS block (so full_Q_running covers
                                # all processed cols), then gather W trailing only.
                                # Total bandwidth ~= rows*columns vs 2*rows*columns
                                # for a naive full-gather of both Q and W per refresh.
                                chunk_full = torch.empty(
                                    (self.rows, count),
                                    device=self.dev, dtype=Q_local_2d.dtype,
                                )
                                _dist.all_gather_into_tensor(
                                    chunk_full,
                                    Q1.reshape(n_local, count).contiguous(),
                                )
                                full_Q_running[:, i1:i2] = chunk_full
                                W_trailing = torch.empty(
                                    (self.rows, self.columns - i2),
                                    device=self.dev, dtype=W_local_2d.dtype,
                                )
                                _dist.all_gather_into_tensor(
                                    W_trailing, W_local_2d[:, i2:].contiguous(),
                                )
                                full_Q = full_Q_running
                            else:
                                full_Q = Q_local_2d
                                W_trailing = W_local_2d[:, i2:]

                            if invperm is None:
                                stitched_fp32 = torch.empty_like(W)
                                stitched_fp32[:, :i2] = full_Q[:, :i2]
                                stitched_fp32[:, i2:] = W_trailing
                                update = grad_refresh_fn(stitched_fp32, i2)
                                if update is not None:
                                    W_local_2d[:, i2:].sub_(update[row_sl])
                            else:
                                # act_order path: closure expects NATURAL-order
                                # weight covering ALL columns; we cannot avoid a
                                # full gather of W here because the trailing slice
                                # in PERMUTED order maps to scattered natural cols.
                                # Fall back to full-W gather for act_order.
                                if world > 1:
                                    full_W = torch.empty_like(W)
                                    _dist.all_gather_into_tensor(
                                        full_W, W_local_2d.contiguous(),
                                    )
                                else:
                                    full_W = W_local_2d
                                Q_nat = full_Q[:, invperm]
                                W_nat = full_W[:, invperm]
                                quant_nat_cols = perm[:i2]
                                stitched_nat = W_nat.clone()
                                stitched_nat[:, quant_nat_cols] = Q_nat[:, quant_nat_cols]
                                # Pass perm so the closure re-keys natural-order grad
                                # into permuted coord before Adam (see block_gd.py
                                # rationale; mirrors old GPTQPlus permuted-state Adam).
                                update = grad_refresh_fn(stitched_nat, i2, perm=perm)
                                if update is not None:
                                    W_local_2d[:, i2:].sub_(update[row_sl])

            # All-gather final Q_local_2d into the (rows, columns) Q replica
            # so the module.weight write below sees the same Q on every rank.
            if world > 1:
                _dist.all_gather_into_tensor(Q, Q_local_2d.contiguous())
            else:
                Q = Q_local_2d
        else:
            # NUM_GROUPS > 1: per-row-group block update. Each row group g uses
            # H_per_group[g] for its inner block compensation; the column-level
            # quant (find_params + fake_quantize) is still per-row and uniform.
            rpg = self.rows_per_group
            for i1 in range(0, self.columns, blocksize):
                with nvtx.nvtx_range(f"block_{i1 // blocksize}"):
                    i2 = min(i1 + blocksize, self.columns)
                    count = i2 - i1
                    W1 = W[:, i1:i2].clone()
                    Q1 = torch.zeros_like(W1)
                    Err1 = torch.zeros_like(W1)
                    W1_g = W1.view(self.num_groups, rpg, count)
                    Err1_g = Err1.view(self.num_groups, rpg, count)
                    Hinv1_g = Hinv_per_group[:, i1:i2, i1:i2]
                    with nvtx.nvtx_range("block.inner_cols"):
                        for i in range(count):
                            w_col = W1[:, i].unsqueeze(1)
                            q_fake, _, _ = self.quantizer.fake_quantize(w_col)
                            q_col = q_fake.flatten()
                            Q1[:, i] = q_col
                            d_g = Hinv1_g[:, i, i]
                            d_per_row = d_g.repeat_interleave(rpg)
                            err = (W1[:, i] - q_col) / d_per_row
                            Err1[:, i] = err
                            err_g = err.view(self.num_groups, rpg)
                            hinv_row_g = Hinv1_g[:, i, i:]
                            update_g = err_g.unsqueeze(-1) * hinv_row_g.unsqueeze(1)
                            W1_g[:, :, i:].sub_(update_g)
                    Q[:, i1:i2] = Q1
                    if i2 < self.columns:
                        with nvtx.nvtx_range("block.outer_compensate"):
                            Hinv_outer_g = Hinv_per_group[:, i1:i2, i2:]
                            outer_update_g = torch.bmm(Err1_g, Hinv_outer_g)
                            W_g = W.view(self.num_groups, rpg, self.columns)
                            W_g[:, :, i2:].sub_(outer_update_g)

                    if grad_refresh_fn is not None and i2 < self.columns:
                        with nvtx.nvtx_range("block.refresh"):
                            if invperm is None:
                                stitched_fp32 = W.clone()
                                stitched_fp32[:, :i2] = Q[:, :i2]
                                update = grad_refresh_fn(stitched_fp32, i2)
                                if update is not None:
                                    W[:, i2:].sub_(update)
                            else:
                                Q_nat = Q[:, invperm]
                                W_nat = W[:, invperm]
                                quant_nat_cols = perm[:i2]
                                stitched_nat = W_nat.clone()
                                stitched_nat[:, quant_nat_cols] = Q_nat[:, quant_nat_cols]
                                # Pass perm so the closure re-keys grad to permuted
                                # coord (see block_gd.py rationale).
                                update = grad_refresh_fn(stitched_nat, i2, perm=perm)
                                if update is not None:
                                    W[:, i2:].sub_(update)

        if invperm is not None:
            with nvtx.nvtx_range("quant.invperm"):
                Q = Q[:, invperm]
        with nvtx.nvtx_range("quant.weight_writeback"):
            self.linear.weight.data.copy_(Q.to(self.linear.weight.dtype))

    def free(self) -> None:
        """Release GPU buffers; call once a layer is fully quantised."""
        self.H = None
        self.act_square = None
        self.saliency_cpu = None
