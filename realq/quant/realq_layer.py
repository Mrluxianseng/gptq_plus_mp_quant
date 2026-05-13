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
        # DEBUG: dump first add_batch's X for sub-task 3 alignment.
        import os as _os
        _name = getattr(self, "_dbg_name", None)
        if _name is not None and self.index == 0 and int(_os.environ.get("REALQ_DBG", "0")):
            _os.makedirs("./debug/realq_dbg", exist_ok=True)
            torch.save({"X_first_batch": inp.detach().cpu(),
                        "sal_first_batch": sal.detach().cpu()},
                       f"./debug/realq_dbg/{_name}_X.pt")
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
        rank_mode = (
            group_parallel_quant == "rank"
            and dist_utils.get_world_size() > 1
        )
        # NUM_GROUPS>1 rank mode requires the H buffer to have been
        # constructed in shard form via __init__ (so add_batch knew to
        # reduce_scatter per group instead of all-reducing the full tensor).
        if rank_mode and self.num_groups > 1 and not self.hessian_group_sharded:
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
        # DEBUG: dump for first layer to compare against old code.
        import os
        _name = getattr(self, "_dbg_name", None)
        if _name is not None and int(os.environ.get("REALQ_DBG", "0")):
            os.makedirs("./debug/realq_dbg", exist_ok=True)
            torch.save({"H": H_per_group.cpu(), "W": W.cpu()},
                       f"./debug/realq_dbg/{_name}_pre.pt")

        # ----------- find_params (row-parallel under rank mode) ------------
        # Per-row scale/zero from the un-permuted W. This MUST run BEFORE
        # act_order: old code calls find_params on the original column layout,
        # then permutes both W and H. Even though minmax is permutation-
        # invariant per row, w_clip's MSE search loops in a way whose tensor
        # strides change with permutation and produce non-bit-exact results.
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

        # DEBUG: dump scale/zero after find_params.
        if _name is not None and int(os.environ.get("REALQ_DBG", "0")):
            torch.save({"scale": self.quantizer.scale.detach().cpu(),
                        "zero": self.quantizer.zero.detach().cpu()},
                       f"./debug/realq_dbg/{_name}_scale.pt")

        # ----------- act_order ------------
        # act_square is all-reduced in finalize_hessian, so perm is identical
        # on every rank. Apply to columns of W and to the (C, C) dims of the
        # local H buffer; rows are NOT permuted.
        perm = invperm = None
        if act_order:
            perm = torch.argsort(self.act_square, descending=True)
            W = W[:, perm]
            H_per_group = H_per_group[:, perm][:, :, perm]
            invperm = torch.argsort(perm)

        # ----------- Hinv ------------
        # Per-(local) group Hinv. NUM_GROUPS=1 takes a separate path that
        # calls ``cholesky_inverse_with_damp`` on the (C, C) tensor directly
        # so the result is bit-equal to sub-task 4 (no extra .empty_like +
        # indexed write step that caused observable last-bit drift on Qwen3).
        if self.num_groups == 1:
            _, Hinv_single = cholesky_inverse_with_damp(H_per_group[0], percdamp=percdamp)
        else:
            Hinv_per_group = torch.empty_like(H_per_group)
            for g in range(H_per_group.shape[0]):
                _, hinv_g = cholesky_inverse_with_damp(H_per_group[g], percdamp=percdamp)
                Hinv_per_group[g] = hinv_g

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

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1
                W1_local = W_local[:, i1:i2].clone()
                Q1_local = torch.zeros_like(W1_local)
                Err1_local = torch.zeros_like(W1_local)
                Hinv1 = Hinv_single[i1:i2, i1:i2]
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
                W_local[:, i2:] -= Err1_local.matmul(Hinv_single[i1:i2, i2:])

                if grad_refresh_fn is not None and i2 < self.columns:
                    # Sync the local rank's slice into a full-rows Q + W
                    # before refresh, so the autograd backward sees the
                    # complete weight. The refresh closure returns an fp32
                    # update tensor; we apply it directly to the fp32
                    # working ``W_local`` so per-step Adam deltas don't get
                    # truncated by bf16 ULPs (CRITICAL for tiny effective
                    # learning rates — see REFACTOR_NOTES "bf16 round-trip"
                    # entry). ``module.weight.data`` stays untouched
                    # outside of being read once for the autograd forward.
                    if rank_mode and world > 1:
                        import torch.distributed as _dist
                        full_Q = torch.empty_like(W)
                        full_W = torch.empty_like(W)
                        _dist.all_gather_into_tensor(full_Q, Q_local.contiguous())
                        _dist.all_gather_into_tensor(full_W, W_local.contiguous())
                    else:
                        full_Q = Q_local
                        full_W = W_local
                    if invperm is None:
                        # No act_order: stitched weight is straightforward.
                        # ``trailing_col_start`` = i2; update applies to
                        # ``W_local[:, i2:]`` (rank-local rows × trailing
                        # cols). For rank mode the update spans all rows
                        # but we only slice the rank's row range.
                        stitched_fp32 = full_W.clone()
                        stitched_fp32[:, :i2] = full_Q[:, :i2]
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

            # All-gather the per-rank Q_local into the full-rows Q so the
            # final ``module.weight`` write below replicates correctly.
            if rank_mode and world > 1:
                import torch.distributed as _dist
                _dist.all_gather_into_tensor(Q, Q_local.contiguous())
            else:
                Q = Q_local
        elif rank_mode:
            # NUM_GROUPS > 1 + rank mode: B 方案 (n_local_rows, count) 扁平表示.
            # Each rank only allocates its own row slice and uses a per-row
            # ``local_hessian_group_idx`` long tensor to look up the right
            # Hinv block per local row. Per-block Q is sync'd across ranks
            # (sync_weight_rows_from_col_ optimization) so block_gd refresh
            # only needs to gather W's trailing columns instead of full W.
            import torch.distributed as _dist
            n_local = self.local_row_end - self.local_row_start
            local_rows = torch.arange(
                self.local_row_start, self.local_row_end, device=self.dev,
            )
            # Per-row mapping: local_row r -> position of its global group in
            # the local H buffer. Used to gather Hinv blocks per local row.
            local_global_group_idx = torch.div(
                local_rows, self.rows_per_group, rounding_mode="floor",
            ).long()
            local_hessian_group_idx = self.hessian_group_to_pos.to(self.dev)[
                local_global_group_idx
            ]
            if bool((local_hessian_group_idx < 0).any().item()):
                raise RuntimeError(
                    "Hessian group shard does not cover every local row group: "
                    f"rank={rank} local_groups="
                    f"{torch.unique(local_global_group_idx).detach().cpu().tolist()} "
                    f"hessian_groups={self.hessian_group_ids.tolist()}."
                )

            W_local = W[row_sl].clone()                           # (n_local, columns)
            Q_local = torch.zeros_like(W_local)
            # Full-rows Q replica updated incrementally per block via
            # all_gather. Used at refresh time to stitch together the
            # leading (already-quantised) columns. Stays None when refresh
            # is not enabled (we sync Q once at end of all blocks instead).
            full_Q_running = None
            if grad_refresh_fn is not None and world > 1:
                full_Q_running = torch.zeros_like(W)

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1
                W1_local = W_local[:, i1:i2].clone()              # (n_local, count)
                Q1_local = torch.zeros_like(W1_local)
                Err1_local = torch.zeros_like(W1_local)
                # Per-row Hinv block (count, count). Built from each local
                # row's owning Hessian group. Memory: n_local * count * count.
                Hinv1_block = Hinv_per_group[:, i1:i2, i1:i2]      # (num_local_groups, count, count)
                Hinv1_per_row = Hinv1_block[local_hessian_group_idx]  # (n_local, count, count)

                for i in range(count):
                    w = W1_local[:, i]
                    d = Hinv1_per_row[:, i, i]                     # (n_local,)
                    w_col = w.unsqueeze(1)
                    q_fake, _, _ = self.quantizer.fake_quantize(
                        w_col,
                        st_idx=row_sl.start,
                        end_idx=row_sl.stop,
                    )
                    q = q_fake.flatten()
                    Q1_local[:, i] = q
                    err = (w - q) / d
                    Err1_local[:, i] = err
                    # In-block compensation: every remaining col [i, count)
                    # gets err * Hinv_row[:, i, i:] subtracted. Per-row Hinv
                    # so this is element-wise multiply on (n_local, count - i).
                    W1_local[:, i:].sub_(err.unsqueeze(1) * Hinv1_per_row[:, i, i:])

                Q_local[:, i1:i2] = Q1_local

                # Outer (cross-block) compensation: error in this block
                # propagates to trailing columns via Hinv[:, i1:i2, i2:].
                # Each local row pulls its own Hinv outer block.
                if i2 < self.columns:
                    Hinv_outer_block = Hinv_per_group[:, i1:i2, i2:]   # (num_local_groups, count, columns - i2)
                    Hinv_outer_per_row = Hinv_outer_block[local_hessian_group_idx]   # (n_local, count, columns - i2)
                    outer_update = torch.bmm(
                        Err1_local.unsqueeze(1), Hinv_outer_per_row,
                    ).squeeze(1)                                      # (n_local, columns - i2)
                    W_local[:, i2:].sub_(outer_update)

                # block_gd refresh
                if grad_refresh_fn is not None and i2 < self.columns:
                    if world > 1:
                        # sync_weight_rows_from_col_ optimization: only
                        # gather Q for THIS block (so full_Q_running covers
                        # all processed cols), then gather W trailing only.
                        # Total bandwidth ~= rows*columns vs 2*rows*columns
                        # for a naive full-gather of both Q and W per refresh.
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
                        stitched_fp32 = torch.empty_like(W)
                        stitched_fp32[:, :i2] = full_Q[:, :i2]
                        stitched_fp32[:, i2:] = W_trailing
                        update = grad_refresh_fn(stitched_fp32, i2)
                        if update is not None:
                            W_local[:, i2:].sub_(update[row_sl])
                    else:
                        # act_order path: closure expects NATURAL-order
                        # weight covering ALL columns; we cannot avoid a
                        # full gather of W here because the trailing slice
                        # in PERMUTED order maps to scattered natural cols.
                        # Fall back to full-W gather for act_order.
                        if world > 1:
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
                        full_update = grad_refresh_fn(stitched_nat, 0)
                        if full_update is not None:
                            update_permuted = full_update[:, perm]
                            W_local[:, i2:].sub_(update_permuted[row_sl, i2:])

            # All-gather final Q_local into the (rows, columns) Q replica
            # so the module.weight write below sees the same Q on every rank.
            if world > 1:
                _dist.all_gather_into_tensor(Q, Q_local.contiguous())
            else:
                Q = Q_local
        else:
            # NUM_GROUPS > 1: per-row-group block update. Each row group g uses
            # H_per_group[g] for its inner block compensation; the column-level
            # quant (find_params + fake_quantize) is still per-row and uniform.
            rpg = self.rows_per_group
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                W1_g = W1.view(self.num_groups, rpg, count)
                Err1_g = Err1.view(self.num_groups, rpg, count)
                Hinv1_g = Hinv_per_group[:, i1:i2, i1:i2]
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
                    Hinv_outer_g = Hinv_per_group[:, i1:i2, i2:]
                    outer_update_g = torch.bmm(Err1_g, Hinv_outer_g)
                    W_g = W.view(self.num_groups, rpg, self.columns)
                    W_g[:, :, i2:].sub_(outer_update_g)

                if grad_refresh_fn is not None and i2 < self.columns:
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
                        full_update = grad_refresh_fn(stitched_nat, 0)
                        if full_update is not None:
                            update_permuted = full_update[:, perm]
                            W[:, i2:].sub_(update_permuted[:, i2:])

        if invperm is not None:
            Q = Q[:, invperm]
        # DEBUG: dump Q at end of quantize.
        if _name is not None and int(os.environ.get("REALQ_DBG", "0")):
            torch.save({"Q": Q.detach().cpu()},
                       f"./debug/realq_dbg/{_name}_Q.pt")
        self.linear.weight.data.copy_(Q.to(self.linear.weight.dtype))

    def free(self) -> None:
        """Release GPU buffers; call once a layer is fully quantised."""
        self.H = None
        self.act_square = None
        self.saliency_cpu = None
