"""Output-row rank-parallel sharding helpers.

When ``Config.group_parallel_quant == "rank"`` each DP rank owns a
contiguous slice of the linear's OUTPUT ROWS. ``RealQLayer.quantize``
runs find_params + the per-row inner block update on that slice only;
the resulting per-rank Q rows are all-gathered at the END of quantize so
the linear's ``weight`` ends up identical on every rank.

This is a correctness-preserving sharding: for the same calibration data
+ same H + deterministic float32 ops, ``rank`` mode and ``none`` mode
produce identical Q. The point is throughput — under ``rank`` each
rank does ``rows / world_size`` of the per-row work instead of every
rank doing the full ``rows``.

What is NOT sharded:
- The Hessian itself is fully replicated on every rank (already all-
  reduced during ``finalize_hessian``).
- The per-block outer compensation runs on every rank's local row slice;
  no extra collective is needed because each rank's slice is independent.
- block_gd refresh runs on the FULL replicated weight (``module.weight``
  is gathered before refresh) so the autograd graph sees the same set of
  parameters on every rank.
"""
from __future__ import annotations


def row_slice_for_rank(rank: int, world: int, rows: int) -> slice:
    """Contiguous output-row slice owned by ``rank`` under rank mode.

    Requires ``rows % world == 0`` (matches old code's
    ``--group_parallel_quant=rank`` precondition: "outputs must be
    divisible by world_size").
    """
    if world <= 1:
        return slice(0, rows)
    if rows % world != 0:
        raise ValueError(
            f"group_parallel_quant=rank requires out_features ({rows}) "
            f"divisible by world_size ({world})."
        )
    per_rank = rows // world
    return slice(rank * per_rank, (rank + 1) * per_rank)
