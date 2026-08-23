"""Output-row rank-parallel sharding helpers.

When ``Config.group_parallel_quant == "rank"`` each DP rank owns a
contiguous slice of the linear's OUTPUT ROWS. ``RealQLayer.quantize``
runs find_params + the per-row inner block update on that slice only;
the resulting per-rank Q rows are all-gathered at the END of quantize so
the linear's ``weight`` ends up identical on every rank.

This preserves the row-separable real-valued algorithm: under ``rank`` each
rank does ``rows / world_size`` of the per-row work instead of every rank
doing the full ``rows``. It is not a universal bit-equivalence guarantee:
rank/none can use different floating-point reduction trees, tensor layouts,
and kernels even when they implement the same equations.

What is NOT sharded:
- Every rank still computes each output group's local Hessian contribution.
  With ``num_groups > 1`` those blocks are reduced/scattered per calibration
  batch and each owner retains only the groups needed by its output rows;
  the ``num_groups == 1`` path remains replicated.
- The per-block outer compensation runs on every rank's local row slice;
  no extra collective is needed because each rank's slice is independent.
- Before every block_gd refresh, quantized-prefix and working trailing rows
  are gathered into a full stitched weight, so every rank's autograd closure
  sees the same complete parameter tensor.
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
