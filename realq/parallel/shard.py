"""Row-sharding helpers for GROUP_PARALLEL_QUANT=rank mode.

In rank mode each linear layer's output rows are split contiguously across
ranks. find_params, block-internal quantization, and Hessian inverse are
sharded; block-outer compensation runs locally with the full group tensor on
every rank.

Sub-task 1 only needs the slice computation; the actual sharded update lands
in sub-task 4.
"""
from utils import dist_utils


def row_slice_for_rank(num_rows: int, rank: int | None = None, world: int | None = None) -> slice:
    """Contiguous output-row slice for the given rank.

    Requires `num_rows % world == 0` — RealQ's rank mode assumes equal-sized
    output-row shards. The caller should validate this once per linear layer.
    """
    if world is None:
        world = dist_utils.get_world_size()
    if rank is None:
        rank = dist_utils.get_rank()
    if num_rows % world != 0:
        raise ValueError(
            f"row_slice_for_rank: num_rows ({num_rows}) must be divisible by "
            f"world_size ({world}) — required by GROUP_PARALLEL_QUANT=rank mode."
        )
    chunk = num_rows // world
    return slice(rank * chunk, (rank + 1) * chunk)


def sample_shard_slice(num_samples: int, rank: int | None = None, world: int | None = None) -> slice:
    """Contiguous sample shard for DP. Mirrors dist_utils.shard_slice."""
    return dist_utils.shard_slice(num_samples, rank=rank, world=world)
