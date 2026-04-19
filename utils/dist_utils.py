from typing import Optional, Any
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from utils import memory_utils


def init_process_group():
    # Tell NCCL exactly which GPU this rank owns so collectives don't "guess"
    # the device from rank % device_count and hang when the mapping is not
    # that simple (e.g. CUDA_VISIBLE_DEVICES listing multiple GPUs per rank).
    # Caller must have already run torch.cuda.set_device(local_rank).
    kwargs = dict(backend="nccl", timeout=datetime.timedelta(hours=8))
    if torch.cuda.is_available():
        kwargs["device_id"] = torch.device(f"cuda:{torch.cuda.current_device()}")
    dist.init_process_group(**kwargs)
    dist.barrier()


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if is_dist_available_and_initialized():
        return dist.get_rank()
    return 0


def is_main():
    return get_rank() == 0


def broadcast_parameters(module: nn.Module, src: Any = 0, group: Optional[Any] = None):
    for param in module.parameters():
        dist.broadcast(param.data, src=src, group=group)


def gather_into_tensor(tensor, dim: int = 0):
    world_size = get_world_size()
    if is_main():
        gathered_shape = (*tensor.shape[:dim], world_size * tensor.shape[dim], *tensor.shape[dim + 1 :])
        gathered_tensor = torch.empty(gathered_shape, device=tensor.device, dtype=tensor.dtype)
        gathered_tensor_chunks = list(gathered_tensor.chunk(world_size, dim=dim))
    else:
        gathered_tensor = None
        gathered_tensor_chunks = None
    dist.gather(tensor, gathered_tensor_chunks)
    return gathered_tensor


def print_on_main(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


def distribute_model(model) -> None:
    no_split_module_classes = [
        "Qwen3DecoderLayer",
        "Qwen3MoeForCausalLM",
        "LlamaDecoderLayer",
    ]
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )
    memory_utils.cleanup_memory()


# ---------------------------------------------------------------------------
# DP helpers (data-parallel over calibration samples).
#
# Invariants:
# - When world_size == 1 all helpers are no-ops and return the input unchanged,
#   so the single-GPU path is bit-identical to the pre-DP code.
# - Shard layout is always contiguous: rank r owns samples
#   [r * total // world_size, (r + 1) * total // world_size).
# - The caller is responsible for ensuring `total % world_size == 0`.
# ---------------------------------------------------------------------------


def shard_slice(total: int, rank: Optional[int] = None, world: Optional[int] = None) -> slice:
    """Contiguous shard of an iterable of length `total` for the given rank.

    Raises ValueError if total is not divisible by world_size — DP designs
    assume equal-sized shards so the gradient averaging math stays simple.
    """
    if world is None:
        world = get_world_size()
    if rank is None:
        rank = get_rank()
    if total % world != 0:
        raise ValueError(
            f"shard_slice: total ({total}) must be divisible by world_size ({world}) "
            f"— either trim the input or drop to a compatible world_size."
        )
    chunk = total // world
    return slice(rank * chunk, (rank + 1) * chunk)


def shard_size(total: int, world: Optional[int] = None) -> int:
    """Per-rank shard length. Mirrors shard_slice.stop - shard_slice.start."""
    if world is None:
        world = get_world_size()
    if total % world != 0:
        raise ValueError(
            f"shard_size: total ({total}) must be divisible by world_size ({world})."
        )
    return total // world


def allreduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce. No-op when world_size == 1. Returns the tensor."""
    if get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def allreduce_sum_scalar(scalar) -> float:
    """Sum a Python scalar across ranks. Uses fp64 to minimise rounding error."""
    if get_world_size() <= 1:
        return float(scalar)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.tensor([float(scalar)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def allreduce_mean_scalar(scalar, count: int = 1) -> float:
    """Global weighted mean over ranks: global_mean = sum(scalar_r * count_r) / sum(count_r).

    When each rank uses `count=1` this is a simple mean across ranks.
    """
    if get_world_size() <= 1:
        return float(scalar)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pair = torch.tensor([float(scalar) * count, float(count)], device=device, dtype=torch.float64)
    dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    if pair[1].item() == 0:
        return 0.0
    return (pair[0] / pair[1]).item()


def broadcast_tensor_(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """In-place broadcast. No-op when world_size == 1."""
    if get_world_size() > 1:
        dist.broadcast(tensor, src=src)
    return tensor


def barrier():
    if is_dist_available_and_initialized():
        dist.barrier()


def assert_bit_exact(tensor: torch.Tensor, tag: str = "", rtol: float = 0.0, atol: float = 0.0):
    """Cross-rank consistency check: tensor on every rank must equal rank-0's.

    Designed to catch fasterquant divergence immediately. Meant for debug runs;
    keep off in production because it doubles traffic.
    """
    if get_world_size() <= 1:
        return
    ref = tensor.clone().contiguous()
    dist.broadcast(ref, src=0)
    if rtol == 0.0 and atol == 0.0:
        ok = torch.equal(tensor, ref)
    else:
        ok = torch.allclose(tensor, ref, rtol=rtol, atol=atol)
    if not ok:
        diff = (tensor - ref).abs()
        raise RuntimeError(
            f"[assert_bit_exact] Rank {get_rank()} diverges at `{tag}`: "
            f"max|diff|={diff.max().item():.3e}, mean|diff|={diff.mean().item():.3e}, "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
        )
