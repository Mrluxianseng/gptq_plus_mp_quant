"""Smoke-test NCCL setup and a collective across all torchrun ranks."""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        value = torch.tensor([rank + 1], dtype=torch.int64, device=f"cuda:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) // 2
        if value.item() != expected:
            raise AssertionError(
                f"rank {rank}: NCCL all_reduce returned {value.item()}, expected {expected}"
            )
        print(
            f"NCCL_PREFLIGHT rank={rank}/{world_size} local_rank={local_rank} "
            f"device={torch.cuda.get_device_name(local_rank)} all_reduce={value.item()} OK",
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
