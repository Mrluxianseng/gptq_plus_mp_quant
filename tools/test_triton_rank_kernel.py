"""CUDA parity smoke for the row-sharded Triton GPTQ inner-column kernel.

Run with four GPUs:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --standalone --nproc_per_node=4 tools/test_triton_rank_kernel.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from gptq_utils.triton_gptq_kernels import fused_gptq_column_


def reference(W, GH, Z, H, scale, maxq, qlo, second_order_scale):
    W = W.clone()
    GH = GH.clone()
    Q = torch.zeros_like(W)
    W_int = torch.zeros_like(W)
    Err = torch.zeros_like(W)
    for col in range(W.shape[1]):
        w = W[:, col]
        q_int = torch.clamp(torch.round(w / scale), qlo, maxq)
        q = (scale * q_int).to(w.dtype)
        err = (w - q - GH[:, col]) / H[:, col, col]
        Q[:, col] = q
        W_int[:, col] = q_int
        Err[:, col] = err
        W[:, col:] -= second_order_scale * (
            err[:, None] * H[:, col, col:] + GH[:, col:]
        )
        GH[:, col:] -= Z[:, col, None] * H[:, col, col:]
    return W, GH, Q, W_int, Err


def main():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    rows_per_rank, cols = 7, 32
    total_rows = rows_per_rank * world
    generator = torch.Generator(device="cpu").manual_seed(9127)
    W_all = torch.randn(total_rows, cols, generator=generator, dtype=torch.float32)
    GH_all = torch.randn(total_rows, cols, generator=generator, dtype=torch.float32) * 0.03
    Z_all = torch.randn(total_rows, cols, generator=generator, dtype=torch.float32) * 0.02
    A = torch.randn(total_rows, cols, cols, generator=generator, dtype=torch.float32) * 0.03
    H_all = torch.bmm(A, A.transpose(1, 2))
    H_all.diagonal(dim1=-2, dim2=-1).add_(1.0)
    scale_all = torch.full((total_rows,), 0.08, dtype=torch.float32)

    start, end = rank * rows_per_rank, (rank + 1) * rows_per_rank
    W = W_all[start:end].to(device).contiguous()
    GH = GH_all[start:end].to(device).contiguous()
    Z = Z_all[start:end].to(device).contiguous()
    H = H_all[start:end].to(device).contiguous()
    scale = scale_all[start:end].to(device).contiguous()
    maxq, qlo, second_order_scale = 7, -8, 1.0

    expected = reference(W, GH, Z, H, scale, maxq, qlo, second_order_scale)
    Q = torch.zeros_like(W)
    W_int = torch.zeros_like(W)
    Err = torch.zeros_like(W)
    for col in range(cols):
        fused_gptq_column_(
            W, GH, Z, H, scale, Q, W_int, Err, col,
            maxq=maxq, qlo=qlo, second_order_scale=second_order_scale,
        )
    torch.cuda.synchronize(device)
    actual = (W, GH, Q, W_int, Err)
    names = ("W", "GH", "Q", "W_int", "Err")
    for name, got, want in zip(names, actual, expected):
        if name == "W_int":
            if not torch.equal(got, want):
                raise AssertionError(f"rank {rank}: {name} differs from torch reference")
        elif not torch.allclose(got, want, rtol=0.0, atol=2e-6):
            error = (got - want).abs().max().item()
            raise AssertionError(f"rank {rank}: {name} max_abs_diff={error:.3e}")
    print(f"rank={rank}/{world}: batched-Hessian Triton parity passed", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
