# Triton rank-sharding integration

`triton_fused` now has an entry path for REAL-Q's `group_parallel_quant=rank`
mode. Each torchrun worker passes its owned output rows and their batched
inverse-Hessian rows to the Triton kernel. Sequential column order and the
existing rank all-gather of quantized block tensors are preserved.

The paired benchmark keeps four workers and the paper-aligned calibration
configuration, but changes both control and candidate to the same rank-sharded
path. Metrics record each rank's elapsed quantization time and peak allocated/
reserved memory, plus the worst rank. The entry wrapper also checks that every
rank ended with the same quantized state hash.

Before a full campaign, run the small CUDA parity smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 tools/test_triton_rank_kernel.py
```

The fused path is limited to symmetric per-row weights (`--w_groupsize=-1`),
`block_gd`, blocks no wider than 128, and no atomic or two-sided rounding.
The local workstation has CPU-only PyTorch, so the Triton CUDA smoke and full
four-GPU benchmark must still be run on the server.
