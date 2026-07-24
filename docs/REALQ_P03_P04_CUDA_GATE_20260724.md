# REAL-Q P03/P04 CUDA exactness gate (2026-07-24)

> **Superseded evidence notice (2026-07-24):** the first-pass logs described
> below are retained for audit history but are not admissible correctness
> evidence. They manually injected a one-group Hessian, did not exercise the
> production `NUM_GROUPS=4` sharded-Hessian/multi-group-bmm branch, used Python
> `assert` for some gates, and lacked a fail-closed source/GPU manifest. A
> hardened rerun will be recorded in this document after it completes. Do not
> use the first-pass verdict or hashes to enable an optimization.

## Verdict

Historical first-pass result only: P03 (`w_clip_update_impl=where_out`) and P04
(`w_group_param_layout=compact`) passed every synthetic CUDA gate described
below on the integrated source `dfa3dbef2270c67361d11e9de40e5024c774c07d`.
Every mathematical output was compared as contiguous raw bytes; no tolerance
was used.

The tests found no numerical difference in the exercised production domain.
P04 deliberately changes grouped-qparam storage and the first two NCCL
all-gather shapes. It does not change the number of collectives, the expanded
qparams, observer-level integer `Q`, returned scales, Fisher-MSE refresh
inputs, Adam updates
or moments, or final weights.

This is an exactness gate, not a full-model quality or timing result. The
default implementations remain the legacy-safe settings until the separate
full-model checkpoint and performance gates finish.

## Source and hardware

- Branch: `codex/round2-cuda-gate-20260724`
- Base: `dfa3dbef2270c67361d11e9de40e5024c774c07d`
- Probe: `tools/p03_p04_cuda_probe.py`
- PyTorch: `2.6.0a0+ecf3bae40a.nv25.01`
- CUDA runtime reported by PyTorch: `12.8`
- Backend: NCCL
- Determinism: TF32 disabled, deterministic algorithms enabled,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Only physical GPUs 4–7 were used:

| Physical GPU | UUID | Model |
| --- | --- | --- |
| 4 | `GPU-a2934718-b34d-d8c9-a88a-c342cddcef21` | NVIDIA L20C |
| 5 | `GPU-f1a38e30-e9fd-cd44-5193-919ef3738502` | NVIDIA L20C |
| 6 | `GPU-b69ea445-63eb-5f6c-4e78-f87280749d75` | NVIDIA L20C |
| 7 | `GPU-a27bc59a-b0ea-46cf-230f-360dbd1fc84c` | NVIDIA L20C |

Physical GPUs 0–3 were never made visible to these processes.

## Coverage

### P03 observer and P02 dispatch composition

Both ordinary per-row and grouped observers were tested in FP32. The grouped
case has a short final group (`C=10`, group size 4). For every case, guarded
and where-out implementations had raw-identical:

- `scale`;
- `zero`;
- fake-dequantized weight;
- integer `Q`;
- scale returned by `fake_quantize`.

The matrix includes:

- Cartesian legacy selection;
- finite symmetric P02 union selection, where P03 is deliberately bypassed;
- asymmetric P02 fallback to Cartesian;
- NaN/Inf P02 fallback to Cartesian.

The P02 fixture used `grid=4`. It establishes equality for that fixture only;
it is not a proof of candidate-count reduction or a performance result.

### P04 observer and coordinate composition

Expanded and compact layouts were checked for:

- per-row no-op behavior;
- symmetric and asymmetric standalone `WeightQuantizer` domains;
- exact groups, a short final group, and `groupsize > C`;
- expanded reconstruction of compact `scale` and `zero`;
- generic `quantize`, fake-dequantized weight, integer `Q`, and returned scale;
- act-order natural-column mapping through both the public path and P01's
  prevalidated private path on a nontrivial row slice.

REAL-Q itself intentionally rejects asymmetric quantized weights. Therefore
asymmetric P04 coverage is at the standalone `WeightQuantizer` boundary, while
all `RealQLayer` end-to-end gates use the supported symmetric weight mode.

### End-to-end `RealQLayer`

World size one exercised both `group_parallel_quant=none` and `rank`:

- per-row quantization;
- grouped quantization with a short tail;
- grouped quantization with `groupsize > C`;
- all combinations of P01 off/on, P02 Cartesian/union, P03 guarded/where-out,
  and, where applicable, P04 expanded/compact.

This produced 80 candidate comparisons. One legacy baseline was reused within
each case/mode, and a final legacy repeat checked allocator/order stability.

DP2 and DP4 used real NCCL row sharding. Each world size tested:

- 8 per-row P01/P02/P03 combinations;
- 16 grouped-short-tail P01/P02/P03/P04 combinations;
- real Fisher-MSE backward;
- gradient all-reduce;
- bias-corrected Adam;
- P06 exact act-order stitching;
- stitched refresh inputs, refresh updates, `exp_avg`, `exp_avg_sq`, expanded
  qparams, and final weights;
- exact collective sequence, shapes, and count;
- replicated final-weight hashes across ranks;
- a final legacy repeat after all candidates.

All comparisons passed raw-byte exactly.

Integer `Q` was checked at the observer boundary only. End-to-end distributed
checks compared expanded qparams, fake-dequantized final weights, refresh
inputs/updates, and Adam moments; they did not extract an integer-Q tensor.
Hashes from different DP world sizes use different seeded fixtures and are not
evidence of world-size invariance.

## Results

| World | Visible physical GPUs | Mode | Comparisons | Peak allocated per rank |
| --- | --- | --- | ---: | ---: |
| 1 | 4 | none + rank | 80 | 33,580,544 bytes (32.025 MiB) |
| 2 | 4,5 | rank + Fisher/Adam | 24 | 67,150,336 bytes (64.040 MiB) |
| 4 | 4,5,6,7 | rank + Fisher/Adam | 24 | 67,151,360 bytes (64.041 MiB) |

For the distributed grouped-short-tail case (`rows=8`, `C=9`, group size 4):

- expanded scale/zero all-gather:
  `(8,9) <- (rows/world,9)`;
- compact scale/zero all-gather:
  `(8,3) <- (rows/world,3)`;
- total all-gather count: 7 in both layouts;
- remaining refresh and final-Q collective shapes were identical.

Final-weight hashes:

| World | Case | SHA-256 |
| --- | --- | --- |
| 1 | row | `3120a7d46468a0f7e03b1461d7577424961aed35b1789ade807a3be4c69acef5` |
| 1 | group short tail | `897fe4b113eb8669145e5fbc868c691968f59ab6f0de82e5201c3cf6dc9ffbff` |
| 1 | group size greater than width | `2f1373795d2aa7837f810222e2a3c7beb833751d18cd526f2972330bc18083a7` |
| 2 | row + Fisher/Adam | `a83988d38c4014610c83920e5906a1f965f6315bf82409d3b0fb3dec0c0fdae0` |
| 2 | group short tail + Fisher/Adam | `ba49e2c60dc1c66dbb2ef5a2c930ede89f3b13f77675b9c70108869893777d02` |
| 4 | row + Fisher/Adam | `1c970357b3ecb36946f7139364b4e2ba6247aa3300dbb188a61e1a56c00fa9ae` |
| 4 | group short tail + Fisher/Adam | `a21cc4d5ec247bdb37f045f60634244df15b22062bd63f977a4b03c74e6973aa` |

## Integrated P06 rerun

The existing P06 probe was rerun from the same integrated checkout on physical
GPUs 4–7. All three cases passed in both execution orders
(`legacy -> exact` and `exact -> legacy`), including stitched inputs, updates,
Adam moments, final weights, collective contracts, and rejection boundaries.

| Groups | Columns | Refreshes | Legacy gathers | Exact gathers | SHA-256 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 10 | 2 | 7 | 5 | `e620238fbb774731a0c26409019ccb79dccc89f8bc70a483da11aacf1feab140` |
| 2 | 9 | 2 | 7 | 5 | `76f8beaf3f433df955e4085fd8fc91ebdd3319926cc4ebc020f08f1bc5a6dd62` |
| 4 | 8 | 1 | 4 | 3 | `3901397135045f3c85b73656c985ada32bbf03f4b34eeccaae68328db71c05ae` |

## Reproduction

Run from the repository root:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
REALQ_P03_P04_PROBE_DEVICE=cuda \
CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=. \
torchrun --standalone --nproc_per_node=1 \
  tools/p03_p04_cuda_probe.py

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
REALQ_P03_P04_PROBE_DEVICE=cuda \
CUDA_VISIBLE_DEVICES=4,5 \
PYTHONPATH=. \
torchrun --standalone --nproc_per_node=2 \
  tools/p03_p04_cuda_probe.py

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
REALQ_P03_P04_PROBE_DEVICE=cuda \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH=. \
torchrun --standalone --nproc_per_node=4 \
  tools/p03_p04_cuda_probe.py

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
REALQ_P06_PROBE_DEVICE=cuda \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH=. \
torchrun --standalone --nproc_per_node=4 \
  tools/p06_distributed_cpu_probe.py
```

Superseded raw logs retained outside the source worktree:

- `/minimax-avatar-new/zhangqian/realq/experiment_data/p03_p04_cuda_gate_20260724/world1.log`
- `/minimax-avatar-new/zhangqian/realq/experiment_data/p03_p04_cuda_gate_20260724/world2.log`
- `/minimax-avatar-new/zhangqian/realq/experiment_data/p03_p04_cuda_gate_20260724/world4.log`
- `/minimax-avatar-new/zhangqian/realq/experiment_data/p03_p04_cuda_gate_20260724/p06_world4.log`
