# REAL-Q P03/P04/P06 CUDA exactness gate (2026-07-24)

## Verdict

The hardened CUDA gates passed:

- P03, `w_clip_update_impl=where_out`, was raw-byte identical to the guarded
  implementation in every exercised observer and `RealQLayer` case.
- P04, `w_group_param_layout=compact`, changed grouped-qparam storage and the
  first two qparam all-gather shapes, but its expanded qparams and all
  downstream mathematical tensors were raw-byte identical to the expanded
  layout.
- P06 exact act-order stitching was raw-byte identical to the legacy
  all-gather path in both execution orders while reducing the tested
  all-gather count.

This admits the three implementations through their targeted synthetic CUDA
exactness gate. It is not a full-model perplexity, benchmark, or end-to-end
memory result and does not by itself prove quality or speed on an actual
model.

## Evidence integrity

The admissible P03/P04 runs bind to clean commit
`ed9f8ade5b54f0ca3351f02a92dc0fea50812b09`. The admissible P06 run binds to
clean commit `f6ec38d79566f6c8180644eebfdf53b8a2371214`, which only adds P06
peak-memory recording on top of the hardened probe.

`tools/run_cuda_probe_evidence.py` failed closed unless all of the following
held:

- the checkout was clean before and after the command and stayed on the exact
  expected commit;
- the exact probe SHA-256 and harness SHA-256 matched;
- the subprocess exited with status zero and emitted one valid result;
- `sys.flags.optimize == 0` and `__debug__ is True`;
- the selected physical GPU IDs and UUIDs matched the rank mapping;
- no selected GPU had an existing compute process before launch;
- the raw log, result JSON, exact argv, environment, source-state hashes, and
  global `nvidia-smi` snapshots were recorded.

The probes use explicit exceptions rather than Python `assert`; startup also
rejects `python -O` before importing PyTorch. Dispatch instrumentation required
the selected implementations to run, preventing an ignored switch from
passing by merely reproducing the baseline.

The P03/P04 probe SHA-256 is
`59a09fd02d8d40124ea48774c124dea59996b50332e02c9ca539264f4fa52104`.
The P06 probe SHA-256 is
`4a1973f1784f66181861c7eaaf9c1557b62a78428d617918cb95c0f534d30868`.
The evidence-harness SHA-256 is
`d50257800fae6112a14643ac25e396fc837012bd1d0969715f115a694d9d5c26`.

## Hardware and deterministic setup

- PyTorch: `2.6.0a0+ecf3bae40a.nv25.01`
- CUDA runtime reported by PyTorch: `12.8`
- Backend: NCCL
- TF32 disabled; deterministic algorithms enabled
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `python_optimize=0`; `python_debug=true`

Only physical GPUs 4–7 were selected:

| Physical GPU | UUID | Model |
| --- | --- | --- |
| 4 | `GPU-a2934718-b34d-d8c9-a88a-c342cddcef21` | NVIDIA L20C |
| 5 | `GPU-f1a38e30-e9fd-cd44-5193-919ef3738502` | NVIDIA L20C |
| 6 | `GPU-b69ea445-63eb-5f6c-4e78-f87280749d75` | NVIDIA L20C |
| 7 | `GPU-a27bc59a-b0ea-46cf-230f-360dbd1fc84c` | NVIDIA L20C |

Physical GPUs 0–3 were not made visible to any probe process.

## P03/P04 coverage

### Observer boundary and real dispatch

P03 compared guarded and where-out clipping for per-row and grouped observers,
including a short final group. It covered Cartesian selection, finite
symmetric P02 union selection, asymmetric fallback, and non-finite fallback.
The P02 fixture used `grid=4`; this proves equality for the fixture, not a
candidate-count reduction or a speedup.

P04 compared expanded and compact qparams for:

- per-row no-op behavior;
- exact groups, a short final group, and group size greater than width;
- symmetric and asymmetric standalone `WeightQuantizer` domains;
- generic quantization, fake-dequantized weights, observer-level integer
  `Q`, returned scales, and reconstructed expanded `scale`/`zero`;
- act-order natural-column mapping through the public path and P01's
  prevalidated private path on a nontrivial row slice.

REAL-Q rejects asymmetric quantized weights, so asymmetric P04 coverage stops
at the standalone observer boundary. End-to-end `RealQLayer` cases use the
supported symmetric weight mode.

Instrumentation required actual calls to:

- P01 `_prepare_fake_quantize_inner` and per-column
  `_fake_quantize_prevalidated`;
- P02 `_select_symmetric_union_scale`;
- P03 `torch.where(..., out=...)`;
- P04 compact-layout resolution.

### End-to-end `RealQLayer`

World size one exercised `group_parallel_quant=none` and `rank` over per-row,
short-tail grouped, and group-size-greater-than-width cases. It uses a manual
one-group Hessian fixture and therefore is only the single-rank composition
gate. It made 80 candidate comparisons.

DP2 and DP4 used real `NUM_GROUPS=2` and `NUM_GROUPS=4`. Both called the real
`add_batch -> reduce_scatter -> finalize_hessian` path, required
`hessian_group_sharded=True`, and entered the rank multi-group-bmm branch.
Each distributed run covered:

- 8 per-row P01/P02/P03 combinations;
- 16 grouped-short-tail P01/P02/P03/P04 combinations;
- real Fisher-MSE backward, packed gradient all-reduce, bias-corrected Adam,
  and P06 exact act-order stitching;
- the exact Hessian reduce-scatter/all-reduce and quantization all-gather
  sequences, shapes, dtypes, and counts;
- one reused legacy baseline per case plus a final legacy repeat after all
  candidates.

Every candidate and final repeat compared the following locally as contiguous
raw bytes: finalized Hessian, finalized `act_square`, expanded scale/zero,
stitched refresh inputs, refresh updates, Adam `exp_avg`/`exp_avg_sq`, and
final fake-dequantized weights. Replicated qparams, stitched inputs, updates,
moments, and final weights were also digest-compared across ranks.

Integer `Q` was checked only at the observer boundary. Different DP world
sizes use different seeded fixtures, so hashes across world sizes are not
evidence of world-size invariance.

## P03/P04 results

| World | Physical GPUs | Hessian/groups | Candidate comparisons | Peak allocated per rank |
| ---: | --- | --- | ---: | ---: |
| 1 | 4 | manual fixture, `NUM_GROUPS=1` | 80 | 33,585,152 bytes (32.029 MiB) |
| 2 | 4,5 | real sharded, `NUM_GROUPS=2` | 24 | 67,156,992 bytes (64.046 MiB) |
| 4 | 4,5,6,7 | real sharded, `NUM_GROUPS=4` | 24 | 67,158,016 bytes (64.047 MiB) |

Distributed collective contracts:

| World | Case | Refreshes | Hessian reduce-scatters | All-reduces | All-gathers | Packed gradient records |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | row | 2 | 2 | 5 | 7 | 2 × FP32 `[81]` |
| 2 | grouped short tail | 2 | 2 | 5 | 7 | 2 × FP32 `[73]` |
| 4 | row | 2 | 4 | 5 | 7 | 2 × FP32 `[81]` |
| 4 | grouped short tail | 2 | 4 | 5 | 7 | 2 × FP32 `[73]` |

For the grouped-short-tail case (`rows=8`, `columns=9`, group size 4), P04
changed only the first two qparam gathers:

- expanded: `(8,9) <- (rows/world,9)`;
- compact: `(8,3) <- (rows/world,3)`.

The all-gather count remained seven and all later collective shapes were
unchanged.

Final-weight SHA-256 values:

| World | Case | SHA-256 |
| ---: | --- | --- |
| 1 | row | `3120a7d46468a0f7e03b1461d7577424961aed35b1789ade807a3be4c69acef5` |
| 1 | grouped short tail | `897fe4b113eb8669145e5fbc868c691968f59ab6f0de82e5201c3cf6dc9ffbff` |
| 1 | group size greater than width | `2f1373795d2aa7837f810222e2a3c7beb833751d18cd526f2972330bc18083a7` |
| 2 | row + Fisher/Adam | `5e173f0249985f79137e8fe9aa5d4ef4f337f1c8952c23a1c7c7fc3a4e963be5` |
| 2 | grouped short tail + Fisher/Adam | `564cc06429198acd66b2ba5825bc8031ae072d76ca29784171aa34630aac1351` |
| 4 | row + Fisher/Adam | `ea37ac240ae6ee66692dceda9cd18281ec82a2835e37ae6a5fd8d3614b6182ee` |
| 4 | grouped short tail + Fisher/Adam | `292bb75087518b4237fdac5dad6e5ce6a7b52afef6332a38e01ed3727fe5b2e6` |

## P06 world-four result

The P06 CUDA probe ran every case in both orders (`legacy -> exact` and
`exact -> legacy`). It compared stitched inputs, updates, Adam moments, final
weights, collective contracts, and rejection boundaries.

| Groups | Columns | Refreshes | Legacy gathers | Exact gathers | Peak allocated per rank | SHA-256 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 10 | 2 | 7 | 5 | 67,148,288 bytes (64.038 MiB) | `e620238fbb774731a0c26409019ccb79dccc89f8bc70a483da11aacf1feab140` |
| 2 | 9 | 2 | 7 | 5 | 67,164,672 bytes (64.053 MiB) | `76f8beaf3f433df955e4085fd8fc91ebdd3319926cc4ebc020f08f1bc5a6dd62` |
| 4 | 8 | 1 | 4 | 3 | 67,159,552 bytes (64.048 MiB) | `3901397135045f3c85b73656c985ada32bbf03f4b34eeccaae68328db71c05ae` |

The maximum across cases and all four ranks was 67,164,672 bytes
(64.053 MiB) per rank.

## Admissible artifacts and hashes

| Gate | Evidence root | Manifest SHA-256 | Raw log SHA-256 | Result JSON SHA-256 |
| --- | --- | --- | --- | --- |
| P03/P04 world 1 | `20260724T151905Z_ed9f8ade5b54_p03-p04_w1` | `c1c13272f94a2e7544535b488e787ed63cfe043aae60abfd893ff8c8923f9715` | `97c8a900bf154d827bc51ced921b12eedeeca755ee832f4b333cba8ea4a3104a` | `53beb64637fb72daf63b110735f21c20fc191e1ab0b50be87aef0e55b969d959` |
| P03/P04 world 2 | `20260724T152022Z_ed9f8ade5b54_p03-p04_w2` | `59828df8a080eec84638bfb1aa30ba4b6c8aa35728bae0608180b87501d936d1` | `62c96a6a94f52389b3754e4b1187f27169886e00c0a0ed42b8e620a835eb2285` | `abf31b64b93673a0c614dce257745322350351b4e0b6050d8a98fa4f05bcb994` |
| P03/P04 world 4 | `20260724T152053Z_ed9f8ade5b54_p03-p04_w4` | `cbde80e1bc9575a1dd89e66919dd1d27dc1536ca6dce2cc024ddd75b163b328a` | `17225bf56bcc9611a4fed8ca3a57acb251507c877ad96da9855db66e43c2dff3` | `14901dcdf1c43e14646b6648e3dae5f02cdddf5334168ca4c00596ab1f7687de` |
| P06 world 4 | `20260724T152428Z_f6ec38d79566_p06_w4` | `fcc21490abb3a193fea9118d1b281025c1153d893d463f9b90ebbc1c99e21350` | `0ec8d35f144fd74b30ae43ab05d3d66c99a0c6ecc8551a9e816fb6a3b7701d8c` | `fb34e23ababe9ec871811358cc7c0aeb887e84f900497a2a7b672f499f06e2e1` |

All roots are under:

`/minimax-avatar-new/zhangqian/realq/experiment_data/p03_p04_cuda_gate_hardened_20260724`

Every manifest records `status=passed`, command return code zero, empty
validation errors, a clean identical source state before and after the run,
and the exact physical GPU mapping.

## Reproduction

Run from a clean repository checkout. The evidence runner intentionally
rejects physical GPUs outside 4–7.

```bash
python tools/run_cuda_probe_evidence.py \
  --probe p03-p04 \
  --gpus 4 \
  --world-size 1 \
  --master-port 30571 \
  --output-parent /absolute/path/to/evidence

python tools/run_cuda_probe_evidence.py \
  --probe p03-p04 \
  --gpus 4,5 \
  --world-size 2 \
  --master-port 30572 \
  --output-parent /absolute/path/to/evidence

python tools/run_cuda_probe_evidence.py \
  --probe p03-p04 \
  --gpus 4,5,6,7 \
  --world-size 4 \
  --master-port 30574 \
  --output-parent /absolute/path/to/evidence

python tools/run_cuda_probe_evidence.py \
  --probe p06 \
  --gpus 4,5,6,7 \
  --world-size 4 \
  --master-port 30580 \
  --output-parent /absolute/path/to/evidence
```

## Superseded historical evidence

The first-pass directory
`/minimax-avatar-new/zhangqian/realq/experiment_data/p03_p04_cuda_gate_20260724`
is retained only for audit history and contains `README_SUPERSEDED.md`. It is
not admissible because it manually injected a one-group Hessian, missed the
production four-group sharded branch, used removable Python assertions, and
lacked a fail-closed source/GPU manifest.

The earlier hardened P06 run
`20260724T152136Z_ed9f8ade5b54_p06_w4` passed, but is superseded by the
admissible P06 artifact above because it did not record per-case/per-rank CUDA
peak memory. Its raw files remain unmodified and its directory contains
`README_SUPERSEDED.md`.
