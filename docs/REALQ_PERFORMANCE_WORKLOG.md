# REAL-Q performance optimization worklog

## Scope and evidence boundary

This log begins after the formal schedule A/B arms published immutable metrics
and released their source locks. Performance work uses a fresh experiment
root and never mutates, relabels, or reuses formal A/B artifacts.

The objective is to reduce Stage-1 wall time and peak memory without changing
REAL-Q's mathematical algorithm. Every claimed benefit is classified as:

1. a theoretical reduction in operations, traffic, or storage;
2. an isolated measured improvement under the protocol below;
3. an end-to-end measured improvement after correctness gates pass.

A theoretical reduction is never reported as measured speedup. Nsight
instrumented time is never used as the primary wall-time result.

## Optimization classes

| Class | Meaning | Default policy |
|---|---|---|
| `T` | Instrumentation or measurement only | enabled only in profiling commands |
| `X` | Intended bit-exact structural optimization | legacy remains default until all exact gates pass |
| `N` | Same real-valued algorithm but allocation, layout, replay, kernel, or reduction order may change bits | explicit switch, default off, drift documented |
| `A` | Changes objective, population, schedule, clipping, optimizer, quantizer rule, or update order | rejected from this track |

If an `X` candidate differs by one byte at an exact gate, it is repaired,
removed, or reclassified as `N`. It is not described as approximately exact.

## Non-negotiable invariants

An optimization must preserve every applicable item:

- model, tokenizer, dataset, sampled tokens, static Fisher/saliency cache,
  rotation, seeds, and initial model state;
- sample membership/order, refresh cadence, loss-slide behavior, and LR
  schedule;
- Fisher objective/dtype, activation-loss clipping population, threshold,
  detach behavior, and arithmetic order;
- Hessian, damping, act-order permutation, natural-column group mapping, and
  short-tail handling;
- quantizer range, rounding, clamp, scale/zero arithmetic, strict `<` winner
  update, and first-winner tie behavior;
- block boundaries, prefix locking, trailing updates, Adam state lifetime,
  bias correction, gradient clipping, and update order;
- A/K/V hook sites, aware/unaware teacher isolation, clip ratios, and dtype
  promotion;
- final quantized parameters and persistent buffers;
- RNG draw count and order;
- reduction topology and summation order for anything claimed bit-exact.

Copy-only collective changes can be exact candidates. Any change to a
reduction message or summation tree is numerical-equivalence work and remains
opt-in.

## Correctness gates

### E0: source and input identity

- same baseline ID and source commit;
- resolved configs differ only in the candidate switch;
- identical tracked-diff, launcher, model, token, static-cache, and rotation
  hashes;
- identical rank count and physical GPU quartet;
- cache-hit logs contain no fetch, save, recompute, fallback, or OOM retry;
- declared quantization stop reached and process exits successfully.

### E1: operator-level raw-byte equality

Test per-row and group-128 weights, act-order on/off, full and short groups,
clip on/off, zero rows, clamp boundaries, exact ties, signed zero, finite
fuzz, and legacy-defined non-finite behavior. Compare dtype, shape, integer
codes, scale, zero, and fake-quant output separately. `allclose` is not
sufficient.

### E2: deterministic integration equality

A real tiny Hugging Face decoder must match complete refresh identities,
sample IDs, slide alphas, component/total losses, final parameter/buffer
bytes, and versioned checkpoint. Cross-layer-state candidates use at least
three transformer blocks.

### E3: Qwen3-4B one-layer checkpoint equality

Run baseline A/A first on the same physical GPU quartet. Compare complete
canonical state key sets, dtype, shape, and raw bytes. Require canonical state
SHA256 equality and `max_abs_diff == 0`. Archive-file SHA alone is not an
exact-state oracle because container metadata may differ.

### E4: representative end-to-end regression

Before promoting an exact implementation, run the relevant representative
quantization/evaluation case and require identical checkpoint state, KL/PPL,
schedule, refresh count, and provenance with no new warning or fallback.

## Frozen performance fixture

This is a performance fixture, not a paper reproduction:

- local Qwen3-4B;
- four ranks on one fixed GPU quartet;
- four samples, sequence length 128;
- immutable token/static caches;
- QuaRot, `num_groups=4`, block 128, P99, full Fisher vocabulary;
- `grad_lr=0`, `final_layer_grad_lr=0`;
- loss slide off;
- stop after complete transformer block 0;
- evaluation off;
- save complete checkpoint only for correctness runs.

Required cases:

| Case | Weight mode | A/K/V | Purpose |
|---|---|---|---|
| `group_stress` | W2 G128, act-order, weight clip | A/K/V16 unaware | grouped params, clip search, inner loop, collectives |
| `per_row` | W2 per-row, act-order, weight clip | A/K/V16 unaware | per-row oracle/control |
| `group_akv_aware` | W2 G128, act-order, weight clip | A/K/V4 aware, clips 0.9 | aware replay and activation layout |

Correctness cases may run concurrently on disjoint quartets. Timed and Nsight
runs run alone. The planned A/A group repeat uses the same physical GPUs
0--3, not merely a different quartet of the same model.

## Measurement protocol

Primary performance metric:

```text
Stage-1 layer-0 critical-path wall
= maximum per-rank elapsed time between synchronized layer boundaries
```

For baseline and each candidate:

1. one untimed warm-up;
2. at least three isolated timed repetitions;
3. all repetitions plus median, minimum, maximum, and A/A noise envelope;
4. identical GPU quartet, topology, caches, CPU allocation, and config;
5. reset CUDA peak counters before the measured range;
6. record allocated/reserved and sampled device-memory peaks;
7. compare one candidate with its immediately adjacent baseline.

Keep separate:

- `algorithm_wall`: synchronized Stage-1/layer range;
- `process_wall`: launcher start through exit;
- `profiled_wall`: instrumented run, never primary speedup;
- `artifact_wall`: optional checkpoint/report completion.

An optimization is accepted only if its median benefit exceeds A/A noise and
does not cause material memory, startup, stability, or fallback regression.

## Nsight Systems method

Use Nsight 2024.6.2 for attribution:

1. NVTX-only on all ranks (`trace=nvtx`) for distributed range/straggler
   attribution.
2. CUDA plus NVTX on rank 0 only (`trace=cuda,nvtx`); the other ranks run
   normally.

Do not request `nccl`; that trace category is unsupported in this environment.
Export `nvtx_pushpop_sum`, `nvtx_gpu_proj_sum`, `cuda_api_sum`, and
`cuda_gpu_kern_sum` where applicable. Record exact command, traced rank,
rank/GPU mapping, report/table hashes, range counts, dominant APIs/kernels,
profile overhead, and Nsight version.

## Candidate ledger

All switches remain proposed until implementation lands.

| ID | Candidate / switch | Initial class | Theoretical reduction | Main risk | Initial default |
|---|---|---:|---|---|---|
| P01 | trusted fake-quant inner path, `quantizer_inner_fastpath` | X | validate static qparams once instead of every inner-column call | stale/invalid qparams | off |
| P02 | symmetric clip-search union, `w_clip_search_impl` | X | expensive candidates 625 to at most 50 per row/group | tie order, non-finite fallback | legacy Cartesian |
| P03 | clip winner `where`, `w_clip_update_impl` | X | remove host-visible `torch.any` guard | NaN/tie masked-write semantics | guarded |
| P04 | compact grouped weight qparams, `w_group_param_layout` | X | storage `2RC` to `2R ceil(C/G)` | act-order/tail/API mapping | expanded |
| P05 | layer-resident FP32 Fisher, `fisher_fp32_cache` | X initially | BF16-to-FP32 expansion once per layer | allocator/workspace changes | off |
| P06 | trailing-only act-order stitch, `act_order_stitch_impl` | X | remove redundant full-weight gather | boundary or collective asymmetry | full gather |
| P07 | compact A/K/V qparams, `act_qparam_layout` | N | avoid activation-sized scale/zero tensors | broadcast kernel and allocator drift | expanded |
| P08 | reuse next-layer FP replay, `reuse_next_fp_outs` | N | regular teacher forwards `2L-2` to `L` | replay/allocator/stale-state drift | off |
| P09 | trailing gradient reduction, `grad_allreduce_scope` | N | reduce only active suffix | NCCL message and reduction order | full |

### Theoretical accounting already audited

- P01 restores a hoist present in the old per-row code and extends trusted
  grouped use. It must preserve the current tensor `maxq` arithmetic.
- P02 uses `scale=max(a_i,b_j)` only for symmetric finite inputs and
  reconstructs the original nested-loop first winner. Asymmetric or
  non-finite inputs retain the legacy search.
- P05 removes repeated conversion traffic estimated at about 506.65 GiB per
  rank for formal Q4 and 1.772 TiB per rank for formal Q8 under the audited
  refresh/chunk assumptions. These are theoretical traffic counts, not speed.
- P06 reduces audited logical gather payload by about 65% for Q4/Q8 model
  shapes. This does not imply a 65% wall-time gain.
- P07 changes layout and potentially TensorIterator kernel choice, so it is
  numerical class N until full evidence proves otherwise.
- P08 reduces regular teacher replays from 70 to 36 for a 36-block model, but
  reused values may differ from recomputation because allocator/workspace
  context changes.
- P09 changes collective message sizes and can change floating reduction
  order; it remains default-off.

## Promotion and rollback

Promote an X implementation only after E0--E4 pass, the median improvement
exceeds A/A noise, and no unacceptable memory/stability regression appears.
Tests, documentation, and rollback switch land together.

Keep N default-off regardless of speed until the report includes why equality
is not guaranteed, affected kernels/allocations/replays/reductions, checkpoint
mismatch count/fraction/max absolute difference, A/A variability, evaluation
drift, and the exact enabling switch.

Reject or leave disabled if an exact gate fails, identity is invalid, refresh
count/loss/optimizer/integer codes change, benefit is within noise, cost moves
outside the measured range, or OOM/fallback/hang/nondeterminism worsens.

## Artifact contract

Each run directory contains:

```text
<run_id>/
  manifest.json
  resolved_config.json
  command.txt
  run.log
  timing.json
  gpu_memory.csv
  theory_counts.json
  checkpoint/                 # correctness only
  checkpoint_compare.json
  nsys/
    command.txt
    report.nsys-rep
    stats/
```

The manifest records source/diff/dirty paths, exact argv/config, Canoe job and
hardware identity, versions, GPU UUID mapping, model/tokens/static/rotation
hashes, seeds, warm-up/repetition/scope, per-rank wall and peaks, state hash,
Nsight hashes/settings, theoretical counters, correctness result, decision,
and rollback reason.

## Measured results

Populate only after isolated repetitions complete. Negative, neutral, and
failed candidates remain in the table.

| Candidate | Baseline run | Candidate run | Wall repetitions | Median change | Memory | Nsight attribution | Correctness | Decision |
|---|---|---|---|---:|---|---|---|---|

## Experiment entries

### 2026-07-24 - baseline preparation

- Formal A/B completed before source changes.
- Fresh performance root is required under
  `/minimax-avatar-new/zhangqian/realq/experiment_data/perf_stage1_debugging_zhangqian/`.
- Qwen3-4B smoke static/token caches were independently resolved and will be
  copied under names expected by the production cache key.
- Correctness wave 1: `group_stress` on GPUs 0--3 and `per_row` on GPUs 4--7.
- Correctness wave 2: group A/A repeat on the same GPUs 0--3 and
  `group_akv_aware` on GPUs 4--7.
- Estimated checkpoint storage is 30--35 GB for four full-model artifacts.
- The first attempted root,
  `j-7x9o0je4pk_20260724T121230Z_cb5c469794a4`, was deliberately marked
  `INTERRUPTED` after detecting a separately launched user process. It is
  invalid for every correctness or timing claim; no user process was killed.

### 2026-07-24 - pre-optimization correctness baseline

Successful immutable root:

```text
/minimax-avatar-new/zhangqian/realq/experiment_data/perf_stage1_debugging_zhangqian/j-7x9o0je4pk_20260724T121554Z_cb5c469794a4
```

The root passed with return code zero at source commit
`cb5c469794a4e187a85c2ce44beb3b05e8685909`. The tracked diff remained
empty before and after, the input cache tree was unchanged, all four cases
hit both token and collective static caches, and no fetch, cache write,
recompute, fallback, traceback, child failure, or CUDA OOM marker occurred.
The source model identity was
`64b5baa184e0fb3676b4d52c6b662a1a20ebf8e1e51ffb495e7d47b60063130d`.

| Case | GPUs | Process wall (s) | Sampled per-GPU peak (MiB) | Checkpoint bytes | Canonical state SHA256 |
|---|---|---:|---|---:|---|
| `pre_group128` | 0--3 | 127.881826043 | 17556, 19132, 19362, 19362 | 8,823,945,165 | `9587755d521e46b3e65b2139864dab6985674ceda219475c24b0606841cfb595` |
| `pre_group128_repeat` | 0--3 | 121.324124515 | 17556, 17980, 18044, 17618 | 8,823,945,165 | `9587755d521e46b3e65b2139864dab6985674ceda219475c24b0606841cfb595` |
| `pre_row` | 4--7 | 125.003221112 | 11838, 11520, 11776, 11776 | 8,823,945,165 | `7b2a5f84bcfe9214a81321c5fe9a7749f72ea8ef2c20312b36f401a9f9379d84` |
| `pre_aware_a4k4v4` | 4--7 | 151.707985236 | 18348, 19170, 19746, 19426 | 8,823,952,077 | `2ba29a821d09549685c2bae5cae0b24131432df591a19bda5b3dc14f05e6361c` |

These process-wall values were collected from concurrent correctness waves
and are diagnostic only. They are explicitly ineligible for a speedup claim.
Primary timing starts only after synchronized layer-boundary instrumentation
lands and uses isolated warm-up plus three repetitions.

The group-128 A/A comparison on the same physical GPUs passed exactly:

- 435 canonical tensor keys and 399 weight keys compared;
- no missing, extra, or mismatched key;
- canonical state SHA256 equal and maximum absolute difference exactly zero;
- archive SHA256 differed, as expected for containers with metadata:
  `2dd9a8544c9938cb3003aee81cd98135e9540566342e7438edea18f1dfb54518`
  versus
  `dfc4a25e2455256f4a27d08ae9cebbe0dd8263535adb0ff123e1650251961bb2`.

This baseline establishes E0 and E3 oracles for the first structural
optimization candidates. It does not yet satisfy the isolated timing
protocol or E4.
