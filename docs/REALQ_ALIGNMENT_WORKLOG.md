# REAL-Q paper/code alignment worklog

This document is the durable audit and validation record for aligning the
legacy implementation (`gptq_utils/gptq_plus_utils.py`) and the refactored
implementation (`realq/`) with `main.tex`.

## Scope and acceptance criteria

The numerical scope is:

- aggregated Fisher MSE;
- Adam-driven block gradient descent;
- the existing legacy loss slide-window design (intentionally preserved);
- QuaRot/rotation;
- activation, value-cache, and key-cache aware/unaware fake quantization;
- A/K/V clipping and activation-delta (`a_loss_ratio`) clipping;
- reverse-cosine layer learning-rate scheduling;
- per-row and grouped weight quantization;
- KL, perplexity, and zero-shot evaluation.

Activation checkpointing and purely performance-oriented FSDP changes are
outside the current ten-hour numerical-alignment timebox unless they block a
required correctness test.

Acceptance requires all of the following:

1. Paper formulas and experiment defaults are represented accurately, except
   that the legacy slide-window indexing and terminal-layer policy remain a
   deliberate compatibility constraint for this audit and are reported as a
   paper mismatch rather than silently treated as paper-correct.
2. Legacy and refactored implementations use the same numerical semantics for
   every explicitly tested configuration and numerical mechanism listed in
   the validation matrix; this does not assert an exhaustive Cartesian test
   of unrelated backends, model families, or bit widths.
3. Every block-GD refresh loss is captured and compared.  Each matched old/new
   step must have symmetric relative difference below 1%.
4. Repeating the same implementation with the same seed must keep every
   matched step loss below 1% difference (the target is exact equality where
   deterministic kernels permit it).
5. Tests cover per-row and group-128 weights, activation/K/V aware and unaware
   modes, A/K/V clipping, activation-loss clipping, rotation on/off, and the
   retained slide-window behavior.

For two finite scalar losses `a` and `b`, the symmetric relative difference is

```text
2 * abs(a - b) / max(abs(a) + abs(b), 1e-12)
```

This avoids making the result depend on which implementation is called the
reference. Non-finite losses, missing steps, duplicated step identities, and
configuration mismatches are hard failures.

## Execution environment

- Development checkout:
  `/minimax-avatar-new/zhangqian/realq/gptq_plus`
- Branch at audit start: `zq`
- Baseline commit: `847465c`
- User-owned untracked paper source: `main.tex` (kept out of implementation
  commits)
- Canoe job: `j-e894fp9rne`
- Canoe name: `zhangqian_debugging_0724_0251`
- Cluster/queue: `pa-cne02-prod-01` /
  `minimax-avatar-data2-h800new`
- Resources: one node, eight GPUs, 100 CPUs, 1000 GiB memory
- Lifetime command: 604800 seconds (one week)
- Initial state at 2026-07-24 02:52 CST: `Pending`,
  `NotEnoughResources`; the requested queue and all resource fields were
  verified from the submitted job spec.
- Running state used for validation: node `e01-cn-8h74p0pxk0r`, eight NVIDIA
  L20C GPUs, Python 3.12.3, PyTorch 2.6.0a0+nv25.01, CUDA 12.8.

## Audit verdict

“Legacy and refactored code agree” and “the paper describes the code” are two
different questions:

- After the corrections below, the tested old/new numerical paths are exact,
  not merely within the requested 1% threshold.
- The current loss-slide formula and pseudocode do **not** describe the
  intentionally retained legacy schedule. This is a high-severity
  paper/reproducibility mismatch.
- The paper and code have no independent query/Q fake quantizer. `A4KV4`
  means linear-input A, post-RoPE K, and `v_proj`-output V quantization. Q is
  only Hadamard-rotated with K to preserve the attention inner product.
- The paper tables are not exactly reproducible from the current repository:
  several numerically material settings are undisclosed or conflict across
  `Config`, the convenience launcher, and the legacy sweep. See
  `docs/REALQ_PAPER_PROTOCOL.md`.
- The paper's claim that calibration sampling is the only stochastic source
  is too strong for the implementation: Stage 0 samples categorical Fisher
  labels and QuaRot generates a random transform. Dedicated fixed seeds make
  those products replayable and isolate a calibration-seed sweep, but do not
  make the underlying operations non-random.

## Baseline findings

| Area | Legacy baseline | Refactored baseline | Required action |
|---|---|---|---|
| Aggregated Fisher MSE | Core quadratic is correct | Core quadratic is correct | Add direct equivalence tests |
| Adam block-GD | Core Adam update is correct | Core Adam update is correct | Align final-layer clipping and trace every step |
| Slide window | Legacy refresh-count design | Matches legacy | Preserve and regression-test |
| Reverse cosine | Uses the wrong half-cosine formula | Same wrong formula | Use `sin(pi*x/2)`; disable the ramp in aware mode |
| Fisher distribution | Top-20 by default | Top-20 by default | Full vocabulary by default |
| Final KL | Top-20 by default | Top-20 by default | Full vocabulary by default |
| Evaluation KL | Hard-coded top-20 | Shared hard-coded top-20 | Full-vocabulary KL |
| A/K/V aware teacher | Correctly disables fake quant on FP references | FP Fisher and teacher paths are contaminated | Restore FP isolation |
| A/K/V clipping | Operators work; several scripts omit V clip | Operators work; no paper preset | Validate and provide explicit paper settings |
| Activation-loss clip | Operator is correct; default disabled | Operator is correct; default unexpectedly 0.95 | Default to 1.0; test explicit 0.95 |
| Rotation | Tied checkpoints skip the global rotation after untying | Same shared bug | Rotate the now-untied embedding/head normally |
| GuidedQuant saliency | Uses channel-mean squared gradient inside each row group | Same paper mismatch | Use the group squared Euclidean norm (channel sum) and invalidate old static caches |
| Per-row weight quant | Correct | Correct | Regression-test |
| Group-128 weight quant | Correct, including act-order group mapping | Scale/column shapes are incompatible | Repair and compare against legacy |
| Cholesky retry | Damping accumulates across retries | Fresh retry is correct | Repair legacy retry |
| Static cache identity | Omits concrete model/rotation identity | Same class of collision | Strengthen key and all-rank hit policy |
| Seed/cache behavior | Token cache omitted seed | Includes seed | Separate/cache all seed-dependent inputs |
| Multi-GPU lm-eval | Destroys PG and dispatches model | Standard path omits both | Restore legacy lifecycle |

## Final function-by-function finding

| Feature | Legacy after audit | Refactor after audit | Paper consistency |
|---|---|---|---|
| Aggregated Fisher MSE | Full token-aggregated, non-diagonal Fisher and full quadratic | Exact same value/gradient semantics | Yes; independent non-diagonal oracle |
| Adam Block-GD | One bias-corrected Adam step per refresh, trailing columns only; state resets per linear and persists across that linear's shrinking suffix | Same moments, reset boundary, bias correction, clipping, and shrinking active suffix | Core Adam/update/reset matches; clip operator/threshold and 32-sample scheduler cadence are not disclosed |
| Loss slide | Historical refresh-only alpha and penultimate-layer skip | Exact historical behavior | **No**; current formula/pseudocode describe a different schedule |
| Reverse cosine | `sin(pi*x/2)` after correction | Same; constant LR for aware A/K/V | Literal equation matches; endpoint is ambiguous because the separately handled final block consumes the only `x=1` index |
| QuaRot | Full global and local transforms; path-loaded and directly supplied storage-tied heads are cloned/untied before rotation | Shared implementation and wrapper topology | Yes within floating-point tolerance; tied-source/untied Llama and Qwen3 whole-model logits oracles now execute the full transform |
| GuidedQuant saliency Hessian | Per-token/module group `sum(grad²)` after correction, including the standalone GuidedQuant cache and optional dynamic correction | Same shared paper-norm primitive | Yes for the stated squared Euclidean norm; old channel-mean caches are schema-invalidated |
| A quantization | Per-token linear-input fake quant; aware/unaware timing works | Same sites and timing | High-level per-token A is consistent; exact hook sites/axes are an implementation choice |
| V quantization | Per-token `v_proj`-output fake quant | Same | High-level per-token V is consistent; exact hook site is not specified |
| K-cache quantization | Post-RoPE K fake quant, aware/unaware | Same | High-level per-token K is consistent; exact post-RoPE site and independent aware switch are not specified |
| Query/Q quantization | Not implemented | Not implemented | Not claimed by the paper; Q is rotation-only |
| A/K/V range clip | Symmetric quantization with explicit clip ratios | Same, conditional low-bit default 0.9 | Stated symmetric/per-token ratios match; signed integer range and rounding rule are not disclosed |
| Activation-loss clip | Explicit `local_backward_chunk` historical mode and `global_refresh` partition-invariant mode; current/next components remain separate and detached | Same modes, values, gradients, and activation arithmetic order | Matches detached P95; paper does not define the percentile population, so paper-gap runs lock the historical local mode |
| Per-row weights | Correct | Correct | Yes for W4A16 |
| Grouped weights | Correct group-128/act-order/short tail | Repaired to match all tested legacy paths | Group-128 operator is consistent; W2/W3 full-pipeline runs were not performed |
| Final KL | Full-vocabulary fp32 KL | Same | Yes |
| KL/PPL evaluation | Correct shifted-token PPL and full-vocabulary fp32 KL after fixes | Same evaluator | Metric math matches; evaluation chunk, tokenizer/revisions, and software version are undisclosed; synthetic held-out smoke only |
| Ten-task evaluation | Shared exact task list and metric-key handling | Restored rank-0 lifecycle | Task discovery verified; full datasets unavailable on node |
| Checkpoint reload | Versioned safe manifest plus explicit legacy opt-in | Same format and runtime reconstruction | Stronger than the original paper requirement |

## Validation matrix

The small-model integration fixture uses a deterministically initialized tiny
Llama causal LM, deterministic synthetic token IDs, short sequences, and at
least three transformer blocks so the retained slide policy can be exercised.
It is intentionally a real Hugging Face decoder rather than a mock linear
stack.

| Case | W | Group | Rotate | A/K/V | Mode | Clips | Slide |
|---|---:|---:|---:|---|---|---|---:|
| `w4_row_fp` | 4 | per-row | off | 16/16/16 | unaware | disabled | off |
| `w4_row_rotate_slide` | 4 | per-row | on | 16/16/16 | unaware | loss 0.95 | on |
| `w4_g128_actorder` | 4 | 128 | off | 16/16/16 | unaware | weight clip | on |
| `w4a4kv4_unaware_clip09` | 4 | 128 | on | 4/4/4 | unaware | loss + A/K/V 0.9 | on |
| `w4a4kv4_unaware_clip1` | 4 | 128 | on | 4/4/4 | unaware | disabled | on |
| `w4a4v4_aware` | 4 | 128 | on | 4/16/4 | aware A/V | loss 0.95 + A/V 0.9 | on |
| `w4k4_aware` | 4 | 128 | on | 16/4/16 | aware K | loss 0.95 + K 0.9 | on |
| `w4a4kv4_aware` | 4 | 128 | on | 4/4/4 | aware | loss 0.95 + A/K/V 0.9 | on |

The first bring-up may use only the first transformer layer and smaller hidden
dimensions. The final matrix must retain at least one three-layer slide case
and one final-layer full-vocabulary KL case.

## Change log

### 2026-07-24

- Completed three-way static audit of paper, legacy implementation, and
  refactored implementation.
- Submitted the named one-node/eight-GPU Canoe environment and verified its
  immutable job spec.
- Began three independent correction tracks: group quantization, paper
  formulas/defaults, and aware/rotation isolation.
- Added opt-in `alignment_trace_path` / `alignment_run_id` controls to both
  implementations. Each block-GD refresh now emits the same identity
  `(layer, module, block, col_start, col_end, adam_step)`, the exact global
  sample-mean loss used for that Adam update, its current/next slide
  components, the explicit slide alpha (including the first `1.0` step), and
  global calibration sample IDs. With tracing disabled, the historical
  gradient/count collective layout is unchanged.
- Tightened the trace comparator to reject configuration mismatches, empty
  traces, component-loss drift, missing/extra/duplicate identities, sample
  mismatch, and slide-alpha mismatch in addition to the primary `<1%` loss
  requirement. Trace metadata includes separate calibration, rotation, and
  refresh seed domains, static-precompute/Hessian batch sizes, Fisher
  distribution/clipping controls, and normalized algorithm switches (global
  loss, Fisher refresh, Adam, Block-GD, global sample shuffle, analytical-term
  disablement, and second-order scale).
- Corrected the reverse-cosine schedule to
  `sin(pi/2 * layer_idx/(num_layers-1))`; aware activation quantization uses
  the paper's constant LR.
- Made full-vocabulary Fisher and final/evaluation KL the defaults and moved
  distribution math to fp32.
- Corrected global P99 saliency clipping and initially added a
  partition-invariant global-refresh P95 activation-loss path.
- A paper-runtime cross-check then established that the paper did not specify
  global P95 axes and that every pre-audit implementation used
  rank/backward-chunk-local P95 without a prepass. Added
  `a_loss_clip_scope={local_backward_chunk,global_refresh}` to both
  implementations, made the historical mode the default and paper-gap
  manifest setting, retained global mode as an explicit alternative, and
  included the scope in trace provenance (schema 3).
- Focused post-change matrix
  (`../experiment_data/p95_scope_alignment_20260724/matrix_v2/matrix_report.json`):
  real Hugging Face three-layer Llama fixture, W4 per-row, rotate, slide, and
  `a_loss_ratio=0.95` with explicit `local_backward_chunk`. Two legacy and two
  refactored runs produced four passing comparisons over 27 refreshes each;
  every loss relative difference and final tensor-state absolute difference
  was exactly zero, and checkpoint manifests matched. The report binds the
  tested dirty source to diff SHA256
  `6903d98caf6b9a20c058d9e2c6ff3a1611991a27b2c14289b8272801906e9db1`;
  the coherent commit that follows supersedes that working-tree identity.
- Repaired refactored group-128 parameter shapes, short-tail handling,
  act-order natural-column mapping, rank-group synchronization, and batched
  Hessian inversion arithmetic.
- Isolated FP teacher/precompute/replay paths from aware A/V/K fake
  quantization and fixed rotation of cloned-but-untied embeddings/heads,
  including the direct model-object API rather than only path-loaded
  checkpoints.
- Split calibration, rotation, and refresh RNG domains; made token/static/eval
  caches identity-safe, all-rank-consistent, validated, and atomically written.
- Added a safe versioned quantized-checkpoint manifest. Runtime A/V/K behavior,
  rotation, model/tokenizer/dtype identity, and weight provenance now survive
  old/new checkpoint round trips; unsafe legacy pickle loading is opt-in.
- Fixed KL/PPL next-token and fp32 evaluation math, exact ten-task resolution,
  and the refactored multi-rank `lm-eval` lifecycle.
- Found a real 7.0138% old/new failure in the first matrix. The cause was
  activation-loss clipping arithmetic order: legacy used bf16
  `delta -> clip -> fp32 quadratic`, while the refactor clipped fp32 delta.
  Restoring the legacy order made the value and gradient bit-exact and closed
  the end-to-end failure.
- Added independent mathematical oracles for non-diagonal aggregated Fisher,
  Fisher value/gradient, two bias-corrected Adam steps and prefix locking,
  reverse cosine, A/K/V quantizers and clips, global activation-loss clipping,
  group/per-row weights, closed-form full-vocabulary KL value/gradient, PPL
  shift, and tied/untied Llama/Qwen3 whole-model QuaRot logit invariance.
- Corrected GuidedQuant saliency from the historical per-group channel mean
  to the paper's squared Euclidean norm (`sum(grad²)`) in the legacy static
  collector, refactored static collector, standalone GuidedQuant cache, and
  optional dynamic-saliency correction. The refactored cache schema and both
  legacy cache tags were advanced so a pre-correction cache cannot silently
  survive the semantic change. A direct collector-to-`X^T diag(s) X` oracle
  prevents the group-size scale error from being hidden by GPTQ's scale
  invariance.
- Added a fixture `--tokens-only` mode that materializes all legacy/refactored
  cache names for a new global sample count without rewriting model,
  tokenizer, config, or manifest files. This closed an eight-rank offline
  preflight failure caused by requesting 16 samples from an eight-sample
  synthetic fixture.
- Made the convenience launcher choose the paper's weight grouping by bit
  setting: per-row for W4A16 and group 128 for W2/W3 or any A/K/V-low-bit
  configuration. It remains a template, not an exact paper-row launcher.
- Aligned the refactored K-aware configuration contract with legacy GPTQ+:
  `k_cache_quant_aware_gptq=True` now requires an actual low-bit K path rather
  than silently becoming a K16 no-op.

## Validation evidence

- Unit/regression suite on the allocated GPU node: `134 passed`, including
  direct legacy/refactored saliency-hook and downstream Hessian oracles,
  dynamic-saliency-to-finalized-Hessian scale coverage, tied-object full
  rotation, and cache-schema invalidation.
- Safe checkpoint tests: new-to-new and new-to-legacy-entry round trips each
  preserved all 51 tiny-Llama state keys bit-for-bit.
- Root-cause closure
  (`/output/realq_alignment/rootcause_fix_v1/matrix_report.json`):
  `rotate + slide + a_loss=0.95`, four comparisons
  (legacy repeat, refactor repeat, and old/new repeat 0/1), 27 refreshes each,
  maximum loss difference 0, complete tensor-state difference 0, checkpoint
  manifest match.
- Final frozen one-GPU matrix
  (`/output/realq_alignment/matrix_v5/matrix_report.json`): source commit
  `1a88d2bba821e4d93068843fe1d441a7bfd31df9`, empty tracked-diff SHA-256
  (`e3b0c442...b855`), and status containing only user-owned `main.tex`.
  All eight cases above ran twice per implementation: 32 comparisons and 864
  matched refresh-comparison points. The report and an independent pass over
  every raw JSONL record found exact equality for the main/current/next
  losses, refresh identity and Adam step, sample IDs, and slide alpha. All 32
  comparisons had 30/30 canonical tensor keys bit-exact with no missing/extra
  keys, maximum weight absolute difference 0, and exact checkpoint
  manifests/configs.
- Final frozen eight-GPU matrix
  (`/output/realq_alignment/matrix_8gpu_v4/matrix_report.json`): the same
  source commit and clean provenance, world size 8 on eight NVIDIA L20C GPUs,
  with 16 global calibration/refresh samples. The representative
  `w4_row_rotate_slide` and fully aware `w4a4kv4_aware` cases produced eight
  comparisons and 216 matched refresh-comparison points. Raw loss/components,
  identities, Adam steps, all 16 sample IDs, slide alpha, 30/30 canonical
  tensor keys, and manifests were all bit-exact (maximum loss and weight
  differences 0).
- Clean one-GPU matrix
  (`/output/realq_alignment/matrix_v3/matrix_report.json`): all eight cases
  above, two repetitions per implementation, 32 comparisons and 864 matched
  refresh-loss points. Every case passed; the maximum symmetric relative loss
  difference and complete tensor-state absolute difference were both 0, and
  every checkpoint manifest comparison passed. The matrix started from the
  corrected numerical source at commit `a4e5e48`; its runner predated the
  provenance fields added in `29c9879`, so this source boundary is recorded
  here rather than inferred from the report.
- Clean eight-GPU matrix
  (`/output/realq_alignment/matrix_8gpu_v2/matrix_report.json`): source commit
  `c5a0dfcfccc813ade8e2efb6cb0287ab3d77a928`, empty tracked-diff hash, and
  repository status containing only user-owned `main.tex`. The two
  representative cases (`w4_row_rotate_slide` and `w4a4kv4_aware`) produced
  eight comparisons and 216 matched refresh-loss points on eight NVIDIA L20C
  GPUs. Every loss difference, tensor/weight difference, sample-index
  difference, slide-alpha difference, and manifest difference was exactly
  zero. The first `matrix_8gpu_v1` attempt stopped during fixture preflight
  because the isolated node had only eight-sample token caches for a requested
  global sample count of 16; it did not enter either quantizer. `--tokens-only`
  created the missing cache variants without altering model/tokenizer
  artifacts, after which `v2` passed.
- Production KL/PPL evaluator smoke
  (`/output/realq_alignment/eval_smoke_rootcause.json`): deterministic
  held-out synthetic tokens, full-vocabulary KL and shifted-token PPL;
  old/new KL `0.003448204603046179`, PPL `570.77783203125`, both differences
  exactly 0.
- Fully aware A4KV4 production-evaluator smoke
  (`/output/realq_alignment/eval_smoke_aware.json`): restored the legacy and
  refactored matrix checkpoints through the real rotation and aware A/K/V
  runtime wrappers, then ran the real decoder on deterministic held-out
  tokens. Both produced KL `0.006344683468341827` and PPL
  `576.2203979492188`; both relative differences were 0.
- Final-HEAD fully aware A4KV4 production-evaluator smoke
  (`/output/realq_alignment/eval_smoke_aware_v5.json`): restored the final
  matrix-v5 legacy/refactored checkpoints and executed the real decoder,
  rotation/runtime A/K/V wrappers, full-vocabulary fp32 KL, and shifted-token
  PPL on deterministic held-out tokens. Both produced KL
  `0.006439953576773405` and PPL `583.7960205078125`; both symmetric relative
  differences were exactly 0.
- Real `lm-eval` task discovery on the node loaded 2,953 task definitions.
  Each paper task resolved uniquely:
  `piqa`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`,
  `lambada_openai`, `ceval-valid`, `boolq`, `openbookqa`, `social_iqa`.
  This validates task naming/discovery, not full dataset scoring.
- Coherent commits:
  - `a4e5e48` — numerical semantics, caches, evaluation, and checkpoints;
  - `29c9879` — independent oracles and alignment matrix runner;
  - `e0aaf80` — final audit assertions and provenance capture;
  - `c5a0dfc` — paper-norm saliency and distributed fixture hardening;
  - `5c75cff` — dynamic-saliency/cache, tied-object rotation, and protocol
    closure;
  - `e8539ca` — reject invalid K-aware/K16 no-op configurations;
  - `1a88d2b` — keep checkpoint-manifest tests on valid placeholder configs.

## Known paper/code mismatch: legacy loss slide

For a module with `N_m=ceil(C_m/B)` column blocks, the paper counts all
`sum_m N_m` blocks. The implementation has no trailing weights to update after
the final block, so it counts only
`sum_m max(N_m-1, 0)` actual refreshes and stretches alpha from 1 to 0 over
that shorter sequence.

The clean tiny-Llama trace makes the discrepancy directly observable. Each
transformer layer has seven linears: q/k/v/o/up/gate have two blocks each and
down has four, so the paper has `B_total=16` while the implementation has nine
trailing-column refreshes. The actual first-layer alpha sequence is

```text
legacy/refactor: [1, 7/8, 6/8, 5/8, 4/8, 3/8, 2/8, 1/8, 0]
paper at the same effective refreshes:
                 [1, 13/15, 11/15, 9/15, 7/15, 5/15, 3/15, 2/15, 1/15]
```

The final paper block would have alpha 0 but cannot trigger a trailing-column
update. In the same trace, all nine layer-1 (penultimate-layer) refreshes have
no next-layer component or alpha, as do the final-layer refreshes.

The paper also slides every non-final transformer block into its successor,
whereas both implementations skip the penultimate-to-final transition and
reserve true KL for the final block. Old and new are mutually consistent, but
neither is fully consistent with `main.tex:195-203,457-463`.

## Reverse-cosine endpoint ambiguity

The corrected code implements the equation in `main.tex:622-627` literally
over all `L` transformer-block indices:

```text
x = layer_index / (L - 1)
scale = sin(pi*x/2)
```

The same appendix separately says that the final transformer block uses its
own final-layer learning rate and true KL. Consequently the scheduled branch
is used only for indices `0 .. L-2`; its deepest actual non-final block has
`x=(L-2)/(L-1)` and never reaches the reported scheduled endpoint. An equally
plausible interpretation would normalize the `L-1` non-final blocks over
`0 .. L-2`, making the last non-final block reach the target. Old and new use
the first interpretation exactly. The paper needs to disambiguate the
denominator before changing this behavior.

## Validation boundaries and paper-reproduction blockers

- Full zero-shot task scoring is unavailable on the isolated node because the
  task datasets are not present locally; only real task registration plus
  mocked evaluator lifecycle can be validated without network data.
- End-to-end integration matrices use the tiny Llama fixture. Qwen3 has
  tied/untied whole-model rotation oracles, but no full
  static-precompute-to-Block-GD-to-A/K/V-to-evaluation integration run, so the
  tiny-Llama result must not be extrapolated to every paper model family.
- The paper's 70B FSDP plus activation-checkpointing memory/timing protocol was
  not reproduced. The eight-GPU tiny-model run validates distributed
  numerical semantics, not the 70B systems or performance claims.
- Exact paper-table reproduction remains blocked on the undisclosed settings
  listed in `docs/REALQ_PAPER_PROTOCOL.md`.

## Formal schedule A/B completion and performance handoff

The controlled formal campaign completed on 2026-07-24 under source commit
`2702d78836cc1a3eb9f8ab5f9c7a0b2b9c26ff14`, empty tracked diff, and wrapper
SHA256
`0a7c70f9fe9dcd34a816c75ca13025aacf6b94df906deec2b29795e1ae88e0b3`.

The Q4 literal-sine arm produced raw KL/PPL
`0.22162500023841858 / 14.133820533752441`; historical cos-squared produced
`0.09558671712875366 / 13.663459777832031`. Literal sine therefore increased
KL by 131.8575% and PPL by 3.4425% in the locked current protocol.

The Q8 aware arms ordered as constant, literal sine, then historical
cos-squared:

```text
constant aware: 0.9197931289672852 / 20.18940544128418
literal sine:   0.9449643492698669 / 20.760927200317383
historical:     0.9737851023674011 / 21.33316421508789
```

Constant-aware reduced KL/PPL by 5.5445%/5.3614% versus historical
cos-squared. The best arm remained 19.9209% above the paper KL anchor and
2.5884% above its PPL anchor. The primary and independent validators passed;
the independent pass ran 782 checks with no failure and recomputed the eight
current Q8 cache hashes.

Full evidence is in
[`REALQ_AB_PERF_RESULTS.md`](REALQ_AB_PERF_RESULTS.md). Performance work starts
only after this boundary and follows
[`REALQ_PERFORMANCE_WORKLOG.md`](REALQ_PERFORMANCE_WORKLOG.md): exact candidates
must pass raw-byte and real-model checkpoint gates; potentially
numerically-different candidates remain explicit default-off switches.
