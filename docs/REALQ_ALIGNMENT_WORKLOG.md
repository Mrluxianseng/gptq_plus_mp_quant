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
   every supported configuration.
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
| Adam Block-GD | One bias-corrected Adam step per refresh, trailing columns only | Same moments, bias correction, clipping, and shrinking active suffix | Yes; independent two-step oracle |
| Loss slide | Historical refresh-only alpha and penultimate-layer skip | Exact historical behavior | **No**; current formula/pseudocode describe a different schedule |
| Reverse cosine | `sin(pi*x/2)` after correction | Same; constant LR for aware A/K/V | Yes |
| QuaRot | Full global and local transforms; cloned untied heads rotate normally | Shared implementation and wrapper topology | Yes within floating-point tolerance; tied/untied Llama and Qwen3 whole-model logits oracles |
| A quantization | Per-token linear input fake quant; aware/unaware timing works | Same sites and timing | Yes |
| V quantization | Per-token `v_proj` output fake quant | Same | Yes |
| K-cache quantization | Post-RoPE K fake quant, aware/unaware | Same | Yes |
| Query/Q quantization | Not implemented | Not implemented | Not claimed by the paper; Q is rotation-only |
| A/K/V range clip | Symmetric quantization with explicit clip ratios | Same, conditional low-bit default 0.9 | Yes |
| Activation-loss clip | One global-refresh P95 detached scale, in legacy activation arithmetic order | Same value and gradient after correction | Yes |
| Per-row weights | Correct | Correct | Yes for W4A16 |
| Grouped weights | Correct group-128/act-order/short tail | Repaired to match all legacy paths | Yes for paper low-bit/A4KV4 settings |
| Final KL | Full-vocabulary fp32 KL | Same | Yes |
| KL/PPL evaluation | Correct shifted-token PPL and full-vocabulary fp32 KL after fixes | Same evaluator | Yes; actual synthetic held-out smoke exact |
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
| `w4a4v4_aware` | 4 | 128 | on | 4/16/4 | aware A/V | A/V 0.9 | on |
| `w4k4_aware` | 4 | 128 | on | 16/4/16 | aware K | K 0.9 | on |
| `w4a4kv4_aware` | 4 | 128 | on | 4/4/4 | aware | A/K/V 0.9 | on |

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
- Corrected global P99 saliency clipping and global-refresh P95 activation-loss
  clipping so neither depends on micro-batch or rank partitioning.
- Repaired refactored group-128 parameter shapes, short-tail handling,
  act-order natural-column mapping, rank-group synchronization, and batched
  Hessian inversion arithmetic.
- Isolated FP teacher/precompute/replay paths from aware A/V/K fake
  quantization and fixed rotation of cloned-but-untied embeddings/heads.
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

## Validation evidence

- Unit/regression suite on the allocated GPU node: `123 passed`.
- Safe checkpoint tests: new-to-new and new-to-legacy-entry round trips each
  preserved all 51 tiny-Llama state keys bit-for-bit.
- Root-cause closure
  (`/output/realq_alignment/rootcause_fix_v1/matrix_report.json`):
  `rotate + slide + a_loss=0.95`, four comparisons
  (legacy repeat, refactor repeat, and old/new repeat 0/1), 27 refreshes each,
  maximum loss difference 0, complete tensor-state difference 0, checkpoint
  manifest match.
- Clean one-GPU matrix
  (`/output/realq_alignment/matrix_v3/matrix_report.json`): all eight cases
  above, two repetitions per implementation, 32 comparisons and 864 matched
  refresh-loss points. Every case passed; the maximum symmetric relative loss
  difference and complete tensor-state absolute difference were both 0, and
  every checkpoint manifest comparison passed. The matrix started from the
  corrected numerical source at commit `a4e5e48`; its runner predated the
  provenance fields added in `29c9879`, so this source boundary is recorded
  here rather than inferred from the report.
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
- Real `lm-eval` task discovery on the node loaded 2,953 task definitions.
  Each paper task resolved uniquely:
  `piqa`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`,
  `lambada_openai`, `ceval-valid`, `boolq`, `openbookqa`, `social_iqa`.
  This validates task naming/discovery, not full dataset scoring.
- Coherent commits:
  - `a4e5e48` — numerical semantics, caches, evaluation, and checkpoints;
  - `29c9879` — independent oracles and alignment matrix runner.

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

## Evidence still outstanding at this point in the log

- Clean eight-rank end-to-end runs for representative slide/P95 and fully
  aware A/K/V cases.
- Full zero-shot task scoring is unavailable on the isolated node because the
  task datasets are not present locally; only real task registration plus
  mocked evaluator lifecycle can be validated without network data.
- Exact paper-table reproduction remains blocked on the undisclosed settings
  listed in `docs/REALQ_PAPER_PROTOCOL.md`.
