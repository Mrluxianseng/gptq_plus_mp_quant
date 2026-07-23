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
   that the legacy slide-window indexing and terminal-layer policy are retained
   by explicit user request.
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
| `w4a4kv4_unaware` | 4 | 128 | on | 4/4/4 | unaware | A/K/V 0.9 | on |
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

## Pending evidence

- Unit-test output and exact environment package versions.
- Per-case, per-step old/new loss comparison reports.
- Same-seed repeatability reports for both implementations.
- Final weight/output comparison and held-out evaluation smoke results.
- Commit IDs for each coherent correction set.
