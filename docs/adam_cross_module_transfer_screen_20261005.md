# REAL-Q cross-module Adam-state transferability screen (2026-10-05)

> **Status (2026-10-06):** This file contains early transfer pilots and historical values. Later matched experiments use the current REAL-Q entry point, log-space shrinkage, and independent (m/v) bias ages. The consolidated method-by-method conclusions and latest valid comparisons are in [`adam_optimization_journal.md`](adam_optimization_journal.md). Do not pool the older campaign figures with those current matched pairs or present them as current-code replications.

## Question and scope

REAL-Q resets Adam state at linear-module boundaries, so its authors identify cross-module optimizer-state transfer as future work. The concrete engineering constraint here is to avoid an extra warmup backward pass for each module. This pilot checks whether a previous layer's same-role second-moment state contains a useful coarse prior for the next layer. It does **not** test a quantization-quality improvement.

The probe is opt-in through `REALQ_ADAM_TRANSFER_DIAG_DIR`. It observes the normal refresh gradient and Adam second moment, makes no optimizer-state or parameter changes, and adds no backward pass. It compares normalized per-output-row profiles for `self_attn.o_proj` and `mlp.down_proj`. For each adjacent layer pair it matches the same refresh index and requires identical active input-suffix width, avoiding the initial pilot's confound between a donor's final narrow suffix and a target's broad first suffix.

## Pilot configuration

- Model: local Qwen3-0.6B; W4, `w_clip=true`, `num_groups=4`, `blocksize=128`, `act_order=true`.
- Calibration: WikiText-2, 8 samples, sequence length 64; seeds 42 and 43.
- Quantized prefix: layers 0 through 8; evaluation intentionally skipped.
- Compared refreshes: first three refreshes for adjacent same-role modules.
- Environment: local WSL Ubuntu, ext4 Python environment, RTX 4060 Laptop GPU (8 GB).

This is a mechanism screen with a deliberately small calibration set, not a thesis-quality result or a full-model benchmark.

## Results

Pearson correlations below are averages over the three matched refresh indices; seed values are shown separately. Spearman correlations followed the same broad pattern.

| Donor → target | Module | Pearson, seed 42 | Pearson, seed 43 | Reading |
|---|---|---:|---:|---|
| 0 → 1 | `o_proj` | 0.18 | 0.26 | weak |
| 0 → 1 | `down_proj` | 0.05 | 0.12 | weak |
| 1 → 2 | `o_proj` | 0.09 | 0.17 | weak |
| 1 → 2 | `down_proj` | 0.02 | 0.10 | weak |
| 2 → 3 | `o_proj` | 0.04 | 0.10 | weak |
| 2 → 3 | `down_proj` | 0.04 | 0.32 | seed-sensitive |
| 3 → 4 | `o_proj` | 0.28 | 0.27 | moderate and repeatable |
| 3 → 4 | `down_proj` | 0.55 | 0.43 | moderate and repeatable |
| 4 → 5 | `o_proj` | 0.35 | 0.24 | modest/moderate |
| 4 → 5 | `down_proj` | 0.55 | 0.42 | moderate and repeatable |
| 5 → 6 | `o_proj` | 0.11 | 0.28 | weak/seed-sensitive |
| 5 → 6 | `down_proj` | 0.25 | 0.41 | modest |
| 6 → 7 | `o_proj` | 0.27 | 0.28 | modest and repeatable |
| 6 → 7 | `down_proj` | 0.40 | 0.47 | moderate and repeatable |
| 7 → 8 | `o_proj` | 0.46 | 0.47 | moderate and repeatable |
| 7 → 8 | `down_proj` | 0.56 | 0.58 | moderate and repeatable |

The row-profile similarities are layer-dependent rather than universal. They are more consistent in the later tested pairs, especially `down_proj`, but weak in the first two transitions. A global “copy the previous module's moments” rule is therefore unsupported.

The magnitude of the donor's mean rowwise `vhat` relative to the target's current mean rowwise `g²` is also not consistently one. It ranges from far below one to hundreds of times larger for some pairs/refreshes. Copying raw `v` is unsafe. A future candidate would need a normalized row prior and target-side scale calibration, using the target gradient already computed in the normal refresh.

## Interpretation and theory connection

Adam's second moment is an exponential moving estimate of squared gradients. A state prior has a plausible role only when it predicts part of the target gradient's scale structure. The repeated moderate row-profile correlations in several deeper adjacent pairs support testing a **selective, scale-calibrated rowwise prior**. Weak early-layer correlations argue for a no-transfer fallback. The correlations do not prove that the prior improves Adam's update or final KL/PPL; the rowwise projection also discards within-row coordinate structure.

This direction is aligned with REAL-Q's stated future work on optimizer-state transfer across modules. CPAL's study of nonzero second-moment initialization and Adam-Rel's analysis of Adam under changing gradient magnitudes support examining moment initialization and bias-correction age, but neither validates this cross-layer transfer mechanism: [REAL-Q](https://arxiv.org/abs/2609.00049), [CPAL second-moment initialization](https://mlanthology.org/cpal/2025/abuduweili2025cpal-revisiting/), [Adam on Local Time](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2733d3b0dde1d74995f35a9cf442d38-Abstract-Conference.html).

This pilot supports only the premise that some same-role output-row structure may transfer. It does not establish benefit, a safe bias-correction age, or a universally applicable transfer rule. The project's Adam state age and discrete refresh semantics must be preserved explicitly in the next candidate.

## Gated candidate: 3-seed partial-prefix A/B

An opt-in candidate was implemented only for layer 8 `mlp.down_proj`. On its first ordinary refresh it uses layer 7's matched row profile only if the online Pearson correlation exceeds 0.4. It normalizes the donor row profile, scales it by the target gradient's already-available mean `g²`, seeds only `v` (keeps `m=0`), and advances the bias-correction age by one virtual prior step. There is no extra backward. The default path remains unchanged. All three candidate logs show exactly one transfer event, with observed correlations 0.577, 0.592, and 0.529.

Each pair uses the same seed/configuration and a 16-sample WikiText-2 KL/PPL evaluation after quantizing layers 0–8. These numbers are only a small prefix pilot, not full-model thesis evidence.

| Seed | Control KL | Candidate KL | KL change | Control PPL | Candidate PPL | PPL change |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.127095 | 0.127379 | +0.000284 (worse) | 60.081 | 59.958 | -0.123 |
| 43 | 0.145597 | 0.144230 | -0.001366 | 62.093 | 61.835 | -0.258 |
| 44 | 0.129471 | 0.128538 | -0.000932 | 59.938 | 59.666 | -0.272 |
| Mean | 0.134054 | 0.133383 | -0.000672 (-0.50%) | 60.704 | 60.486 | -0.218 (-0.36%) |

The partial-prefix result is mixed on KL and too small to support a conclusion. We therefore also ran the candidate and control through **all 28 layers** on two matched seeds, still with the same deliberately small 8-sample calibration set and 16-sample evaluation:

| Seed | Control full-model KL | Candidate full-model KL | KL change | Control PPL | Candidate PPL | PPL change |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.403959 | 0.403208 | -0.000751 (-0.19%) | 71.326 | 70.966 | -0.360 (-0.50%) |
| 43 | 0.476089 | 0.460194 | -0.015895 (-3.34%) | 79.093 | 77.042 | -2.051 (-2.59%) |
| 44 (held out from target selection) | 0.381644 | 0.385682 | +0.004038 (+1.06%, worse) | 71.481 | 72.400 | +0.918 (+1.28%, worse) |
| Mean | 0.420564 | 0.416361 | -0.004203 (-1.00%) | 73.967 | 73.469 | -0.498 (-0.67%) |

The held-out seed reverses both metrics despite passing the row-correlation gate (correlation 0.529). This shows that coarse output-row similarity alone is insufficient to predict quality benefit. The mean across three seeds is still slightly positive, but this candidate is **not robustly validated** and should not be claimed as a completed thesis contribution. A plausible weakness is that the prior matches rowwise scale structure but ignores within-row coordinate structure; the Adam denominator is elementwise, so row-profile agreement need not imply a helpful elementwise preconditioner.

The observed quantization times were similar (control/candidate 5:04/4:57 at seed 42 and 5:20/4:54 at seed 43; seed 44 was also similar), but this small run is not a speed benchmark. No additional backward pass was introduced. The next useful work is mechanism diagnosis of the held-out failure—especially whether within-row donor/target `v` profiles align—before deciding whether to refine or reject the transfer. A larger calibration/evaluation run should wait until that check gives a stronger rationale.

## Artifacts

- Read-only logger: `realq/refresh/adam_transfer_diagnostic.py`
- Hook and opt-in update: `realq/refresh/block_gd.py` (diagnostic requires `REALQ_ADAM_TRANSFER_DIAG_DIR`; update requires `REALQ_ADAM_V_TRANSFER_MODE=normalized_row_prior` plus explicit target settings)
- Unit tests: `tests/test_adam_transfer_diagnostic.py`
- Raw runs: `outputs/adam-transfer-row-profile-v2-seed{42,43}-n8-t64-l8/`
- Paired prefix A/B runs: `outputs/adam-vinit-{control,rowprior-downproj-l8}-seed{42,43,44}-l8-n8t64/`
- Paired full-model A/B runs: `outputs/adam-vinit-full-{control,rowprior-downproj-l8}-seed{42,43,44}-n8t64/`
- Earlier two-seed 0–4 pilot used a mismatched donor suffix and is retained only as a debugging record; it must not be used as evidence.

## REAL-Q entry-path audit and forced-effect diagnostic (2026-10-06)

The transfer hook lives in `realq/refresh/block_gd.py`; the legacy
`scripts/gptq_plus_lr_sweep.sh` launcher invokes the separate top-level
`ptq.py`/`gptq_utils` path and does not exercise this hook. The first attempts
to reproduce the update-effect diagnostic used that legacy launcher, so their
absence of transfer logs was a runner mismatch rather than evidence about the
REAL-Q hook. Subsequent diagnostics used `python -m realq.ptq` directly.

On the REAL-Q path, Qwen3-0.6B, W4, seed 44, 8 WikiText-2 calibration samples
of 64 tokens, layers 0–8, and the default sliding-loss configuration, the
target entry marker appeared exactly once. With the original correlation
threshold 0.4, the online row-profile correlation was 0.17315 and the
candidate correctly skipped transfer. With the gate deliberately lowered to
zero for a mechanism-only probe, the same refresh applied the prior and
reported:

| Diagnostic | Value |
|---|---:|
| Donor/target scale ratio | 1.882 |
| Cosine: transferred vs ordinary first-step Adam update | 0.90068 |
| Relative L2 change in update | 0.43677 |
| Coordinatewise candidate/base `vhat` ratio, p05 / median / p95 | 0.608 / 1.549 / 124.998 |
| Target within-row coefficient of variation of `g²` | 1.445 |

This confirms that the row prior can substantially reshape the elementwise
preconditioner even though it uses only row-level similarity. In the default
gated run it is rejected for this seed, so this is not a quality result and
does not justify lowering the threshold. No KL/PPL evaluation was run for the
forced mechanism probe. The first forced-effect run incorrectly advanced the
shared `adam_step` for both moments even though only `v` was initialized; this
also changed `m` bias correction and overstated the relative update change
(0.62563). That bookkeeping defect is fixed: the target's `m` correction stays
at step 1 while only `v` uses virtual step 2. The next-refresh age audit
confirmed `m_step=2, v_step=3`; the table reports the corrected rerun. The
results and raw logs are in
`outputs/adam-vtransfer-realq-entryaudit-seed44-slide/`,
`outputs/adam-vtransfer-realq-forced-effect-seed44-v3/`, and
`outputs/adam-vtransfer-realq-forced-effect-seed44-v3/adam_vtransfer_realq_forced_effect_seed44_v3/`.

The next candidate should make the prior conservative at the coordinate
level—for example, a shrinkage blend in log-`vhat` space between the target's
first-step `g²` and the normalized donor row prior—then measure held-out KL
after the real quantization trajectory. Such a blend must be justified by
repeatable post-quantization gains; this diagnostic alone only motivates
controlling the update perturbation. `py_compile` passed for the modified
modules. The WSL environment has no `pytest` installation, so the unit suite
could not be run there.

## Coordinate-shrunk prior: fresh matched REAL-Q test (2026-10-06)

The historical layer-8 full-model records above are not directly reproduced by
the current entry/configuration: for seed 44, their saved online correlation
was 0.529, while a fresh run through `python -m realq.ptq` gave 0.173 for the
same named layer/module. The historical campaign did not preserve enough exact
runner/source provenance to attribute this discrepancy. Treat its result as
historical only; do not use it to set the current gate or claim a repeatable
gain.

The current matched screen used the actual REAL-Q entry point, one seed (44),
8 WikiText-2 calibration windows of 64 tokens, W4, 128-column blocks, and
quantized layers 0–5. Both arms used the same calibration/evaluation caches,
refresh seed, sample order, FP reference, and 16-window/128-token WikiText-2
evaluation. The sole treatment was the opt-in Adam-`v` prior at layer 5
`mlp.down_proj`; its current online profile correlation was 0.411815 and it
passed the pre-existing 0.4 gate. The donor influence was blended in log space
with the already-computed target coordinatewise `g²`, using donor weight 0.5.
This adds no backward pass.

| Arm | KL | PPL | Quantization wall time |
|---|---:|---:|---:|
| Original REAL-Q Adam | 0.11036846 | 58.52295 | 139.19 s |
| 0.5 donor-weight prior | 0.11105838 | 58.81338 | 126.84 s |
| Candidate change | +0.00068992 (+0.63%) | +0.29044 (+0.50%) | not interpreted |

The candidate log confirms exactly one transfer event and correct independent
bias ages (`m_step=1`, virtual `v_step=2`; next refresh `m_step=2`, `v_step=3`).
Despite blending, its first update differed substantially from ordinary Adam:
update cosine 0.9151, relative update L2 change 0.4450, coordinatewise corrected
`v` ratio p05/median/p95 = 0.718/2.490/27.738. The target's coordinatewise
`g²` within-row coefficient of variation was 26.13, while the donor was 4.16×
the target mean scale before normalization. Row-profile correlation therefore
did not constrain the fine-grained coordinatewise preconditioner enough; the
candidate slightly worsened both measured quality metrics in this one-seed
prefix screen. This does not prove every transfer is harmful, but it rejects
the current 0.5 blend as a useful result and provides no basis to promote it.

The run completed both arms, source hashes remained unchanged, and the
candidate event count was one. The current WSL environment still lacks
`pytest`; `py_compile` and the real GPU execution covered the edited path, but
the newly added zero-shrinkage unit test has not been run under pytest. Raw
artifacts are in
`outputs/adam-vtransfer-shrinkage-seed44-layer5-down-shrink05/`; the paired
runner is `tools/run_adam_vtransfer_shrinkage_pair.py`.

The next scientifically useful step is to stop treating row correlation as a
proxy for coordinatewise Adam usefulness. Before any further quality run,
measure target/donor `vhat` similarity at the coordinate or structured
sub-block level on identical active suffixes, and predict the exact Adam
update perturbation from saved refresh tensors. If those measurements do not
identify a stable, less destructive transferable component, reject
cross-layer `v` transfer and investigate within-module refresh-state
continuation instead. Do not run an unrestricted shrinkage-factor sweep.
