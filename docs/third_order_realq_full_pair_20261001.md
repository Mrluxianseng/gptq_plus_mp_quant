# Matrix-free third-order correction in REAL-Q

Date: 2026-10-01  
Status: no repeatable KL improvement established. User halted further model runs. One deterministic-SDPA candidate was positive under math-SDPA evaluation; its repeat stopped before final evaluation, and no deployment-SDPA matched pair was completed.

Latest static audit: [三阶失败原因审查](third_order_failure_analysis_20261001.md). It derives the estimator's exact remaining-error identity and distinguishes confirmed limitations from unproven causal explanations. No further model experiment was launched for that audit.

## Method and scope

At one selected REAL-Q layer, the opt-in refresh estimates the third-order output-gradient contraction from symmetric suffix evaluations, without constructing a rank-3 tensor:

```text
g_candidate = g(y0) + F d + 1/2 * [g(y0+d) + g(y0-d) - 2*g(y0)]
```

Here `d` is the actual quantized block residual, `F d` is REAL-Q's existing Fisher-MSE output gradient, and each `g(.)` is the full-vocabulary KL gradient through the frozen FP suffix. At finite `d`, the central difference also contains fourth- and higher-order terms. The correction is enabled only for `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj` at layer 4. The rest of the model uses the existing REAL-Q path. Full runs use Qwen3-0.6B, W4, 32 WikiText-2 calibration sequences ×256 tokens, seed 44, and the same eight fixed 1024-token evaluation windows.

## Results before deterministic-SDPA fix

The control repeated exactly under this backend: KL `0.22546510398`, PPL `32.7383155823`. The third-order candidate was repeated three times with the same saved config. The candidate outcomes vary substantially:

| Run | Exact KL | PPL | KL change vs control |
|---|---:|---:|---:|
| Control | 0.22546510 | 32.73831558 | — |
| Cubic run 1 | 0.23257132 | 32.91956711 | +3.15% |
| Cubic run 2 | 0.22407226 | 32.57554626 | −0.62% |
| Cubic run 3 | 0.23024461 | 32.63245392 | +2.12% |
| Cubic mean ± sample SD | 0.22896273 ± 0.004391 | 32.709189 ± 0.19148 | +1.55% mean |

This does **not** establish a KL improvement. The mean KL regresses, and the between-run spread is larger than the apparent best-run gain. PPL mean is nearly unchanged (−0.09%), also with large variation. Do not select run 2 as a success.

An alternative diagnostic replaces the Fisher output-gradient approximation with the exact downstream KL output gradient at layer 4. This is a diagnostic of the output-gradient approximation, not an algorithmic improvement claim. Across three runs, mean KL was `0.22860544` (+1.39% vs control) and mean PPL `32.535722` (−0.62%). Exact local output gradients therefore did not yield a lower final KL after the rest of the REAL-Q quantization trajectory.

## What the audits say

The first-refresh output-gradient audit at layer 4 is stable across cubic repetitions on each backend. Under the original backend, candidate cosine against the exact downstream-KL gradient was about `0.36` for `up_proj`, `0.18` for `gate_proj`, and `0.26` for `down_proj`. Under deterministic math SDPA it was `0.468`, `0.192`, and `0.634`. The candidate is closer than plain Fisher in those local measurements, but remains imperfect and projection-dependent. Under math SDPA, relative gradient errors computed from logged norms/cosines are approximately `1.00`, `2.01`, and `0.78`, respectively. The residual ratio `||d||/||y0|| ≈ 0.41–0.43` warrants caution about finite-step contamination, but does not by itself establish Taylor failure without derivative/remainder evidence.

BF16 symmetry was measured directly for `y0 ± d`. Relative positional asymmetry was `0.24–0.41%` for the three projections. This does not bound the resulting gradient error, so it cannot rule out numerical error as an important factor.

## Reproducibility issue found and fixed

`utils/reproducibility.py` defined `configure_deterministic_sdpa()` to turn off flash and memory-efficient SDPA backward kernels, but `configure_reproducibility()` never called it. The pipeline requested deterministic algorithms with `warn_only=True`, allowing a memory-efficient attention backward path without stopping the run. I connected the existing helper to deterministic setup; logs now record flash SDPA off, memory-efficient SDPA off, and math SDPA on. The three older cubic runs and exact-downstream diagnostics predate that change and are exploratory.

The first deterministic-SDPA pair completed with the same saved REAL-Q configuration except the layer-4 correction flag. Final evaluation also used math SDPA. These are valid paired observations for that backend, but do not establish performance on the default inference backend:

| Run | Exact KL | PPL | End-to-end time |
|---|---:|---:|---:|
| Control, correction off | 0.23593497 | 33.28921127 | 15m 36s |
| Cubic, layer 4 | 0.23062909 | 32.81828690 | 25m 24s |

The cubic candidate reduced KL by **2.25%** and PPL by **1.42%** in this one pair, at about **63% longer runtime**. The same-config repeat was halted at layer 27, before evaluation, so this is not a replicated result. The backend switch itself changes baseline absolute KL/PPL relative to the old attention backend, so only within-backend paired comparisons are valid.

Because math SDPA also changes inference numerics, I added a separate phase that uses default SDPA to generate teacher logits, deterministic math SDPA during quantization, and default SDPA for final KL/PPL. This code passed syntax/unit checks, but the full pipeline with this train/eval backend split was not run. The repeated candidate was launched before this change and stopped at layer 27, so it provides no final metric.

## Code and checks

* CLI checks: `process_args.py`, `--third_order_refresh_layer` and `--third_order_refresh_estimator`.
* Estimator integration and cache use: `gptq_utils/gptq_plus_utils.py`, inside `collect_true_weight_gradient`.
* Replayer for exact saved run arguments: `tools/run_third_order_symmetry_smoke.py` (`--estimator off` reproduces the baseline path).
* Analytic checks: `tests/test_third_order_taylor.py` — 2 checks pass.
* Syntax checks pass for the modified REAL-Q and runner files.

The initial full run review caught a missing `fp_inps_final` cache dependency; that was fixed before these matched runs. All completed runs quantized all 28 layers and reached final evaluation without OOM, traceback, or non-finite gradients. The later repeat was intentionally stopped before final evaluation at layer 27 per user instruction. The new teacher/quantization/eval SDPA split remains unvalidated end to end.

## Root-cause assessment

The strongest confirmed limitation is algebraic: `gc = gF + (g+ + g-)/2`, hence `gc-g+ = gF-(g+-g-)/2`. The correction retains Fisher's approximation to the odd part of the gradient, including the leading `(F-H)d` error. An O(d²) cubic-gradient addition does not systematically repair an O(d) curvature error. The earlier toy test used an exact Hessian-vector product as its quadratic base, unlike this integrated Fisher-based estimator.

The audit also has two observability limitations: it records only the first batch per module, and the injected-gradient implementation deliberately retains the original Fisher scalar in loss logs. Neither those scalar losses nor output-space cosine establish that clipped Adam weight updates improve KL. No simple factor-of-two or batch/token reduction mistake was found in the reviewed path.

Finite residual size, BF16 rounding, the FP-versus-quantized suffix mismatch, and the optimizer/discrete trajectory are possible additional causes, not established causal rankings. Direct exact-FP-suffix-gradient diagnostics did not improve mean final KL on the old backend; they are not an upper bound on final quantized-model quality.

Old attention-backward warnings are a reproducibility concern. However, a fresh offline comparison found all 2,000 common `block-metrics` rows of the two math-SDPA candidate logs identical at printed precision, as well as the three first-batch audits. This is not tensor-level identity or a completed final-metric repetition. No third full-model run was launched or is recommended at this stage.

## Logs

* Original control: `outputs/third_order_realq_matched_control_20261001/Qwen3-0.6B/third_order_realq_matched_control_20261001/logs/log_261001_124848.txt`
* Cubic runs: `outputs/third_order_realq_matched_candidate_20261001/.../log_261001_131655.txt`, `outputs/third_order_cubic_repeat2_20261001/.../log_261001_152828.txt`, and `outputs/third_order_cubic_repeat3_20261001/.../log_261001_155148.txt`
* Exact-downstream diagnostics: `outputs/third_order_exact_downstream_full_20261001/.../log_261001_141710.txt`, `outputs/third_order_exact_downstream_repeat2_20261001/.../log_261001_145220.txt`, and `outputs/third_order_exact_downstream_repeat3_20261001/.../log_261001_151027.txt`
* Deterministic math-SDPA pair: `outputs/third_order_sdpa_deterministic_control_20261001/.../logs/` and `outputs/third_order_sdpa_deterministic_cubic_20261001/.../logs/`. The repeat `third_order_sdpa_deterministic_cubic_repeat2_20261001` has no final metric because it was stopped by request.

## Decision rule

Do not widen the correction to more layers or call it a repeatable KL improvement. The completed old-backend mean regressed KL, and the one positive math-eval pair is not fully replicated. Keep experiments stopped as requested. Further discussion should first address the retained lower-order approximation error and the missing evidence between output gradients and actual weight updates; the current evidence does not establish that a particular new direction is necessary or will succeed.
