# Handoff: REAL-Q acceleration and matrix-free third-order gradient work

Updated 2026-10-01. This note is a map of implementation, evidence, limitations, and the next useful steps for a new owner.

## Important workspace state

The repository has a dirty working tree with many concurrent research changes, including changes predating these two workstreams. Do not run `git reset`, `git clean`, broad checkout/revert commands, or assume the current modifications belong to one experiment. The handoff documents and experiment outputs are local; much of `outputs/` is git-ignored and will not travel through a normal Git clone. Copy the relevant output directories/manifests when handing the project to another checkout.

## Acceleration work

### 1. Fused GPTQ sequential inner-column update — validated opt-in

**Purpose.** Reduce Python/CUDA launch overhead in the formal GPTQ inner loop while preserving its sequential column dependencies, block-GD refreshes, quantized weights, and quality metrics.

**Implementation map.**

* Triton kernel: `gptq_utils/triton_gptq_kernels.py`, function `fused_gptq_column_`.
* Quantizer validation, dispatch, and inner-loop call site: `gptq_utils/gptq_plus_utils.py` around the `GPTQPlus.fasterquant` configuration near lines 4476–4497 and sequential loop near 4740 / 5031.
* CLI: `process_args.py`, `--gptq_inner_kernel {torch,triton_fused}`. Default remains `torch`.
* The implementation rejects unsupported settings: non-`block_gd`, group-parallel path, atomic quantization, two-sided rounding, unsupported per-row symmetric setup, and block size above 128.

The kernel preserves serial dependence from column `i` to `i+1`. It fuses per-column quantize/dequantize, error formation, in-block second-order compensation, and GHinv rank-one update. A prior candidate had a stride bug for compact output buffers; that candidate's results were discarded. The final implementation passes separate strides and was rechecked on the non-contiguous layout.

**Evidence.** The clean full 28-layer Qwen3-0.6B W4 paired run measured 698.33 s (PyTorch) vs. 311.65 s (Triton) quantization time, a 2.24× speedup. Total runner time including setup/evaluation improved 2.11×. The 507-tensor state hash, WikiText-2 KL (0.2254651), and PPL (32.7383156) matched exactly. Peak allocated memory was effectively unchanged; reserved memory rose by about 63 MB. This is the strongest acceleration result and is validated for this configuration; broader settings still need tests.

**Reproduction/evidence files.**

* Main report: `docs/triton_gptq_inner_loop_optimization_20260929.md`.
* Source profile and launch-count analysis: `docs/gptq_inner_kernel_profile_20260929.md`.
* Kernel microbenchmark harness: `tools/prototype_fused_gptq_column.py`.
* Full control/candidate run records: `outputs/phase_profile_triton-inner-control-full28-cleanrecheck-timeout4h/` and `outputs/phase_profile_triton-inner-candidate-full28-stridefixed-timeout4h/`.
* Source/cmd manifests: `outputs/gptq_kernel_control_manifest_recheck_20260930.json` and `outputs/gptq_kernel_candidate_manifest_stridefixed_20260930.json`.
* Nsight summaries and traces: `outputs/phase_profile_triton-inner-nsys-control-finalsrc/` and `outputs/phase_profile_triton-inner-nsys-candidate-finalsrc/`.

Do not use the old `phase_profile_triton-inner-candidate-full28-finalsrc-timeout4h` result as evidence: it predates the stride fix. `tests/test_triton_column_block.py` imports `realq.quant.triton_column_block`, a separate implementation; for the production `gptq_utils` route, the corrected full-model hash/KL/PPL pair and `tools/prototype_fused_gptq_column.py` are the relevant evidence.

### 2. Exact symmetric-union MSE clip search — validated opt-in, short-run speed result

**Purpose.** Accelerate symmetric MSE clipping/initial-quantizer calibration without changing its selected quantization parameters on the proven input domain. A legacy Cartesian search checks up to 25 negative × 25 positive endpoints (625 pairs); the symmetric-union formulation checks at most 50 endpoint candidates.

**Implementation map.**

* Search implementation: `utils/quant_utils.py`, `WeightQuantizer` search path around line 867.
* CLI wiring: `process_args.py`, `--pre_clip_search_impl` and `--quantizer_search_impl`; both default to `cartesian_legacy`.
* GPTQ+ call sites: `gptq_utils/gptq_plus_utils.py` around lines 12174 and 12517.
* Focused correctness tests: `tests/test_quantizer_optimization_contracts.py`, `tests/test_p02_adversarial_review.py`, and `tests/test_w_clip_where_out.py`.

The exact optimized path is limited to finite, symmetric FP32 inputs; unsupported inputs retain the legacy fallback. Layer-0 Qwen3 weights had bit-identical scales, zero-points, and clipped weights. Three five-layer candidates matched the 507-tensor checkpoint hash and exact-prefix KL/PPL of the legacy baseline. Two paired five-layer runs using both switches averaged 173.3 s versus 223.9 s for the legacy pair (22.6% faster), with unchanged peak memory. This is a short-run result, not a full-model speed claim.

**Evidence.** `docs/preclip_exact_union_profile_20260929.md`, plus `outputs/phase_profile_qwen3-06b-seed44-layers0to4-both-union/` and `...both-union-retry2/`.

### 3. Frozen FFN replay/cache — do not claim as an acceleration yet

Implementation is in `gptq_utils/frozen_ffn_replay.py` / `gptq_utils/frozen_prefix_replay.py`; probes are in `tools/frozen_ffn_replay_probe.py` and `tools/profile_prefix_refresh.py`. Some isolated down-projection refresh pairs were 13–15% faster, but complete quantization runs did not show repeatable end-to-end speedup. An unexplained no-verify state divergence occurred in the all-FFN cache experiment. Keep it experimental/off by default and do not list it as a validated contribution. See `docs/frozen_ffn_replay_audit_20260929.md`.

## Third-order / matrix-free gradient approximation

### What is implemented, and where

There are two earlier feasibility demos and a real-quantization-residual probe. Since then, an opt-in `--third_order_refresh_layer` branch has been integrated into the REAL-Q block-GD refresh path; it remains off by default. Full-model runs show inconsistent quality outcomes and do not yet demonstrate a repeatable KL improvement.

1. `tools/run_third_order_taylor_demo.py` and `docs/third_order_taylor_hvp_demo_20261001.md` implement a smooth synthetic SwiGLU-tail experiment. They form Hessian-vector products (HVPs) and use a central difference of `H(y) d` to approximate the third-order contraction. `tests/test_third_order_taylor.py` checks a small explicit third-derivative contraction and one fixed cubic-gradient case. The demo found gradient improvement in its tested synthetic cases, but KL scalar accuracy was not monotonic and three HVPs cost about 2.7–3.1× one HVP.
2. `tools/run_third_order_real_qwen_probe.py` checks the exact downstream KL/HVP calculation through a real Qwen3 suffix, but uses controlled random perturbations, **not quantization residuals**. The FP32 replay matched full-model logits on its check. Results and limits are in the same demo document.
3. `tools/run_quantized_third_order_gradient_compare.py` is the real-residual probe. It hooks `analyze_grad_cosine`'s GPTQ+ reference pass immediately after the chosen layer is quantized, reads the actual quantized-prefix input and output residual, uses the cached full-FP final hidden state as teacher, and keeps the downstream suffix FP. It compares the actual layer-weight gradients for `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj`.

For `y0` the FP output, `yq` the output after real quantization, `d=yq-y0`, and `g(y)=∂KL_downstream/∂y`, the probe compares:

```text
REAL-Q Fisher:          gF = F d
third-order candidate:  g3 = g( y0 ) + F d
                            + 1/2 [g(y0+d) + g(y0-d) - 2g(y0)]
exact downstream KL:    g(yq)
```

The symmetric second difference of the gradient estimates `T[d,d]` without storing a rank-3 tensor. It contains fourth- and higher-order remainder terms, so it is a finite-residual approximation, not an exact third derivative. Unlike the earlier HVP demo, this probe does not use an explicit `H(y)d`; it estimates the contracted cubic correction directly from gradients at three points. Each output-space gradient is mapped through the **same actual target-block Jacobian** to the FFN weight gradients. Gradients are raw, before optimizer elementwise clipping.

### Current actual-residual evidence

The small runs used Qwen3-0.6B, GPTQ+ W4, no pre-clip, four Wikitext-2 sequences of length 64 for static Fisher, and two measured sequences drawn from that same four-sequence pool. Results are a smoke test, not independent validation.

* **Layer 4:** residual norm / FP output norm = 0.0622. Fisher cosine to exact KL gradient was approximately zero or negative for all three FFN projections; adding the correction raised cosine to 0.171–0.261 and modestly lowered relative error (from 2.05–2.64 to 1.89–2.52). This is a promising small-residual signal, not a proof of an optimization gain.
* **Layer 20:** residual norm / FP output norm = 8.64. Fisher and corrected cosines were both near zero, with relative errors 598–1018. This is far outside a local Taylor regime, so it cannot fairly test whether the local cubic approximation works; the correction does not rescue this high-residual case.
* The reference-point KL gradient norms were about `1e-7` and `1e-8`, checking that the teacher and FP reference paths coincide numerically. No PPL or end-to-end quantization-quality improvement was measured.

Raw result files are `outputs/third_order_real_quant_layer4/Qwen3-0.6B/third_order_real_quant_layer4/third_order_gradient_compare.json` and `outputs/third_order_real_quant_layer20/Qwen3-0.6B/third_order_real_quant_layer20/third_order_gradient_compare.json`. Method/result detail is in `docs/third_order_real_quant_gradient_compare_20261001.md`.

**Full REAL-Q integration result:** `docs/third_order_realq_full_pair_20261001.md` records repeated 28-layer W4 runs. Before fixing deterministic SDPA selection, three cubic runs had mean KL 1.55% worse than control, with substantial run-to-run spread; exact-downstream-gradient diagnostics also regressed mean KL. Under math SDPA, one full pair improved KL 2.25%, but its same-config repeat was stopped at layer 27 before evaluation. All 2,000 common block-metrics rows match at printed precision; there is no second final metric. The reproducibility helper is now connected, and `ptq.py` uses default SDPA for teacher and final evaluation around math-SDPA quantization; this backend split has not been validated in a full run. No repeatable quality gain is established.

**Latest static failure audit:** `docs/third_order_failure_analysis_20261001.md` derives `gc-g+ = gF-(g+-g-)/2`: the candidate retains the leading Fisher curvature error. It also notes that earlier toy tests used an exact HVP base, production loss logs retain the original Fisher scalar, and actual Adam-update alignment was not recorded. Residual size, BF16 rounding, and FP-suffix mismatch remain possible factors, not proven dominant causes. No new GPU experiment was run; keep experiments stopped per user instruction.

## Suggested handoff entry point

Start with `docs/triton_gptq_inner_loop_optimization_20260929.md`, `docs/preclip_exact_union_profile_20260929.md`, `docs/third_order_real_quant_gradient_compare_20261001.md`, and `docs/third_order_realq_full_pair_20261001.md`. The Triton path is the best validated acceleration; exact-union search is an additional bounded speed optimization. Third-order remains exploratory and has not shown a repeatable KL gain. Keep runtime acceleration and KL-quality improvement as separate claims.
