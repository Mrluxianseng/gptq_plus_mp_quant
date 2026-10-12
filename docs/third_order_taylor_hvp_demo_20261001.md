# Third-order Taylor / HVP demo (2026-10-01)

## Why this experiment

REAL-Q uses a block-output Fisher quadratic as a tractable proxy for the downstream KL objective. Its paper reports weaker surrogate-gradient alignment on SwiGLU FFN projections and lists improving fidelity under severe nonlinearities as future work. The question here is whether a third-order correction can be estimated without constructing a rank-3 tensor.

This document records a **mechanism demo only**. It is not an end-to-end quantization result and does not establish a KL/PPL gain for REAL-Q.

## Approximation

Let `L(y)` be the exact downstream KL as a function of one block's output, with the teacher and student identical at the reference `y0`. Thus `∇L(y0)=0`. The local gradient expansion is

```text
∇L(y0 + Δ) = H(y0) Δ + 1/2 T(y0)[Δ, Δ] + O(||Δ||^3).
```

The quadratic baseline uses `H(y0)Δ`. To avoid storing `H` or `T`, compute Hessian-vector products and use the directional derivative identity

```text
T(y0)[Δ, Δ] = D_Δ ( H(y) Δ ) |_(y=y0)
             ≈ [H(y0 + εΔ)Δ - H(y0 - εΔ)Δ] / (2ε).
```

Each `H(y)Δ` is obtained with reverse-mode automatic differentiation by differentiating `∇L(y) · Δ` once more. The estimate holds `Δ` fixed in both HVPs. It uses O(activation memory + hidden-size vectors), not O(H³) tensor storage. The code's scalable path computes one HVP for the quadratic term and two additional HVPs for the central-difference cubic correction. This demo uses the exact local KL Hessian `H(y0)` as its quadratic baseline; it does **not** substitute REAL-Q's token-aggregated empirical Fisher cache. It isolates whether the third-order contraction is numerically sound and does not measure the additional mismatch from REAL-Q's Fisher aggregation.

For an FP32 full tensor, `T ∈ R^(H×H×H)` would take 4 GiB at `H=1024` for just one token. The demo never allocates it. A small `H=8` reference explicitly forms the Hessian, differentiates `ΔᵀHΔ`, and verifies the HVP finite-difference contraction against that exact contracted third derivative.

## Demo setup

`tools/run_third_order_taylor_demo.py` builds a deterministic nonlinear downstream map consisting of RMS normalization, a SwiGLU FFN, a residual connection, and a language-model head. It compares:

- the exact KL gradient at `y0 + Δ`;
- the local quadratic gradient `H(y0)Δ`;
- the third-order corrected gradient `H(y0)Δ + 1/2 T[Δ,Δ]`;
- the KL scalar's second- and third-order Taylor predictions;
- CUDA peak allocated memory and elapsed time for one vs. three HVPs.

It uses hidden sizes 128, 512, and 1024; eight tokens; five relative perturbation norms (0.5%, 1%, 5%, 10%, 20%); three deterministic random seeds; and three timing repetitions in the main accuracy run. A separate timing run at hidden size 1024 uses 30 repetitions. The finite-difference step sweep uses `ε ∈ {0.0625, 0.125, 0.25, 0.5, 1.0}` times `||Δ||`.

Hardware/software: WSL Ubuntu 24.04, PyTorch 2.11.0+cu128, NVIDIA GeForce RTX 4060 Laptop GPU. The synthetic-tail experiment is a small smooth model, not Qwen3.

## Results

Across the three seeds and three hidden sizes (nine cases per perturbation magnitude), the corrected **gradient** had lower relative error than the quadratic gradient in 9/9 cases at every magnitude:

| `||Δ|| / ||y0||` | Mean relative gradient error, quadratic → cubic | KL absolute-error wins, cubic vs. quadratic |
|---:|---:|---:|
| 0.5% | 0.0066 → 0.0001 | 4/9 |
| 1% | 0.0132 → 0.0002 | 6/9 |
| 5% | 0.0661 → 0.0058 | 5/9 |
| 10% | 0.1323 → 0.0233 | 4/9 |
| 20% | 0.2665 → 0.0924 | 3/9 |

The scalar KL approximation did **not** improve consistently. It improved its absolute error in 4/9, 6/9, 5/9, 4/9, and 3/9 cases respectively. At 10–20% perturbations, higher-order terms beyond cubic can dominate, and a Taylor truncation is not monotonic. The evidence supports a local gradient-correction demo, not a general claim that a cubic KL surrogate is better.

At `H=1024`, `B×T=8`, with 30 timing repetitions, the one-HVP quadratic path took about 2.1–2.3 ms; the three-HVP cubic path took 5.9–6.6 ms (2.7–3.1×). CUDA peak allocation was 74.72 MiB vs. 74.82 MiB, about 0.10 MiB extra in this tiny-tail benchmark. The dominant runtime cost is repeated backward/HVP work; the rank-3 tensor's storage is avoided. This is not a full-model speed estimate.

At `H=8`, the finite-difference contraction differed from the explicitly differentiated third-order contraction by relative error `2.60e-4` with `ε=0.25||Δ||`. In the step sweep up to `H=1024`, gradient-error differences across `ε=0.0625…0.5` were negligible at the tested perturbation scales; at 20%, `ε=1.0` was slightly less stable.

## Real Qwen3-0.6B suffix probe

To check that the mechanism survives a real Transformer tail, `tools/run_third_order_real_qwen_probe.py` loads local Qwen3-0.6B in FP32, captures the output of block 20 for a one-token input, then differentiates the exact KL through the remaining seven blocks and LM head. Before measuring, replaying those seven blocks from the captured activation reproduces the full-model logits exactly (maximum absolute difference `0.0`). Three independent random perturbation directions are tested at each relative magnitude. They are controlled probes, **not residuals from REAL-Q quantization**.

| `||Δ|| / ||y0||` | Mean gradient relative error, quadratic → cubic | KL absolute error, quadratic → cubic | KL wins |
|---:|---:|---:|---:|
| 1% | 0.130 → 0.009 | 1.053e-3 → 5.617e-5 | 3/3 |
| 5% | 0.590 → 0.196 | 9.666e-2 → 3.979e-2 | 2/3 |
| 10% | 1.030 → 0.750 | 5.999e-1 → 6.178e-1 | 1/3 |

The corrected gradient is closer in all nine probes, while the scalar KL approximation stops being reliable as perturbations grow. On this 7-block suffix, one quadratic HVP averaged 0.056–0.058 s; the two extra HVPs for the cubic term took 0.109–0.112 s. Peak CUDA allocation was 2297.57 MiB vs. 2297.58 MiB. This confirms low *additional* tensor memory on the actual model and makes the compute tradeoff concrete; it remains an FP32, single-token, synthetic-residual probe rather than quantization-loop timing.

## Interpretation and next gate

This does answer the narrow feasibility question: third-order information can be contracted directionally without rank-3 memory, and in this controlled nonlinear KL example it corrects the local gradient much better than a quadratic-only expansion. It also costs about three times as many HVPs.

It is **not ready to be inserted into the quantization loop**. Before claiming it addresses REAL-Q's bottleneck, the same measurement must be performed at a real non-final transformer-block output along a real quantization residual, with the actual frozen downstream model. The key comparison is the angle/error between (a) the REAL-Q aggregated Fisher gradient, (b) the HVP-based third-order-corrected gradient, and (c) the exact downstream-KL gradient at that point. Only if this holds on several real FFN modules and refresh states should an optimizer integration be attempted; end-to-end KL must then be the acceptance metric.

## Reproduction

```bash
python tools/run_third_order_taylor_demo.py \
  --hidden-sizes 128,512,1024 \
  --relative-deltas 0.005,0.01,0.05,0.10,0.20 \
  --finite-diff-steps 0.25 \
  --batch-tokens 8 --repeats 3 \
  --output outputs/third_order_taylor.json
```

The test file can run directly with the project Python environment (it also works under pytest):

```bash
python tests/test_third_order_taylor.py
```

Real-model suffix probe (needs the local Qwen3-0.6B checkpoint and about 2.3 GiB allocated CUDA memory in this setup):

```bash
python tools/run_third_order_real_qwen_probe.py \
  --layer 20 --relative-deltas 0.01,0.05,0.10 \
  --delta-seeds 9000,9001,9002 \
  --output outputs/third_order_qwen3_probe.json
```

## References

- [REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent](https://arxiv.org/abs/2609.00049) — Fisher surrogate and its nonlinear FFN fidelity limitation/future direction.
- [Pearlmutter, “Fast Exact Multiplication by the Hessian” (1994)](https://doi.org/10.1162/neco.1994.6.1.147) — Hessian-vector products without materializing a Hessian.
- [Taylor-Mode Automatic Differentiation for Higher-Order Derivatives in JAX](https://openreview.net/pdf?id=SkxEF3FNPH) — directional higher-order derivatives as an alternative to full derivative tensors.
