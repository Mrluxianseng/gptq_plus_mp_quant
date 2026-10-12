# REAL-Q inner-loop GPU profile (2026-09-29)

## Question

Locate kernel-launch and fusion opportunities in the formal GPTQ column loop, while separating that work from REAL-Q's per-block gradient refresh.

## Profile setup

- WSL Ubuntu 24.04, RTX 4060 Laptop GPU (8 GB), Nsight Systems CLI 2026.5.1.
- Qwen3-0.6B, one Transformer layer (`--stop-layer 0`), WikiText-2 calibration, 32 samples × 256 tokens, W4, block size 128, activation order, four quantization groups.
- REAL-Q update mode: `block_gd` with Adam and `fisher_diag_mse`; exact symmetric-union search enabled for pre-clipping and initial quantizer parameter search; evaluation skipped to keep the trace focused.
- Added one NVTX push/pop around each sequential inner column loop. It adds no tensor operation and does not change the quantization formula. The runner verified source files stayed unchanged during the run.
- Peak allocated/reserved CUDA memory: 2,559,177,728 / 2,615,148,544 bytes. The post-run state hash was `00ec6c11475130cab1abf46d38a19a186cd1cd885cff39ea963ce0d9519eb664`, identical to the preceding one-layer profile without this NVTX marker. Evaluation was skipped, so this run has no KL/PPL result.

## Main observation: the sequential loop is launch-heavy

An isolated 128-column inner-loop range in `layers.0.mlp.down_proj` recorded 2,556 CUDA kernel launches: approximately 20 launches per quantized column. Nsight measured 19.6–28.9 ms of CUDA launch API time for that range; the summed device time of the kernels in the representative range was about 5.48 ms. The NVTX host range took roughly 67–87 ms across the sampled terminal subgroup ranges. These are instrumented timings, not a speedup estimate, but the gap and launch count identify host dispatch and many tiny kernels as concrete optimization targets.

The kernel summary shows separate 128-call round and clamp kernels, 128-call elementwise binary/unary kernels, roughly 254 calls each for GEMV-style kernels, and hundreds of additional generic elementwise launches. This agrees with the source loop: each column separately fake-quantizes the current weight, forms its error, performs the `err × Hinv` update on the remaining columns, writes the error, and performs the `Z × Hinv` update to `GHinv`.

The current NVTX marker is inside the subgroup loop, so a selected range represents one subgroup's 128-column pass. The four last range instances (92–95) are the terminal block's subgroup passes. Individual range times vary under tracing; kernel counts are the more robust signal here.

## Keep refresh cost separate

`fasterquant.block.total` includes both the inner column loop and the post-loop block update/refresh. In this run, the down-projection had 24 block ranges and 23 true-gradient-refresh ranges (about 1.72 s summed in the profile recorder). Its block-range time therefore cannot be treated as inner-loop time. The new `fasterquant.block.inner_column_loop` marker isolates the serial column work from that refresh. This distinction matters because a fused column kernel cannot remove the repeated model forward/backward work of `block_gd` refreshes.

## Engineering direction indicated by the trace

The best next prototype is a Triton kernel for the current supported frozen-inner path (`block_gd` here), retaining the sequential column order. For one column it can compute the scalar quantize/dequantize and error per row, then apply both rank-one tail updates over `(row, remaining-column)` tiles, and write `Q`, integer weights, scale, and `Err1`. That should replace much of the sequence of PyTorch pointwise and small-GEMV launches. It must not batch different quantized columns together because column `i+1` consumes the updates from column `i`.

The current results justify prototyping this engineering optimization; they do not yet establish an end-to-end speedup. Before adopting it, compare the reference and candidate on quantized integer weights, `Q`, `Err1`, `W1`, `GHinv1`, and final model state. Fused arithmetic may change float rounding, so also verify that no quantization decisions change. Then benchmark repeated unprofiled runs on the same layer and compare the same output/KL/PPL checks. The initial acceptance gate should be correctness plus measurable end-to-end quantization-time reduction, not kernel microbenchmark speed alone.

## Artifacts

- Nsight trace: `outputs/phase_profile_nsys-innerloop-qwen3-06b-layer0-both-union/nsys-kernel-trace.nsys-rep`
- SQLite export: `outputs/phase_profile_nsys-innerloop-qwen3-06b-layer0-both-union/nsys-kernel-trace.sqlite`
- Representative kernel summaries: `outputs/phase_profile_nsys-innerloop-qwen3-06b-layer0-both-union/inner-column-92-all-stats.txt` through `inner-column-95-all-stats.txt`
- Profile run summary: `outputs/phase_profile_nsys-innerloop-qwen3-06b-layer0-both-union/status.json`
- Source manifest: `outputs/phase_profile_innerloop_source_20260929.json`

