# Fused GPTQ inner-column kernel validation (2026-09-30)

> **Final-source follow-up (2026-10-02):** the full-model result below was an earlier single-pair check and is superseded for thesis claims by the three final-source repetitions documented in [the final acceleration and memory report](thesis_final_acceleration_memory_optimization.md). Those three pairs yielded 2.17× median quantization speedup and 2.08× median end-to-end speedup, with identical 507-tensor hash, KL and PPL in every pair. The final report also records the combined Triton + exact chunked-KL run and the 28-layer output fingerprint audit.

## Change

Added an experimental `--gptq_inner_kernel triton_fused` path for the sequential inner-column update. The default remains `torch`. The fused path is deliberately limited to `block_gd`, non-atomic quantization, no two-sided rounding, per-row symmetric low-bit quantization, FP32 working tensors, and block sizes up to 128. It keeps the column order and all block-level gradient refreshes unchanged.

The Triton kernel uses runtime column indexing so all columns reuse one compiled kernel per shape. It fuses per-column quantize/dequantize, error formation, the second-order update to the remaining weights, and the `GHinv` rank-one update. It uses round-to-nearest-even division (`div_rn`) and disables FP fusion to preserve PyTorch's FP32 operation sequence. The input weight view and output buffers have separate row strides: in the actual block loop, `W1` is a slice of the full weight matrix (stride 4096 in the smoke test), while `Q1`, `W_int1`, and `Err1` are compact (stride 128).

## Correctness and timing

The regression smoke test now uses a non-contiguous `W1` view with shape 3072×128 and parent row stride 4096, while the output buffers use compact row stride 128. Five deterministic seeds produced bit-identical quantized values, errors, per-column weight states, and `GHinv`. A three-run median on this representative layout measured 32.327 ms for the PyTorch loop and 7.850 ms for Triton (4.12×). The microbenchmark is diagnostic; the model-level runs below are the relevant evidence.

| Scope, final source revision | PyTorch control | Triton candidate | State hash | Peak allocated/reserved memory |
|---|---:|---:|---|---:|
| Layer 0 | 67.28 s (Nsight-instrumented) | 35.28 s (Nsight-instrumented) | identical: `00ec6c11475130cab1abf46d38a19a186cd1cd885cff39ea963ce0d9519eb664` | 2,559,177,728 / 2,615,148,544 B |
| Layers 0–4 | 172.94 s | 71.85 s | identical: `fe2945201ea303dc34a962d14b86d508bb98e5a30d75c9c573a02c3cef4ecbda` | 2,559,177,728 / 2,615,148,544 B |
| Full 28 layers, clean recheck | 698.33 s | 311.65 s | identical: `389fe728cabd23e60ff25d2600e6a49ed2b483b48ebbae6a5743380b3678a0d3` | 5,913,079,296 / 6,499,079,296 B control; 5,913,075,200 / 6,561,988,608 B Triton |

The matched five-layer candidate was 2.41× faster. In the full-model pair, quantization time was 2.24× faster (55.4% less time), and total runner wall time including setup and evaluation was 2.11× faster. The `block_gd` last-layer true-gradient refresh took about 70.5 s in the clean control and 71.5 s in the corrected candidate, so the gain comes from the quantization path rather than skipping refresh work. Both full runs used the same WSL model, calibration/evaluation data, seed, settings, exact pre-clip/quantizer search, and source hashes. They produced the same 507-tensor state hash and exactly the same WikiText-2 KL/PPL: 0.2254651 / 32.7383156. Allocated memory was effectively unchanged; reserved memory increased by about 63 MB.

There was one 5001 s full-control timing outlier with the same state hash and KL/PPL. Two later control runs measured 705.38 s and 698.33 s; the former briefly overlapped a short smoke test, while the latter was isolated. The repeated isolated result is the control used above. The cause of the 5001 s outlier is unverified, so it is not used in the speedup claim.

An initial full candidate run was invalid: its Triton kernel used `W1`'s row stride for compact `Q1`, `W_int1`, and `Err1` buffers. That produced a different model hash and different KL/PPL, so those measurements are discarded. The kernel was corrected to pass each output tensor's stride separately, then the full candidate was rerun and matched the control exactly. The stricter non-contiguous smoke test guards this case.

## Nsight launch-count check

For one selected 128-column `layers.0.mlp.down_proj` subgroup loop:

- In the final-source PyTorch trace, one selected 128-column `layers.0.mlp.down_proj` subgroup loop recorded 2,556 `cudaLaunchKernel` calls and about 3.35 ms of summed device kernel time.
- The final-source Triton trace recorded 128 `cuLaunchKernelEx` calls and 127 GPU kernel instances in the filtered range; it also recorded three module-load calls for JIT setup. The 127/128 difference is the profiler's GPU-activity count versus host launch count; the launch API count is the direct evidence of one launch per column.
- The filtered host range was about 89.0 ms for PyTorch and 17.6 ms for Triton. Nsight instrumentation perturbs host timings, so use these ranges to understand launch structure, not as the end-to-end speed claim.

The `fasterquant.block.total` range still contains the separate post-block gradient refresh. The fused kernel does not change or eliminate that work. In the full paired runs, the number and time of last-layer true-gradient refreshes were effectively equal.

## Output equivalence and limits

The corrected full-model control and Triton candidate both completed normal evaluation with identical weight-state hashes, KL, and PPL. This is a runtime optimization; it does not improve quantization quality by design.

The implementation remains opt-in and raises an error for unsupported settings rather than silently falling back. It is currently limited to `block_gd`, non-atomic quantization, no two-sided rounding, per-row symmetric low-bit quantization, FP32 working tensors, and block sizes up to 128. The full-model result is validated on this Qwen3-0.6B/W4 configuration; broader configurations still need their own stride and numeric checks.

## Artifacts

- Kernel: `gptq_utils/triton_gptq_kernels.py`
- CLI wiring and sequential-loop dispatch: `gptq_utils/gptq_plus_utils.py`, `process_args.py`
- Synthetic correctness and timing harness: `tools/prototype_fused_gptq_column.py`
- Control/candidate manifests: `outputs/gptq_kernel_control_manifest_runtimecol_20260929.json`, `outputs/gptq_kernel_candidate_manifest_runtimecol_20260929.json`
- Full control/candidate manifests: `outputs/gptq_kernel_control_manifest_recheck_20260930.json`, `outputs/gptq_kernel_candidate_manifest_stridefixed_20260930.json`
- Full clean-control and corrected-candidate results: `outputs/phase_profile_triton-inner-control-full28-cleanrecheck-timeout4h/status.json`, `outputs/phase_profile_triton-inner-candidate-full28-stridefixed-timeout4h/status.json`
- Final-source five-layer results: `outputs/phase_profile_triton-inner-control-prefix5-finalsrc/status.json`, `outputs/phase_profile_triton-inner-candidate-prefix5-finalsrc/status.json`
- Final-source Nsight traces: `outputs/phase_profile_triton-inner-nsys-control-finalsrc/nsys-kernel-trace.nsys-rep`, `outputs/phase_profile_triton-inner-nsys-candidate-finalsrc/nsys-kernel-trace.nsys-rep`
- Final-source filtered kernel summaries: `outputs/phase_profile_triton-inner-nsys-control-finalsrc/inner-column-kernel-stats.txt`, `outputs/phase_profile_triton-inner-nsys-candidate-finalsrc/inner-column-kernel-stats.txt`
