# REAL-Q Stage-0 activation-memory study (2026-10-04)

## Motivation and external systems literature

REAL-Q is the method under study and the baseline, not external literature supporting the optimization. Its Stage-0 end-to-end saliency/Fisher collection runs a full-model backward pass before quantization, so autograd-saved activations are a separate memory target from later GPTQ/Block-GD state. External systems work such as Capuchin studies tensor eviction/prefetch and recomputation ([paper](https://www.microsoft.com/en-us/research/?p=690597)); Chen et al. and Checkmate study activation rematerialization ([Chen et al.](https://arxiv.org/abs/1604.06174), [Checkmate](https://arxiv.org/abs/1910.02653)). These provide technical context, not evidence of benefit for this PTQ workload.

The present prototype applies PyTorch `saved_tensors_hooks` only around the model forward used by Stage 0. Non-leaf CUDA tensors at or above a configurable size are copied to CPU when autograd saves them and copied back when backward needs them. The default is disabled; the experimental option is `--static_saved_tensor_offload_mib`.

The implementation mechanism follows the PyTorch saved-tensor CPU-offload API, which explicitly trades host-device movement for GPU memory ([documentation](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)). ZeRO-Offload is a related heterogeneous-computing reference, but it primarily offloads gradients, optimizer states, and optimizer computation; it is not evidence for activation offload, so it should not be presented as direct support for this prototype.

## Local full-model paired audit

Hardware: WSL, Qwen3-0.6B, RTX 4060 Laptop GPU (8 GiB). Both arms used the same REAL-Q configuration: W4, 32 calibration samples of 256 tokens, 32 backward samples, all 28 layers, seed 44, PyTorch GPTQ inner kernel. The candidate changed only Stage-0 CPU offload (threshold 1 MiB). The runner calls the candidate arm `triton` for historical reasons; in this audit its manifest was explicitly changed to `--gptq_inner_kernel torch`, so this is an offload-only comparison.

| Metric | Control | Stage-0 offload | Difference |
|---|---:|---:|---:|
| Quantization time | 1157.87 s | 1229.56 s | +6.19% slower |
| End-to-end wall time | 1196.56 s | 1267.75 s | +5.95% slower |
| Peak CUDA allocated | 5,913,079,296 B | 5,913,079,296 B | equal |
| Peak CUDA reserved | 6,184,501,248 B | 6,289,358,848 B | +104,857,600 B |
| State tensors | 507 | 507 | equal |
| State SHA-256 | `96e6281619f2304fb97c885962fcbae9e7dea8fe0e3f256aa225fc41201a8d4e` | same | exact |
| WikiText-2 KL | 0.23623386025428772 | same | exact |
| WikiText-2 PPL | 33.27994918823242 | same | exact |

All 28 layer output fingerprints matched. Thus the offload preserves this run's quantized model and measured outputs. It does **not** lower this configuration's end-to-end peak memory: the later quantization stage dominates. The slight reserved-memory increase also means the strict “no peak metric above control” target is not met by this short-sequence full run.

Stage-0 itself copied 14,400 saved tensors with 25.00 GiB cumulative payload at the 1 MiB threshold. Its progress bar took about 29 s with offload versus 12 s in control. The full-run penalty is consistent with synchronous CPU transfer cost; one paired run is not enough to estimate stable speed impact. A separate seq-2048 collector experiment had shown Stage-0 peak allocated memory falling from 7.0645 GiB to 4.5410 GiB (−35.72%), while layer logits were bitwise equal. That result is local to the collector and does not establish an end-to-end peak reduction once later quantization stages are included.

Conclusion: keep this as a default-off, memory-targeted Stage-0 option for cases where Stage-0 is the actual global bottleneck. For the current 32×256 path it misses the global memory objective and slows Stage-0, so it is not ready to be claimed as an overall optimization. The next promising variant is selective/overlapped transfer (or recomputation checkpointing) whose transfer volume is controlled; it must be measured against every-stage peaks and full-run time again.

### Threshold tradeoff probe (stop after layer 0)

A follow-up paired profile kept the same 32×256 input and quantized only layer 0. Control peak allocated/reserved was 2,731,209,728 / 2,868,903,936 B, with 51.45 s quantization time. At a 4 MiB threshold, the candidate copied 1,792 tensors / 7.00 GiB cumulatively; its peak was 2,613,769,216 / 2,743,074,816 B (−4.3% allocated and −4.4% reserved), and the partial model state SHA-256 matched control. Quantization took 60.27 s (+17.2%). This is a single run per arm and excludes evaluation, so the timing is directional. Compared with the full-run 1 MiB arm, raising the threshold cut cumulative payload from 25 GiB to 7 GiB, but this partial profile still does not find a good memory-and-speed balance. A wider sweep would need paired repetitions and a long-sequence setting where Stage-0 dominates the real peak.

## Telemetry and component-timing fixes, verified after the paired run

* `GpuMonitor` now creates campaign files exclusively (`open(..., "x")`) so a second run cannot silently truncate or mix into an existing telemetry file. The stop path waits for the writer to finish before parsing.
* Telemetry parsing rejects repeated headers/incomplete rows and reports samples by GPU index. A live WSL monitor smoke emitted three valid rows and parsed them into per-GPU summary statistics. A unit test confirms an existing telemetry file is preserved and the second writer fails loudly.
* Wall-clock component timing is written to `wall_profile_sections.json` using complete section names. The paired parser aggregates suffixes across layers instead of parsing a width-truncated text table. Both full-run arms emitted 5,898 structured profile sections; component totals were present in `paired_repetitions.json`.
* Fixed the output-trace wrapper lookup to accept the actual generated name `benchmark_entry.py` as well as `entry.py`; this mismatch had previously aborted the paired audit before the run began.
* Validation: `python -m unittest discover -s tests -p 'test_triton_benchmark_telemetry.py' -v` (5 tests passed), `py_compile` on the touched Python modules, plus the full paired audit with exact state hash, KL/PPL and per-layer fingerprints.

Artifacts: `outputs/local-stage0-offload-20261004-retry1-campaign/paired_repetitions.json`; controls/candidate profiles under `outputs/phase_profile_local-stage0-offload-20261004-retry1-r2-{control,triton}/`.

## Triton speed path: traced local end-to-end pair

After fixing the outer runner to always request and require per-layer fingerprints, a fresh campaign completed with `--trace-layer-outputs`. Control used the PyTorch GPTQ inner kernel; candidate used the Triton fused inner-column kernel. The benchmark's complete configuration stayed fixed (Qwen3-0.6B, W4, 32×256 calibration, all 28 layers, seed 44).

| Metric | PyTorch | Triton fused |
|---|---:|---:|
| Quantization time | 848.93 s | 377.35 s |
| Full run wall time | 885.02 s | 412.55 s |
| Peak allocated | 5,913,079,296 B | 5,913,075,200 B |
| Peak reserved | 6,184,501,248 B | 6,184,501,248 B |
| State SHA-256 | `96e6281619f2304fb97c885962fcbae9e7dea8fe0e3f256aa225fc41201a8d4e` | same |
| WikiText-2 KL / PPL | 0.23623386 / 33.27995 | same |
| Layer output hashes | 28 layers, each run 8 times | all 28 exactly equal |

This single traced pair gives 2.25× quantization speed and 2.15× full-run speed, with the same measured peak memory. The structured component profile attributes the change to `fasterquant.block.inner_column_loop`: 505.71 s in control and 38.96 s in Triton (about 13.0× for that component). The gradient-refresh component stayed similar (218.57 s vs 214.60 s). This is strong preliminary evidence that the kernel accelerates the intended loop while preserving the measured outputs, but timing claims still need paired repetition: only one traced pair is currently complete. The first outer campaign also showed a similar 2.22× quant speed, but it omitted layer traces and is excluded from correctness-qualified evidence.

Artifacts: `outputs/local-triton-layertrace-20261004-campaign/summary.json`, `paired_repetitions.json`, `gpu_telemetry.csv`, and the control/candidate phase profiles. Its 257 telemetry rows parsed with no invalid records. The outer benchmark now passes `--trace-layer-outputs` and marks a campaign invalid if those fingerprints are absent or differ.

### Repeated traced Triton audit (two matched pairs)

A fresh two-pair campaign was run after the trace-enforcement and telemetry/profile-parser fixes. It used the same W4, 32×256 calibration, 8×1024 evaluation, seed 44, all 28 layers and `blocksize=128` configuration as the single-pair run above. Both pairs completed and the campaign was marked valid. In each pair, all 507 tensor-state hashes, KL/PPL, and every layer fingerprint matched exactly; all 28 layers had eight recorded forward outputs per arm.

| Metric | Pair 1 | Pair 2 | Mean / summary |
|---|---:|---:|---:|
| Quantization time, PyTorch | 840.39 s | 856.00 s | 848.20 s |
| Quantization time, Triton | 381.15 s | 381.04 s | 381.09 s |
| Quantization speedup | 2.205× | 2.246× | 2.226× (sample SD 0.029×) |
| End-to-end speedup | 2.106× | 2.140× | 2.123× (sample SD 0.024×) |
| Peak allocated delta (Triton − PyTorch) | −4,096 B | −4,096 B | effectively equal |
| Peak reserved delta (Triton − PyTorch) | 0 B | 0 B | equal |
| Mean sampled device memory (PyTorch → Triton) | 3,674.8 → 3,843.7 MiB | 3,473.9 → 3,968.1 MiB | paired-arm means: 3,574.4 → 3,905.9 MiB (+9.3%) |

The component profile supports the mechanism: mean `fasterquant.block.inner_column_loop` time fell from 503.74 s to 39.55 s (about 12.7×), while `true_gradient_refresh` stayed near 218 s in both arms. The Triton kernel materially accelerates the compensated inner-column loop, not the refresh work. This is a repeatable local result across two pairs, but only on this RTX 4060 Laptop GPU and this reduced 32×256 calibration setup; it is not yet a server-scale thesis result. The outer telemetry parser read 516 rows with zero invalid rows. We aligned telemetry timestamps to each arm's recorded wall interval (samples per arm: 173, 82, 177, 83); the candidate's mean sampled device memory was about 9.3% higher, despite equal CUDA allocator peaks. This is a signal that the speed result is not a memory reduction and warrants a longer, independently paired memory study. The device-wide 5-second samples include allocator/runtime overhead and only estimate the run-time average; they do not replace PyTorch's exact allocator peak counters.

The WSL campaign could not resolve the Windows worktree's `.git` pointer, so its summary's `git_commit` field contains a Git error. The native Windows checkout reports commit `e3767b9`; the campaign still recorded per-source hashes and `sources_unchanged=true` for each arm. Future campaign reports should pass a host-resolved revision or make the WSL runner understand the worktree pointer before using the summary as a complete provenance record.

Artifacts: `outputs/local-triton-repeat2-layertrace-20261004-campaign/summary.json`, `paired_repetitions.json`, `gpu_telemetry.csv`, and the four `outputs/phase_profile_local-triton-repeat2-layertrace-20261004-r{2,3}-{control,triton}/` directories.

## Literature notes for the next memory experiment

* The most directly relevant loss-memory paper I should have surfaced earlier is Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (ICLR 2025). It removes the need to materialize the full token-by-vocabulary logit matrix for cross-entropy, demonstrating that vocabulary-sized loss intermediates can dominate memory. This is a close systems precedent for REAL-Q's final-block KL refresh, although it is not a drop-in replacement: KL against a teacher requires the full teacher/student distributions or an exact streaming equivalent. Our existing chunked-KL path still materializes logits, but chunks the large FP32 softmax/KL intermediates and recomputes chunk gradients; the full 32×256 paired audit recorded above in `docs/memory_optimization_assessment_20261001.md` measured 53.2% lower allocated peak with exact state/KL/PPL. A more ambitious next step is a tiled teacher/student KL backward that streams vocabulary tiles and recomputes logits, with a strict gradient-equivalence test before a full quantization run.
* Activation checkpointing (Chen et al.) saves memory by recomputing forward activations during backward; it avoids large host-device copies but spends extra compute. In REAL-Q Stage 0, evaluate it at Transformer-layer granularity first, with the model in eval mode and the same deterministic batches, because the later quantization peak dominated the 32×256 full run and indiscriminate Stage-0 memory reduction cannot lower that global peak.
* REAL-Q itself explicitly lists custom memory-efficient backward kernels and intermediate activation offloading as system-level future work. PyTorch's selective checkpoint API exposes recompute/save/CPU-offload policies; a controlled hybrid could recompute cheap pointwise/normalization activations and retain/offload selected expensive intermediates. This is a literature-guided hypothesis; it still needs an end-to-end peak measurement and exact output checks.
* The Triton result is consistent with the official Triton guidance to benchmark custom kernels against native references. The correctness gate here is stronger than timing alone: exact state hashes, full KL/PPL, and every captured layer output were required.

## References to consider for the thesis bibliography

1. Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. “Training Deep Nets with Sublinear Memory Cost.” *arXiv:1604.06174*, 2016. https://arxiv.org/abs/1604.06174
2. Jie Ren et al. “ZeRO-Offload: Democratizing Billion-Scale Model Training.” *USENIX ATC*, 2021. https://www.usenix.org/conference/atc21/presentation/ren-jie
3. PyTorch Contributors. “Hooks for autograd saved tensors.” PyTorch Tutorials. https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
4. Qian Zhang et al. “REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent.” arXiv:2609.00049, 2026. https://arxiv.org/abs/2609.00049
5. PyTorch Contributors. “torch.utils.checkpoint.” PyTorch documentation. https://docs.pytorch.org/docs/stable/checkpoint
6. Triton Contributors. “Triton Tutorials: Vector Addition and Kernel Benchmarking.” https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
7. Erik Wijmans, Brody Huval, Alexander Hertzberg, Vladlen Koltun, and Philipp Krähenbühl. “Cut Your Losses in Large-Vocabulary Language Models.” *ICLR*, 2025. https://arxiv.org/abs/2411.09009
