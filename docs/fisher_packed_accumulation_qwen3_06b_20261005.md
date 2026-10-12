# Exact packed Fisher accumulation on Qwen3-0.6B

## Goal and relation to REAL-Q

REAL-Q forms full empirical Fisher matrices for the hidden states and reports their GPU cache cost as `2 * L * d^2` bytes when stored in BF16. For Qwen3-0.6B (`L=28`, `d=1024`), this is about 56 MiB. REAL-Q also notes that its Stage-0 full-precision backward pass and the retained Fisher matrices contribute memory overhead, and lists memory-efficient backward, activation offload, and fused block optimization as future work. See *REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent*, Appendix D.6 and the future-work discussion; paper: <https://arxiv.org/abs/2609.00049>.

This experiment targets the **GPU FP32 accumulation buffers** used while collecting the Fisher matrices in Stage-0. In this repository the accumulated matrices are subsequently converted to the existing full BF16 CPU cache; the persistent cache is therefore already off GPU. This differs from the paper's theoretical GPU-resident BF16 cache accounting. The change does not approximate Fisher, remove off-diagonal terms, or change the downstream REAL-Q cache/loss API. After accumulation and distributed reduction, each full matrix is reconstructed and converted to the existing full BF16 CPU cache. Thus the measured target is the temporary GPU accumulator, not the paper's complete BF16 GPU-cache footprint.

For a symmetric `d x d` matrix, packing the diagonal and lower triangle stores `d(d+1)/2` values instead of `d^2`. At `d=1024`, that is 524,800 rather than 1,048,576 FP32 values. Across 28 layers:

| Representation | GPU accumulator size |
| --- | ---: |
| Full FP32 matrices | 112.000 MiB |
| Packed symmetric FP32 matrices | 56.055 MiB |
| Reduction in this component | 49.95% |

The theoretical ratio approaches 50% as `d` grows. This does **not** imply that total process VRAM or whole-run peak drops by 50%; the accumulator is only one term in peak memory.

## Implementation

The opt-in flag `--static_fisher_packed_accumulation` stores each layer's GPU Fisher accumulator as a one-dimensional lower-triangle vector. The existing per-layer all-reduce operates on this compact vector. The implementation then reconstructs a full symmetric matrix, normalizes it, and follows the original full BF16 CPU cache path. CPU triangle indices are used so index tensors do not occupy persistent GPU memory. The flag is off by default; other Fisher collection paths are unchanged.

The symmetry assumption was checked using an actual CUDA Gram matrix of the same hidden width, and pack/unpack preserves the full matrix and its quadratic form. The full paired run below is the end-to-end guard against downstream numerical or control-flow differences.

## Local paired pilot

Machine: WSL2, RTX 4060 Laptop GPU, 8 GB. Model: local Qwen3-0.6B. Both runs used the same seed, W4A16 per-row quantization (`w_groupsize=-1`), the same REAL-Q code path, and the same manifest; candidate differed only by the packed-accumulation flag. The manifest used 32 WikiText-2 calibration samples of length 256, 32 backward samples, and 8 evaluation samples of length 1024. This is a **reduced local pilot**, not the paper's main calibration setting of 2048 samples at sequence length 2048. The quantization quality and memory numbers below must not be presented as paper-protocol results.

| Metric | Control | Packed candidate |
| --- | ---: | ---: |
| Full 28-layer quantization time | 1302.878 s | 1125.236 s |
| Peak CUDA allocated | 2,574,575,104 B | 2,543,747,584 B |
| Peak CUDA reserved | 2,722,103,296 B | 2,724,200,448 B |
| Quantized state SHA-256 | `99589bb56bfe9f3c1bce8381b3d36cdc101e96367aedd9eacb4617fda1404b81` | identical |
| Final WikiText-2 KL | 0.2330884635 | identical |
| Final WikiText-2 PPL | 32.61846542 | identical |

Allocated peak fell by 30,827,520 bytes (1.20%). Reserved peak increased by 2,097,152 bytes (0.08%), which is effectively unchanged at this single-run precision. The candidate's quantization time was lower in this pair, but one run per side with fixed order is insufficient evidence for a speedup; no speed claim is made.

For layer-output verification, the recorder captured eight forward outputs for each of 28 layers in each run. Every per-layer fingerprint matched. The canonical combined fingerprint SHA-256 was `01a9dc59db1a787afbab7214f679ba4dc349d933a0d7fc92c53d23a8087c75ed` for both runs. State hashes, KL, and PPL also matched exactly in the recorded precision.

Artifacts are under `outputs/local-fisher-packed-32x256-retry1-campaign/` and `outputs/phase_profile_local-fisher-packed-32x256-retry1-{control,packed}/`.

## Conclusion and limits

The storage transformation works on the local full-model pilot: it halves the Stage-0 GPU FP32 Fisher accumulator while preserving all Fisher entries and the measured quantization outputs. Whole-run allocated peak improves only 1.20%, and reserved peak does not improve. This is consistent with the Fisher accumulator being a small fraction of the observed end-to-end peak.

This is not yet a paper-scale result. Before making a thesis claim, run repeated matched pairs and a formal-setting memory audit, recording stage-wise allocated/reserved peaks. To reduce the paper's retained BF16 cache itself by about half would require a separate packed-cache representation and lazy per-layer reconstruction; this change deliberately does not claim that result.
