# REAL-Q memory scaling and exact packed Fisher accumulation (2026-10-05)

## Purpose and external technical literature

REAL-Q is the method being studied and the experimental baseline, not an external systems-work reference for the optimization contribution. This experiment targets exact GPU-memory reductions in its Stage-0 Fisher collection and backward path. The paper's stated memory bottleneck provides project context; it is not counted below as literature evidence for the implementation.

The external systems literature includes checkpointing/rematerialization, which trades recomputation for less activation storage (Chen et al., [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174); Jain et al., [Checkmate](https://arxiv.org/abs/1910.02653)), and tensor eviction/prefetch plus recomputation (Peng et al., [Capuchin](https://www.microsoft.com/en-us/research/?p=690597)). These works motivate alternative activation-memory policies; they do not validate this implementation's gains. The current prototype specifically uses PyTorch's documented saved-tensor CPU hooks as its implementation mechanism ([saved tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)). Cut Cross-Entropy is a separate precedent for large-vocabulary loss intermediates, not a drop-in KL implementation ([Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009)).

### Additional directly relevant systems papers screened

* [SSDTrain](https://arxiv.org/abs/2408.10013) studies activation offload for LLM training and overlaps data movement with compute. Its asynchronous, scheduled transfers are a stronger design reference than this prototype's synchronous hook copies, although its NVMe training setup differs from one-shot PTQ calibration.
* [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) studies selective recomputation and sequence parallelism. Selective recomputation is relevant to Stage-0 backward; sequence parallelism is not a direct single-GPU drop-in.
* [FlashAttention](https://arxiv.org/abs/2205.14135) uses exact tiled attention to avoid materializing large attention intermediates. It is relevant only if profiling shows the current Stage-0 attention implementation materializes those tensors and does not already use a memory-efficient attention backend.
* [GMLake](https://arxiv.org/abs/2401.08156) targets allocator fragmentation with low-level virtual-memory management. It is relevant to reserved-memory peaks only if traces establish fragmentation as a material cause; it is not a simple application-level REAL-Q change.
* Older runtime schedulers provide additional design space: [vDNN](https://arxiv.org/abs/1602.08124) virtualizes GPU/CPU memory, [SuperNeurons](https://arxiv.org/abs/1801.04380) combines liveness analysis, tensor pooling, and cost-aware recomputation, and [SwapAdvisor](https://par.nsf.gov/servlets/purl/10191573) jointly plans swapping, allocation, and operator scheduling. These are useful systems references, but their CNN-era/framework assumptions need adaptation and their reported savings should not be transferred to REAL-Q.
* The original [GPTQ paper](https://arxiv.org/abs/2210.17323) is useful for understanding the existing per-layer/block quantization schedule and Hessian workspace lifetime. It is part of the quantization-method lineage, not a separate memory-optimization result for this project.

These papers expand the candidate design space, but applicability must be decided from phase-level measurements. They are not assumed to improve REAL-Q without a matched exactness and full-process memory/time comparison.

### Cross-model growth is the optimization priority

The target is enabling larger models, so prioritize each phase by its memory growth from the smaller to the larger model, not just by the smaller model's absolute peak. Existing matched 8-sample × 64-token Qwen3 profiles use the same quantization settings and both stop after layer 0; the figures below are allocated/reserved maxima sampled within those phases:

| Phase | Qwen3-0.6B alloc / reserved (GiB) | Qwen3-1.7B alloc / reserved (GiB) | Absolute growth (alloc / reserved) | Growth factor (alloc) |
|---|---:|---:|---:|---:|
| Stage-0 label/Fisher preparation (`static_fisher_batch_label_sampling`) | 1.740 / 1.777 | 4.498 / 4.541 | +2.758 / +2.764 GiB | 2.58× |
| Layer-0 quantization and refresh | 1.411 / 1.463 | 4.274 / 4.490 | +2.863 / +3.027 GiB | 3.03× |

This preliminary matched partial profile already shows why “optimize the largest 0.6B phase” is not the right rule: layer-0 quantization grows slightly more in absolute allocated bytes and much more proportionally. It does not yet rank the full 28-layer pipeline, because the 0.6B and 1.7B profiles above both stop after layer 0. A full matched smaller-model run is still needed before setting the final priority across all later-layer peaks. Artifacts: `outputs/phase_profile_fisher-scale-qwen3-06b-8x64-20261005-control/` and `outputs/phase_profile_fisher-scale-qwen3-17b-8x64-20261005-control/`.

### Map REAL-Q's stated bottleneck to the implementation before choosing a paper

REAL-Q's Appendix E identifies Stage-0 full-precision model residency and per-layer `O(d^2)` Fisher statistics, and names memory-efficient backward kernels, activation offloading, and fused Block-GD as future system directions. In this repository, `collect_static_end_to_end_saliency_and_fisher` runs the full-model backward; without FSDP it moves the full model to the CUDA device. Its output-gradient hooks form FP32 `grad_flat.T @ grad_flat` and keep one accumulator per transformer layer in `fisher_data` until post-backward aggregation. Only after aggregation is each Fisher converted to BF16 CPU storage. Thus, the on-disk/returned cache is CPU-resident, but the Stage-0 working accumulators are GPU-resident together during the backward. Exact packed accumulation already halves that accumulator component; it does not reduce resident model weights or saved activations.

Stage 1 has a distinct model-width-scaled workspace: the quantizer allocates FP32 `H` with shape `(num_hessian_groups, input_columns, input_columns)` per active module, in addition to the layer weights/activation state and Block-GD temporaries. MLP `down_proj` uses the intermediate width, so its H workspace grows quadratically with that width. This is a code-derived mechanism, not yet a complete tensor-by-tensor attribution of the phase peak.

The full 1.7B 8×128 offload-only pair now completed with exact layer outputs, state hash, KL, and PPL. Stage-0 sampled allocated peak fell from 4.479 to 4.226 GiB (−5.7%), but the candidate's global allocated peak moved to layer-27 quantization (4.404 GiB). Whole-run allocated peak fell only 3.02%, reserved peak rose 0.90%, and quantization time increased from 1867.56 to 1891.01 s (+1.26%). Therefore saved-activation offload alone is not an adequate whole-pipeline result under the strict peak criterion; it exposes the next Stage-1 peak rather than removing it.

The current evidence-based target is consequently two-part: reduce Stage-0 model/activation residency and stream the exact per-layer Fisher accumulators off the GPU as the model grows; separately reduce Stage-1 per-module Hessian and Block-GD workspace, whose matched layer-0 peak grew by about 2.86 GiB from 0.6B to 1.7B. Literature searches should now be scoped to those measured mechanisms (FSDP/parameter streaming and exact statistic streaming for Stage 0; memory-bounded Hessian processing and fused/blockwise backward for Stage 1), rather than treating generic activation offload as the single answer. Full 28-layer cross-model phase comparisons are still needed to set the final ranking.

## Exact packed accumulation

For a symmetric `d x d` matrix, the FP32 accumulator stores the lower triangle including the diagonal, then reconstructs the exact symmetric matrix before following the existing cache path. The number of accumulator elements changes from `d^2` to `d(d+1)/2`; no Fisher entry is discarded. At the two Qwen3 sizes:

| Model | Blocks / hidden size | Full FP32 accumulators | Packed FP32 accumulators | Accumulator saving |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 28 / 1024 | 112 MiB | 56.055 MiB | 49.95% |
| Qwen3-1.7B | 28 / 2048 | 448 MiB | 224.219 MiB | 49.95% |

The theoretical accumulator grows 4x from 0.6B to 1.7B because `L` is unchanged and `d` doubles. This is a temporary GPU accumulation cost. In this repository the final full BF16 Fisher matrices are cached on CPU; therefore this change does not halve the paper's theoretical resident BF16 cache footprint on GPU.

The implementation is opt-in via `--static_fisher_packed_accumulation` in [fisher_storage.py](../gptq_utils/fisher_storage.py) and the Stage-0 collector in `gptq_plus_utils.py`.

## Matched small-config scale smoke

The 0.6B and 1.7B runs below used the same local reduced configuration: one RTX 4060 Laptop GPU (8 GB), W4A16 per-row, 8 WikiText-2 calibration samples of 64 tokens, backward batch 1, and 2 evaluation samples of 128 tokens. They ran Stage-0 over the full model and quantized transformer layer 0; they are not full-model quantization results and are not the paper's 2048-sample calibration protocol. Control and packed candidate differed only in exact Fisher storage representation. All 28 layer-output fingerprints captured by the final full-model evaluation, the quantized-state SHA, short evaluation KL/PPL, and source-integrity check matched.

| Model | Control peak allocated | Packed peak allocated | Allocated reduction | Control reserved | Packed reserved | Reserved reduction |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 1,887,963,136 B | 1,829,300,224 B | 3.11% | 1,908,408,320 B | 1,845,493,760 B | 3.30% |
| Qwen3-1.7B | 4,848,603,136 B | 4,663,734,272 B | 3.81% | 4,875,878,400 B | 4,821,352,448 B | 1.12% |

The phase trace locates the 1.7B high-water mark in Stage-0 Fisher backward; the 0.6B smoke also peaks during Stage-0. The matched whole-process allocated peak is 2.57x larger on 1.7B under this small configuration. This includes model-resident weights and backward workspaces, so it is not a direct estimate of a single component's scaling law. Vocabulary size remains the same in both configurations; the Fisher accumulator grows 4x, while hidden-state-shaped activations grow with hidden width and model-resident weights grow with model size.

Artifacts: `outputs/fisher-scale-qwen3-06b-8x64-20261005-campaign/summary.json` and `outputs/fisher-scale-qwen3-17b-8x64-20261005-campaign/summary.json`.

## Full 0.6B result

A full 28-layer Qwen3-0.6B comparison on the existing reduced 32-sample x 256-token calibration configuration combined packed Fisher with recomputed token-chunk LM-head KL. The validated control and candidate had identical layer-output fingerprints, 507-tensor quantized-state hash, WikiText-2 KL, and PPL; both profile runners reported unchanged source hashes.

| Metric | Control | Candidate |
|---|---:|---:|
| Peak allocated | 2,574,575,104 B | 2,515,912,192 B |
| Peak reserved | 2,722,103,296 B | 2,575,302,656 B |
| KL / PPL | 0.2330884635 / 32.61846542 | identical |
| Quantization time | 1302.878 s | 1195.759 s |

Allocated peak fell 2.28%; reserved peak fell 5.39%. The cumulative allocator trace places the candidate's maximum in `static_fisher_batch_backward`, Stage-0. The single timing pair is not sufficient to claim a speedup; the isolated recomputed-KL microbenchmark was slower, and full timing is sensitive to run conditions. The purpose of this pair is memory and exactness validation.

The earlier run `local-memory-combo-32x256-retry4-20261005` produced identical output hashes but is excluded as a valid paired result because `utils/loss_utils.py` changed during that run and `sources_unchanged` was false. The clean rerun above has `sources_unchanged=true`.

Artifacts: `outputs/local-memory-combo-32x256-final-20261005-campaign/summary.json` and `outputs/phase_profile_local-memory-combo-32x256-final-20261005-recomputed/`.

## Full 1.7B, 28-layer result

The follow-up ran all 28 transformer layers on the same 8x64-token local configuration used in the scale smoke. The control and packed-Fisher candidate differed only in accumulator representation. Both runs completed with unchanged source hashes. The exact 28-layer-output trace, 507-tensor quantized state SHA, and final WikiText-2 KL/PPL matched.

| Metric | Control | Packed Fisher | Reduction |
|---|---:|---:|---:|
| Peak allocated | 4,848,603,136 B | 4,668,254,720 B | 3.72% |
| Peak reserved | 4,886,364,160 B | 4,844,421,120 B | 0.86% |
| Quantization time | 1858.452 s | 1870.856 s | candidate +0.67% |
| Final KL / PPL | 0.5254853964 / 35.92361069 | identical | exact |

In the control, the maximum allocated peak was in `static_fisher_batch_backward` (the second Stage-0 backward batch); the reserved-memory high-watermark occurred later, during layer 16 quantization. In the packed candidate, the largest allocated sample occurred during layer 14 quantization, with both global allocated and reserved peaks remaining below control. This is evidence the smaller Stage-0 footprint propagates into a modest full-process peak reduction on 1.7B, but the final gain is small because the peak shifts to the regular quantization loop. Timing differs by less than one percent in a single pair and is not a speed claim.

Artifacts: `outputs/fisher-scale-qwen3-17b-full8x64-20261005-campaign/summary.json` and the control/candidate profiles under `outputs/phase_profile_fisher-scale-qwen3-17b-full8x64-20261005-{control,packed}/`.

## Current interpretation and next experiment

Packing Fisher measurably lowers allocated peak in the tested settings and becomes more useful as the hidden width grows, but the observed end-to-end savings remain below 4%. The reason is that packed storage halves only the Fisher accumulator, while model weights and Stage-0 autograd activations form most of the peak. The next evidence-driven step is to test the existing Stage-0 saved-tensor CPU offload against a clean full-pipeline control now that KL refresh no longer dominates the peak. If that lowers the global peak while preserving layer traces/state/KL/PPL, activation offload is a second exact system-level contribution; if not, proceed to selective activation recomputation/checkpointing, which REAL-Q also lists as future work. Report host-memory transfer and runtime overhead alongside GPU peak so the resource shift is visible.

The next evidence-driven step is to compare Stage-0 saved-tensor CPU offload against the valid packed-Fisher full 1.7B control. Report host-memory transfer and runtime overhead alongside GPU peak so the resource shift is visible.

### Stage-0 offload pilot on 1.7B

An isolated offload pilot used the same packed-Fisher method in both arms, with 8 calibration samples of 128 tokens, backward batch 1, and quantization through layer 0. The offload candidate used a 1 MiB saved-tensor threshold and moved 3,152 autograd-saved tensors (3.41 GiB cumulative payload) to host memory. Source hashes were unchanged; all 28 traced model-layer outputs, quantized-state SHA, and final KL/PPL matched.

| Metric | Packed control | Packed + activation offload | Reduction |
|---|---:|---:|---:|
| Full measured allocated peak through layer 0 | 4,876,121,088 B | 4,667,974,144 B | 4.27% |
| Full measured reserved peak through layer 0 | 4,909,432,832 B | 4,804,575,232 B | 2.14% |
| Stage-0 sampled allocated high-watermark | 4,818,826,240 B | 4,472,664,576 B | 7.18% |
| Stage-0 sampled reserved high-watermark | 4,909,432,832 B | 4,580,179,968 B | 6.71% |
| Quantization time through layer 0 | 93.35 s | 89.83 s | one pair; no speed claim |

The global candidate peak moved from Stage-0 into layer-0 quantization. The observed global reductions therefore are smaller than the Stage-0 reductions and do not yet establish an all-28-layer result. Two earlier 64-token offload attempts are excluded as no-op configurations: the run logs show `tensors=0` even at 1 MiB, which is why the test was moved to 128 tokens. Artifacts: `outputs/fisher-stage0-offload-qwen3-17b-8x128-20261005-campaign/summary.json` and its two phase profiles.

A full 28-layer 1.7B offload comparison at 8x128 tokens is the next check. It will establish whether later-layer quantization creates a larger peak and whether the Stage-0 savings survive over the complete pipeline.

## Next bottleneck: per-module Hessian workspace and symmetrization

The stage profile points to two growing regions, not one universal peak: Stage-0 full-model/backward residency and Stage-1 per-module Hessian work. With `num_groups=4`, `GPTQPlus.H` is FP32 with shape `(4, input_width, input_width)`. The model configs give Qwen3-0.6B hidden/intermediate widths 1024/3072 and Qwen3-1.7B widths 2048/6144. That means one attention/up/gate Hessian is 16 MiB vs 64 MiB, while one `down_proj` Hessian is 144 MiB vs 576 MiB. These are exact tensor sizes from the implementation shape, before temporary activations or model weights are counted. The down projection therefore adds 432 MiB of persistent Hessian state from 0.6B to 1.7B, with a 4x scaling factor.

The code also previously symmetrized with `0.5 * (H + H.T)`. Python evaluates the sum tensor and then a second scaled output while the old `H` is still live. For the 1.7B `down_proj` example, that can transiently keep three 576 MiB matrices alive (about 1.69 GiB total for this operation); the analogous 0.6B transient is about 432 MiB. The source now uses one `empty_like` output buffer, `torch.add(..., out=buffer)`, then in-place scaling, so the operation needs at most the old and new H at once. It keeps the same elementwise add-then-scale order. A CPU check was bitwise equal on multiple matrix shapes. **Environment note updated 2026-10-07:** the Windows Python remains CPU-only, but the WSL Ubuntu project virtual environment does have CUDA-enabled PyTorch (`2.11.0+cu128`, `torch.cuda.is_available() == True`); WSL CUDA profiling is available. The previously planned CUDA A/B for the symmetrization change remains an open validation item.

Opt-in `--enable_memory_profile` now marks: model-to-device residency, Fisher GPU accumulator bytes after each backward, each module's Hessian allocation and batch workspace, the before/after Hessian symmetrization interval, and GPTQ Hessian-inverse work. These markers synchronize CUDA by design and belong in diagnostic runs; do not use their elapsed times as the speed benchmark. The next paired run should use the same 8×128 Qwen3-1.7B through layer 0, then repeat only the winning exact implementation against 0.6B under the same settings. It must compare layer outputs/state/KL/PPL and peak allocated and reserved memory. After locating the layer-0 module's winning allocation, run full 28 layers to ensure the peak does not migrate above control.

### Literature fit and limits

The memory techniques should be matched to the component. The GPTQ paper describes the blockwise second-order quantization lineage and is useful context for why `H` and its inverse exist, but it does not prescribe this repository's multi-group 3D Hessian storage or offer a direct fix for its symmetrization temporary ([Frantar et al., GPTQ](https://arxiv.org/abs/2210.17323)). For model-weight residency in Stage-0, ZeRO-Infinity studies CPU/NVMe offload and memory hierarchy for very large model training; it is a possible design reference for staging frozen parameters, but its optimizer/training runtime cannot be claimed as a ready-made PTQ solution ([Rajbhandari et al., ZeRO-Infinity](https://arxiv.org/abs/2104.07857)). For saved activations, SSDTrain overlaps activation offload and compute in large-model training, but its asynchronous SSD design differs from the current one-shot calibration hook ([Jiang et al., SSDTrain](https://arxiv.org/abs/2408.10013)). The local exact saved-tensor CPU-offload pilot shows that activation offload alone does not satisfy our full-pipeline peak constraint.

At this point the strongest directly actionable code target is the exact temporary-memory reduction around wide-module Hessians, followed by measuring the `H` batch product and Cholesky/inverse peak separately. Literature does not yet justify replacing the full Hessian with a lower-rank or diagonal proxy under the user's exactness requirement; such approximations would change the quantizer and need a separate accuracy study. The REAL-Q paper remains the baseline/specification for this project, not an external related-work citation.
