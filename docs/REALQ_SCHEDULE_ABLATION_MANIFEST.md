# REAL-Q schedule A/B experiment manifest

## Status and evidence boundary

This document fixes the protocol for the reverse-cosine A/B experiments
initiated on 2026-07-24. The previously undisclosed numerical choices below
were confirmed by the author for the current REAL-Q implementation. They
supplement, rather than amend, the settings disclosed in `main.tex`; they do
not by themselves establish that the published table rows were generated with
the same values.

The experiments are therefore reported as a **paper-gap comparison**, not an
exact paper-table reproduction. Exact historical model, tokenizer, dataset,
and `lm-eval` task-config revisions remain unavailable. The held-out
WikiText-2 KL/PPL comparison is nevertheless controlled: all schedule arms of
one case use the same current source commit, model artifact, verified dataset
files, sampled token tensors, static Fisher/saliency cache, seeds, and every
non-schedule hyperparameter.

## Author-confirmed numerical protocol

| Setting | Fixed value | Operational meaning |
|---|---:|---|
| `grad_clip` | `1.0` | After global sample averaging and any act-order reindexing, clamp each trailing-column gradient element to `[-1, 1]` before Adam. |
| `final_layer_grad_clip` | `None` | Inherit `grad_clip`; the effective final-transformer-block threshold is also `1.0`. This does **not** disable clipping. |
| `percdamp` | `0.01` | GPTQ Hessian diagonal damping. |
| `act_order` | `true` | Quantize columns in descending Hessian-diagonal order. |
| `w_clip` | `true` | Enable the MSE search for the symmetric weight fake-quantization range. |
| `grad_hessian_topk` | `-1` | Use the full vocabulary for the categorical Fisher-label gradient. |
| `kl_topk` | `-1` | Use full-vocabulary KL in the final transformer block and evaluation. |
| `eval_seq_len` | `2048` | Chunk the held-out WikiText-2 test tokens into 2048-token sequences. |
| Calibration / rotation / refresh seeds | `1 / 0 / 0` | Keep the current REAL-Q split seed domains and explicitly pin every run. |
| Fisher categorical-label base seed | source-fixed `0` | Deterministic per-global-sample label draws; not a CLI field. |
| `rotate` | `true` | Apply the current QuaRot path. |
| `optimized_rotation_path` | `None` | Generate the rotation from `rotation_seed=0`; do not load an optimized rotation artifact. |
| Reverse-cosine semantics | current implementation | For non-aware runs use the literal all-`L` `sin(pi*x/2)` schedule with denominator `L-1` and base ratio `0.01`; the final block then uses its separate LR. For a real low-bit aware A/V/K path, use constant LR in non-final blocks. |
| Loss-slide semantics | current implementation | Refresh only column blocks that retain trailing columns; interpolate alpha over those actual refreshes; do not slide the penultimate block into the final block. |

The chosen semantics preserve the current refactored code. Loss-slide remains
legacy-compatible; the schedule arms intentionally compare the historical
formula with the corrected/current behavior. They do not remove the
discrepancies with the paper wording documented in
`REALQ_PAPER_PROTOCOL.md`.

## Hardware-tuned systems profile

Target node: 4 ranks per arm on NVIDIA L20C GPUs with 183,359 MiB per GPU.
These knobs change chunking rather than the mathematical sample population.
They remain fixed across all arms of a case because different bf16 kernel or
reduction shapes can still introduce small floating-point drift.

| Setting | Qwen3-4B W4A16 | Qwen3-8B W2A4KV4 aware | Rationale |
|---|---:|---:|---|
| `global_loss_bsz` | `48` (local 12) | `32` (local 8) | Four-rank global Stage-0 batch. The 4B local-12 probe completed at 168,612 MiB/GPU; local 16 reached 182,624 MiB/GPU and then OOMed on an additional 9.27-GiB allocation. The 8B local-8 probe completed at 129,794 MiB/GPU. |
| `hessian_accum_bsz` | `256` local | `64` local | 8B has exactly 64 local samples. For 4B, 256 avoids the estimated near-capacity down-projection temporary at 512 while substantially reducing replay chunks. |
| `backward_samples` | `32` global | `32` global | Paper algorithm setting; one Adam step always uses 32 selected global samples. |
| `backward_bsz` | `128` global (local cap 32) | `128` global (local cap 32) | Accommodates even a maximally skewed 32-sample rank filter in one local chunk. Values above 128 cannot reduce chunk count. |
| `final_layer_backward_bsz` | `32` global (local cap 8) | `32` global (local cap 8) | Limits the much larger full-vocabulary FP32 KL temporary in the final block. |
| `lm_eval_batch_size` | `64` | `64` | High-throughput starting point for the ten zero-shot tasks; irrelevant to WikiText KL/PPL. Any OOM-driven fallback must be recorded and applied consistently to compared arms. |

The Stage-0 probes used one complete global batch solely to measure the peak
batch shape: 48 samples for Qwen3-4B and 32 for Qwen3-8B. The formal runs use
the paper's 2048 and 256 samples respectively, processed in repeated batches
with the same peak shape (plus one smaller 4B tail batch). The rejected 4B
local-16 probe and the accepted local-12 probe are retained under
`../experiment_data/batch_probe_20260724/`.

## Fail-closed execution contract

`tools/run_schedule_ablation_variant.py` requires an explicit case name and
validates every author-confirmed, paper-specified, algorithmic, seed, batch,
evaluation, and disabled-mode field against this manifest. A default or
partially specified REAL-Q command is rejected. Each case also has a fixed
directory layout under an absolute experiment root:

`tools/launch_schedule_ablation.py` mechanically expands one warm or measured
case into its four-rank `torchrun` argv. It is only a convenience layer: the
experiment entry point reparses and validates all 73 Config fields rather
than trusting the launcher.

- shared static Fisher/saliency cache: `<case>/cache/static`;
- shared sampled-token cache: `<case>/cache/tokens`;
- shared reference-logit cache: `<case>/cache/runtime`;
- shared post-rotation fingerprint:
  `<case>/cache/rotation_fingerprint.json`;
- atomic warm-completion manifest, binding all of the above content hashes to
  the source/model/data protocol: `<case>/cache/warm_complete.json`;
- unique arm output: `<case>/outputs/<variant>`;
- unique arm metrics: `<case>/metrics/<variant>.json`.

A warm-up run must finish reference-logit generation, rotation fingerprinting,
sampled-token generation, and all four static-cache shards before it can
atomically publish `warm_complete.json`. Measured arms refuse a missing,
in-progress, or provenance-mismatched warm-up and forbid cache recomputation:
they require a valid token-cache hit, reference-cache loader hit, and
collective all-rank static-cache hit. Cache files are hashed before and after
each arm and any mutation invalidates the run.

QuaRot can in principle differ by ULP across physical GPUs. The warm-up
therefore hashes every byte of every post-rotation parameter and buffer.
Measured arms compare their complete post-rotation model SHA256 against that
fingerprint before quantization; a mismatch aborts all ranks. Concurrent
GPU 0--3 versus 4--7 execution is admissible only after bitwise equality has
actually been established.

Warm and metrics files use exclusive reservation locks and cannot be
overwritten. Rank-0 filesystem validation outcomes are broadcast before any
rank proceeds, so a validation failure cannot strand the other ranks at a
later barrier. The JSON records the exact Config, argv, source
commit/diff/wrapper hashes, dataset/token identities, cache hit evidence and
hashes, full post-rotation model hash, runtime versions, GPUs, and unrounded
KL/PPL. Every rank's raw evaluation values are also retained. Cross-GPU
agreement permits only a predeclared numerical-kernel tolerance
(`abs=1e-7, rel=5e-6` for raw KL; `abs=1e-4, rel=5e-6` for PPL); the observed
spread and accepted bound are recorded, and rank 0 is the reported value.

## Controlled cases

### Case A: Qwen3-4B W4A16

- Paper setting: W4A16, symmetric per-row weights (`w_groupsize=-1`),
  A/K/V at 16 bits, QuaRot, 2048 calibration samples of length 2048.
- REAL-Q settings: `num_groups=4`, `blocksize=128`,
  `saliency_clip_percentile=0.99`, `a_loss_ratio=0.95`,
  `backward_samples=32`, loss slide enabled.
- LR: reported non-final LR `2e-4`, base ratio `0.01`, separate final-block
  LR `1e-5`.
- Activation awareness flags: both `false`.
- Arms:
  - `legacy_cos2_scheduled`: historical
    `0.5 * (1 - cos(pi*x))`.
  - `paper`: current production `sin(pi*x/2)`.
- Paper anchor, used only as a gap check: KL `5.44` in paper-table units
  (`0.0544` raw), PPL `13.44`.

`paper_sin_scheduled` is mathematically identical to `paper` for W4A16 and is
not a useful third measured arm. Their scalar schedules are unit-tested for
identity.

### Case B: Qwen3-8B W2A4KV4 activation-aware

- Paper setting: W2 weights with group size 128; symmetric, per-token
  A/K/V4 with clip ratios `0.9`; 256 calibration samples of length 2048.
- Both `act_quant_aware_gptq` and `k_cache_quant_aware_gptq` are `true`.
- REAL-Q settings: `num_groups=4`, `blocksize=128`,
  `saliency_clip_percentile=0.99`, `a_loss_ratio=1.0`,
  `backward_samples=32`, loss slide enabled.
- LR: reported constant-aware/non-final LR `5e-6`, separate final-block LR
  `1e-6`.
- Arms:
  - `legacy_cos2_scheduled`: historical curve, incorrectly retained in aware
    mode.
  - `paper_sin_scheduled`: corrected sine curve, but deliberately still
    scheduled in aware mode. This isolates the curve-formula change.
  - `paper`: current production behavior, constant in aware non-final blocks.
    This isolates disabling the schedule for aware quantization.
- Paper anchor, used only as a gap check: KL `76.7` in paper-table units
  (`0.767` raw), PPL `19.68`.

## Data and evaluation identity

- Model artifacts are the local `modelzoo/Qwen3/Qwen3-4B` and
  `modelzoo/Qwen3/Qwen3-8B` directories; each metrics JSON records the
  mutation-sensitive artifact identity, while the complete post-rotation
  state is content-hashed as described above. Qwen3-4B is a tied-embedding
  checkpoint and therefore has no separate `lm_head.weight` tensor. The
  RealQ loader intentionally unties it for rotation: although Transformers
  emits an intermediate “newly initialized” warning, the loader immediately
  clones the source embedding into the head. This was checked full-tensor and
  bitwise (`max_abs_diff=0`). Qwen3-8B stores an explicit untied head.
- Dataset: `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`.
- Revision:
  `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Expected parquet SHA256:
  - train:
    `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`
  - validation:
    `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`
  - test:
    `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`
- Primary metric: held-out WikiText-2 full-vocabulary FP32 mean token KL.
- Secondary metric: shifted-token perplexity on the same test split.
- These controlled schedule arms set `lm_eval=false`; the ten-task paper
  evaluation is not part of the primary A/B. `lm_eval_batch_size=64` remains
  fixed for a separately versioned task-suite run.
- JSON reports both raw KL and paper-table units (`raw KL * 100`) without
  rounding the stored floats.

## Interpretation thresholds fixed before observing results

For the relative KL change between schedule arms:

- below 1%: negligible;
- 1% to below 5%: small/moderate;
- 5% to below 10%: significant;
- 10% or more: very significant.

PPL is secondary because the evaluated Qwen3 checkpoints are post-trained and
the full-precision checkpoint is not necessarily a PPL optimum.

## Residual run-to-run uncertainty

The installed PyTorch attention backward reports that its
memory-efficient CUDA kernel is nondeterministic. REAL-Q requests
deterministic algorithms with `warn_only=True`, so the warning is recorded but
the kernel is not replaced. Sharing the one warm Stage-0 cache removes this
source from the Fisher/saliency comparison, but each measured arm still runs
its own Block-GD attention backward. The four ranks are participants in one
all-reduced trajectory, not four independent repeats; their final metric
spread therefore checks distributed consistency but does not estimate
arm-to-arm run noise.

Consequently, a large A/B separation can support a directional conclusion,
whereas a result in the predeclared small/negligible bands is not evidence of
a schedule effect by itself. If wall time permits, the current `paper` arm
will be repeated under a separately warmed experiment root; otherwise the
reported conclusion must explicitly retain this nondeterminism limitation.
