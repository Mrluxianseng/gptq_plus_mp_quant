# REAL-Q paper protocol: specified settings and reproducibility gaps

This file records what can and cannot be reconstructed from `main.tex`.  It is
an audit record, not a claim that the paper tables are currently reproducible.
The paper source is intentionally not modified by the implementation audit.

The author-confirmed settings selected on 2026-07-24 for the current
reverse-cosine A/B experiments are recorded separately in
[`REALQ_SCHEDULE_ABLATION_MANIFEST.md`](REALQ_SCHEDULE_ABLATION_MANIFEST.md).
Those settings close choices for the new runs but do not retroactively turn
previously undisclosed paper-table settings into paper disclosures.

## Settings explicitly specified by the paper

| Setting | Paper requirement |
|---|---|
| Weight quantization | Symmetric. W4A16 is per-row; group size 128 is used for W3/W2 and W*x*A4KV4 (`main.tex:235`). |
| Activation/KV quantization | Per-token A/K/V, with `a_clip=k_clip=v_clip=0.9`; weight group size 128 (`main.tex:333-335`). |
| Calibration sequence length | 2048 (`main.tex:237,301`). |
| Calibration | 2048 WikiText-2 sequences for W4A16/W4A4KV4, except 256 for LLaMA-3.1-70B and low-bit/W2A4KV4/W3A4KV4 (`main.tex:235,643`). |
| Dynamic optimizer | Adam, beta1=0.9, beta2=0.999, epsilon=1e-8, bias correction, 32 gradient samples per block step (`main.tex:235,620`). |
| Optimizer-state lifetime | Reset for each new linear module, then retained while the active trailing-column suffix shrinks across that module's block refreshes (`main.tex:188,870`). |
| Block size | 128 columns (`main.tex:58,76,394`). |
| Layer schedule | Reverse cosine `sin(pi*x/2)`, starting at 0.01 of the reported scheduled LR; disabled for activation-aware runs (`main.tex:622-627,719`). |
| Final block | True full-vocabulary KL against the LM head and its own LR (`main.tex:620,635`). |
| Activation-loss clip | P95 detached scaling for Qwen3-0.6B/1.7B/4B (`main.tex:629`). |
| Saliency clip | P99 before saliency-weighted Hessian construction (`main.tex:631`). |
| Group saliency | Squared Euclidean norm of the output gradient restricted to each row group, hence a channel sum rather than mean (`main.tex:133-138`). |
| Saliency groups | `N_g=4` in every experiment (`main.tex:133`). |
| Fisher storage | One full aggregated matrix per transformer block, stored in bfloat16 (`main.tex:803-811`). |
| Rotation | QuaRot for every method (`main.tex:235`). |
| Evaluation | Held-out WikiText-2 KL/PPL plus ten zero-shot tasks (`main.tex:235,237`). |
| Hardware | RTX Pro 6000 96 GB; paper reports 4 GPUs for LLaMA-3.1-8B and Qwen3-0.6B/1.7B/4B/8B, and 8 GPUs for LLaMA-3.1-70B and Qwen3-32B (`main.tex:776-793`). |
| Reported calibration seed | Seed 1 for the reported Qwen3-0.6B W4A16 row (`main.tex:836-848`). |

The model/setting-specific non-final and final learning rates are listed in
`main.tex:719-755`; a single `grad_lr` default cannot represent that table.

## Information not disclosed by the paper

Exact table reproduction additionally depends on values that the paper does
not state:

- non-final and final gradient-clipping operator and thresholds;
- GPTQ damping, activation order, weight-clipping search, and Hessian
  accumulation batch size;
- internal backward gradient-accumulation chunk size (including whether the
  final block uses a smaller chunk while retaining the same 32 samples);
- evaluation context/chunk length (the implementation's `eval_seq_len=2048`
  is not explicitly specified as an evaluation setting by the paper);
- the population/axes over which activation-loss P95 and saliency P99 are
  computed;
- Fisher categorical-label base seed and draw policy, plus calibration,
  rotation, and refresh/sample-shuffle seed domains for every table row;
- exact refresh-sample selection/order, shuffle cadence, and whether its
  scheduler cursor is shared across modules and transformer blocks;
- the calibration split and exact concat/random-span/overlap sampling policy;
- exact A/V/K fake-quant hook sites and scale axes, and whether “aware” covers
  K through an independent switch;
- concrete model/tokenizer/dataset revisions;
- `lm-eval` version, task-config revisions, and evaluation batch size;
- the exact legacy loss-slide indexing and terminal-layer policy.

The audited reproduction protocol currently fixes two additional multi-GPU
choices that are not stated by the paper:

- `dp_global_shuffle=true`: one model-wide refresh scheduler selects global
  calibration IDs, then each rank filters to its contiguous data shard. This
  changes per-step sample membership relative to the old stratified mode and
  is therefore an algorithm/protocol choice, not a performance-only toggle.
- `group_parallel_quant=rank`: output-row quantization and, for multiple
  output groups, Hessian reduction are rank-parallel. It preserves the same
  real-valued row-separable algorithm, but its reduction/kernels are not
  universally bit-exact with `none`.

For clarity, the audited implementation exposes the unspecified
activation-loss percentile population rather than silently choosing one:

- `a_loss_clip_scope=local_backward_chunk` (the default and historical
  paper-code path) computes current/next thresholds independently inside
  each rank's backward chunk after flattening its
  `(sample, token, hidden channel)` elements;
- `a_loss_clip_scope=global_refresh` (the newer partition-invariant path)
  computes current/next thresholds independently over all selected
  `(sample, token, hidden channel)` elements in one refresh, globally across
  ranks and all gradient-accumulation chunks. It requires an additional
  no-grad forward prepass and is not the paper runtime path;
- saliency P99 is computed separately for each linear module, over its complete
  `(calibration sample, token, saliency group)` tensor, globally across ranks
  and independent of static-precompute micro-batch partitioning.

These values must be added to the paper or to an author-approved experiment
manifest before any launcher can truthfully be called an exact paper
reproduction entry point.

## Repository conflicts found by the audit

| Knob | Paper | Refactored default/template | Legacy sweep |
|---|---|---|---|
| Calibration seed | Reported Qwen3-0.6B row uses 1 | `Config` defaults to 42; convenience template now uses 1 | parser defaults to 42; sweep does not pin it |
| Calibration samples | Usually 2048; specified exceptions use 256 | `Config` 128; template 2048 | LR sweep 256 |
| Adam gradient samples per step | 32 | 32 | 32 |
| Backward accumulation chunk | Paper gives the 32-sample step but not an internal chunk size | non-final/final 32 | non-final 32, final 8 |
| Static Fisher/saliency batch | Not disclosed | `Config` / template 16 | LR sweep 8 |
| Hessian accumulation batch | Not disclosed | `Config` 64; template inherits it | LR sweep 32 |
| Gradient clip | Not disclosed | 1.0 / 1.0 | `5e-5` / `5e-4` |
| Learning rate | Per model/setting table | one convenience default `3e-4` | sweep-specific values |
| Small-Qwen activation-loss P95 | 0.95 for Qwen3-0.6B/1.7B/4B | `Config` / template 1.0 unless explicitly overridden | sweep 1.0 unless explicitly overridden |
| Activation-loss P95 population | Not disclosed | `local_backward_chunk` by default; explicit `global_refresh` alternative | Historical local backward-chunk behavior |
| Weight grouping | W4A16 per-row; W2/W3 and W*x*A4KV4 group 128 | template now selects this from bit widths; `Config` alone defaults per-row | sweep defaults per-row unless overridden |
| Eval datasets | Held-out WikiText-2 plus ten tasks | KL/PPL defaults to WikiText-2 and `lm_eval=False`; template does not enable tasks | parser defaults multiple KL/PPL datasets but task scoring still requires its enable flag |

`realq/scripts/run.sh` is therefore explicitly a convenience template.  It
must not be cited as the command that generated a paper row. In particular,
setting only `A_BITS`, `K_BITS`, and `V_BITS` does not enable the corresponding
activation-aware flags, and model-specific activation-loss clipping and
learning rates still require explicit arguments.

Legacy parser defaults and refactored `Config` defaults are not intended to be
equivalent. They differ materially in weight bits, LR/final LR, enabled
rotation/slide/weight clipping/act-order switches, dataset, sample count, and
several batch sizes. Exact old/new comparisons in this audit pin every
numerically relevant knob; they do not imply that invoking each entry point
with no arguments yields the same experiment.

The paper also says that “each method” is evaluated in aware and unaware
activation variants (`main.tex:335`), while the following text/table reports
both variants for REAL-Q but only aware baselines (`main.tex:339,349-371`).
This is an internal reporting ambiguity that must be resolved before the
activation-table protocol is fully reproducible.

## Reverse-cosine endpoint ambiguity

The reverse-cosine equation at `main.tex:622-627` normalizes block index `i`
by `L-1` across all `L` transformer blocks, which makes index `L-1` reach the
reported scheduled final LR.
The same setup states that this final transformer block instead uses a
separate final-layer LR and true KL. The implementation follows the equation
literally and then overrides the final block; hence the last *non-final* block
uses

```text
sin(pi/2 * (L-2)/(L-1))
```

times the ramp range and does not reach the scheduled endpoint. If the intended
schedule spans only the `L-1` non-final blocks, its denominator should instead
be `L-2`. The paper must choose one interpretation; the audit does not silently
renormalize it.

## Randomness and deterministic replay

The statement at `main.tex:836` that calibration-set sampling is the only
stochastic source is not literally true of the implementation. Calibration
itself samples random spans from the concatenated WikiText-2 corpus. Stage 0
also draws categorical target labels from the full-precision distribution for
the empirical Fisher, using a deterministic per-global-sample seed with
currently undisclosed base seed 0. QuaRot generates a random Hadamard
transform from its own rotation seed. Block-GD sample ordering has a separate
refresh seed.

The corrected code records the configurable calibration, rotation, and
refresh seeds, so a fixed complete configuration is replayable and changing
the calibration seed leaves those algorithmic seed products fixed. The Fisher
label base seed remains source-fixed at 0 rather than being an independent
manifest field; changing it requires source provenance and a cache-schema
bump. The defensible paper claim is therefore that calibration sampling is the
only seed *varied in that sweep*, not the only random operation in the
pipeline.

## Systems-protocol boundary

The paper's 70B procedure uses FSDP for Stage 0 and its memory discussion also
claims activation checkpointing (`main.tex:638,831`). This audit deliberately
timeboxes performance-only work and has not reproduced the paper's 70B
activation-checkpointed memory or wall-clock measurements. The eight-GPU
validation establishes distributed numerical agreement on the tiny-Llama
fixture; it is not evidence for the 70B systems claims.

## Loss-slide discrepancy

The current paper defines the total schedule length as every column block in
every linear module and slides every non-final transformer block into its
successor (`main.tex:195-203,457-463`).  Both implementations preserve the
historical implementation instead:

```text
R_m = max(ceil(C_m / B) - 1, 0)
R_total = sum_m R_m
alpha(k) = 1 - k / (R_total - 1),  k = 0 .. R_total-1
```

Only blocks with trailing, still-unquantized columns produce a Block-GD
refresh, and the penultimate transformer block does not slide into the final
block.  This is an intentional old/new compatibility constraint in the
current code, but it is not consistent with the current paper text.  The
paper must describe this legacy schedule, or both implementations must change.

## Query quantization terminology

The paper's `A4KV4` notation specifies input activations plus K/V cache
quantization.  It does not define a separately fake-quantized query tensor,
and the code has no `q_bits`, `q_clip`, or query-aware mode.  The Q tensor is
Hadamard-rotated together with K so their inner product is preserved; only K
is fake-quantized after RoPE.  “Q-aware quantization” must not be used as a
synonym for A quantization at the input of `q_proj`.
