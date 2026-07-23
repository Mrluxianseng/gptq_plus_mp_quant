# REAL-Q paper protocol: specified settings and reproducibility gaps

This file records what can and cannot be reconstructed from `main.tex`.  It is
an audit record, not a claim that the paper tables are currently reproducible.
The paper source is intentionally not modified by the implementation audit.

## Settings explicitly specified by the paper

| Setting | Paper requirement |
|---|---|
| Weight quantization | Symmetric. W4A16 is per-row; group size 128 is used for W3/W2 and W*x*A4KV4 (`main.tex:235`). |
| Activation/KV quantization | Per-token A/K/V, with `a_clip=k_clip=v_clip=0.9`; weight group size 128 (`main.tex:333-335`). |
| Calibration | 2048 WikiText-2 sequences for W4A16/W4A4KV4, except 256 for LLaMA-3.1-70B and low-bit/W2A4KV4/W3A4KV4 (`main.tex:235,643`). |
| Dynamic optimizer | Adam, beta1=0.9, beta2=0.999, epsilon=1e-8, bias correction, mini-batch 32 (`main.tex:235,620`). |
| Block size | 128 columns (`main.tex:58,76,394`). |
| Layer schedule | Reverse cosine `sin(pi*x/2)`, starting at 0.01 of the reported scheduled LR; disabled for activation-aware runs (`main.tex:622-627,719`). |
| Final block | True full-vocabulary KL against the LM head and its own LR (`main.tex:620,635`). |
| Activation-loss clip | P95 detached scaling for Qwen3-0.6B/1.7B/4B (`main.tex:629`). |
| Saliency clip | P99 before saliency-weighted Hessian construction (`main.tex:631`). |
| Group saliency | Squared Euclidean norm of the output gradient restricted to each row group, hence a channel sum rather than mean (`main.tex:133-138`). |
| Rotation | QuaRot for every method (`main.tex:235`). |
| Evaluation | Held-out WikiText-2 KL/PPL plus ten zero-shot tasks (`main.tex:235,237`). |
| Reported calibration seed | Seed 1 for the reported Qwen3-0.6B W4A16 row (`main.tex:836-848`). |

The model/setting-specific non-final and final learning rates are listed in
`main.tex:719-755`; a single `grad_lr` default cannot represent that table.

## Information not disclosed by the paper

Exact table reproduction additionally depends on values that the paper does
not state:

- non-final and final gradient-clipping thresholds;
- GPTQ damping, activation order, weight-clipping search, and Hessian
  accumulation batch size;
- final-block backward mini-batch size if it differs from the stated Adam
  mini-batch 32;
- the population/axes over which activation-loss P95 and saliency P99 are
  computed (the implementation uses the complete global refresh population);
- calibration, rotation, and refresh/sample-shuffle seed domains for every
  table row;
- concrete model/tokenizer/dataset revisions;
- `lm-eval` version, task-config revisions, and evaluation batch size;
- the exact legacy loss-slide indexing and terminal-layer policy.

These values must be added to the paper or to an author-approved experiment
manifest before any launcher can truthfully be called an exact paper
reproduction entry point.

## Repository conflicts found by the audit

| Knob | Paper | Refactored default/template | Legacy sweep |
|---|---|---|---|
| Calibration seed | Reported Qwen3-0.6B row uses 1 | `Config` defaults to 42; convenience template now uses 1 | parser defaults to 42; sweep does not pin it |
| Calibration samples | Usually 2048; specified exceptions use 256 | `Config` 128; template 2048 | LR sweep 256 |
| Adam mini-batch | 32 | backward and final inherit 32 | backward 32, final 8 |
| Gradient clip | Not disclosed | 1.0 / 1.0 | `5e-5` / `5e-4` |
| Learning rate | Per model/setting table | one convenience default `3e-4` | sweep-specific values |
| Eval datasets | Held-out WikiText-2 plus ten tasks | KL/PPL default only WikiText-2 | parser defaults multiple KL/PPL datasets |

`realq/scripts/run.sh` is therefore explicitly a convenience template.  It
must not be cited as the command that generated a paper row.

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
