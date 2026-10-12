# Shadow quantization mechanism audit — 2026-09-29

## Question

At selected REAL-Q column-block refreshes, does the next-block state predicted
by the shadow quantizer match the state that GPTQ actually reaches after the
current Adam update, and does that mismatch materially rotate the refresh
gradient?

## Protocol

Qwen3-0.6B, W4, seed 44, 32 WikiText2 calibration sequences of length 256,
`blocksize=128`, `act_order`, and the existing `fisher_diag_mse` refresh were
used. The audit captured predicted and actual states at consecutive boundaries.
At the later boundary, the actual-state gradient was recomputed using the exact
sample indices from the earlier shadow refresh. The target linear-module output
was captured on both passes. The audit-only extra backward does not step Adam or
advance the sample scheduler.

Selected checks: layer-0 `self_attn.q_proj` blocks 0 and 1; layer-0
`mlp.down_proj` blocks 0 and 1; layer-1 `mlp.down_proj` blocks 0 and 1; and
layer-4 `mlp.down_proj` blocks 0 and 1. The layer-4 run quantized through layer
4; the other runs quantized through layer 1.

## Preliminary results

| Layer / module | predicted block | weight relative L2 | gradient cosine | gradient relative L2 | module-output relative L2 |
|---|---:|---:|---:|---:|---:|
| 0 / `self_attn.q_proj` | 0 | 0.176% | 0.999991 | 0.465% | 0.064% |
| 0 / `self_attn.q_proj` | 1 | 0.265% | 0.999990 | 0.478% | 0.082% |
| 0 / `mlp.down_proj` | 0 | 0.232% | 0.9999996 | 0.113% | 0.123% |
| 0 / `mlp.down_proj` | 1 | 0.212% | 0.9999992 | 0.165% | 0.124% |
| 1 / `mlp.down_proj` | 0 | 0.789% | 0.999867 | 2.864% | 0.326% |
| 1 / `mlp.down_proj` | 1 | 0.529% | 0.999902 | 2.371% | 0.277% |
| 4 / `mlp.down_proj` | 0 | 1.381% | 0.999963 | 0.953% | 0.541% |
| 4 / `mlp.down_proj` | 1 | 1.053% | 0.999975 | 0.757% | 0.472% |

The compared sample-index lists and column intervals match exactly in every
completed event. The three completed audit runs produce the same final partial-
prefix model hash, KL, and PPL, so the audit-only backward did not perturb the
quantization trajectory in these runs. Peak allocated CUDA memory was
2,559,177,728 bytes. These are mechanism diagnostics, not quality comparisons:
the model was quantized only through layer 1 and layers 2 onward remained FP.

The layer-4 run also completed successfully. Its final quantized state hash is
identical to the earlier matched 32-sample, four-layer shadow-preview candidate
(`54853fa0...`); KL/PPL are likewise identical at `0.10253544897 / 31.0459443`.
Both Adam-instrumented runs (blocks 0–1 and 2–3) have that same final state
hash, KL/PPL, and peak CUDA allocation. This confirms that the added diagnostic
backward does not change the quantization result in this reproduced candidate.
Runtime is not compared: the diagnostic recomputes matched-sample gradients.

## Update-aware shadow replay

The original preview quantizes the next block before applying the pending Adam
update. I added an audit-only replay that clones the live floating tail, applies
the same already-computed first-order update to that clone, and then runs the
existing GPTQ next-block preview. It does not change the live state or add a
backward pass. Compared with the actual next-block weights at the following
boundary, the original preview had 1.381%, 1.053%, 1.038%, and 0.868% relative
L2 error for blocks 0–3. The update-aware replay had exactly zero error and
zero maximum absolute difference for all four blocks. Since it uses the same
update and GPTQ quantization routine in the same order, this confirms the
source of the measured state mismatch: the earlier preview omitted the Adam
update that happens before the real block is quantized.

This is a mechanism result, not a KL improvement: the replay was diagnostic and
did not supply a new loss or change the quantization trajectory. A simple
post-hoc subtraction from an already-quantized preview did not improve the
weight error; the order has to be update first, GPTQ quantization second.

## Adam-preconditioned direction check

For layer-4 `mlp.down_proj` blocks 0 and 1, the audit captured the Adam
`exp_avg`, `exp_avg_sq`, step, and warm-start offset before the live refresh
update. It then applied the exact Adam formula, same LR (`1.1915e-5`), same
gradient clipping, and that same saved moment state separately to the shadow
gradient and the later actual-state gradient. This is a counterfactual matched-
state comparison: it isolates how Adam transforms the gradient discrepancy;
it is not the historical update from two independently advanced Adam states.

| Predicted block | raw gradient cosine | raw gradient relative L2 | same-state Adam update cosine | Adam update relative L2 | coordinate sign disagreement |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.999963 | 0.953% | 0.987574 | 15.765% | 0.797% |
| 1 | 0.999975 | 0.757% | 0.998383 | 5.704% | 0.360% |
| 2 | 0.999964 | 0.895% | 0.999401 | 3.474% | 0.241% |
| 3 | 0.999977 | 0.716% | 0.999715 | 2.402% | 0.169% |

The update norms are nearly equal in each pair (`0.004290 / 0.004290` and
`0.004227 / 0.004214`). At this cold-start stage, Adam's first update is close
to a clipped sign-normalized gradient, so a small fraction of coordinate sign
flips can enlarge relative update error even while the overall cosine remains
high. The effect is largest on the first block and falls across the next three
blocks as each still-floating column's Adam moments accumulate. This fits a
cold-start explanation, but four adjacent blocks do not show that it causes a
material final KL regression.

## Interpretation and limits

The pre-Adam shadow and post-Adam actual states are close in the tested first
two blocks, including after errors from previous layers. By layer 4, the
predicted next-block weights differ by about 1.1–1.4% relative L2 and module
outputs by about 0.5%; the raw gradient cosine remains above 0.99996. The stale
shadow state is therefore measurably less exact deeper in the network, but the
tested raw gradients still point in nearly the same direction. Adam's cold
start makes the per-coordinate update more sensitive: the same-state Adam
cosine drops to 0.9876 on the first block, then rises above 0.9994 on blocks
2–3 as moments accumulate. The corresponding sign-disagreement fraction falls
from 0.80% to 0.17%. That is evidence that raw-gradient cosine alone can
understate cold-start update mismatch, but it does not establish a harmful
end-to-end effect or explain earlier KL regressions. The update-aware state
replay removes the next-block weight mismatch exactly in these deterministic
checks, but because the diagnostic leaves the training loss unchanged, it does
not yet show that using the replay to alter optimization improves KL.

It does not establish that all layers/modules behave this way. The present
measurements cover only the first four blocks of selected modules, with one
calibration seed and one minibatch schedule. Adam updates were checked only for
the first four layer-4 down-proj blocks, using shared moments; no alternative
updates were applied to the model.

## One-step Adam-aware refresh candidate

I added `--preview_next_quant_block_adam_aware`. It first computes a provisional
gradient on the old preview, uses it to estimate the Adam step, applies that
step to a disposable live-weight copy, runs GPTQ preview again, and recomputes
the gradient on that state using the *same* calibration sample IDs. The second
gradient is the one applied to the real weights. This is one correction pass,
not a fixed-point solve, and adds one backward per refresh.

A matched five-layer-prefix smoke pair completed with the same calibration
seed (44), 32 calibration samples, 8 evaluation samples, and shared source
hashes. At the two audited layer-4 down-proj blocks, sample IDs and columns
matched across arms. Results:

| Arm | KL | PPL | Quantization time | Peak allocated CUDA memory |
|---|---:|---:|---:|---:|
| Old preview control | 0.10253545 | 31.04594 | 287.1 s | 2.56 GB |
| One-step Adam-aware | 0.10187542 | 31.16105 | 371.1 s | 2.56 GB |

KL improved by 0.64%, while PPL worsened by 0.37%; quantization time rose by
29.3% and peak allocated memory did not increase. At the audited blocks, the
candidate's provisional shadow-weight relative L2 error fell from 1.381% to
0.381% and from 1.053% to 0.231%; gradient relative L2 errors fell from 0.953%
to 0.076% and from 0.757% to 0.053%. A replay using the final applied gradient
again matched the actual next-block weights exactly, as expected from applying
the same update and GPTQ routine in the same order.

This was only a short-prefix signal: it used one seed and five quantized
layers, PPL moved in the wrong direction, and the control still used the stale
preview path rather than unmodified REAL-Q. The full-model pair below replaces
this preliminary comparison for deciding whether the method works.

## Full-model REAL-Q control versus Adam-aware preview

After the preview correction, I ran a matched all-layer pair on the Qwen3-0.6B
configuration from the smoke manifest. The control is unmodified REAL-Q with
shadow preview disabled. The candidate enables the one-step Adam-aware shadow
preview on layers 0–26. Both arms quantize all 28 layers; layer 27 uses the
existing true-KL refresh path in both arms. Calibration/evaluation settings
match (seed 44, 32 calibration samples, eight Wikitext-2 evaluation samples,
same source hashes apart from each copied entry.py path; the copied files have
identical hashes).

| Arm | Full-model KL | PPL | Quantization time | Wall time | Peak allocated CUDA memory |
|---|---:|---:|---:|---:|---:|
| REAL-Q control, preview off | 0.22546510 | 32.73832 | 6312.0 s | 6356.3 s | 5.91 GB |
| Adam-aware preview, layers 0–26 | 0.22884201 | 32.91666 | 7321.2 s | 7391.4 s | 5.91 GB |

The candidate worsened full-model KL by 1.50% and PPL by 0.54%, while adding
16.0% quantization time. Peak allocated memory was unchanged. The final layer
was not shadow-previewed, but it was fully quantized in both arms and used the
true-KL update path; thus the reported end-to-end metrics include its actual
effect on the final model.

The two layer-4 down-proj mechanism records have matching sample IDs and column
ranges. The Adam-aware predicted next-block weights match the later executed
weights exactly in both records, but this equality is a replay check, not proof
that the gradient was computed at the final executed state. The algorithm first
uses a provisional gradient `g0` to preview `Q(W - Adam(g0))`, then computes
`g1` on that preview and applies `Adam(g1)` to the real floating tail. The
actual next block is therefore `Q(W - Adam(g1))`, which need not be the state
on which `g1` was evaluated. The audit separately replays GPTQ after the real
`g1` update and gets exact next-block weights, as expected from using the same
update and quantizer; it does not close this one-pass gradient-state mismatch.

The candidate's ordinary predicted gradient differs from the later executed
gradient by relative L2 0.076% and 0.053%; same-state Adam-update cosine is
0.99878 and 0.99987, with sign disagreement 0.129% and 0.052%. Thus the replay
shows that the GPTQ state can be reproduced, while the gradient remains a
one-pass approximation. Correcting that local replay did not translate into a
full-model KL gain.

This is a negative full-model result for one seed and a small evaluation set,
not proof that every variant must fail. It is enough to reject the current
one-step Adam-aware preview as a thesis contribution in its present form: it
misses the primary KL objective and costs about 16% more time. Do not tune its
parameters or spend another full-model run on this unchanged mechanism. Keep
the code and artifacts as a documented negative result; a future revision
would need a new mechanism and a cheaper falsification criterion before
another full-model comparison.

## Artifacts

- Q/K projection run: `outputs/shadow_quant_mechanism_audit_20260929/`
- Layer-0 down projection: `outputs/shadow_quant_mechanism_audit_downproj_20260929/`
- Layer-1 down projection: `outputs/shadow_quant_mechanism_audit_downproj_layer1_20260929/`
- Layer-4 down projection: `outputs/shadow_quant_mechanism_audit_downproj_layer4_20260929/`
- Layer-4 down projection with Adam-state audit: `outputs/shadow_quant_mechanism_audit_adam_downproj_layer4_20260929/`
- Later layer-4 down-proj blocks: `outputs/shadow_quant_mechanism_audit_adam_downproj_layer4_later_20260929/`
- Update-aware shadow replay: `outputs/shadow_quant_mechanism_audit_adam_aware_downproj_layer4_20260929/`
- One-step candidate control: `outputs/shadow_quant_mechanism_audit_adam_aware_pair_control_20260929/`
- One-step candidate: `outputs/shadow_quant_mechanism_audit_adam_aware_pair_candidate_20260929/`
- Full-model REAL-Q control (preview off): `outputs/shadow_quant_mechanism_audit_full28_control_retry_20260929/`
- Full-model Adam-aware candidate: `outputs/shadow_quant_mechanism_audit_full28_adamaware_candidate_20260929/`
- Instrumentation: `tools/run_shadow_quant_mechanism_audit.py`,
  `tools/summarize_shadow_quant_mechanism_audit.py`
