# REAL-Q Adam refresh: corrected interpretation and next experiment

## Audit of the completed partial-reset screening

The current default is `full_block_refresh=False`. The prefix screening used the single-linear Fisher refresh path. Its partial resets zeroed an EMA buffer while keeping a shared old `adam_step` for both bias corrections.

- Resetting m alone gives m_hat = (1-beta1)*g/(1-beta1^t), not g. At t=16, beta1=0.9, its magnitude is about 0.123*g. Thus this intervention changes direction history AND update magnitude.
- Resetting v alone gives v_hat = (1-beta2)*g^2/(1-beta2^t). Relative to a freshly and consistently corrected v=g^2, it increases the update magnitude by sqrt((1-beta2^t)/(1-beta2)), ignoring epsilon. At t=16, beta2=0.999, this is approximately 3.99x.
- The completed results reject those concrete interventions on this small configuration. They do not establish that retaining all history is optimal, or that a correctly normalized partial restart cannot help.
- The default final-layer KL path in `realq/refresh/kl_loss.py` has separate Adam arithmetic and no corresponding history-policy hook. The completed layer-0-to-4 screen did not test it. Full-block paths also need policy/diagnostic counter consistency tests before use.

## Research question

Ordinary Adam is already a discrete-time optimizer. REAL-Q additionally interleaves rounding, GPTQ compensation, changes to active columns, and sometimes changes to the loss mixture/samples. The hypothesis to test is whether momentum becomes misaligned after those transitions, rather than whether discrete updates inherently invalidate EMA.

## Primary sources and applicability

- Cautious Optimizers: https://arxiv.org/abs/2411.16085 . Mask coordinates where the proposed descent direction conflicts with the current gradient; retain moments and use the paper's explicit normalization. First candidate if conflict diagnostics show material harmful update mass. Training results do not establish downstream-KL or post-GPTQ gains.
- Adam on Local Time (NeurIPS 2024): https://arxiv.org/abs/2412.17113 . Nonstationary gradient magnitudes and locally reset correction time. Its RL epoch boundaries differ from individual REAL-Q column boundaries; our always-reset-age arm is not a faithful transfer of its task schedule.
- SPAM (ICLR 2025): https://arxiv.org/html/2501.06842v2 . Spike-aware clipping, periodic momentum resets, and post-reset learning-rate warmup. The reported sensitivity to overly frequent resets and absent warmup makes a direct every-refresh transplant unsuitable. Learning-rate warmup is distinct from additional backward-only warmup passes.
- Scheduling Weight Transitions for Quantization-Aware Training (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Scheduling_Weight_Transitions_for_Quantization-Aware_Training_ICCV_2025_paper.html . Motives for measuring actual discrete transitions rather than assuming floating-point improvement survives rounding. QAT differs from GPTQ compensation; use as a measurement/design reference.

## Bounded next experiment

1. At a few real refresh snapshots across early, middle and late blocks, record current g, prior m/v, active-column mapping and the actual Adam subtraction vector d. Report cosine, conflict fraction, and conflict contribution sum(max(-g*d,0))/sum(abs(g*d)). Compare coordinates only in the shared still-floating suffix. Keep diagnostic samples fixed when attributing changes to GPTQ rather than sample changes.
2. Fork each identical snapshot into no-refresh, original Adam, and Cautious-Adam arms. Reuse the same backward result. Add an Adam-direction control scaled to the candidate's update norm, to distinguish direction effects from a change in effective step size. Use the published mask normalization and guard an all-masked vector.
3. Measure held-out true downstream KL immediately after the update and again after executing the actual next GPTQ block. Save the full required GPTQ/quantizer state so the replay is exact. Extra diagnostic downstream evaluations are measured separately and are not production backward overhead. PPL is secondary. A few actual KL-gradient probes can be added only if surrogate descent and true-KL outcomes disagree.
4. Only advance a candidate with repeatable post-GPTQ KL benefit to full-model, multiple calibration seeds and longer held-out evaluation, including the independent final-layer KL update path. Check disabled-candidate equivalence, frozen-prefix immutability, policy coverage, finite states, backward counts, peak memory and runtime.

A consistently corrected m-only restart, with independent m/v ages, is a secondary mechanism control. A coherent bias correction does not itself guarantee benefit under nonstationarity. Do not infer an engineering contribution from a restart switch or a learning-rate change alone.

## Focus after the sign-update discussion

Prioritize a direction-only diagnostic using Grams (https://arxiv.org/html/2412.17107v1): replace the actual Adam subtraction vector u by sign(g)*abs(u), keeping the same current gradient, moment transition, bias ages, learning rate, and active coordinates. On nonzero-gradient coordinates, the candidate preserves every coordinate's absolute update magnitude, which makes this a cleaner directional intervention than resetting m or using constant-magnitude signSGD. Treat zero-gradient coordinates explicitly and report their update mass.

The first snapshot arms should be original Adam, Grams-style direction replacement, and no weight refresh (a diagnostic anchor only). Cautious masking is a follow-up if complete direction reversal is harmful or selective suppression is motivated. These are borrowed methods, not newly invented update formulas. Transfer to REAL-Q is only a potential engineering contribution if integration and repeated post-quantization quality/resource results justify it.

Sample snapshots from early/middle/late depth, attention and FFN modules, and early/late refreshes within a module. Reuse an exact snapshot, including quantizer/GPTQ working buffers, for each branch. Restrict candidates to the identical still-floating suffix and use held-out evaluation samples. Score surrogate loss, downstream KL before/after update, and downstream KL after the true next GPTQ block. Do not choose a winner using the final test set or select only favorable layers/seeds. Extend evaluation depth and calibration context before drawing a general negative or positive conclusion from the previous tiny screen.
