# Sequence Fisher hybrid: fixed small-scale protocol

The claim under test is **repeatable KL improvement specifically from retaining
cross-token terms** in the block-output quadratic loss. This is an experimental
engineering change to REAL-Q's Adam refresh loss, not an established solution to
nonlinear Taylor truncation error. Existing closed-form compensation, saliency,
weight order, Adam settings, and calibration data are held fixed.

## Definition

For calibration sequence n, FP block-output score g[n,k,t,h] is the gradient of
SUM-reduced sampled-label NLL. Labels are independently drawn from teacher
probabilities at every output position. Scores are divided by the existing
precompute gradient scale (1000), detached, and stored in BF16. The original
draw zero alone still builds REAL-Q's saliency and aggregate Fisher; additional
draws have their statistic hooks disabled. Draws use private deterministic
generators, not the optimizer/calibration random stream.

With delta equal to current block output minus the FP output for that sample:

    Lsequence = sum_nk (sum_th g[n,k,t,h] delta[n,t,h])^2 / (2 N K T)
    Ltoken    = sum_nkt (sum_h g[n,k,t,h] delta[n,t,h])^2 / (2 N K T)
    Lhybrid   = (1 - mix) Laggregate + mix Lsequence

The token control replaces Lsequence with Ltoken, using identical score draws,
sample indexing, normalization, and mix. Thus its difference from the sequence
arm is precisely the t != s cross terms. Different calibration sequences are
never summed together before squaring. Finite K has sampling variance, and
BF16 storage adds rounding; this is not the full exact Hessian.

Only gate_proj/up_proj/down_proj refreshes in nonfinal Transformer blocks use
the hybrid. The forward pass remains the real Transformer block. It already
contains SwiGLU and attention nonlinearities. The new term changes the
approximation to downstream feedback, not that forward Jacobian.

## Preregistered pilot

- Qwen3-0.6B, W4, groups=4, blocksize=128, act_order.
- Only Transformer blocks 0--4 quantized; later blocks stay FP.
- Calibration: WikiText2, 32 sequences, length 256; seeds 42 and 44.
- Evaluation: same eight held-out WikiText2 windows, length 1024, offset 0.
- K=4 and mix=0.5 fixed before observing results; no parameter search.
- Three arms: REAL-Q aggregate; aggregate + token; aggregate + sequence.
- Baseline has cross compensation, two-sided rounding, nonlinear MSE, and
  quantization preview disabled.
- Primary metric: full-vocabulary KL. PPL is supplementary. Record elapsed
  quantization time, static precompute time, peak CUDA allocated/reserved memory,
  state fingerprints, and outputs of the first four calibration sequences after
  each quantized block.

Before the pilot: exact toy KL-Hessian audit, dense/projection gradient audit,
cache indexing/scale checks, and a one-block real-model no-op pair (4 samples,
mix=0). No-op must preserve original statistics, model hash, KL and PPL exactly.
All comparison arms use frozen source fingerprints. Original saliency/Fisher
fingerprints must match within each calibration seed. Token/sequence sketches
must have matching cache fingerprints.

## Interpretation

Sequence must improve over REAL-Q across seeds to justify continuation; it must
also beat the matched token control to support an interaction-specific claim.
Two calibration seeds and eight shared evaluation windows are a screening test,
not statistical proof or evidence of full-W4 generalization. A failure or a
mixed result is reported as such, not reclassified after tuning parameters.

## First pilot results (2026-09-29)

The 4-sample, one-block, mix-zero run passed: original saliency and aggregate
Fisher hashes matched; final model SHA256, KL (0.021650202572345734), and PPL
(19.724040985107422) were bitwise identical between mode off and sequence
mode/mix zero.

All six five-block runs then finished with identical source hashes and intact
saliency/Fisher hashes per calibration seed. Candidate sketch cache hashes also
matched between token and sequence modes. No CUDA OOM or nonfinite scores were
reported. Peak allocated CUDA memory was 2,559,177,728 bytes in every arm. The
observed W4 prefix quantization time ranged from 156.6 to 192.1 s; this is a
small-prefix measurement and machine/run-order variation is substantial.

| Calibration seed | REAL-Q KL | Token-only KL | Sequence KL | Sequence vs REAL-Q | Sequence vs token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.09844537 | 0.10151564 | 0.10079902 | +2.39% | -0.71% |
| 44 | 0.10330717 | 0.10370904 | 0.10216203 | -1.11% | -1.49% |

Thus the sequence arm beat its matched token-only ablation on KL in both
seeds, but did **not** repeat an improvement over REAL-Q. PPL also disagreed
with KL in seed 42: sequence PPL was 1.04% higher than REAL-Q, while in seed
44 it was 0.94% lower. The pilot does not establish a working improvement.

The layer traces localize the largest effects to blocks 2--4. Their output MSE
is dominated by the first token in both seeds (about 7,500--8,600 versus under
0.01 for the remaining-token average); the REAL-Q refresh losses show the same
jump. This supports a real model behavior rather than a logging shape error,
but it also makes these four calibration sequences a noisy/sensitive screen.
Later-layer outputs use each run's naturally accumulated quantized prefix, so
their difference includes upstream changes as it does in the actual pipeline.
Per-layer outputs and token error summaries are in `audited_summary.json`.

The high FP reference value was checked independently: a separate standard HF
Qwen3 forward reproduced every saved layer-2 FP target exactly in BF16 for the
four inspected samples from both seeds (max absolute difference 0). Therefore
the large first-token reference activation is not a trace-capture or layer
indexing bug. See `fp_reference_audit.json`. The saved four samples all show
the position-0 spike; this audit does not prove why the model/input produces it
or characterize all 32 calibration sequences.

Observed quantization time changed by +4.0% and +14.8% for sequence vs
REAL-Q in seeds 42 and 44; the latter likely includes run-order/thermal noise.
Peak *allocated* CUDA memory showed no increase at this scale; peak *reserved*
was about 40 MiB higher for sequence. Static precompute took 19.4/20.2 s for
sequence vs 12.0/15.0 s for REAL-Q in these two runs. A proper performance claim
needs interleaved replication and a larger/full quantization run.

## Follow-up replication (started 2026-09-29)

The same three arms are repeated without changing K, mix, data size, or
quantization settings, on calibration seeds 46, 48, and 50. Evaluation remains
the same eight held-out windows. The runner records window-wise KL as well as
aggregate KL/PPL; it does not alter the evaluator's loss. Source/statistic/cache
integrity checks repeat. This is replication, not a parameter search. If the
sequence arm does not improve aggregate KL over REAL-Q and its matched token
arm on at least four of five seeds, stop this design rather than tune it against
these results.

The 3-seed replication completed. Across all five calibration seeds, the
sequence hybrid reduced KL versus REAL-Q in 2/5 seeds; its mean paired KL
change was +1.68% (higher is worse). It beat the matched token arm in only
2/5 seeds. The token hybrid reduced KL versus REAL-Q in 2/5 seeds and had a
mean change of -0.09%, effectively neutral at this scale. The hybrid sequence
proposal therefore failed its preregistered continuation criterion.

Per-window KL does not reveal a single evaluation window explaining the result:
the sequence arm worsened 7/8 windows in seed 46, 7/8 in seed 48, and improved
5/8 in seed 50, with large opposing window changes. Seed 50's aggregate
sequence KL was only 0.38% lower than REAL-Q while it was 1.53% higher than the
token arm. Exact window arrays are retained in each run folder.

Because the hybrid replaced half of the aggregate loss with a noisy sequence
estimate, the next distinct candidate keeps the aggregate Fisher at full
strength and adds only the matched cross-token correction:

    Lcross = Laggregate + (Lsequence - Ltoken)

The difference cancels all same-token terms and retains t != s terms from the
same score samples. The coefficient is fixed at one because the estimator is
added as its unscaled cross contribution, not fit as another loss mix. This is
a new hypothesis, not a result; it is being screened on seeds 46, 48, and 50
against same-source REAL-Q controls, with the same eight evaluation windows.

### Cross-only screen outcome

The three paired runs completed, and all source, cached Fisher/saliency, probe
cache, and completion checks passed. The cross addition worsened KL on all
three calibration seeds; its paired average was about +3.25%. It also worsened
PPL in each seed. The 8-window KL improved on only 1/8, 2/8, and 3/8 windows.
Quantization time increased by 7.9%, 14.9%, and 11.3%, while peak allocated
memory was unchanged. Per-layer target-output comparisons are in
`outputs/sequence_fisher_cross_add_20260929/audited_summary.json`.

This candidate should be stopped. The probe labels are independently sampled
from each token's fixed teacher categorical distribution. Conditioned on the
input and current model Jacobian, each token score has zero mean and independent
draws, so the expected t != s score outer products are zero. With only four
probes, `Lsequence - Ltoken` therefore adds a noisy finite-sample term instead
of a stable omitted curvature component. The observed all-seed degradation is
consistent with that mechanism; coefficient tuning would not establish a
better estimator. The screen does not show that sequence interactions in a
different sampling or loss construction are universally useless.

Reproduce the screen and audit with:

    /home/uky/gptq_plus/.venv/bin/python -u tools/run_sequence_fisher_pilot.py --stage cross_test --output outputs/sequence_fisher_cross_add_20260929
    /home/uky/gptq_plus/.venv/bin/python tools/summarize_sequence_fisher_pilot.py outputs/sequence_fisher_cross_add_20260929

## Reproduction

Run in WSL from the repository root using the existing virtual environment:

    /home/uky/gptq_plus/.venv/bin/python tools/test_sequence_fisher.py
    /home/uky/gptq_plus/.venv/bin/python -u tools/run_sequence_fisher_pilot.py --stage noop --output outputs/sequence_fisher_noop_20260929
    /home/uky/gptq_plus/.venv/bin/python -u tools/run_sequence_fisher_pilot.py --stage pilot --output outputs/sequence_fisher_pilot_20260929

Output directories must be new to prevent overwriting experimental records.
