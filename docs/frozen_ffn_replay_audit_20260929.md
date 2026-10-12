# Frozen-prefix replay audit — 2026-09-29

## Scope

Test whether reusing the frozen prefix of Qwen3's FFN during REAL-Q's repeated
weight-gradient refreshes can reduce end-to-end quantization time without
changing the quantized model. The model was Qwen3-0.6B, W4, WikiText2, seed 42,
32 calibration sequences of length 256, block size 128, and the existing
`fisher_diag_mse` refresh path. Each arm used the same captured source hashes,
calibration inputs, evaluation set, and command. Full held-out evaluation used
292 WikiText2 windows. This is an execution optimization test; it does not
change the loss or claim a KL improvement.

## All-FFN cache

`FrozenFFNPrefix` reuses the attention/residual and normalized MLP input for
gate/up refreshes, and reuses the gate-up product for down-projection refreshes.
The down cache is invalidated at the gate/up-to-down phase boundary. The cache
is capped at 128 MiB.

| Run | Cache mode | Final state vs REAL-Q control | KL / PPL | Quantization time |
| --- | --- | --- | --- | ---: |
| Original control | off | reference | 0.22264633 / 27.98901558 | 919.512 s |
| Initial candidate | on, no verify | different | 0.22346988 / 28.04050064 | 943.825 s |
| Full verification | on, verify every hit | exact | 0.22264633 / 27.98901558 | 938.363 s* |
| Repeat 1 | on, no verify | exact | 0.22264633 / 27.98901558 | 920.554 s |
| Repeat 2 | on, no verify | exact | 0.22264633 / 27.98901558 | 931.829 s |
| Control repeat | off | exact to original control | 0.22264633 / 27.98901558 | 926.182 s |
| Repeat 3 | on, no verify | exact | 0.22264633 / 27.98901558 | 930.965 s |

The 7,560 cache hits in the verification run each matched the ordinary full
block forward bit-for-bit. The 28 post-quantization block outputs also matched
the control outputs bit-for-bit. In no-verify mode, three of four complete
candidate runs produced exactly the control's full-model state hash; one run
did not. That outlier first showed a small loss difference at layer 3's
`down_proj` block 0, followed by output divergence from layer 3 onward. The
verification run's exact result confirms the cached forward computation, but
does not explain the one end-to-end state divergence in an unverified run. We
do not yet have enough evidence to claim stable full-FFN equivalence.

There was no repeatable speedup. The no-verify candidate quantization times
were 920.6–943.8 s against control runs of 919.5–926.2 s. Full-process wall
times also showed no consistent reduction. The verified run is not a valid
speed benchmark because it executes an additional ordinary forward at every
cache hit. Peak cache storage was 107,020,288 bytes (about 102 MiB); peak CUDA
allocated memory was 5,913,079,296 bytes (about 5.51 GiB), within the 8-GiB
device budget.

**Decision:** keep the all-FFN replay path experimental and disabled by
default. Do not present it as a speed or quality contribution. Investigate the
unexplained no-verify outlier before relying on it in a quantization run.

## Down-projection-only cache

The first two complete candidate/control pairs had identical final model
states, KL, and PPL. Their quantization times fell by 2.15% and 1.25%; full
process times fell by 1.49% and 1.19%. An additional pair without an explicit
cuBLAS workspace setting diverged: candidate KL was 0.223377 versus 0.222646
for control, candidate PPL was 28.0192 versus 27.9890, and candidate quantization
was 3.66% slower. A follow-up pair with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` matched all 507 model tensors, KL, and PPL;
its reductions were only 0.43% for quantization and 0.29% end to end. Across the
three exact pairs, the mean reductions were 1.28% and 0.99%, respectively.
Thus three of four complete pairs matched exactly, while one showed a real discrepancy.
The full audit retained source hashes and per-arm outputs in
`outputs/frozen_prefix_timing_replication_20260929/extended_audit_20260929.json`.

To isolate the cache computation, a real-refresh diagnostic compared baseline
and replay at identical live weights, alternated execution order, required
bitwise equality of gradient tensors and exact equality of loss/count/extras,
then returned the baseline result to preserve the quantization trajectory.
All 621 comparisons passed with cuBLAS fixed and all 621 passed without it.
The paired refresh calls were 15.0% and 13.4% faster respectively, with an
86-MiB peak cache. This verifies the local gradient path on the actual REAL-Q
refresh workload; it does not explain the single full-run mismatch or establish
the same gain end to end.

This remains a modest engineering prototype, not a quality improvement or a
new mathematical method. Caching frozen activations has precedent in
fine-tuning systems such as [AutoFreeze](https://arxiv.org/abs/2102.01386); the
project-specific work is the replay boundary and invalidation behavior for
REAL-Q's module/column refresh loop. The full-run outlier means a stable
end-to-end speed claim is premature. Do not generalize to other models or
calibration sizes without additional paired runs.

## Research decision

Keep the down-only cache as a promising opt-in engineering prototype; explain
the one full-run discrepancy before calling it a repeatable end-to-end
optimization. Do not merge the full-FFN extension into a default path. There is
still no validated KL-improving method from this experiment. The next useful
step is to isolate the source of the full-run discrepancy and profile how much
time is spent in repeated block forwards versus closed-form quantization and
gradient updates. Any extension should preserve the full-model state and show
repeated paired timings before it is described as a contribution.

## Artifacts

- Full control/candidate: `outputs/frozen_ffn_replay_full_replication_20260929/`
- Full per-hit verification: `outputs/frozen_ffn_replay_full_verify_20260929/`
- Two no-verify repeats: `outputs/frozen_ffn_replay_full_repeat_20260929/`
- Additional control/candidate repeat: `outputs/frozen_ffn_replay_full_anchor_20260929/`
- Down-only timing pairs: `outputs/frozen_prefix_timing_replication_20260929/audited_summary.json`
- Extended down-only timing and gradient audit: `outputs/frozen_prefix_timing_replication_20260929/extended_audit_20260929.json`
- Real down-projection gradient checks, with and without fixed cuBLAS: `outputs/frozen_prefix_gradient_diag_20260929/run/paired_refresh_summary.json` and `outputs/frozen_prefix_gradient_diag_nocublas_20260929/run/paired_refresh_summary.json`
- Five-layer exact-forward smoke: `outputs/frozen_ffn_replay_smoke_20260929/audited_summary.json`
- Synthetic output/gradient checks: `tools/test_frozen_ffn_replay.py`
