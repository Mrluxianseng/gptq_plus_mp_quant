# Exact MSE clip-search profile (2026-09-29)

## Change under test

The pre-clip stage calls `WeightQuantizer.find_params` with symmetric Q4, MSE clipping, FP32 weights, `grid=50`, and `maxshrink=0.5`. The legacy search evaluates the Cartesian pair of 25 negative and 25 positive shrink candidates (625 pairs per row). The existing `symmetric_union_exact` backend uses the fact that symmetric quantization depends on the larger absolute endpoint, reducing this to at most 50 endpoint candidates. It dispatches only for finite symmetric FP32 weights and retains the Cartesian fallback outside that proven domain.

Two independent opt-ins now use this backend: `--pre_clip_search_impl` for manual pre-clipping and `--quantizer_search_impl` for GPTQ's initial quantizer calibration (`self.quantizer.find_params(W)`). Both default to `cartesian_legacy`; all previous commands preserve their behavior. Neither changes GPTQ's sequential per-column error compensation or Adam refresh.

## Correctness evidence

- Direct comparison on actual Qwen3-0.6B layer-0 `q_proj`, `k_proj`, `gate_proj`, `up_proj`, and `down_proj` weights found bit-identical clipping scales, zero-points, and clipped weights for both backends.
- Three five-layer end-to-end candidates (two using the same exact backend through a profile-only hook, one through the new CLI path) each produced the same 507-tensor checkpoint hash: `fe2945201ea303dc34a962d14b86d508bb98e5a30d75c9c573a02c3cef4ecbda`.
- All candidates reported exact-prefix KL `0.10330717265605927` and PPL `30.991367340087891`, matching the legacy baseline profile runs.
- Source hashes were checked before and after each profiled run; no source file changed during execution.
- The legacy five-layer profile spent 21.99 s in GPTQ's initial `quantizer_find_params`; with both exact-union switches enabled, these calls fell below the profile's top-300 reporting cutoff in most modules. The combined option's effect is also reflected in the end-to-end pair below.

## Timing and memory

| Run | Pre-clip backend | Quantization time | Peak allocated | Peak reserved |
|---|---|---:|---:|---:|
| legacy profile 1 | Cartesian | 222.60 s | 2,559,177,728 B | 2,615,148,544 B |
| legacy profile 2 | Cartesian | 225.11 s | 2,559,177,728 B | 2,615,148,544 B |
| candidate profile 1 | exact union | 188.50 s | 2,559,177,728 B | 2,615,148,544 B |
| candidate profile 2 | exact union | 199.41 s | 2,559,177,728 B | 2,615,148,544 B |
| candidate wired CLI | exact union | 197.58 s | 2,559,177,728 B | 2,615,148,544 B |
| combined exact-search run 1 | pre-clip + GPTQ init | 176.15 s | 2,559,177,728 B | 2,615,148,544 B |
| combined exact-search run 2 | pre-clip + GPTQ init | 170.43 s | 2,559,177,728 B | 2,615,148,544 B |

The pre-clip-only wired run is 12.2% faster than the detailed legacy profile. Enabling both exact searches gives a mean of 173.3 s for two runs, 22.6% below the two-run legacy mean of 223.9 s. The larger combined gain is plausible because it accelerates both the pre-clip search and GPTQ's separate initial quantizer search; full-model replications are still needed before claiming a stable full-run percentage.

The CUDA-event profile attributes about 40.7 s across the five legacy layers to `layer.weight_preclip`; the wired exact-union run spends about 1.17 s there. Legacy GPTQ initial quantizer calibration adds about 22.0 s. Thus the measured search reductions are large, while peak GPU allocation does not change. This is a runtime optimization, not a memory-footprint reduction.

## Artifacts and limits

- Legacy profiles: `outputs/phase_profile_qwen3-06b-seed44-layers0to4-retry1-20260929/`, `outputs/phase_profile_qwen3-06b-seed44-layers0to4-detail-20260929/`.
- Candidate profiles: `outputs/phase_profile_qwen3-06b-seed44-layers0to4-unionclip-20260929/`, `outputs/phase_profile_qwen3-06b-seed44-layers0to4-unionclip-retry2-20260929/`, `outputs/phase_profile_qwen3-06b-seed44-layers0to4-unionclip-wired/`, `outputs/phase_profile_qwen3-06b-seed44-layers0to4-both-union/`, `outputs/phase_profile_qwen3-06b-seed44-layers0to4-both-union-retry2/`.
- All runs used the same REAL-Q seed-44 Qwen3-0.6B command through layer 4, 32 calibration examples of length 256, W4, act-order, block size 128, and the same evaluation settings.
- This establishes correctness and useful short-run acceleration for the profiled configuration. It does not establish end-to-end full-28-layer speedup or benefit on asymmetric, nonfinite, grouped, or different-dtype clipping inputs; those domains keep the legacy fallback or need separate measurement.
- `py_compile` passed for the edited Python files. The project's focused pytest file could not be launched because the WSL environment does not have `pytest` installed.
