# REAL-Q formal schedule A/B and timing results

## Status

The formal reverse-cosine A/B campaign completed on 2026-07-24. All five
metric artifacts passed the main validator. An independent validator then
passed 782 checks with zero failures and re-hashed the eight unique current
Qwen3-8B cache artifacts (about 9.4 GB) byte for byte.

This is a controlled **paper-gap experiment**, not an exact reproduction of
the historical paper tables. The complete fixed protocol and the reasons for
that evidence boundary are in
[`REALQ_SCHEDULE_ABLATION_MANIFEST.md`](REALQ_SCHEDULE_ABLATION_MANIFEST.md).

## Executive conclusions

1. The reverse-cosine change was not an index shifted by one place. The two
   compared non-final schedules use the same layer index and denominator but
   different curves:

   ```text
   current literal sine:  r + (1-r) sin(pi*x/2)
   historical cos²:       r + (1-r) sin²(pi*x/2)
   x = layer_index / (L-1), r = 0.01
   ```

2. On Qwen3-4B W4A16, the difference was very large in the primary metric.
   Literal sine produced 131.86% more KL than historical cos-squared in this
   locked protocol. PPL increased by 3.44%. Historical cos-squared was still
   75.71% above the paper KL anchor, so reverting the formula alone does not
   close the paper gap.

3. On Qwen3-8B W2A4KV4 aware, the ordering was the opposite:
   constant-aware was best, literal sine was second, and historical
   cos-squared was worst. Constant-aware reduced KL by 5.54% and PPL by 5.36%
   relative to historical cos-squared. This supports disabling the layer
   schedule for a real low-bit aware A/K/V path.

4. The Q4 schedule result and the Q8 aware result are not contradictory. Q4
   isolates curve shape in an A16/K16/V16 unaware path. Q8 additionally
   isolates whether an aware path should be scheduled at all.

5. These percentages are single-trajectory effect sizes, not statistical
   significance. Four ranks jointly form one all-reduced trajectory; they are
   not four independent repeats. The installed attention backward kernel also
   warns that it is nondeterministic.

## Frozen evidence identity

| Field | Value |
|---|---|
| Source commit | `2702d78836cc1a3eb9f8ab5f9c7a0b2b9c26ff14` |
| Tracked diff SHA256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Source status | only user-owned untracked `main.tex` |
| Formal wrapper SHA256 | `0a7c70f9fe9dcd34a816c75ca13025aacf6b94df906deec2b29795e1ae88e0b3` |
| Canoe job | `j-7x9o0je4pk` |
| Canoe name | `zhangqian_debugging_0724_1301` |
| Queue / cluster | `minimax-avatar-h800new` / `pa-cne02-prod-01` |
| Node | 8 x NVIDIA L20C, 183,359 MiB each |
| Runtime | Python 3.12.3, PyTorch `2.6.0a0+ecf3bae40a.nv25.01`, CUDA 12.8, Transformers 4.56.2 |
| Formal artifact root | `/minimax-avatar-new/zhangqian/realq/experiment_data/schedule_ablation_localclip_formal_20260724` |

Every metric JSON records identical source state at the start and end of its
run. Within each model case, configs are identical after removing only
`exp` and `output_dir`. Dataset identity, sampled calibration tokens, static
Fisher/saliency cache, reference logits, warm manifest, model identity, and
post-rotation model state are identical across arms.

## Qwen3-4B W4A16

The case uses 2,048 calibration samples of length 2,048, symmetric per-row
W4, A/K/V16 unaware, QuaRot, P99 saliency clipping, historical
`local_backward_chunk` P95 activation-loss clipping, loss slide enabled,
Adam refresh batches of 32, non-final base LR `2e-4`, and final-block LR
`1e-5`.

### Metrics

| Arm | Schedule in non-final blocks | Raw KL | Paper units (`KL x 100`) | PPL |
|---|---|---:|---:|---:|
| Literal sine (`paper`) | `sin(pi*x/2)` | 0.221625000238 | 22.1625000238 | 14.1338205338 |
| Historical (`legacy_cos2_scheduled`) | `sin²(pi*x/2)` | 0.095586717129 | 9.55867171288 | 13.6634597778 |
| Paper table anchor | historical artifact unknown | 0.0544 | 5.44 | 13.44 |

The label `paper` means the current behavior closest to the literal schedule
equation, not proof that the historical paper run used this exact code or
environment.

### Controlled differences

| Comparison | Metric | Absolute difference | Directed relative difference | Symmetric relative difference |
|---|---|---:|---:|---:|
| literal sine minus historical cos² | KL | +0.126038283110 | +131.8575% | 79.4663% |
| literal sine minus historical cos² | PPL | +0.470360755920 | +3.4425% | 3.3842% |
| literal sine minus paper anchor | KL | +0.167225000238 | +307.3989% | 121.1666% |
| literal sine minus paper anchor | PPL | +0.693820533752 | +5.1624% | 5.0325% |
| historical cos² minus paper anchor | KL | +0.041186717129 | +75.7109% | 54.9205% |
| historical cos² minus paper anchor | PPL | +0.223459777832 | +1.6626% | 1.6489% |

By the thresholds fixed before observing the metrics, the 131.86% primary-KL
effect is very significant in amplitude. It does not establish statistical
significance because this campaign contains one trajectory per arm.

## Qwen3-8B W2A4KV4 aware

The case uses 256 calibration samples of length 2,048, symmetric W2 group-128,
per-token A/K/V4 with separate clip ratios `0.9`, and both A/V-aware and
K-aware weight quantization. The reported non-final LR is `5e-6`; the final
block uses KL with LR `1e-6`.

Let:

- `C` be current constant-aware behavior;
- `S` be literal sine while deliberately retaining scheduling in aware mode;
- `L` be historical cos-squared while deliberately retaining scheduling.

### Metrics

| Arm | Non-final behavior | Raw KL | Paper units (`KL x 100`) | PPL |
|---|---|---:|---:|---:|
| `C`: constant-aware (`paper`) | constant `5e-6` | 0.919793128967 | 91.9793128967 | 20.1894054413 |
| `S`: literal sine | scheduled `sin(pi*x/2)` | 0.944964349270 | 94.4964349270 | 20.7609272003 |
| `L`: historical cos² | scheduled `sin²(pi*x/2)` | 0.973785102367 | 97.3785102367 | 21.3331642151 |
| Paper table anchor | historical artifact unknown | 0.767 | 76.7 | 19.68 |

### Controlled differences

| Isolation | Metric | Absolute difference | Directed relative difference | Symmetric relative difference |
|---|---|---:|---:|---:|
| `S - L`: curve formula only | KL | -0.028820753098 | -2.9597% | 3.0041% |
| `S - L`: curve formula only | PPL | -0.572237014771 | -2.6824% | 2.7188% |
| `C - S`: disable scheduling in aware mode | KL | -0.025171220303 | -2.6637% | 2.6997% |
| `C - S`: disable scheduling in aware mode | PPL | -0.571521759033 | -2.7529% | 2.7913% |
| `C - L`: complete current-vs-historical effect | KL | -0.053991973400 | -5.5445% | 5.7026% |
| `C - L`: complete current-vs-historical effect | PPL | -1.143758773804 | -5.3614% | 5.5091% |

`C - L` falls in the pre-registered significant amplitude band for KL.
Constant-aware is directionally best on both metrics.

### Remaining paper gap

| Arm | KL gap from 0.767 | Relative KL gap | PPL gap from 19.68 | Relative PPL gap |
|---|---:|---:|---:|---:|
| Constant-aware | +0.152793128967 | +19.9209% | +0.509405441284 | +2.5884% |
| Literal sine | +0.177964349270 | +23.2027% | +1.080927200317 | +5.4925% |
| Historical cos² | +0.206785102367 | +26.9602% | +1.653164215088 | +8.4002% |

The best current arm still does not reproduce the paper number. Schedule
semantics explain part of the variation among current runs, but not the
remaining historical gap.

## Validation result

The main and independent validators checked:

- metric schema 4, case and variant identity;
- source commit, tracked-diff SHA, wrapper SHA, and start/end immutability;
- exact config equality except the declared arm output identity;
- exact equality of dataset, token, static, reference-logit, warm, rotation,
  and model projections across arms;
- cache-hit requirements and start/end cache immutability;
- 36 unique schedule calls, layers 0 through 35 exactly once;
- the literal sine, historical cos-squared, and constant-aware scalar formula
  at every applicable layer;
- aware state in layers 0 through 34 and the non-aware final KL block with LR
  `1e-6`;
- finite metrics, exact `kl_x100 = 100 * kl_raw`, and all-rank consistency;
- absence of reservation locks and successful launcher exit.

The independent validator passed 782 checks and observed rank spread zero for
all five formal artifacts. It recomputed the SHA256 of the four Q8 static
shards, sampled-token file, reference-logit cache, warm manifest, and rotation
fingerprint; all matched their manifests.

### Artifact hashes

| Artifact | SHA256 |
|---|---|
| Q4 warm manifest | `156a21f9eeb98ee61358f40cd46b73517e175c7dd63b26755fe6e643081b2dc6` |
| Q4 rotation fingerprint file | `b8580e0af1ef60b2fd74ef9d14b97fd06618024f7063edf6f5a5dd1b3edade97` |
| Q4 post-rotation state | `d04968565fb80f3758405ec6a0171d2d4d18f2b1623afbbbc6c2dba8c55337f8` |
| Q4 literal-sine metric JSON | `c54afc15e8b084e6bec7bb1193c92695d62c615c4b7a3a2979678dab0d03406d` |
| Q4 historical metric JSON | `6b78b9197023db60d7d654846399bc0673e017efd60ec94d16dcfdf09f54d567` |
| Q8 warm manifest | `6795d17382a489c99f81be6f4df6e218ec5ad9afa03cc8db24d2346d2a4ad3e3` |
| Q8 rotation fingerprint file | `f556b188b0aed0acc6d889ef8262b68081bde571ed55215f9934f33fc6d77a81` |
| Q8 post-rotation state | `e993c4af07f2bc144478013964b7b9bc7f6a5a67e0120e21c6d164932efef313` |
| Q8 constant-aware metric JSON | `b11260c5f2fc1b2285f45d792ce009b6c9848d0a7200d44488f972cb9b4127e8` |
| Q8 literal-sine metric JSON | `0ca520c05cc1f12884da3e169bdaad67eee7cf1aab85d584fb9be457325edf94` |
| Q8 historical metric JSON | `1e41238e79913c2beee66ccd622f58155099384e9ee579000aa3565c46aa2539` |

## Timing comparison with the paper

The paper reports Stage 0 plus Stage 1 W4A16 wall time on four NVIDIA RTX Pro
6000 96-GB GPUs. For Qwen3-4B it reports 1.2 hours, or 72 minutes and 4.8
GPU-hours.

The closest comparable scope here is the measured Stage-0 precompute core plus
the exact Stage-1 quantization progress range:

| Q4 arm | Stage-0 core | Stage-1 | Stage-0 + Stage-1 | Versus 72 min | Four-GPU compute |
|---|---:|---:|---:|---:|---:|
| Literal sine | 2:06 | 38:37 | 40:43 | 1.768x; 43.45% less wall time | 2.714 GPU-h |
| Historical cos² | 2:06 | 38:59 | 41:05 | 1.753x; 42.94% less wall time | 2.739 GPU-h |

Broader observed scopes give a more conservative range:

| Q4 arm | Approx. launch-to-metric | Approx. launch-to-complete artifact | Speedup versus 72 min at artifact scope |
|---|---:|---:|---:|
| Literal sine | 45:30 | 46:17.8 | 1.555x |
| Historical cos² | 45:45 | 46:33.7 | 1.546x |

The defensible summary is about **1.6x to 1.8x faster**, or roughly **36% to
43% less wall time**, depending on scope. This is not a code-only speedup:
GPU generation/capacity, CUDA/PyTorch/Transformers versions, filesystem,
batching, and exact historical runner behavior differ. The two Q4 formula
arms differed by only 22 seconds in Stage 1 (0.94%), so schedule formula is
not the runtime cause.

The paper's Qwen3-8B timing row is W4A16 at 2.2 hours. The current Q8 formal
case is W2A4KV4 aware and is therefore not comparable to that timing row.
For reference only, Stage 0 was 23 seconds and Stage 1 was:

| Q8 arm | Stage-1 wall | Execution note |
|---|---:|---|
| Constant-aware | 57:37 | isolated arm |
| Literal sine | 57:13 | concurrent with historical arm |
| Historical cos² | 56:42 | concurrent with literal-sine arm |

Concurrent Q8 timings are diagnostic and must not be used as speedup evidence.

## Memory telemetry

Five-second telemetry for the two concurrent Q8 counterarms observed the same
per-rank maxima: 124,488 / 89,954 / 89,954 / 124,490 MiB. The isolated
constant-aware arm observed 124,488 / 89,954 / 167,016 / 124,490 MiB, with
the maximum occurring in the final KL block. The 167,016-MiB peak is 91.09%
of the 183,359-MiB device capacity.

These are sampled device-memory values, not allocator-exact peaks. The
performance campaign records both sampled device memory and PyTorch
allocated/reserved peaks.

## Limitations

- The formal primary A/B evaluates WikiText-2 KL/PPL, not the ten-task
  zero-shot suite. The task names and lifecycle were audited separately.
- Historical model/tokenizer/dataset software revisions and all hidden
  paper-run hyperparameters are unavailable.
- `local_backward_chunk` preserves paper-era activation-loss clipping but its
  percentile population was not specified in the paper.
- Current loss-slide behavior is intentionally retained even though it does
  not match the paper pseudocode exactly.
- Attention backward warns of nondeterminism. No small arm difference should
  be treated as causal without repeat trajectories.
- Hardware and software differences prevent attributing the measured timing
  gap solely to the refactor.
