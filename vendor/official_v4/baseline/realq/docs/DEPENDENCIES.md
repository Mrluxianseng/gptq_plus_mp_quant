# Dependency consolidation (2026-09-05)

The publication preserves the current working-tree changes and copies the
runtime import closure into this package. Source helpers come from the
parent `gptq_plus` checkout; their logic is preserved while imports change.

| Former source | Bundled location |
| --- | --- |
| `utils/*.py` reachable from REAL-Q | `realq.utils.*` |
| `utils/had_mat.pt` | `realq.utils` package data, byte-for-byte copy |
| `gptq_utils/quant_aware_utils.py` | `realq.utils.quant_aware_utils` |
| Reachable `realq_moe` modules | `realq.utils.moe` |
| `tools/realq_auto_tune.py` | `realq.utils.auto_tune` |

The outer helpers remain in place for legacy experiments. No algorithm
parameters, gradient updates or numerical kernels are changed by the import
migration. The MoE copy preserves the existing optional dispatch; this
publication does not claim a new GPU validation of that path.

## Python dependencies

`requirements-core.txt` lists the core runtime and lm-eval.
`requirements.txt` adds FlashAttention-4, its pinned quack dependency,
the pinned fast-Hadamard-transform source revision, build tools and the
reasoning evaluator requirements from `benchmarks/requirements.txt`.
PyTorch's distribution supplies its matching Triton dependency on Linux.
The bundled Hadamard implementation also has a PyTorch fallback.

The complete dependency file lists directly required packages and key
validated versions; it is not a lockfile of every transitive dependency.
CUDA drivers/toolkit, a C++ compiler, Git and optionally Nsight Systems
(`nsys` for profiling) are system tools, not pip packages. Install the core
requirements first so CUDA extension builds can import the selected PyTorch.

## LiveCodeBench source dependency

Official LiveCodeBench code execution additionally uses this pinned source
checkout. Its upstream packaging includes only the top-level `lcb_runner`
package in the wheel and omits the evaluator subpackages, so installing its
wheel is insufficient. REAL-Q's generation/export path does not need it.
The Python runtime dependencies for the evaluator are included in the
requirements files; upstream model-serving backends are not used.

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git \
  datasets/reasoning_eval/vendor/LiveCodeBench
git -C datasets/reasoning_eval/vendor/LiveCodeBench \
  checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
```

Pass that checkout via `--lcb-source` to
`python -m realq.benchmarks.official_eval`, or via
`--reasoning_lcb_source_dir` in the quantization CLI. Keep official code
execution inside the isolated environment required by the existing scorer.

## Packaging and validation scope

The repository root installs as `realq`; subpackages retain their existing
names and `utils/had_mat.pt` is included in wheels and source distributions.
Validation for this migration covers package builds, isolated imports,
entry-point help and small CPU checks. No full model quantization or GPU
benchmark is part of this publication task.

The publication checks passed: wheel build and isolated installation,
static import checks across 106 packaged Python files with outer project
imports explicitly blocked, dense and MoE pipeline imports, all three
CLI help paths (quantization, data preparation and the historical tuner),
Hadamard matrix loading/orthogonality, a tiny CPU column-block quantization
with a refresh callback, and the Fisher loss value and gradient. The 46
original Python sources were compared with the initial working-tree
snapshot and differ only in their project-local import paths.
