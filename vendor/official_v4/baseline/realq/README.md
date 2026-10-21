# REAL-Q

REAL-Q augments saliency-guided, GPTQ-style quantization with dynamic
block-wise gradient descent. After each column block's analytical error
compensation, an Adam update corrects the remaining unquantized weights.
The refresh objective uses the full aggregated Fisher matrix, optionally
blends two adjacent Transformer blocks through a loss sliding window, and
uses true KL divergence for the final Transformer block.

This repository contains the refactored implementation, including its
shared utilities and the existing Qwen3 MoE dispatch path. All project-local
runtime imports are under `realq`; the former outer `gptq_plus` checkout is
not required.

## Install

Use Python 3.10 or newer in a virtual environment. The validated runtime is
PyTorch 2.9.1 with CUDA 12.8, Transformers 4.56.2 and lm-eval 0.4.4.

```bash
git clone https://github.com/QianerPKU/REAL-Q.git
cd REAL-Q
python -m venv .venv
source .venv/bin/activate
python -m pip install setuptools wheel packaging ninja
python -m pip install -r requirements-core.txt
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m realq.ptq --help
```

`requirements.txt` includes GPU acceleration and reasoning-evaluation
dependencies. For the SDPA path with the bundled PyTorch Hadamard fallback,
the core requirements and package installation are sufficient. See
[dependency details](docs/DEPENDENCIES.md) for CUDA build prerequisites and
the pinned LiveCodeBench source checkout used by its official evaluator.

When working inside the original checkout, activate its existing `.venv`
and install with `python -m pip install --no-deps --no-build-isolation -e realq`.

## Run

```bash
python -m realq.ptq --model /path/to/model --help
# Supply a model/setting-specific learning rate for actual quantization.
GRAD_LR=1e-5 NSAMPLES=256 W_GROUPSIZE=128 \
  bash scripts/run.sh /path/to/model 4 0
```

The launcher is a convenience template. Its defaults are not a complete
paper-table reproduction recipe; learning rates and calibration settings
depend on the model and quantization setting. The historical auto-tuning
launcher is retained, but the current experiment workflow uses supervised,
staged LR selection rather than unattended full-grid launches.

## Code map

- `ptq.py`, `pipeline.py`: CLI and pipeline orchestration.
- `precompute/`: saliency and aggregated Fisher statistics.
- `quant/`: Hessian construction, column quantization and compensation.
- `refresh/`: Block-GD, Fisher/KL losses and optimizer kernels.
- `runner/`: layer traversal, calibration streams and refresh scheduling.
- `utils/`: bundled model/data/rotation/quantization helpers; `utils/moe/`
  contains the existing sparse-model implementation.
- `benchmarks/`: reasoning generation and evaluation.

The directory itself is mapped to the Python package `realq` during
installation, so a clone named `REAL-Q` still supports `python -m realq.ptq`.
