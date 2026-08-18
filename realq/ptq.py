"""RealQ command-line entry point.

Usage:
    torchrun --nproc_per_node=4 -m realq.ptq --model models/Qwen/Qwen3-0.6B
    python -m realq.ptq --model models/Qwen/Qwen3-0.6B  # single-GPU

The entry stays small: parse → init logging → init distributed → pipeline.run.
"""
from __future__ import annotations

import os

# datasets needs this for some loaders; same setting as the old ptq.py.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
# Must be set before torch/CUDA is imported or creates a cuBLAS context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402  (env var must come first)
from utils.reproducibility import configure_deterministic_sdpa  # noqa: E402

# LR-search workers can request a fully deterministic SDPA path.  The shared
# reproducibility helper uses ``warn_only=True`` because general experiments
# may prefer the faster memory-efficient kernel, but an optimizer comparing
# very close KL values must not rank candidates using atomic-backward noise.
if os.environ.get("REALQ_DETERMINISTIC_SDPA") == "1":
    configure_deterministic_sdpa()

from realq import config as cfg_mod
from realq import pipeline
from realq.parallel import env as parallel_env
from realq.utils import log as log_utils
from realq.utils import nvtx as nvtx_utils
from utils.reproducibility import configure_reproducibility

torch.backends.cuda.matmul.allow_tf32 = False


def main() -> None:
    cfg = cfg_mod.parse_cli()
    nvtx_utils.set_enabled(cfg.nsys_profile)
    log_utils.init(cfg.output_dir, cfg.exp)
    configure_reproducibility(cfg.refresh_seed, deterministic=True)
    parallel_env.init()
    pipeline.run(cfg)


if __name__ == "__main__":
    main()
