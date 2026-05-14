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

import torch  # noqa: E402  (env var must come first)

from realq import config as cfg_mod
from realq import pipeline
from realq.parallel import env as parallel_env
from realq.utils import log as log_utils
from realq.utils import nvtx as nvtx_utils

torch.backends.cuda.matmul.allow_tf32 = False


def main() -> None:
    cfg = cfg_mod.parse_cli()
    nvtx_utils.set_enabled(cfg.nsys_profile)
    log_utils.init(cfg.output_dir, cfg.exp)
    parallel_env.init()
    pipeline.run(cfg)


if __name__ == "__main__":
    main()
