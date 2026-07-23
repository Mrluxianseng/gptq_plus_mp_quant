"""Shared reproducibility controls for the legacy and refactored pipelines."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_reproducibility(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG used by REAL-Q and request deterministic CUDA kernels.

    ``CUBLAS_WORKSPACE_CONFIG`` must normally be present before CUDA context
    creation.  Both CLI entry points set it before importing torch; setting it
    here as well makes direct library/test use safe as long as CUDA has not
    already been initialized.

    ``warn_only=True`` is deliberate: third-party attention implementations
    occasionally lack a registered deterministic alternative.  Such a warning
    remains visible in the test log, while fixed-seed repeatability is enforced
    independently by the per-step loss comparator.
    """

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older supported torch releases do not expose warn_only.
            torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)

    logging.info(
        "[reproducibility] seed=%d deterministic=%s CUBLAS_WORKSPACE_CONFIG=%s",
        seed,
        deterministic,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
