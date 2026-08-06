"""Lightweight NVTX wrapper for nsys profiling.

Markers are emitted only when the global flag is enabled (set once at
process start from ``cfg.nsys_profile``). When disabled, ``nvtx_range``
returns a ``nullcontext``, so the call site is essentially free
(~100 ns) and adds no synchronization, no allocation, and no behavior
change. The wrapper is intentionally a single-file module so it can be
imported from anywhere in realq without circular-import concerns.

Usage:
    from realq_moe.utils import nvtx

    with nvtx.nvtx_range("precompute.batch_3"):
        ...
"""
from __future__ import annotations

from contextlib import nullcontext

import torch

_ENABLED = False


def set_enabled(flag: bool) -> None:
    global _ENABLED
    _ENABLED = bool(flag)


def is_enabled() -> bool:
    return _ENABLED


def nvtx_range(name: str):
    """Context manager that emits an NVTX range when enabled.

    When disabled returns a ``nullcontext``; the cost is one global read
    plus one ``nullcontext()`` instantiation, which is well under the
    granularity of any measurement nsys would care about.
    """
    if not _ENABLED:
        return nullcontext()
    return torch.cuda.nvtx.range(name)
