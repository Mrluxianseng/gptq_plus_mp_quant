"""Lightweight NVTX wrapper for nsys profiling.

Markers are emitted only when the global flag is enabled (set once at
process start from ``cfg.nsys_profile``). When disabled, ``nvtx_range``
returns a ``nullcontext``, so the call site is essentially free
(~100 ns) and adds no synchronization, no allocation, and no behavior
change. The wrapper is intentionally a single-file module so it can be
imported from anywhere in realq without circular-import concerns.

Usage:
    from realq.utils import nvtx

    with nvtx.nvtx_range("precompute.batch_3"):
        ...
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

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


def _cuda_profiler_call(name: str) -> None:
    """Synchronously invoke a CUDA profiler-control API and check status."""
    torch.cuda.synchronize()
    status = getattr(torch.cuda.cudart(), name)()
    if status not in (None, 0):
        raise RuntimeError(f"{name} failed with CUDA status {status}.")


@dataclass
class CudaProfilerLayerCapture:
    """Open one inclusive layer capture window for ``nsys``.

    This controller is deliberately independent of NVTX emission.  It drives
    nsys's ``cudaProfilerApi`` capture range and synchronizes at both edges so
    queued kernels from the preceding layer cannot leak into the trace and all
    kernels from the final selected layer are collected before capture stops.
    """

    start_layer: int
    end_layer: int
    active: bool = False
    completed: bool = False

    def before_layer(self, layer_idx: int) -> None:
        if layer_idx == self.start_layer:
            if self.active or self.completed:
                raise RuntimeError("CUDA profiler layer capture started twice.")
            _cuda_profiler_call("cudaProfilerStart")
            self.active = True

    def after_layer(self, layer_idx: int) -> None:
        if layer_idx == self.end_layer:
            if not self.active:
                raise RuntimeError(
                    "CUDA profiler capture end reached before capture start."
                )
            _cuda_profiler_call("cudaProfilerStop")
            self.active = False
            self.completed = True

    def close(self) -> None:
        """Stop an open capture if quantization exits exceptionally."""
        if self.active:
            _cuda_profiler_call("cudaProfilerStop")
            self.active = False
