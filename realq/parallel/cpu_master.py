"""CPU-master layer manager for the cpu_master FSDP path.

When ``analyzer.model._realq_cpu_master`` is set, rank 0 holds the full CPU
model and rank>0 holds a meta skeleton. To run a per-layer GPU op (Hessian
forward, weight quant, KL refresh) every rank needs the same weights on its
local GPU. This manager hides the asymmetry behind a uniform API:

    manager.materialize_layer(idx)          # rank 0: layer.to(dev); broadcast to all ranks.
    manager.release_layer(idx, ...)         # rank 0: layer.to(cpu); rank>0: to_empty(meta).

When ``enabled=False`` (world_size <= 1, or no ``_realq_cpu_master`` tag),
every method degrades to a plain ``.to(dev)`` / ``.to(orig_device)`` so call
sites can use the manager unconditionally — no ``if cpu_master:`` branching.

Direct port of ``gptq_utils.gptq_plus_utils.Stage2CpuMasterLayerManager``
(lines 169-238) and helpers (lines 111-166). Two small differences from the
original:

* ``materialize_runtime_modules`` is idempotent across calls — a second call
  on already-materialised modules is a no-op. The legacy code only called
  it once at runtime-module setup (lm_head + final norm), but RealQ's
  ``quantize_one_layer`` re-issues the call per-layer for the KL-refresh
  branch. Idempotence keeps both call patterns correct.
* No ``update_master`` kwarg on ``release_layer``: the legacy code's
  ``update_master`` was a no-op (rank 0's ``layers[idx]`` IS the master, so
  ``.to(orig_device)`` already updates it).
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn

from realq.parallel import env as parallel_env
from utils import quant_utils


# Module-level helpers (port of gptq_plus_utils.py:111-166).


def _refresh_act_quant_wrapper_aliases(module: nn.Module) -> None:
    """Re-point ActQuantWrapper.weight/bias to the inner module's tensors.

    ``ActQuantWrapper.__init__`` does ``self.weight = module.weight`` (plain
    attribute alias). After ``module.to_empty(device=...)`` the inner Linear's
    weight storage is reallocated, but the wrapper's ``self.weight`` still
    points at the old (now-stale) storage. Refresh restores the alias.
    """
    for child in module.modules():
        if isinstance(child, quant_utils.ActQuantWrapper):
            child.weight = child.module.weight
            child.bias = child.module.bias


def _alloc_module_empty_on_device(module: nn.Module, device: torch.device) -> nn.Module:
    """Allocate empty (uninitialised) storage for every tensor on ``device``."""
    module.to_empty(device=device)
    _refresh_act_quant_wrapper_aliases(module)
    return module


def _iter_module_tensors(module: nn.Module) -> Iterable[torch.Tensor]:
    """Yield each unique parameter then each unique buffer of ``module``."""
    seen = set()
    for _, param in module.named_parameters(recurse=True, remove_duplicate=True):
        if id(param) in seen:
            continue
        seen.add(id(param))
        yield param
    for _, buf in module.named_buffers(recurse=True, remove_duplicate=True):
        if buf is None or id(buf) in seen:
            continue
        seen.add(id(buf))
        yield buf


@torch.no_grad()
def _broadcast_module_from_rank0(
    module: nn.Module,
    dev: torch.device,
    src_module: nn.Module | None = None,
) -> nn.Module:
    """Place ``module`` on ``dev`` with rank 0's tensor values broadcast to all ranks.

    rank 0 reads from ``src_module`` (or ``module`` itself) on CPU and moves
    to ``dev``. rank>0 allocates empty CUDA storage. Then every parameter +
    buffer is ``dist.broadcast``ed src=0. World==1 short-circuits to a plain
    ``.to(dev)``.

    Note: bypasses ``realq.parallel.env`` to call ``dist.broadcast`` directly
    because ``env`` does not expose a per-tensor broadcast helper.
    """
    if parallel_env.get_world_size() <= 1:
        return module.to(dev)

    rank = parallel_env.get_rank()
    if rank == 0:
        source = src_module if src_module is not None else module
        target = source.to(dev)
    else:
        target = _alloc_module_empty_on_device(module, dev)

    for tensor in _iter_module_tensors(target):
        if tensor.device.type != "cuda":
            raise RuntimeError(
                f"cpu_master broadcast expected CUDA tensors, got {tensor.device}."
            )
        if not tensor.is_contiguous():
            tensor.data = tensor.data.contiguous()
        dist.broadcast(tensor.data, src=0)
    _refresh_act_quant_wrapper_aliases(target)
    return target


@torch.no_grad()
def _free_module_to_meta(module: nn.Module) -> nn.Module:
    """Replace ``module``'s tensor storage with meta tensors (≈0 RAM)."""
    module.to_empty(device="meta")
    _refresh_act_quant_wrapper_aliases(module)
    return module


# The manager itself.


class CpuMasterLayerManager:
    """Manages per-layer materialise/release under the cpu_master pattern.

    When ``enabled=False`` every method is a plain ``.to(dev)``/``.to(cpu)``
    so call sites can use the manager unconditionally.

    When ``enabled=True``:
      * rank 0 holds the master copy; ``self.layers[idx]`` IS that master.
      * ``materialize_layer(idx)`` does a rank0→all broadcast; idempotent
        within a single materialised window (tracked by ``self.materialized``).
      * ``release_layer(idx, ...)`` puts rank 0's copy back on CPU and
        clears rank>0's GPU copy back to meta.
      * ``materialize_runtime_modules(...)`` / ``release_runtime_modules(...)``
        do the same for non-block modules (lm_head, final norm, embed_tokens).
        Idempotent via ``self._runtime_materialised`` (a set of module ids).
    """

    def __init__(
        self,
        analyzer,
        dev: torch.device,
        layers: list[nn.Module],
        *,
        enabled: bool | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.model = analyzer.model
        self.dev = dev
        self.layers = layers
        self.rank = parallel_env.get_rank()
        if enabled is None:
            enabled = bool(getattr(self.model, "_realq_cpu_master", False))
        # Auto-disable when world<=1: there's no asymmetry to manage and
        # broadcast would be wasted work.
        if enabled and parallel_env.get_world_size() <= 1:
            enabled = False
        self.enabled = enabled
        self.master_layers = layers if self.enabled and self.rank == 0 else None
        self.materialized: set[int] = set()
        self._runtime_materialised: set[int] = set()

    def _master_layer(self, idx: int) -> nn.Module | None:
        if self.master_layers is None:
            return None
        return self.master_layers[idx]

    def materialize_runtime_modules(self, modules: list[nn.Module]) -> None:
        """Materialise non-block modules (lm_head, norm, embed) on dev.

        Idempotent: a second call on already-materialised modules is a no-op,
        so callers (``quantize_all_layers`` and per-layer KL refresh inside
        ``quantize_one_layer``) can both invoke this without coordinating.
        """
        if not self.enabled:
            for module in modules:
                module.to(self.dev)
            return
        for module in modules:
            mid = id(module)
            if mid in self._runtime_materialised:
                continue
            _broadcast_module_from_rank0(
                module,
                self.dev,
                src_module=module if self.rank == 0 else None,
            )
            self._runtime_materialised.add(mid)

    def release_runtime_modules(
        self, modules: list[nn.Module], orig_device: torch.device
    ) -> None:
        """Release the runtime modules from dev: rank0 → orig_device, rank>0 → meta."""
        if not self.enabled:
            for module in modules:
                module.to(orig_device)
            return
        for module in modules:
            if self.rank == 0:
                module.to(orig_device)
            else:
                _free_module_to_meta(module)
            self._runtime_materialised.discard(id(module))

    def materialize_layer(self, idx: int) -> nn.Module:
        """Place transformer block ``idx`` on dev with rank0's values broadcast."""
        if not self.enabled:
            layer = self.layers[idx].to(self.dev)
            self.layers[idx] = layer
            return layer
        if idx in self.materialized:
            return self.layers[idx]
        layer = _broadcast_module_from_rank0(
            self.layers[idx],
            self.dev,
            src_module=self._master_layer(idx),
        )
        self.layers[idx] = layer
        self.materialized.add(idx)
        return layer

    def release_layer(
        self,
        idx: int,
        layer: nn.Module | None = None,
        *,
        orig_device: torch.device = torch.device("cpu"),
    ) -> nn.Module:
        """Release transformer block ``idx``: rank0 → orig_device (updates master), rank>0 → meta."""
        if not self.enabled:
            if layer is None:
                layer = self.layers[idx]
            self.layers[idx] = layer.to(orig_device)
            return self.layers[idx]
        if layer is None:
            layer = self.layers[idx]
        if self.rank == 0:
            self.layers[idx] = layer.to(orig_device)
            # rank0's self.layers[idx] IS the master; no separate copy needed.
            self.master_layers[idx] = self.layers[idx]
        else:
            self.layers[idx] = _free_module_to_meta(layer)
        self.materialized.discard(idx)
        return self.layers[idx]
