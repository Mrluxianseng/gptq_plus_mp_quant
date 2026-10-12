"""Lossless, opt-in saved-tensor CPU caching for REAL-Q precompute.

Inspired by SSDTrain's storage identity deduplication (arXiv:2408.10013v2).
We do not implement SSD I/O, asynchronous prefetch or weight exclusion.
Weak payload references prevent the cache from extending graph lifetimes.
"""
from __future__ import annotations

import itertools
import weakref

import torch


_storage_ids = itertools.count()
_storage_id_attribute = "_realq_saved_cpu_storage_id_20261010"


class CpuSavedValue:
    def __init__(self, value, device):
        self.value = value
        self.device = device


class ReshapedCpuSavedValue:
    def __init__(self, payload, shape):
        self.payload = payload
        self.shape = shape


class SavedTensorCpuCache:
    """Synchronous, version-aware deduplication of identical saved views.

    Each storage gets a unique identifier rather than using a CUDA pointer
    that the allocator may reuse. Distinct views and versions have distinct
    keys. No key or registry entry retains the original CUDA tensor/storage.
    """

    def __init__(self, threshold_bytes, *, deduplicate=True):
        self.threshold_bytes = int(threshold_bytes)
        self.deduplicate = bool(deduplicate)
        self._payloads = {}
        self.stats = dict(requests=0, requested_bytes=0, copies=0,
                          copied_bytes=0, dedup_hits=0, live_cpu_bytes=0,
                          peak_live_cpu_bytes=0)

    @staticmethod
    def key(tensor):
        storage = tensor.untyped_storage()
        uid = getattr(storage, _storage_id_attribute, None)
        if uid is None:
            uid = next(_storage_ids)
            setattr(storage, _storage_id_attribute, uid)
        if tensor.is_contiguous():
            # Matmul frequently saves a flattened batch view of softmax's
            # output. Identical contiguous storage regions share one copy
            # without copying an entire parent/FSDP parameter storage.
            view = ("contiguous", tensor.storage_offset(), tensor.numel())
        else:
            view = ("strided", tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
        return (uid, tensor.device, tensor.dtype, view, tensor._version)

    def pack(self, tensor):
        nbytes = tensor.numel() * tensor.element_size()
        if tensor.device.type != "cuda" or tensor.is_leaf or nbytes < self.threshold_bytes:
            return tensor
        return self._pack_value(tensor)

    def _pack_value(self, tensor):
        # Kept separate so lifetime/version behavior can be tested on CPU
        # without allocating on a GPU concurrently with a profile run.
        nbytes = tensor.numel() * tensor.element_size()
        self.stats["requests"] += 1
        self.stats["requested_bytes"] += nbytes
        key = self.key(tensor) if self.deduplicate else None
        if key is not None:
            previous = self._payloads.get(key)
            payload = previous() if previous is not None else None
            if payload is not None:
                self.stats["dedup_hits"] += 1
                if tuple(payload.value.shape) == tuple(tensor.shape):
                    return payload
                return ReshapedCpuSavedValue(payload, tuple(tensor.shape))
        payload = CpuSavedValue(tensor.detach().to(device="cpu", copy=True), tensor.device)
        self.stats["copies"] += 1
        self.stats["copied_bytes"] += nbytes
        self.stats["live_cpu_bytes"] += nbytes
        self.stats["peak_live_cpu_bytes"] = max(self.stats["peak_live_cpu_bytes"], self.stats["live_cpu_bytes"])
        if key is not None:
            self._payloads[key] = weakref.ref(payload)
        weakref.finalize(payload, self._release, key, nbytes)
        return payload

    def _release(self, key, nbytes):
        self.stats["live_cpu_bytes"] -= nbytes
        if key is not None:
            current = self._payloads.get(key)
            if current is not None and current() is None:
                self._payloads.pop(key, None)

    @staticmethod
    def unpack(payload):
        if isinstance(payload, ReshapedCpuSavedValue):
            return SavedTensorCpuCache.unpack(payload.payload).reshape(payload.shape)
        if isinstance(payload, CpuSavedValue):
            return payload.value.to(device=payload.device)
        return payload

    def context(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
