"""Research-only saved frozen-parameter migration; production defaults unchanged.

is_leaf includes every requires_grad=False tensor. Identify parameters by
live storage identity and version, not by leaf status or matrix shape alone.
"""
import weakref

import torch

from utils.saved_tensor_offload import SavedTensorCpuCache


class FrozenParameterRegistry:
    def __init__(self):
        self._storages = weakref.WeakValueDictionary()
        self._versions = {}
        self.registration_calls = 0

    def register(self, weight, *, allow_cpu=False):
        if not isinstance(weight, torch.Tensor) or weight.requires_grad or weight.ndim != 2:
            return
        if weight.device.type != "cuda" and not allow_cpu:
            return
        uid = SavedTensorCpuCache.key(weight)[0]
        self._storages[uid] = weight.untyped_storage()
        self._versions[uid] = weight._version
        weakref.finalize(weight.untyped_storage(), self._versions.pop, uid, None)
        self.registration_calls += 1

    def matches(self, tensor):
        if tensor.requires_grad or not tensor.is_leaf or tensor.ndim != 2:
            return False
        uid = SavedTensorCpuCache.key(tensor)[0]
        if uid not in self._storages:
            self._versions.pop(uid, None)
            return False
        return self._versions.get(uid) == tensor._version

    def attach(self, model):
        def capture(module, inputs):
            self.register(getattr(module, "weight", None))
        return [module.register_forward_pre_hook(capture) for module in model.modules()
                if isinstance(getattr(module, "weight", None), torch.Tensor)]


class FrozenParameterCpuCache(SavedTensorCpuCache):
    def __init__(self, threshold_bytes, *, registry, deduplicate=True):
        super().__init__(threshold_bytes, deduplicate=deduplicate)
        self.registry = registry
        self.stats.update(frozen_parameter_requests=0, frozen_parameter_requested_bytes=0,
                          frozen_parameter_copied_bytes=0, frozen_parameter_shapes={})

    def pack(self, tensor):
        nbytes = tensor.numel() * tensor.element_size()
        if tensor.device.type == "cuda" and nbytes >= self.threshold_bytes and self.registry.matches(tensor):
            self.stats["frozen_parameter_requests"] += 1
            self.stats["frozen_parameter_requested_bytes"] += nbytes
            key = str((tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype)))
            self.stats["frozen_parameter_shapes"][key] = self.stats["frozen_parameter_shapes"].get(key, 0) + 1
            before = self.stats["copied_bytes"]
            payload = self._pack_value(tensor)
            self.stats["frozen_parameter_copied_bytes"] += self.stats["copied_bytes"] - before
            return payload
        return super().pack(tensor)
