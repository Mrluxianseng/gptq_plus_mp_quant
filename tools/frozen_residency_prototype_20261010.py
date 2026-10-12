"""All-sized frozen saved parameters for a guarded FSDP backward pilot.

Research only. FSDP frees unsharded storage, so skipping backward unshard is
unsafe unless every needed parameter saving has an independent payload.
Do not combine with checkpoint recomputation that reads module parameters.
"""
from contextlib import contextmanager
import weakref

import torch

from utils.saved_tensor_offload import SavedTensorCpuCache
from utils.frozen_parameter_offload_prototype import FrozenParameterRegistry


class AllFrozenParameterRegistry(FrozenParameterRegistry):
    def register(self, weight, *, allow_cpu=False):
        if not isinstance(weight, torch.Tensor) or weight.requires_grad:
            return
        if weight.device.type != "cuda" and not allow_cpu:
            return
        uid = SavedTensorCpuCache.key(weight)[0]
        storage = weight.untyped_storage()
        self._storages[uid] = storage
        self._versions[uid] = weight._version
        weakref.finalize(storage, self._versions.pop, uid, None)
        self.registration_calls += 1

    def matches(self, tensor):
        if tensor.requires_grad or not tensor.is_leaf:
            return False
        uid = SavedTensorCpuCache.key(tensor)[0]
        if uid not in self._storages:
            self._versions.pop(uid, None)
            return False
        return self._versions.get(uid) == tensor._version

    def attach(self, model):
        def capture(module, inputs):
            for weight in module.parameters(recurse=False):
                self.register(weight)
        return [module.register_forward_pre_hook(capture) for module in model.modules()
                if any(True for _ in module.parameters(recurse=False))]


class AllFrozenParameterCpuCache(SavedTensorCpuCache):
    def __init__(self, threshold_bytes, *, registry, deduplicate=True):
        super().__init__(threshold_bytes, deduplicate=deduplicate)
        self.registry = registry
        self.stats.update(frozen_parameter_requests=0, frozen_parameter_requested_bytes=0,
                          frozen_parameter_copied_bytes=0, frozen_parameter_shapes={})

    def pack(self, tensor):
        if tensor.device.type == "cuda" and self.registry.matches(tensor):
            nbytes = tensor.numel() * tensor.element_size()
            self.stats["frozen_parameter_requests"] += 1
            self.stats["frozen_parameter_requested_bytes"] += nbytes
            key = str((tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype)))
            self.stats["frozen_parameter_shapes"][key] = self.stats["frozen_parameter_shapes"].get(key, 0) + 1
            before = self.stats["copied_bytes"]
            value = self._pack_value(tensor)
            self.stats["frozen_parameter_copied_bytes"] += self.stats["copied_bytes"] - before
            return value
        return super().pack(tensor)


@contextmanager
def fsdp_without_backward_unshard(model):
    from torch.distributed.fsdp import FSDPModule
    states = []
    report = {"configured_fsdp_groups": 0, "parameter_groups_restored": False}
    def configure(module, inputs):
        if states:
            return
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("backward-unshard pilot requires all model parameters frozen")
        for child in model.modules():
            if isinstance(child, FSDPModule):
                group = child._get_fsdp_state()._fsdp_param_group
                if group is not None:
                    states.append((child, group.unshard_in_backward))
                    child.set_unshard_in_backward(False)
        report["configured_fsdp_groups"] = len(states)
        if not states:
            raise RuntimeError("no FSDP parameter group was configured")
    handle = model.register_forward_pre_hook(configure)
    try:
        yield report
        if not states:
            raise RuntimeError("no model forward exercised the FSDP pilot")
    finally:
        handle.remove()
        for child, previous in states:
            child.set_unshard_in_backward(previous)
        report["parameter_groups_restored"] = True
