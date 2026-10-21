"""Process-local synchronous CPU staging for BOTH selected CUDA directions.

NCCL is separately configured to LOC; these counters do not include NCCL.
"""
from __future__ import annotations

import time


class HostTransport:
    def __init__(self, torch):
        if getattr(torch.Tensor, "_portable_host_transport", False):
            raise RuntimeError("Host transport already installed")
        self.torch = torch
        self.original = {name: getattr(torch.Tensor, name) for name in ("to", "cuda", "copy_")}
        self.records = {}
        raw_to, raw_cuda, raw_copy = (self.original[name] for name in ("to", "cuda", "copy_"))

        def target_of(args, kwargs, source):
            value = kwargs.get("device", args[0] if args else None)
            if value is None or isinstance(value, torch.dtype): return source
            if isinstance(value, torch.Tensor): return value.device
            target = torch.device("cuda", value) if isinstance(value, int) else torch.device(value)
            if target.type == "cuda" and target.index is None:
                target = torch.device("cuda", torch.cuda.current_device())
            return target

        def affected(source, target):
            return source.type == target.type == "cuda" and source.index != target.index

        def route(api, value, target, operation):
            start = time.perf_counter_ns()
            host = raw_to(value, device="cpu", non_blocking=False)
            result = operation(host)
            torch.cuda.synchronize(target)
            key = (api, str(value.device), str(target), str(value.dtype))
            row = self.records.setdefault(key, {"api": api, "source": key[1], "destination": key[2],
                "dtype": key[3], "calls": 0, "payload_bytes": 0, "inclusive_ns": 0})
            row["calls"] += 1
            row["payload_bytes"] += value.numel() * value.element_size()
            row["inclusive_ns"] += time.perf_counter_ns() - start
            return result

        def to(value, *args, **kwargs):
            target = target_of(args, kwargs, value.device)
            if not affected(value.device, target): return raw_to(value, *args, **kwargs)
            return route("to", value, target, lambda host: raw_to(host, *args, **kwargs))

        def cuda(value, device=None, non_blocking=False, memory_format=None):
            target = target_of((), {"device": device if device is not None else torch.cuda.current_device()}, value.device)
            options = {"device": device, "non_blocking": non_blocking}
            if memory_format is not None: options["memory_format"] = memory_format
            if not affected(value.device, target): return raw_cuda(value, **options)
            return route("cuda", value, target, lambda host: raw_cuda(host, **options))

        def copy(value, src, non_blocking=False):
            if not affected(src.device, value.device): return raw_copy(value, src, non_blocking=non_blocking)
            return route("copy_", src, value.device, lambda host: raw_copy(value, host, non_blocking=False))

        self.wrappers = {"to": to, "cuda": cuda, "copy_": copy}
        for name, method in self.wrappers.items(): setattr(torch.Tensor, name, method)
        torch.Tensor._portable_host_transport = True

    def close(self):
        changed = any(getattr(self.torch.Tensor, name) is not method for name, method in self.wrappers.items())
        for name, method in self.original.items(): setattr(self.torch.Tensor, name, method)
        delattr(self.torch.Tensor, "_portable_host_transport")
        if changed:
            raise RuntimeError("Transport methods changed during execution")
        return {"route": "both CUDA directions via synchronous CPU", "records": list(self.records.values()),
                "scope": "Python copies only; includes allocation/waits, excludes NCCL"}
