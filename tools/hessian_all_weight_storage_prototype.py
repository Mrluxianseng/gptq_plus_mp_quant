"""Extend stable final storage to every sequential quantized linear module.

Same dtype cast as down_proj's tested copy_ path. This hypothesis addresses
other long-lived weights pinning solver allocation segments. It is not yet a
passed resource optimization. The caller supplies the bound generated method
source, so an already-installed observer is never mistaken for core source.
"""
import hashlib


def install(impl, stable_source, *, enabled, records):
    guard = 'if str(getattr(self, "layer_name", "")).endswith("down_proj"):'
    if stable_source.count(guard) != 1:
        raise RuntimeError("stable final weight storage boundary changed")
    source = stable_source.replace(guard, "if True:  # All sequential linear module destinations")
    namespace = dict(vars(impl))
    exec(compile(source, "hessian_all_weight_storage_prototype", "exec"), namespace)
    candidate = namespace["fasterquant"]
    original = impl.GPTQPlus.fasterquant

    def describe(weight):
        pointer = weight.data_ptr()
        row = dict(pointer=pointer, stride=list(weight.stride()), dtype=str(weight.dtype))
        for segment in impl.torch.cuda.memory_snapshot():
            if segment["address"] <= pointer < segment["address"] + segment["total_size"]:
                row.update(segment_bytes=segment["total_size"],
                           inactive_bytes=sum(b["size"] for b in segment["blocks"] if b["state"] == "inactive"))
                return row
        raise RuntimeError("weight destination absent from native allocator snapshot")

    def wrapped(self, *args, **kwargs):
        weight = self.layer.weight
        before = describe(weight)
        del weight  # Observer must not keep old storage alive through a write.
        result = (candidate if enabled else original)(self, *args, **kwargs)
        weight = self.layer.weight
        after = describe(weight)
        row = dict(layer=getattr(self, "layer_idx", None), module=str(getattr(self, "layer_name", "")),
                   enabled=enabled, before=before, after=after, pointer_preserved=before["pointer"] == after["pointer"])
        records.append(row)
        if enabled and (not row["pointer_preserved"] or before["stride"] != after["stride"] or before["dtype"] != after["dtype"]):
            raise RuntimeError("all-module stable write changed destination/layout/dtype")
        return result

    impl.GPTQPlus.fasterquant = wrapped
    return hashlib.sha256(source.encode()).hexdigest()
