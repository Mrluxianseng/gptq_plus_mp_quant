"""Research-only consumed-H reuse with stable quantized weight storage.

Preserve solver arithmetic/layout via the already checked reuse builder. Avoid
placing a new long-lived model weight in a freed solver/Hessian segment. This
is a location/lifetime hypothesis, not an allocator or numerical guarantee.
"""
import re

from tools.hessian_inverse_lifetime_prototype import build_reusing_fasterquant


def build_stable_fasterquant(impl):
    _, source = build_reusing_fasterquant(impl)
    pattern = (r'(?m)^([ ]+)self\.layer\.weight\.data = Q_final\.reshape\(self\.layer\.weight\.shape\)\.to\(\n'
               r'\1    self\.layer\.weight\.data\.dtype\n\1\)')
    # Match the actual enclosing indentation rather than silently patching a
    # different batched/group path. Failing this guard demands code review.
    matches = list(re.finditer(pattern, source))
    if len(matches) != 1:
        raise RuntimeError("sequential final weight write boundary changed")
    after = '''if str(getattr(self, "layer_name", "")).endswith("down_proj"):
    _weight_destination = self.layer.weight.data
    _weight_source = Q_final.reshape(self.layer.weight.shape)
    if (not _weight_destination.is_contiguous()
            or _weight_destination.device != _weight_source.device
            or _weight_destination.shape != _weight_source.shape):
        raise RuntimeError("unsupported stable weight destination")
    # Same FP32->original-dtype conversion, existing allocation.
    # .data mirrors the original autograd boundary; the current
    # refresh graph has already been consumed at this point.
    _weight_destination.copy_(_weight_source)
    del _weight_destination, _weight_source
else:
    self.layer.weight.data = Q_final.reshape(self.layer.weight.shape).to(
        self.layer.weight.data.dtype
    )'''
    match = matches[0]
    replacement = '\n'.join(match[1]+line for line in after.splitlines())
    source = source[:match.start()] + replacement + source[match.end():]
    namespace = dict(vars(impl))
    exec(compile(source, "hessian_stable_weight_prototype", "exec"), namespace)
    return namespace["fasterquant"], source


def install_fasterquant(impl, *, enabled, records):
    """Shared CPU-only snapshot observer; record ownership, not GPU tensors."""
    original = impl.GPTQPlus.fasterquant
    candidate, source = build_stable_fasterquant(impl)
    torch = impl.torch

    def storage_summary(weight):
        ptr = weight.data_ptr()
        # CPU data from allocator bookkeeping. No retained tensor/snapshot.
        for segment in torch.cuda.memory_snapshot():
            if segment["address"] <= ptr < segment["address"] + segment["total_size"]:
                inactive = sum(b["size"] for b in segment["blocks"] if b["state"] == "inactive")
                return dict(pointer=ptr, segment_bytes=segment["total_size"],
                            inactive_bytes=inactive, dtype=str(weight.dtype),
                            stride=list(weight.stride()), bytes=weight.numel()*weight.element_size())
        raise RuntimeError("weight storage absent from native allocator snapshot")

    def fasterquant(self, *args, **kwargs):
        target = str(getattr(self, "layer_name", "")).endswith("down_proj")
        before = storage_summary(self.layer.weight) if target else None
        result = (candidate if enabled else original)(self, *args, **kwargs)
        if target:
            after = storage_summary(self.layer.weight)
            if enabled and before["pointer"] != after["pointer"]:
                raise RuntimeError("stable weight write allocated a different destination")
            records.append(dict(layer=getattr(self, "layer_idx", None),
                                module=self.layer_name, enabled=bool(enabled),
                                before=before, after=after,
                                pointer_preserved=before["pointer"] == after["pointer"]))
        return result

    impl.GPTQPlus.fasterquant = fasterquant
    return source
