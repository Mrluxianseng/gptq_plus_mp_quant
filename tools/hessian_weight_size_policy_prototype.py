"""CPU preparation for size-aware down-weight placement after a failed gate.

Native allocator medium allocations are packed into larger segments. Keep
stable writes for weights >=20 MiB, but use the original conversion/allocation
for smaller weights. This is a placement hypothesis, not a measured benefit.
Solver reuse and dead-input release remain in the supplied builder chain.
"""
from tools import hessian_stable_weight_prototype as stable

MIN_STABLE_BYTES = 20 * 1024 * 1024


def use_stable_write(weight_bytes):
    return int(weight_bytes) >= MIN_STABLE_BYTES


def build_policy_fasterquant(impl):
    _, source = stable.build_stable_fasterquant(impl)
    anchor = 'if str(getattr(self, "layer_name", "")).endswith("down_proj"):'
    if source.count(anchor) != 1:
        raise RuntimeError("stable down-weight branch changed")
    replacement = ('if (str(getattr(self, "layer_name", "")).endswith("down_proj")\n'
                   '        and self.layer.weight.numel() * self.layer.weight.element_size()\n'
                   f'        >= {MIN_STABLE_BYTES}):')
    source = source.replace(anchor, replacement)
    namespace = dict(vars(impl))
    exec(compile(source, "hessian_weight_size_policy_prototype", "exec"), namespace)
    return namespace["fasterquant"], source


def install_fasterquant(impl, *, enabled, records):
    """Shared CPU metadata observer; explicit requested vs selected coverage."""
    original = impl.GPTQPlus.fasterquant
    candidate, source = build_policy_fasterquant(impl)

    def summary(weight):
        ptr = weight.data_ptr()
        for segment in impl.torch.cuda.memory_snapshot():
            if segment["address"] <= ptr < segment["address"] + segment["total_size"]:
                return dict(pointer=ptr, segment_bytes=segment["total_size"],
                    inactive_bytes=sum(b["size"] for b in segment["blocks"]
                                       if b["state"] == "inactive"),
                    dtype=str(weight.dtype), stride=list(weight.stride()),
                    bytes=weight.numel() * weight.element_size())
        raise RuntimeError("weight absent from native allocator snapshot")

    def fasterquant(self, *args, **kwargs):
        target = str(getattr(self, "layer_name", "")).endswith("down_proj")
        before = summary(self.layer.weight) if target else None
        selected = bool(enabled and target and use_stable_write(before["bytes"]))
        result = (candidate if enabled else original)(self, *args, **kwargs)
        if target:
            after = summary(self.layer.weight)
            if selected and before["pointer"] != after["pointer"]:
                raise RuntimeError("selected stable write changed destination")
            records.append(dict(layer=getattr(self, "layer_idx", None),
                module=self.layer_name, enabled=selected,
                requested_enabled=bool(enabled), policy="native-medium-size-placement",
                min_stable_bytes=MIN_STABLE_BYTES, before=before, after=after,
                pointer_preserved=before["pointer"] == after["pointer"]))
        return result

    impl.GPTQPlus.fasterquant = fasterquant
    return source


def build_entry(source, *, helper_sha):
    """Accept dead/stats raw entry before the full profiler adds wrappers."""
    anchor = 'from tools.hessian_stable_weight_prototype import install_fasterquant as _hinv_install_stable'
    if source.count(anchor) != 1 or "_dead_original_builder=" not in source:
        raise RuntimeError("requires existing dead-input builder before size policy")
    replacement = f'''_size_policy_helper=_HinvPath.cwd()/"tools/hessian_weight_size_policy_prototype.py"
_size_policy_expected={helper_sha!r}
if _hinv_hashlib.sha256(_size_policy_helper.read_bytes()).hexdigest()!=_size_policy_expected:
    raise RuntimeError("size policy helper changed")
from tools.hessian_weight_size_policy_prototype import install_fasterquant as _hinv_install_stable'''
    # The import sits inside a top-level if. Preserve its indentation exactly.
    replacement = replacement.replace("\n", "\n    ")
    result = source.replace(anchor, replacement)
    compile(result, "hessian_weight_size_policy_entry", "exec")
    return result
