"""Capture actual solver boundaries without changing REAL-Q arithmetic.

Capture is an intrusive diagnostic run, never a memory/timing benchmark.
Only the first down_proj subgroup at selected layers is saved. It is a solver
fixture, not an Adam/block/allocator checkpoint. CPU copies preserve dtype
and stride; no live CUDA tensors are retained by the recorder.
"""
import hashlib
import inspect
import json
from pathlib import Path

import torch


def cpu_snapshot(value):
    value = value.detach()
    if value.layout != torch.strided or any(s <= 0 for s in value.stride()):
        raise ValueError("capture requires positive-stride dense storage")
    result = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                 device="cpu")
    result.copy_(value)
    return result


def tensor_identity(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return dict(shape=list(value.shape), stride=list(value.stride()),
                dtype=str(value.dtype), sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def install_capture(impl, directory, layers, provenance):
    original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
    signature = inspect.signature(original)
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    layers = set(layers)
    counts, records = {}, []
    provenance = dict(provenance, torch=str(torch.__version__), torch_cuda=torch.version.cuda)
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))

    def wrapped(self, h_sub, *args, **kwargs):
        layer = int(getattr(self, "layer_idx", -1))
        name = str(getattr(self, "layer_name", ""))
        target = layer in layers and name.endswith("down_proj")
        if not target:
            return original(self, h_sub, *args, **kwargs)
        ordinal = counts.get(layer, 0)
        counts[layer] = ordinal + 1
        if ordinal:
            return original(self, h_sub, *args, **kwargs)
        bound = signature.bind(self, h_sub, *args, **kwargs)
        bound.apply_defaults()
        options = {key: bound.arguments[key] for key in (
            "percdamp", "damp_auto_increment", "need_hinv_init", "inplace_damping")}
        before = cpu_snapshot(h_sub)
        print(f"[bottleneck-capture] layer={layer} module={name} H={tuple(h_sub.shape)}", flush=True)
        result = original(self, h_sub, *args, **kwargs)
        inv, factor, damping, fallback = result
        payload = dict(schema=1, layer=layer, module=name, subgroup=ordinal,
                       options=options, H=before, H_after=cpu_snapshot(h_sub),
                       inverse=None if inv is None else cpu_snapshot(inv),
                       factor=cpu_snapshot(factor), damping=damping, fallback=fallback)
        filename = directory / f"layer{layer}-down-proj-subgroup0.pt"
        temporary = filename.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(filename)
        identity = {key: None if payload[key] is None else tensor_identity(payload[key])
                    for key in ("H", "H_after", "inverse", "factor")}
        records.append(dict(layer=layer, module=name, subgroup=ordinal,
                            file=filename.name, file_sha256=file_sha(filename), tensors=identity,
                            options=options, damping=damping, fallback=fallback))
        temporary = directory / "index.tmp"
        temporary.write_text(json.dumps(dict(schema=1, records=records,
            scope="Actual solver inputs/outputs only; no allocator or Adam state restoration"), indent=2))
        temporary.replace(directory / "index.json")
        print(f"[bottleneck-capture] saved {filename}", flush=True)
        return result

    impl.GPTQPlus._compute_hessian_inverse_with_fallback = wrapped
    return records
