"""Prepare a guarded replay-boundary residency experiment; launches no CUDA.

Only consider after observing the Hessian-staging allocator regression. This
is a fresh prefix diagnostic, not a claim that CPU transfers improve speed.
Both arms observe identical boundaries; candidate rehomes one completed
block's parameters before replay, preserving identities, layout and bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INJECTION = '''
_rehome_layer = "--research_stage2_rehome_completed_layer" in sys.argv
if _rehome_layer:
    sys.argv.remove("--research_stage2_rehome_completed_layer")
_rehome_records = []

def _layer_tensor_record(layer):
    result = {}
    identities = {}
    occupied = {}
    for kind, tensors in (("parameter", layer.named_parameters()), ("buffer", layer.named_buffers())):
        for name, tensor in tensors:
            if tensor.layout != torch.strided:
                raise RuntimeError("residency prototype requires strided tensors")
            if tensor.numel() == 0:
                continue
            if tensor.storage_offset() != 0:
                raise RuntimeError("nonzero-offset module tensor requires an alias-preserving implementation")
            key = kind + ":" + name
            pointer = (str(tensor.device), tensor.untyped_storage().data_ptr())
            if pointer in occupied:
                raise RuntimeError("shared module storage needs a separate alias-preserving implementation")
            occupied[pointer] = key
            value = tensor.detach().cpu().contiguous()
            row = {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
                   "requires_grad": tensor.requires_grad,
                   "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
            if kind == "parameter":
                identities[key] = id(tensor)
                if tensor.grad is not None:
                    grad = tensor.grad.detach().cpu().contiguous()
                    row["grad"] = {"shape": list(grad.shape), "dtype": str(grad.dtype),
                                   "sha256": hashlib.sha256(grad.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
            result[key] = row
    return result, identities

def _layer_segment_observation(layer):
    owners = {}
    for name, tensor in list(layer.named_parameters()) + list(layer.named_buffers()):
        if tensor.is_cuda and tensor.numel():
            owners[name] = tensor.untyped_storage().data_ptr()
    segments = []
    for segment in torch.cuda.memory_snapshot():
        matching = [name for name, pointer in owners.items()
                    if segment["address"] <= pointer < segment["address"] + segment["total_size"]]
        if matching:
            segments.append({"owners": matching, "total_bytes": segment["total_size"],
                             "allocated_bytes": segment["allocated_size"], "active_bytes": segment["active_size"],
                             "inactive_block_bytes": sum(block["size"] for block in segment["blocks"]
                                                         if block["state"] == "inactive")})
    return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
            "inactive_split": torch.cuda.memory_stats().get("inactive_split_bytes.all.current"),
            "layer_owned_segments": segments}

def _residency_boundary_hook(event, context):
    if event != "after_layer_quantized":
        return False
    layer = context["layer"]
    device = context["dev"]
    if torch.is_grad_enabled():
        raise RuntimeError("completed-block residency boundary must be outside an active backward")
    torch.cuda.synchronize(device)
    before_tensors, before_ids = _layer_tensor_record(layer)
    before = _layer_segment_observation(layer)
    started = time.monotonic()
    if _rehome_layer:
        # No change to the calculations preceding this boundary or replay.
        # Evacuate live block values first so empty_cache can release segments
        # which ordinary free cannot release while a new weight pins them.
        layer.to("cpu")
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        layer.to(device)
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    after_tensors, after_ids = _layer_tensor_record(layer)
    if before_tensors != after_tensors or before_ids != after_ids:
        raise RuntimeError("block residency changed parameter identity, layout, gradient or tensor bytes")
    record = {"layer": context["layer_idx"], "rehome_enabled": _rehome_layer,
              "tensor_bytes_layout_and_parameter_identity_exact": True,
              "tensors": before_tensors, "before": before, "after": _layer_segment_observation(layer),
              "boundary_copy_seconds": elapsed, "diagnostic_time_not_speed_evidence": True}
    _rehome_records.append(record)
    (root / f"completed_layer_residency_rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(_rehome_records, indent=2))
    logging.info("COMPLETED_LAYER_RESIDENCY %s", json.dumps({k: v for k, v in record.items() if k != "tensors"}))
    return False
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    base = (ROOT / "tools/realq_memory_profile_entry.py").read_text()
    needle = "original_fwrd = impl.gptq_fwrd"
    inner = "    result = original_fwrd(args, analyzer, dataloader, dev)"
    if base.count(needle) != 1 or base.count(inner) != 1:
        raise RuntimeError("entry boundaries changed")
    wrapped = '''    if getattr(args, "_analysis_hook", None) is not None:
        raise RuntimeError("residency experiment cannot replace an existing analysis hook")
    args._analysis_hook = _residency_boundary_hook
    try:
        result = original_fwrd(args, analyzer, dataloader, dev)
    finally:
        args._analysis_hook = None'''
    entry = args.output / "entry.py"
    entry.write_text(base.replace(needle, INJECTION + "\n" + needle).replace(inner, wrapped))
    compile(entry.read_text(), str(entry), "exec")
    source = ROOT / "outputs/memory_mechanism_followup_20261010/hessian-cpu-layer0-manifests"
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            manifest = json.loads((source / f"candidate-{stage}.json").read_text())
            command = manifest["command"]
            command[command.index("baseline-profile") - 1] = str(entry.resolve())
            if side == "candidate" and stage == "stage2":
                command.append("--research_stage2_rehome_completed_layer")
            manifest["scope_note"] = "Both Hessian CPU; only candidate rehomes completed block before replay. Prefix diagnostic; no speed/full-model claim."
            (args.output / f"{side}-{stage}.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "provenance.json").write_text(json.dumps(
        {"entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(), "real_cuda_validated": False,
         "source": str(Path(__file__).resolve()), "requires_fresh_prefix_audit": True}, indent=2))


if __name__ == "__main__":
    main()
