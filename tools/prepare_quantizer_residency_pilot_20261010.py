"""Prepare a module-end metadata lifetime diagnostic; do not modify live core."""
import hashlib
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_pipeline_memory_profile import set_option

INJECTION = '''
_metadata_cpu = "--research_stage2_quantizer_metadata_cpu" in sys.argv
if _metadata_cpu:
    sys.argv.remove("--research_stage2_quantizer_metadata_cpu")
_original_gptq_free = impl.GPTQPlus.free
_metadata_records = []
def _metadata_lifetime_free(self):
    before = {}
    copied = 0
    gpu_pointers = {}
    for name, tensor in sorted(self.quantizer.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        before[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        if tensor.is_cuda:
            copied += tensor.numel() * tensor.element_size()
            gpu_pointers[name] = tensor.untyped_storage().data_ptr()
    segments = []
    for segment in torch.cuda.memory_snapshot():
        matching = [name for name, ptr in gpu_pointers.items()
                    if segment["address"] <= ptr < segment["address"] + segment["total_size"]]
        if matching:
            segments.append({"buffer_names": matching, "segment_total_bytes": segment["total_size"],
                             "segment_active_bytes": segment["active_size"],
                             "segment_allocated_bytes": segment["allocated_size"],
                             "inactive_block_bytes": sum(block["size"] for block in segment["blocks"]
                                                         if block["state"] == "inactive")})
    if _metadata_cpu:
        self.quantizer.cpu()
        for name, tensor in sorted(self.quantizer.state_dict().items()):
            if tensor.is_cuda:
                raise RuntimeError("quantizer metadata did not leave CUDA")
            digest = hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            if digest != before[name]["sha256"]:
                raise RuntimeError("quantizer metadata changed during CPU transfer")
    _original_gptq_free(self)
    _metadata_records.append({"layer": getattr(self, "layer_idx", None),
                              "module": getattr(self, "layer_name", None), "buffers": before,
                              "cuda_buffer_bytes_before": copied, "offloaded": _metadata_cpu,
                              "metadata_segments_before_free": segments,
                              "allocated_after_free": torch.cuda.memory_allocated(),
                              "reserved_after_free": torch.cuda.memory_reserved(),
                              "inactive_split_after_free": torch.cuda.memory_stats().get("inactive_split_bytes.all.current")})
    (root / f"quantizer_metadata_lifetime_rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(_metadata_records, indent=2))
impl.GPTQPlus.free = _metadata_lifetime_free
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/quantizer-residency-prefix-manifests-20261010")
    args = parser.parse_args()
    folder = args.output
    folder.mkdir(exist_ok=False)
    data = (ROOT / "tools/realq_memory_profile_entry.py").read_text()
    needle = "original_fwrd = impl.gptq_fwrd"
    if data.count(needle) != 1:
        raise ValueError("entry injection point changed")
    entry = folder / "entry.py"
    entry.write_text(data.replace(needle, INJECTION + "\n" + needle))
    source = ROOT / "outputs/memory_mechanism_followup_20261010/hessian-cpu-layer0-manifests"
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            # Both arms use Hessian CPU; isolate metadata before composition.
            manifest = json.loads((source / f"candidate-{stage}.json").read_text())
            command = manifest["command"]
            command[command.index("baseline-profile") - 1] = str(entry)
            if side == "candidate" and stage == "stage2":
                command.append("--research_stage2_quantizer_metadata_cpu")
            manifest["scope_note"] = "Independent metadata-only ablation with shared Hessian CPU; prefix diagnostic, not full-model result or repaired comparison to original control."
            (folder / f"{side}-{stage}.json").write_text(json.dumps(manifest, indent=2))
    (folder / "entry-provenance.json").write_text(json.dumps({"sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
          "scope": "shared diagnostic metadata hashing after completed module; no timed speed claim"}, indent=2))
    print(folder)


if __name__ == "__main__":
    main()
