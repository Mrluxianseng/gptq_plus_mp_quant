"""Fresh prefix diagnostic of a Hinv staging-buffer allocation change."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_full_pipeline_memory_profile import set_option

root = Path(__file__).resolve().parents[1]
source = root / "outputs/portable-memory-pilot-17b-2x512-20261010"
out = root / "outputs/hinv-reuse-layer0-manifests-20261010"
out.mkdir(exist_ok=False)
for side in ("control", "candidate"):
    for stage in ("stage0", "stage2"):
        data = json.loads((source / f"{side}-{stage}.json").read_text())
        data["scope_note"] = "Fresh Stage-0 and layer0 prefix diagnostic, not complete 28-layer evidence. Tests the observed layer0 reserved regression against the original control."
        if stage == "stage2":
            set_option(data["command"], "--quant_stop_layer", "0")
            if side == "candidate":
                data["command"].append("--stage2_hinv_cpu_reuse_buffer")
        (out / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
print(out)
