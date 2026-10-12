"""Independent, fresh Stage-0 frozen-weight saved-value ablations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for model, source_name in (("qwen06b", "portable-memory-pilot-06b-20261010"),
                           ("qwen17b", "portable-memory-pilot-17b-2x512-20261010")):
    source = ROOT / "outputs" / source_name
    out = ROOT / "outputs" / f"frozen-weight-offload-{model}-manifests-20261010"
    out.mkdir(exist_ok=False)
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            data = json.loads((source / f"candidate-{stage}.json").read_text())
            command = data["command"]
            entry = next(i for i, value in enumerate(command) if Path(value).name.endswith("entry.py"))
            command[entry] = str(ROOT / "tools/realq_checkpoint_research_entry.py")
            if side == "candidate" and stage == "stage0":
                command.append("--research_stage0_frozen_weight_offload")
            data["scope_note"] = "Stage-0 frozen-parameter saved-value offload ablation on shared nonleaf offload+dedup; no complete quantization claim."
            (out / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
    print(out)
