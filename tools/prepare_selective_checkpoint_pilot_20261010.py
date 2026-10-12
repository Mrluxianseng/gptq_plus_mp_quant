"""Prepare an independent selective-attention/offload Stage-0 ablation.

Does not run CUDA. Only the research entry is changed; Stage-2 is unchanged.
Execute after the real-weight CUDA selective checkpoint preflight passes.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "outputs/portable-memory-pilot-06b-20261010"
out = ROOT / "outputs/selective-checkpoint-offload-stage0-manifests-20261010"
out.mkdir(exist_ok=False)
for side in ("control", "candidate"):
    for stage in ("stage0", "stage2"):
        data = json.loads((source / f"candidate-{stage}.json").read_text())
        command = data["command"]
        entry = next(i for i, value in enumerate(command) if Path(value).name.endswith("entry.py"))
        command[entry] = str(ROOT / "tools/realq_checkpoint_research_entry.py")
        if side == "candidate" and stage == "stage0":
            command.append("--research_stage0_attention_checkpoint")
        data["scope_note"] = "Stage-0-only selective-attention checkpoint ablation on shared offload+dedup; no complete quantization or throughput claim."
        (out / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
print(out)
