"""Prepare an isolated Stage-0 recomputation experiment with identical entry sources."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "outputs/portable-memory-pilot-06b-20261010"
out = root / "outputs/checkpoint-stage0-manifests-20261010"
out.mkdir(exist_ok=False)
for side in ("control", "candidate"):
    for stage in ("stage0", "stage2"):
        data = json.loads((source / f"control-{stage}.json").read_text())
        data["scope_note"] = "Experimental actual FSDP Stage-0 only; no activation CPU offload; checkpoint marker consumed by the research entry. Not complete model validation."
        command = data["command"]
        command[command.index("baseline-profile")-1] = str(root/"tools/realq_checkpoint_research_entry.py")
        if side == "candidate" and stage == "stage0":
            command.append("--research_stage0_checkpoint")
        (out/f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
print(out)
