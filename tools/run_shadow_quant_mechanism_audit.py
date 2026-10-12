"""Small real-trajectory audit of next-column-block shadow quantization."""
import hashlib
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/d/gptq_plus_realq")
TEMPLATE = ROOT / "outputs/quant_preview_smoke32_20260928/control/manifest.json"
PROBE = ROOT / "tools/overnight_ptq_probe.py"
SOURCES = (
    "gptq_utils/gptq_plus_utils.py",
    "process_args.py",
    "ptq.py",
    "utils/data_utils.py",
    "tools/overnight_ptq_probe.py",
    "tools/run_shadow_quant_mechanism_audit.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="0:self_attn.q_proj:0,1")
    parser.add_argument("--tag", default="qproj")
    parser.add_argument("--stop-layer", type=int, default=1)
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    parser.add_argument("--adam-aware", action="store_true")
    args = parser.parse_args()
    out = ROOT / f"outputs/shadow_quant_mechanism_audit_{args.tag}_20260929"
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    template = json.loads(TEMPLATE.read_text())
    command = list(template["command"])
    entry_index = next(i for i, value in enumerate(command) if value.endswith("/entry.py"))
    entry = out / "entry.py"
    shutil.copyfile(PROBE, entry)
    command[entry_index] = str(entry)
    command[entry_index + 1] = "baseline-profile"
    command[command.index("--exp") + 1] = f"shadow_quant_mechanism_audit_{args.tag}_20260929"
    if "--quant_stop_layer" in command:
        stop_index = command.index("--quant_stop_layer")
        del command[stop_index:stop_index + 2]
    if not args.full_model:
        command.extend(["--quant_stop_layer", str(args.stop_layer)])
    if args.adam_aware:
        command.extend([
            "--preview_next_quant_block",
            "--preview_next_quant_block_adam_aware",
        ])
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    hashes[str(entry)] = digest(entry)
    (out / "manifest.json").write_text(json.dumps({
        "command": command,
        "source_sha256": hashes,
        "calibration_seed": 44,
        "calibration_samples": 32,
        "eval_samples": 8,
        "quant_stop_layer": None if args.full_model else args.stop_layer,
        "full_model": bool(args.full_model),
        "timeout_seconds": args.timeout_seconds,
        "shadow_audit_targets": args.target,
        "purpose": "Compare predicted next-block state to actual post-Adam GPTQ state; matched-sample gradients, module outputs, and Adam-preconditioned gradients using the same saved pre-refresh moment state.",
        "adam_aware_candidate": bool(args.adam_aware),
        "shadow_preview_enabled": bool(args.adam_aware),
    }, indent=2))

    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        RESEARCH_RUN_DIR=str(out),
        REALQ_SHADOW_AUDIT_DIR=str(out / "events"),
        REALQ_SHADOW_AUDIT_TARGET=args.target,
    )
    # The current audit hook intentionally handles one module selector per
    # quantizer invocation; down_proj is run in a second invocation after the
    # q_proj pass has proved the hook and paired-gradient path.
    started = time.monotonic()
    with (out / "run.log").open("w") as log:
        proc = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=30s", str(args.timeout_seconds), *command],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    log_text = (out / "run.log").read_text(errors="replace")
    match = re.search(r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)", log_text)
    unchanged = all(digest(ROOT / name) == value for name, value in hashes.items())
    metrics_path = out / "baseline-profile_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    status = {
        "returncode": proc.returncode,
        "wall_seconds": time.monotonic() - started,
        "sources_unchanged": unchanged,
        "audit_events": len(list((out / "events").glob("event_*.pt"))) if (out / "events").exists() else 0,
        "kl": float(match.group(1)) if match else None,
        "ppl": float(match.group(2)) if match else None,
        "quantization_seconds": metrics.get("quantization_seconds"),
        "peak_allocated": metrics.get("peak_allocated"),
        "peak_reserved": metrics.get("peak_reserved"),
        "finished": time.strftime("%F %T"),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if proc.returncode or not unchanged or not match:
        raise RuntimeError("Audit run failed, changed a source, or lacks final metrics")


if __name__ == "__main__":
    main()
