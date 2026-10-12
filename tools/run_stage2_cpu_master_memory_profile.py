"""Run a provenance-recorded full Stage-2 CPU-master memory experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
TRACKED = (
    "process_args.py",
    "gptq_utils/gptq_plus_utils.py",
    "utils/model_utils.py",
    "utils/memory_utils.py",
    "ptq.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Full Stage-2 run manifest")
    parser.add_argument("--tag", required=True, help="Unique run tag")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--telemetry-interval-ms", type=int, default=100)
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("tag may contain only letters, digits, and hyphens")
    if args.timeout_seconds <= 0 or args.telemetry_interval_ms <= 0:
        raise ValueError("timeout and telemetry interval must be positive")

    manifest_path = Path(args.manifest).resolve()
    source = json.loads(manifest_path.read_text())
    command = list(source["command"])
    exp_i = command.index("--exp") + 1
    command[exp_i] = f"phase_profile_{args.tag}"
    if "--enable_memory_profile" not in command:
        command.append("--enable_memory_profile")
    if "--stage2_cpu_master" not in command:
        command.append("--stage2_cpu_master")

    out = ROOT / "outputs" / f"phase_profile_{args.tag}"
    out.mkdir(parents=True, exist_ok=False)
    entry_i = next(
        i for i, value in enumerate(command)
        if Path(str(value)).name.endswith("entry.py")
    )
    entry_copy = out / "entry.py"
    entry_copy.write_bytes(Path(command[entry_i]).read_bytes())
    command[entry_i] = str(entry_copy)

    tracked = [ROOT / relative for relative in TRACKED] + [entry_copy]
    source_hashes = {str(path): sha256(path) for path in tracked}
    run_manifest = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "command": command,
        "source_sha256_at_launch": source_hashes,
        "stage2_cpu_master": True,
        "memory_profile_enabled": True,
        "telemetry_interval_ms": args.telemetry_interval_ms,
        "timeout_seconds": args.timeout_seconds,
        "purpose": "Measure full Stage-2 CPU-master memory and compare exact output to cached baseline.",
    }
    (out / "manifest.json").write_text(json.dumps(run_manifest, indent=2))
    (out / "nvml_telemetry.csv").write_text("", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        RESEARCH_RUN_DIR=str(out),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
            f"--loop-ms={args.telemetry_interval_ms}",
        ],
        cwd=ROOT,
        stdout=(out / "nvml_telemetry.csv").open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    started = time.monotonic()
    try:
        with (out / "run.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [
                    "timeout", "--signal=TERM", "--kill-after=30s",
                    str(args.timeout_seconds), *command,
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()

    elapsed = time.monotonic() - started
    unchanged = all(sha256(path) == digest for path, digest in zip(tracked, source_hashes.values()))
    log_text = (out / "run.log").read_text(errors="replace")
    metric_match = re.search(r"OVERNIGHT_METRICS (\{.*\})", log_text)
    klppl_match = re.search(
        r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)", log_text
    )
    status = {
        "returncode": proc.returncode,
        "sources_unchanged": unchanged,
        "wall_seconds": elapsed,
        "metrics": json.loads(metric_match.group(1)) if metric_match else None,
        "kl": float(klppl_match.group(1)) if klppl_match else None,
        "ppl": float(klppl_match.group(2)) if klppl_match else None,
        "nvml_telemetry": str(out / "nvml_telemetry.csv"),
        "finished": time.strftime("%F %T"),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if proc.returncode or not unchanged or status["metrics"] is None:
        raise RuntimeError("Stage-2 CPU-master run failed or lacks complete metrics")


if __name__ == "__main__":
    main()
