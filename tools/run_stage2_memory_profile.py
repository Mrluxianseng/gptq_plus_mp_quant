"""Run a provenance-recorded Stage-2 memory profile from a saved manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
TRACKED = (
    "process_args.py",
    "gptq_utils/gptq_plus_utils.py",
    "utils/model_utils.py",
    "utils/memory_utils.py",
    "utils/eval_utils.py",
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
    parser.add_argument("--stage2-cpu-master", action="store_true")
    parser.add_argument("--hinv-cpu-offload", action="store_true")
    parser.add_argument("--hessian-cpu-offload", action="store_true")
    parser.add_argument(
        "--sample-host-rss", action="store_true",
        help="sample aggregate RSS for the run process tree",
    )
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("tag may contain only letters, digits, and hyphens")
    if args.timeout_seconds <= 0 or args.telemetry_interval_ms <= 0:
        raise ValueError("timeout and telemetry interval must be positive")

    source_manifest = Path(args.manifest).resolve()
    source = json.loads(source_manifest.read_text())
    command = list(source["command"])
    exp_i = command.index("--exp") + 1
    command[exp_i] = f"phase_profile_{args.tag}"
    if "--enable_memory_profile" not in command:
        command.append("--enable_memory_profile")
    if args.stage2_cpu_master and "--stage2_cpu_master" not in command:
        command.append("--stage2_cpu_master")
    if args.hinv_cpu_offload and "--stage2_hinv_cpu_offload" not in command:
        command.append("--stage2_hinv_cpu_offload")
    if args.hessian_cpu_offload and "--stage2_hessian_cpu_offload" not in command:
        command.append("--stage2_hessian_cpu_offload")

    out = ROOT / "outputs" / f"phase_profile_{args.tag}"
    out.mkdir(parents=True, exist_ok=False)
    entry_i = next(
        i for i, value in enumerate(command)
        if Path(str(value)).name.endswith("entry.py")
    )
    entry_copy = out / "entry.py"
    entry_copy.write_bytes(Path(command[entry_i]).read_bytes())
    command[entry_i] = str(entry_copy)

    tracked = [ROOT / rel for rel in TRACKED] + [entry_copy]
    hashes = {str(path): sha256(path) for path in tracked}
    run_manifest = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "command": command,
        "source_sha256_at_launch": hashes,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "memory_profile_enabled": True,
        "stage2_cpu_master": args.stage2_cpu_master,
        "hinv_cpu_offload": args.hinv_cpu_offload,
        "hessian_cpu_offload": args.hessian_cpu_offload,
        "telemetry_interval_ms": args.telemetry_interval_ms,
        "timeout_seconds": args.timeout_seconds,
    }
    (out / "manifest.json").write_text(json.dumps(run_manifest, indent=2))
    telemetry_path = out / "nvml_telemetry.csv"
    telemetry_file = telemetry_path.open("w", encoding="utf-8")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
            f"--loop-ms={args.telemetry_interval_ms}",
        ],
        cwd=ROOT,
        stdout=telemetry_file,
        stderr=subprocess.STDOUT,
    )

    env = os.environ.copy()
    env.update(
        RESEARCH_RUN_DIR=str(out),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    started = time.monotonic()
    host_rss_path = out / "host_rss_telemetry.csv"
    host_rss_samples = 0
    max_host_rss_bytes = None
    psutil = None
    if args.sample_host_rss:
        try:
            import psutil as _psutil
        except ImportError as exc:
            raise RuntimeError("--sample-host-rss requires psutil") from exc
        psutil = _psutil
    try:
        with (out / "run.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [
                    "timeout", "--signal=TERM", "--kill-after=30s",
                    str(args.timeout_seconds), *command,
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if psutil is not None:
                with host_rss_path.open("w", newline="", encoding="utf-8") as rss_file:
                    rss_writer = csv.writer(rss_file)
                    rss_writer.writerow(
                        ["timestamp_utc", "root_pid", "process_count", "aggregate_rss_bytes"]
                    )
                    rss_file.flush()
                    while proc.poll() is None:
                        try:
                            root = psutil.Process(proc.pid)
                            tree = [root, *root.children(recursive=True)]
                            rss_bytes = 0
                            live_count = 0
                            for process in tree:
                                try:
                                    rss_bytes += process.memory_info().rss
                                    live_count += 1
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    continue
                            if live_count:
                                rss_writer.writerow(
                                    [datetime.now(timezone.utc).isoformat(), proc.pid,
                                     live_count, rss_bytes]
                                )
                                rss_file.flush()
                                host_rss_samples += 1
                                max_host_rss_bytes = max(
                                    max_host_rss_bytes or 0, rss_bytes
                                )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        time.sleep(min(args.telemetry_interval_ms / 1000.0, 0.5))
                    proc.wait()
            else:
                proc.wait()
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        telemetry_file.close()

    profile_paths = list(
        (ROOT / "outputs").rglob(
            f"cuda_memory_quantization_rank*_pid*.jsonl"
        )
    )
    profile_paths = [
        path for path in profile_paths
        if f"phase_profile_{args.tag}" in str(path)
    ]
    profile_rows = [
        json.loads(line)
        for path in profile_paths
        for line in path.read_text().splitlines()
    ]
    log_text = (out / "run.log").read_text(errors="replace")
    metric_match = re.search(r"OVERNIGHT_METRICS (\{.*\})", log_text)
    klppl_match = re.search(
        r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)", log_text
    )
    nvml_values = []
    with telemetry_path.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) > 1:
                try:
                    nvml_values.append(int(row[1].strip()))
                except ValueError:
                    continue
    unchanged = all(
        sha256(Path(path)) == digest for path, digest in hashes.items()
    )
    status = {
        "returncode": proc.returncode,
        "sources_unchanged": unchanged,
        "wall_seconds": time.monotonic() - started,
        "metrics": json.loads(metric_match.group(1)) if metric_match else None,
        "kl": float(klppl_match.group(1)) if klppl_match else None,
        "ppl": float(klppl_match.group(2)) if klppl_match else None,
        "memory_profile_files": [str(path.relative_to(ROOT)) for path in profile_paths],
        "max_sampled_allocated_bytes": max(
            (row.get("sampled_peak_allocated_bytes", 0) for row in profile_rows),
            default=None,
        ),
        "max_sampled_reserved_bytes": max(
            (row.get("sampled_peak_reserved_bytes", 0) for row in profile_rows),
            default=None,
        ),
        "max_nvml_memory_used_mib": max(nvml_values, default=None),
        "nvml_sample_count": len(nvml_values),
        "max_sampled_aggregate_host_rss_bytes": max_host_rss_bytes,
        "host_rss_sample_count": host_rss_samples,
        "finished": time.strftime("%F %T"),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if proc.returncode or not unchanged or status["metrics"] is None:
        raise RuntimeError("Stage-2 run failed or lacks complete metrics")


if __name__ == "__main__":
    main()
