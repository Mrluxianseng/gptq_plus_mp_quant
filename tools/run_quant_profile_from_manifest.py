"""Run a short, provenance-checked REAL-Q/GPTQ+ phase profile from a saved run.

The source manifest provides the exact baseline command. This runner adds only
the built-in profile switches and an inclusive layer stop, stores all new
artifacts under a fresh output directory, and checks that source files did not
change during the run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from subprocess_log_stream import iter_output_chunks


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Saved baseline run manifest")
    parser.add_argument("--tag", required=True, help="Fresh output-directory suffix")
    parser.add_argument("--stop-layer", type=int, default=4,
                        help="Inclusive transformer layer stop; default profiles layers 0-4")
    parser.add_argument(
        "--nsys-cli",
        help="Optional Nsight Systems CLI path; captures CUDA/NVTX for this short run and writes report files into the run directory.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after quantization to keep the GPU trace focused on the selected quantization prefix.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--allocator-conf",
        default="expandable_segments:True",
        help=(
            "PyTorch CUDA caching-allocator configuration for the child process; "
            "defaults to the original REAL-Q profile setting."
        ),
    )
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("tag may contain only letters, digits, and hyphens")
    if args.stop_layer < 0 or args.timeout_seconds <= 0:
        raise ValueError("stop-layer must be non-negative and timeout positive")

    manifest_path = Path(args.manifest).resolve()
    source = json.loads(manifest_path.read_text())
    command = list(source["command"])
    if not command:
        raise ValueError("manifest has no command")

    tracked = source.get("source_sha256", source.get("hashes", {}))
    checked_sources: dict[Path, str] = {}
    for raw_path, expected in tracked.items():
        p = Path(raw_path)
        if not p.is_absolute():
            p = ROOT / p
        p = p.resolve()
        if p.is_file():
            actual = sha256(p)
            if actual != expected:
                raise RuntimeError(f"source hash mismatch before run: {p}")
            checked_sources[p] = actual

    exp_idx = command.index("--exp") + 1
    command[exp_idx] = f"phase_profile_{args.tag}"
    command.extend([
        "--enable_quant_profile",
        "--quant_profile_target_layers",
        ",".join(str(i) for i in range(args.stop_layer + 1)),
        "--quant_stop_layer", str(args.stop_layer),
    ])
    if args.skip_eval:
        command.append("--skip_eval")

    out = ROOT / "outputs" / f"phase_profile_{args.tag}"
    out.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).resolve()
    entry_index = next(i for i, value in enumerate(command)
                       if str(value).endswith("/entry.py") or str(value).endswith("\\entry.py"))
    original_entry = Path(command[entry_index])
    entry_text = original_entry.read_text()
    needle = "impl.gptq_fwrd=measured_fwrd\nsys.argv[0]='ptq.py'"
    replacement = (
        "impl.gptq_fwrd=measured_fwrd\n"
        "_original_dump_wall_summary = impl.QuantProfileRecorder.dump_wall_summary\n"
        "impl.QuantProfileRecorder.dump_wall_summary = classmethod("
        "lambda cls, top_k=60: _original_dump_wall_summary(top_k=1000))\n"
        "import threading as _memory_sample_threading\n"
        "import csv as _memory_sample_csv\n"
        "import atexit as _memory_sample_atexit\n"
        "_memory_sample_rank = int(os.environ.get('RANK', '0') or 0)\n"
        "_memory_sample_local_rank = int(os.environ.get('LOCAL_RANK', '0') or 0)\n"
        "_memory_sample_path = run_dir / f'cuda_allocator_telemetry_rank{_memory_sample_rank}.csv'\n"
        "_memory_sample_values = []\n"
        "_memory_sample_stop = _memory_sample_threading.Event()\n"
        "def _memory_sample_loop():\n"
        "    while not _memory_sample_stop.is_set():\n"
        "        _now = time.monotonic()\n"
        "        try:\n"
        "            if torch.cuda.is_initialized():\n"
        "                _allocated = torch.cuda.memory_allocated(_memory_sample_local_rank)\n"
        "                _reserved = torch.cuda.memory_reserved(_memory_sample_local_rank)\n"
        "                _memory_sample_values.append((_now, int(_allocated), int(_reserved)))\n"
        "        except Exception:\n"
        "            pass\n"
        "        _memory_sample_stop.wait(1.0)\n"
        "def _memory_sample_flush():\n"
        "    _memory_sample_stop.set()\n"
        "    if _memory_sample_thread.is_alive():\n"
        "        _memory_sample_thread.join(timeout=3.0)\n"
        "    with _memory_sample_path.open('w', newline='', encoding='utf-8') as _stream:\n"
        "        _writer = _memory_sample_csv.writer(_stream)\n"
        "        _writer.writerow(['monotonic_seconds', 'allocated_bytes', 'reserved_bytes'])\n"
        "        _writer.writerows(_memory_sample_values)\n"
        "_memory_sample_thread = _memory_sample_threading.Thread(target=_memory_sample_loop, daemon=True)\n"
        "_memory_sample_thread.start()\n"
        "_memory_sample_atexit.register(_memory_sample_flush)\n"
        "sys.argv[0]='ptq.py'"
    )
    if entry_text.count(needle) != 1:
        raise RuntimeError("baseline entry wrapper did not match the expected profiling hook")
    profile_entry = out / "entry.py"
    profile_entry.write_text(entry_text.replace(needle, replacement))
    command[entry_index] = str(profile_entry)
    profile_manifest = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "command": command,
        "source_sha256": {str(p): h for p, h in checked_sources.items()},
        "runner_sha256": sha256(runner),
        "profile_entry_sha256": sha256(profile_entry),
        "stop_layer_inclusive": args.stop_layer,
        "nsys_cli": args.nsys_cli,
        "skip_eval": args.skip_eval,
        "note": "Original baseline settings retained; profile instrumentation enabled; evaluation settings inherited from the saved command.",
    }
    (out / "manifest.json").write_text(json.dumps(profile_manifest, indent=2))

    env = os.environ.copy()
    env.update(
        RESEARCH_RUN_DIR=str(out),
        GPTQ_PLUS_WALL_PROFILE="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_ALLOC_CONF=args.allocator_conf,
    )
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    started = time.monotonic()
    launch = [
        "timeout", "--signal=TERM", "--kill-after=30s",
        str(args.timeout_seconds),
    ]
    if args.nsys_cli:
        nsys_path = Path(args.nsys_cli)
        if not nsys_path.is_file():
            raise FileNotFoundError(f"Nsight Systems CLI not found: {nsys_path}")
        launch.extend([
            str(nsys_path), "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--stats=false",
            "--export=sqlite",
            "--force-overwrite=true",
            f"--output={out / 'nsys-kernel-trace'}",
        ])
    launch.extend(command)
    with (out / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            launch,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for chunk in iter_output_chunks(proc.stdout):
            log.write(chunk)
            log.flush()
            print(chunk, end="", flush=True)
        proc.stdout.close()
        proc.wait()
    elapsed = time.monotonic() - started
    unchanged = all(sha256(p) == h for p, h in checked_sources.items())
    metrics_path = out / "baseline-profile_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    if metrics is not None:
        allocator_rows_by_rank: dict[str, list[tuple[int, int]]] = {}
        for telemetry_path in sorted(out.glob("cuda_allocator_telemetry_rank*.csv")):
            rank_match = re.search(r"rank(\d+)\.csv$", telemetry_path.name)
            if not rank_match:
                continue
            rank = rank_match.group(1)
            with telemetry_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    try:
                        allocated = int(row["allocated_bytes"])
                        reserved = int(row["reserved_bytes"])
                    except (KeyError, ValueError):
                        continue
                    if allocated >= 0 and reserved >= 0:
                        allocator_rows_by_rank.setdefault(rank, []).append((allocated, reserved))
        if allocator_rows_by_rank:
            rank_summaries = {}
            for rank, values in sorted(allocator_rows_by_rank.items(), key=lambda item: int(item[0])):
                allocated_values = [value[0] for value in values]
                reserved_values = [value[1] for value in values]
                rank_summaries[rank] = {
                    "samples": len(values),
                    "mean_allocated_bytes": statistics.mean(allocated_values),
                    "mean_reserved_bytes": statistics.mean(reserved_values),
                    "max_sampled_allocated_bytes": max(allocated_values),
                    "max_sampled_reserved_bytes": max(reserved_values),
                }
            metrics["cuda_allocator_memory_telemetry"] = {
                "sample_interval_seconds": 1.0,
                "per_rank": rank_summaries,
                "mean_allocated_bytes_across_ranks": statistics.mean(
                    row["mean_allocated_bytes"] for row in rank_summaries.values()
                ),
                "mean_reserved_bytes_across_ranks": statistics.mean(
                    row["mean_reserved_bytes"] for row in rank_summaries.values()
                ),
                "sample_count_total": sum(row["samples"] for row in rank_summaries.values()),
            }
        else:
            metrics["cuda_allocator_memory_telemetry"] = {
                "sample_interval_seconds": 1.0,
                "per_rank": {},
                "sample_count_total": 0,
            }
    log_text = (out / "run.log").read_text(errors="replace")
    klppl = re.search(r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)", log_text)
    status = {
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "sources_unchanged": unchanged,
        "metrics": metrics,
        "partial_prefix_kl": float(klppl.group(1)) if klppl else None,
        "partial_prefix_ppl": float(klppl.group(2)) if klppl else None,
        "profile_summary_present": "Wall-clock section summary" in log_text,
        "finished": time.strftime("%F %T"),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if proc.returncode or not unchanged or metrics is None or not status["profile_summary_present"]:
        raise RuntimeError("profile run failed, source changed, or profile summary is missing")


if __name__ == "__main__":
    main()
