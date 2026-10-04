"""Run the paper-scale, paired REAL-Q PyTorch/Triton benchmark.

The validated configuration is intentionally fixed; only the local model path,
repeat count, run tag, Python executable, and timeout are configurable.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

from subprocess_log_stream import iter_output_chunks


ROOT = Path(__file__).resolve().parents[1]
PAIR_RUNNER = ROOT / "tools" / "run_triton_gptq_reproducibility.py"
WORLD_SIZE = 4


ENTRY_TEMPLATE = '''"""Minimal metrics wrapper for the formal paired benchmark."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from gptq_utils import gptq_plus_utils as impl

mode = sys.argv.pop(1)
assert mode == "baseline-profile", mode
run_dir = Path(os.environ["RESEARCH_RUN_DIR"])
run_dir.mkdir(parents=True, exist_ok=True)
original_fwrd = impl.gptq_fwrd

def measured_fwrd(args, analyzer, dataloader, dev):
    torch.cuda.synchronize()
    started = time.monotonic()
    result = original_fwrd(args, analyzer, dataloader, dev)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        is_main = torch.distributed.get_rank() == 0
    else:
        rank = 0
        world_size = 1
        is_main = True
    local_stats = torch.tensor(
        [elapsed, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        device=torch.cuda.current_device(), dtype=torch.float64,
    )
    if distributed:
        gathered_stats = [torch.empty_like(local_stats) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_stats, local_stats)
    else:
        gathered_stats = [local_stats]
    stats_by_rank = [[float(value) for value in row.tolist()] for row in gathered_stats]
    elapsed_max = max(row[0] for row in stats_by_rank)
    peak_allocated_max = max(row[1] for row in stats_by_rank)
    peak_reserved_max = max(row[2] for row in stats_by_rank)
    digest = hashlib.sha256()
    tensor_count = 0
    for name, tensor in sorted(analyzer.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            .numpy().tobytes()
        )
        tensor_count += 1
    state_hash = digest.hexdigest()
    rank_hashes = [state_hash]
    if distributed:
        rank_hashes = [None] * world_size
        torch.distributed.all_gather_object(rank_hashes, state_hash)
        if len(set(rank_hashes)) != 1:
            raise RuntimeError(f"quantized state differs across ranks: {rank_hashes}")
    if is_main:
        row = {
            "mode": mode,
            "world_size": world_size,
            "quantization_seconds": elapsed_max,
            "quantization_seconds_by_rank": [item[0] for item in stats_by_rank],
            "state_sha256": state_hash,
            "state_sha256_by_rank": rank_hashes,
            "state_tensors": tensor_count,
            "peak_allocated": int(peak_allocated_max),
            "peak_reserved": int(peak_reserved_max),
            "peak_allocated_by_rank": [int(item[1]) for item in stats_by_rank],
            "peak_reserved_by_rank": [int(item[2]) for item in stats_by_rank],
        }
        (run_dir / "baseline-profile_metrics.json").write_text(
            json.dumps(row, indent=2)
        )
    if distributed:
        torch.distributed.barrier()
    return result

impl.gptq_fwrd=measured_fwrd
sys.argv[0]='ptq.py'
runpy.run_path("ptq.py", run_name="__main__")
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str], timeout: int = 20) -> str | None:
    try:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def benchmark_command(python: str, model_path: Path, entry: Path, exp: str,
                      kernel: str) -> list[str]:
    # Match the completed Qwen3-0.6B W4A16 formal campaign protocol.
    args = [
        "baseline-profile", "--exp", exp,
        "--model", str(model_path),
        "--dataset", "wikitext2", "--nsamples", "2048", "--seq_len", "2048",
        "--w_method", "gptq_plus", "--w_bits", "4", "--w_clip",
        "--pre_clip_search_impl",
        ("cartesian_legacy" if kernel == "torch" else "symmetric_union_exact"),
        "--w_groupsize", "-1", "--num_groups", "4", "--blocksize", "128",
        "--act_order", "--rotate", "--rotation_seed", "0", "--refresh_seed", "0",
        "--kl_topk", "-1", "--bsz", "128", "--final_layer_stats_bsz", "16",
        "--hessian_accum_bsz", "128",
        "--alpha", "0.0", "--enable_gptq_plus", "0",
        "--backward_samples", "32", "--backward_bsz", "32",
        "--final_layer_backward_bsz", "32", "--refresh_mb", "2",
        "--g_update_mode", "block_gd",
        "--grad_lr", "3e-4", "--grad_optimizer", "adam", "--grad_lr_layer_schedule", "cosine", "--grad_lr_layer_base_ratio", "0.01",
        "--grad_refresh_loss", "fisher_diag_mse", "--global_loss", "--loss_slide_window",
        "--global_loss_bsz", "32", "--static_fisher_microbatch_bsz", "8",
        "--grad_clip", "5e-5", "--a_loss_ratio", "0.95",
        "--final_layer_grad_clip", "5e-4", "--final_layer_grad_lr", "1e-5",
        "--group_parallel_quant", "rank", "--eval_seq_len", "2048",
        "--eval_datasets", "wikitext2", "--seed", "1",
        "--gptq_inner_kernel", kernel,
    ]
    if kernel == "triton_fused":
        # These two opt-in memory changes passed the local three-pair exactness
        # campaign; keep control at the unmodified REAL-Q execution path.
        args.extend([
            "--static_fisher_activation_checkpointing",
            "--offload_unused_runtime_modules",
        ])
    return [
        python, "-u", "-m", "torch.distributed.run", "--nnodes=1",
        f"--nproc_per_node={WORLD_SIZE}", "--standalone", str(entry), *args,
    ]


def write_manifests(campaign_dir: Path, model_path: Path, python: str,
                    tag_prefix: str) -> tuple[Path, Path]:
    entry = campaign_dir / "entry.py"
    entry.write_text(ENTRY_TEMPLATE, encoding="utf-8")
    paths = []
    for arm, kernel in (("control", "torch"), ("candidate", "triton_fused")):
        command = benchmark_command(
            python, model_path, entry, f"{tag_prefix}-{arm}", kernel
        )
        manifest = {
            "command": command,
            "source_sha256": {},
            "note": (
                "Generated by tools/run_triton_paper_benchmark.py; paper-aligned "
                "full-model REAL-Q W4A16 benchmark. The candidate additionally uses "
                "Stage-0 block activation checkpointing and lazy runtime-module residency; "
                "the control retains the baseline memory path. Source hashes are refreshed "
                "and checked by the paired profile runner."
            ),
        }
        path = campaign_dir / f"{arm}-base-manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1]


def parse_gpu_indices(raw: str) -> list[str]:
    indices = [part.strip() for part in raw.split(",") if part.strip()]
    if len(indices) != WORLD_SIZE or any(not part.isdigit() for part in indices):
        raise ValueError(
            f"expected exactly {WORLD_SIZE} comma-separated physical GPU indices, got {raw!r}"
        )
    if len(set(indices)) != WORLD_SIZE:
        raise ValueError(f"GPU indices must be distinct, got {raw!r}")
    return indices


def collect_environment(model_path: Path, python: str,
                        gpu_indices: list[str]) -> dict[str, Any]:
    probe = (
        "import json, torch; "
        "import importlib.util; "
        "spec=importlib.util.find_spec('triton'); "
        "triton_version='not-installed' if spec is None else __import__('triton').__version__; "
        "print(json.dumps({'torch_version':torch.__version__,"
        "'torch_cuda_version':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),"
        "'visible_cuda_devices':torch.cuda.device_count(),"
        "'torch_device_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],"
        "'triton_version':triton_version}))"
    )
    probe_env = os.environ.copy()
    probe_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_indices)
    probe_result = subprocess.run(
        [python, "-c", probe], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False, env=probe_env,
    )
    if probe_result.returncode:
        raise RuntimeError(
            f"could not inspect the selected Python environment ({python}):\n"
            f"{probe_result.stdout}"
        )
    runtime = json.loads(probe_result.stdout.strip().splitlines()[-1])

    try:
        git_commit = command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
        git_status = command_output(["git", "-C", str(ROOT), "status", "--short", "--branch"])
    except Exception:  # pragma: no cover - git may be unavailable on copied source
        git_commit, git_status = None, None

    gpu_query = command_output([
        "nvidia-smi", "-i", ",".join(gpu_indices),
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    compute_apps = command_output([
        "nvidia-smi", "-i", ",".join(gpu_indices),
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    provenance_names = {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json",
    }
    model_artifacts = []
    for artifact in sorted(model_path.iterdir()):
        if not artifact.is_file():
            continue
        if artifact.name in provenance_names or artifact.name.endswith(
            (".safetensors", ".safetensors.index.json", ".bin", ".bin.index.json")
        ):
            model_artifacts.append({
                "name": artifact.name,
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            })
    return {
        "timestamp_local": dt.datetime.now().astimezone().isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_executable": str(Path(python).resolve()),
        "python_version": sys.version,
        **runtime,
        "nvidia_smi_gpu_query": gpu_query,
        "nvidia_smi_compute_apps_preflight": compute_apps,
        "nvidia_smi_indices": gpu_indices,
        "model_path": str(model_path),
        "model_config_sha256": sha256(model_path / "config.json"),
        "model_artifacts": model_artifacts,
        "benchmark_script_sha256": sha256(Path(__file__).resolve()),
        "paired_runner_sha256": sha256(PAIR_RUNNER),
        "phase_profile_runner_sha256": sha256(ROOT / "tools" / "run_quant_profile_from_manifest.py"),
        "git_commit": git_commit,
        "git_status_short": git_status,
    }


class GpuMonitor:
    def __init__(self, out_dir: Path, gpu_indices: list[str], interval: float) -> None:
        self.out_dir = out_dir
        self.gpu_indices = gpu_indices
        self.interval = interval
        self.active_run_path = out_dir / "active_run.json"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.telemetry_path = out_dir / "gpu_telemetry.csv"
        self.process_path = out_dir / "gpu_processes.log"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 5)

    def _run(self) -> None:
        fields = (
            "utilization.gpu,utilization.memory,memory.used,memory.total,"
            "clocks.gr,power.draw,temperature.gpu"
        )
        header = "timestamp,run_tag,gpu_index,util_gpu_pct,util_mem_pct,memory_used_mib,memory_total_mib,"
        header += "graphics_clock_mhz,power_w,temp_c\n"
        self.telemetry_path.write_text(header, encoding="utf-8")
        self.process_path.write_text(
            "timestamp,pid,process_name,used_gpu_memory_mib\n", encoding="utf-8"
        )
        last_process_sample = 0.0
        while not self.stop_event.is_set():
            now = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
            try:
                active = json.loads(self.active_run_path.read_text(encoding="utf-8"))
                run_tag = active.get("tag") or "idle"
            except (OSError, json.JSONDecodeError):
                run_tag = "unknown"
            row = command_output([
                "nvidia-smi", "-i", ",".join(self.gpu_indices),
                f"--query-gpu=index,{fields}", "--format=csv,noheader,nounits",
            ])
            if row:
                with self.telemetry_path.open("a", encoding="utf-8") as stream:
                    for gpu_row in row.splitlines():
                        stream.write(f"{now},{run_tag},{gpu_row}\n")
                        print(
                            f"[gpu-telemetry] timestamp={now} run={run_tag} {gpu_row}",
                            flush=True,
                        )
            monotonic_now = time.monotonic()
            if monotonic_now - last_process_sample >= 30:
                apps = command_output([
                    "nvidia-smi", "-i", ",".join(self.gpu_indices),
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ])
                with self.process_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"[{now}]\n")
                    stream.write((apps or "<no compute applications reported>") + "\n")
                last_process_sample = monotonic_now
            self.stop_event.wait(self.interval)


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(campaign_dir: Path, tag_prefix: str, pairs: list[dict[str, Any]],
              environment: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    records_path = campaign_dir / "paired_repetitions.json"
    records = json.loads(records_path.read_text()) if records_path.is_file() else pairs
    rows = []
    integrity_ok = len(records) == len(pairs)
    sampled_memory_by_run: dict[str, dict[str, float]] = {}
    telemetry_path = campaign_dir / "gpu_telemetry.csv"
    if telemetry_path.is_file():
        with telemetry_path.open(newline="", encoding="utf-8") as stream:
            for sample in csv.DictReader(stream):
                try:
                    tag = sample["run_tag"].strip()
                    gpu = sample["gpu_index"].strip()
                    used_mib = float(sample["memory_used_mib"].strip())
                except (KeyError, ValueError, AttributeError):
                    continue
                if tag in ("", "idle", "unknown"):
                    continue
                gpu_peaks = sampled_memory_by_run.setdefault(tag, {})
                gpu_peaks[gpu] = max(gpu_peaks.get(gpu, 0.0), used_mib)
    for index, pair in enumerate(records, start=1):
        control, candidate = pair["control"], pair["candidate"]
        control_memory = sampled_memory_by_run.get(control["tag"], {})
        candidate_memory = sampled_memory_by_run.get(candidate["tag"], {})
        comparable_gpus = sorted(set(control_memory) & set(candidate_memory))
        sampled_peak_lower_on_every_gpu = (
            bool(comparable_gpus)
            and all(candidate_memory[gpu] < control_memory[gpu] for gpu in comparable_gpus)
        )
        allocated_peak_lower = (
            candidate["peak_allocated_bytes"] < control["peak_allocated_bytes"]
        )
        state_and_eval_exact = all(control[key] == candidate[key] for key in (
            "state_sha256", "state_tensors", "kl", "ppl"
        ))
        traces = pair.get("layer_output_fingerprints") or {}
        control_trace = traces.get(control["tag"])
        candidate_trace = traces.get(candidate["tag"])
        layer_outputs_exact = (
            control_trace is not None
            and candidate_trace is not None
            and control_trace == candidate_trace
            and [row.get("layer") for row in control_trace] == list(range(28))
        )
        exact = state_and_eval_exact and layer_outputs_exact
        integrity_ok &= exact
        rows.append({
            "pair": index,
            "control_quantization_seconds": control["quantization_seconds"],
            "candidate_quantization_seconds": candidate["quantization_seconds"],
            "quantization_speedup": control["quantization_seconds"] / candidate["quantization_seconds"],
            "control_end_to_end_seconds": control["wall_seconds"],
            "candidate_end_to_end_seconds": candidate["wall_seconds"],
            "end_to_end_speedup": control["wall_seconds"] / candidate["wall_seconds"],
            "state_sha256": control["state_sha256"],
            "state_tensors": control["state_tensors"],
            "kl": control["kl"],
            "ppl": control["ppl"],
            "peak_allocated_bytes_control": control["peak_allocated_bytes"],
            "peak_allocated_bytes_candidate": candidate["peak_allocated_bytes"],
            "peak_reserved_bytes_control": control["peak_reserved_bytes"],
            "peak_reserved_bytes_candidate": candidate["peak_reserved_bytes"],
            "sampled_gpu_peak_memory_mib_control": control_memory,
            "sampled_gpu_peak_memory_mib_candidate": candidate_memory,
            "sampled_gpu_peak_lower_on_every_comparable_gpu": sampled_peak_lower_on_every_gpu,
            "torch_peak_allocated_lower": allocated_peak_lower,
            "candidate_peak_below_control_on_sampled_and_torch_metrics": (
                sampled_peak_lower_on_every_gpu and allocated_peak_lower
            ),
            "exact_output_match": exact,
            "layer_outputs_exact_match": layer_outputs_exact,
            "per_layer_quantization_ms_control": control["per_layer_quantization_ms"],
            "per_layer_quantization_ms_candidate": candidate["per_layer_quantization_ms"],
            "gptq_inner_column_compensation_ms_control": control["gptq_inner_column_compensation_ms"],
            "gptq_inner_column_compensation_ms_candidate": candidate["gptq_inner_column_compensation_ms"],
            "gptq_compensation_path_ms_control": control["gptq_compensation_path_ms"],
            "gptq_compensation_path_ms_candidate": candidate["gptq_compensation_path_ms"],
            "gradient_update_total_ms_control": control["gradient_update_total_ms"],
            "gradient_update_total_ms_candidate": candidate["gradient_update_total_ms"],
            "profile_control_ms": control["profile_section_totals_ms"],
            "profile_candidate_ms": candidate["profile_section_totals_ms"],
        })
    quant_speedups = [row["quantization_speedup"] for row in rows]
    wall_speedups = [row["end_to_end_speedup"] for row in rows]
    telemetry: dict[str, Any] = {"samples": 0}
    if telemetry_path.is_file():
        values: dict[str, list[float]] = {
            "util_gpu_pct": [], "util_mem_pct": [], "memory_used_mib": [],
            "temp_c": [], "graphics_clock_mhz": [], "power_w": [],
        }
        with telemetry_path.open(newline="", encoding="utf-8") as stream:
            for sample in csv.DictReader(stream):
                telemetry["samples"] += 1
                for key, collection in values.items():
                    try:
                        collection.append(float(sample[key].strip()))
                    except (KeyError, ValueError, AttributeError):
                        pass
        for key, collection in values.items():
            if collection:
                telemetry[key] = {
                    "mean": statistics.mean(collection),
                    "median": statistics.median(collection),
                    "min": min(collection),
                    "max": max(collection),
                }
    summary = {
        "campaign": tag_prefix,
        "completed_at_local": dt.datetime.now().astimezone().isoformat(),
        "valid": integrity_ok and len(rows) == config["pairs"],
        "candidate_memory_peaks_below_control": (
            len(rows) == config["pairs"]
            and all(row["candidate_peak_below_control_on_sampled_and_torch_metrics"] for row in rows)
        ),
        "completed_pairs": len(rows),
        "configuration": config,
        "environment": environment,
        "gpu_telemetry_summary": telemetry,
        "pair_results": rows,
        "quantization_speedup": mean_std(quant_speedups) if rows else None,
        "end_to_end_speedup": mean_std(wall_speedups) if rows else None,
    }
    (campaign_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        f"# REAL-Q Triton formal benchmark: {tag_prefix}", "",
        f"- Valid: **{summary['valid']}**; completed pairs: {len(rows)}/{config['pairs']}",
        f"- Candidate whole-run memory peaks below control on all GPUs: **{summary['candidate_memory_peaks_below_control']}**",
        f"- Host/GPU: {environment['host']} / {environment['nvidia_smi_gpu_query'] or environment['torch_device_names']}",
        f"- Commit: `{environment['git_commit']}`",
        f"- Model: `{config['model_path']}`; Qwen3-0.6B W4A16, 28 layers, "
        "Wikitext-2 calibration (2048 x 2048 tokens), symmetric per-row weights", "",
        f"- Distributed execution: {config['world_size']} ranks on GPUs "
        f"{','.join(config['gpu_indices'])}", "",
        "| Pair | Control quant (s) | Triton quant (s) | Quant speedup | Control total (s) | Triton total (s) | Total speedup | State/layers/KL/PPL exact |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['pair']} | {row['control_quantization_seconds']:.2f} | "
            f"{row['candidate_quantization_seconds']:.2f} | {row['quantization_speedup']:.3f}× | "
            f"{row['control_end_to_end_seconds']:.2f} | {row['candidate_end_to_end_seconds']:.2f} | "
            f"{row['end_to_end_speedup']:.3f}× | {row['exact_output_match']} |"
        )
    if rows:
        quant_sd = summary["quantization_speedup"]["sample_std"]
        wall_sd = summary["end_to_end_speedup"]["sample_std"]
        report.extend([
            "", f"- Quantization speedup median: **{summary['quantization_speedup']['median']:.3f}×** "
            f"(mean {summary['quantization_speedup']['mean']:.3f}×, sample SD "
            f"{quant_sd:.3f}×)." if quant_sd is not None else
            f"(mean {summary['quantization_speedup']['mean']:.3f}×; one pair, SD unavailable).",
            f"- End-to-end speedup median: **{summary['end_to_end_speedup']['median']:.3f}×** "
            f"(mean {summary['end_to_end_speedup']['mean']:.3f}×, sample SD "
            f"{wall_sd:.3f}×)." if wall_sd is not None else
            f"(mean {summary['end_to_end_speedup']['mean']:.3f}×; one pair, SD unavailable).",
            "- Exactness is checked after each pair using full state SHA-256 and dynamically recorded tensor count, KL, and PPL; the driver stops on the first mismatch.",
            "- GPU memory is sampled once per second and tagged by control/candidate run. The recorded-peak memory check is separate from `valid`; passing it does not prove memory is lower at every instant or at phase-aligned peaks. PyTorch whole-run peak allocated/reserved counters are also recorded.",
            "- Raw paired data: `paired_repetitions.json`; per-run manifests, logs, profile metrics, and source hashes: `outputs/phase_profile_<tag>/`; tagged GPU samples: `gpu_telemetry.csv`.",
        ])
        report.extend([
            "",
            "Per-pair JSON also records all 28 per-layer GPU times, GPTQ inner-column and outer "
            "compensation-path sections, the true-gradient-refresh plus Adam-application time, "
            "and the raw section breakdown. "
            "Distributed section totals use the slowest rank. Layer activations are compared by "
            "SHA-256 fingerprints (shape/dtype/forward-count included); full activation tensors are not stored.",
        ])
    if telemetry.get("util_gpu_pct"):
        temp_range = telemetry.get("temp_c")
        temp_text = (
            f"temperature range {temp_range['min']:.0f}–{temp_range['max']:.0f} °C."
            if temp_range else "temperature unavailable."
        )
        report.append(
            f"- GPU telemetry ({telemetry['samples']} per-GPU samples across "
            f"{config['world_size']} GPUs): mean utilization "
            f"{telemetry['util_gpu_pct']['mean']:.1f}%, peak "
            f"{telemetry['util_gpu_pct']['max']:.0f}%; {temp_text}"
        )
    (campaign_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("REALQ_MODEL_PATH"),
                        help="Local Qwen3-0.6B model directory (or REALQ_MODEL_PATH)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable from the project virtual environment")
    parser.add_argument("--pairs", type=int, default=3,
                        help="Number of matched control/Triton pairs (default: 3)")
    parser.add_argument("--tag-prefix", default=None,
                        help="Unique artifact tag; generated from local time by default")
    parser.add_argument("--timeout-seconds", type=int, default=14400,
                        help="Timeout for each arm, default four hours")
    parser.add_argument(
        "--gpu-indices",
        default=os.environ.get("NVIDIA_SMI_INDICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")),
        help="Exactly four distinct physical GPU indices, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--monitor-interval", type=float, default=1.0,
        help="GPU memory/utilization sampling interval in seconds (default: 1).",
    )
    parser.add_argument("--allow-gpu-contention", action="store_true",
                        help="Proceed even when a CUDA compute process is visible at preflight")
    args = parser.parse_args()
    if args.pairs <= 0 or args.timeout_seconds <= 0 or args.monitor_interval <= 0:
        parser.error("pairs, timeout, and monitor interval must be positive")
    if not args.model_path:
        parser.error("provide --model-path or set REALQ_MODEL_PATH")
    try:
        gpu_indices = parse_gpu_indices(args.gpu_indices)
    except ValueError as error:
        parser.error(str(error))
    # Apply the same explicit four-device mask to environment probes and every
    # torchrun child; --gpu-indices selects execution as well as telemetry.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_indices)

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        parser.error(f"model path must be a local Hugging Face model directory with config.json: {model_path}")
    try:
        model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"could not read model config.json: {error}")
    if not str(model_config.get("model_type", "")).lower().startswith("qwen3"):
        parser.error(f"expected Qwen3 model_type, got {model_config.get('model_type')!r}")
    if model_config.get("num_hidden_layers") != 28:
        parser.error(
            "this validated script is for the 28-layer Qwen3-0.6B configuration; "
            f"found num_hidden_layers={model_config.get('num_hidden_layers')!r}"
        )
    if not list(model_path.glob("*.safetensors")) and not list(model_path.glob("*.bin")):
        parser.error(f"no local .safetensors or .bin model weights found in {model_path}")
    dataset_path = ROOT / "datasets" / "wikitext"
    if not dataset_path.is_dir():
        parser.error(
            f"Wikitext dataset directory is missing: {dataset_path}. "
            "Place/symlink the local dataset corpus at datasets/wikitext before running."
        )
    python = shutil.which(args.python) or args.python
    tag_prefix = args.tag_prefix or dt.datetime.now().astimezone().strftime("triton-paper-%Y%m%d-%H%M%S")
    if not tag_prefix.replace("-", "").isalnum():
        parser.error("tag-prefix may contain only letters, digits, and hyphens")
    campaign_dir = ROOT / "outputs" / f"{tag_prefix}-campaign"
    if campaign_dir.exists():
        parser.error(f"campaign output already exists; choose a new --tag-prefix: {campaign_dir}")
    try:
        environment = collect_environment(model_path, python, gpu_indices)
    except RuntimeError as error:
        parser.error(str(error))
    if not environment["cuda_available"]:
        parser.error("PyTorch cannot access CUDA; activate the intended project virtual environment")
    if environment["triton_version"] == "not-installed":
        parser.error("Triton is not installed in the selected Python environment")
    if environment["visible_cuda_devices"] != WORLD_SIZE:
        parser.error(
            f"the benchmark requires {WORLD_SIZE} visible CUDA devices, "
            f"but PyTorch sees {environment['visible_cuda_devices']}"
        )
    if len((environment["nvidia_smi_gpu_query"] or "").splitlines()) != WORLD_SIZE:
        parser.error(
            f"nvidia-smi did not report {WORLD_SIZE} selected physical GPUs: "
            f"{environment['nvidia_smi_gpu_query']}"
        )
    visible_apps = environment["nvidia_smi_compute_apps_preflight"]
    has_compute_app = bool(visible_apps) and any(
        line.strip().split(",", 1)[0].strip().isdigit()
        for line in visible_apps.splitlines()
    )
    if has_compute_app and not args.allow_gpu_contention:
        parser.error(
            "CUDA compute process visible before benchmark. Stop it or rerun with "
            f"--allow-gpu-contention after confirming contention is acceptable. Details: {visible_apps}"
        )

    campaign_dir.mkdir(parents=True)

    config = {
        "model_path": str(model_path),
        "model_family": "Qwen3-0.6B",
        "world_size": WORLD_SIZE,
        "gpu_indices": gpu_indices,
        "model_type": model_config.get("model_type"),
        "num_hidden_layers": model_config.get("num_hidden_layers"),
        "layers_quantized": "0-27 inclusive",
        "weight_bits": 4,
        "dataset": "wikitext2",
        "dataset_path": str(dataset_path.resolve()),
        "calibration_samples": 2048,
        "calibration_sequence_length": 2048,
        "evaluation_samples": "all (CLI default 0)",
        "evaluation_sequence_length": 2048,
        "seed": 1,
        "rotation_seed": 0,
        "refresh_seed": 0,
        "gptq_blocksize": 128,
        "weight_group_size": "per-row",
        "groups": 4,
        "calibration_forward_batch_size": 128,
        "hessian_accumulation_batch_size": 128,
        "final_layer_stats_batch_size": 16,
        "global_loss_batch_size": 32,
        "backward_samples_per_refresh": 32,
        "backward_batch_size": 32,
        "refresh_microbatch_size_per_rank": 2,
        "group_parallel_quant": "rank",
        "quantization_work_sharing": "output rows sharded across ranks; rank-local Hessian batches",
        "act_order": True,
        "rotation": True,
        "loss_slide_window": True,
        "update_mode": "block_gd",
        "gradient_refresh_loss": "fisher_diag_mse",
        "gradient_learning_rate": 3e-4,
        "final_layer_gradient_learning_rate": 1e-5,
        "gradient_layer_schedule": "reverse-cosine",
        "gradient_layer_base_ratio": 0.01,
        "activation_loss_clip_percentile": 0.95,
        "control_kernel": "torch",
        "candidate_kernel": "triton_fused",
        "candidate_memory_optimizations": [
            "static_fisher_activation_checkpointing",
            "offload_unused_runtime_modules",
        ],
        "control_memory_optimizations": [],
        "preclip_search": {
            "control": "cartesian_legacy",
            "candidate": "symmetric_union_exact",
            "candidate_set_and_tie_order": "exactly preserved",
        },
        "gptq_quantizer_search": "upstream defaults from origin/zq",
        "pairs": args.pairs,
        "timeout_seconds_per_arm": args.timeout_seconds,
        "model_config_sha256": environment["model_config_sha256"],
    }
    (campaign_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (campaign_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    control_manifest, candidate_manifest = write_manifests(
        campaign_dir, model_path, python, tag_prefix
    )
    monitor = GpuMonitor(campaign_dir, gpu_indices, args.monitor_interval)
    monitor.start()

    runner_command = [
        python, str(PAIR_RUNNER),
        "--control-manifest", str(control_manifest),
        "--candidate-manifest", str(candidate_manifest),
        "--pairs", str(args.pairs),
        "--tag-prefix", tag_prefix,
        "--timeout-seconds", str(args.timeout_seconds),
        "--trace-layer-outputs",
    ]
    returncode = 1
    try:
        print("=== Effective experiment configuration ===", flush=True)
        print("CONFIG=" + json.dumps(config, sort_keys=True), flush=True)
        print("ENVIRONMENT=" + json.dumps(environment, sort_keys=True), flush=True)
        print("CONTROL_MANIFEST=" + str(control_manifest), flush=True)
        print("CANDIDATE_MANIFEST=" + str(candidate_manifest), flush=True)
        print(
            "CONTROL_COMMAND=" + json.dumps(
                json.loads(control_manifest.read_text(encoding="utf-8"))["command"]
            ),
            flush=True,
        )
        print(
            "CANDIDATE_COMMAND=" + json.dumps(
                json.loads(candidate_manifest.read_text(encoding="utf-8"))["command"]
            ),
            flush=True,
        )
        with (campaign_dir / "campaign.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                runner_command, cwd=ROOT, env=os.environ.copy(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for chunk in iter_output_chunks(process.stdout):
                print(chunk, end="", flush=True)
                log.write(chunk)
                log.flush()
            process.stdout.close()
            returncode = process.wait()
    finally:
        monitor.stop()

    pairs_path = campaign_dir / "paired_repetitions.json"
    pairs = json.loads(pairs_path.read_text()) if pairs_path.is_file() else []
    summary = summarize(campaign_dir, tag_prefix, pairs, environment, config)
    campaign_status = {
        "returncode": returncode,
        "valid": summary["valid"] and returncode == 0,
        "finished_local": dt.datetime.now().astimezone().isoformat(),
        "runner_command": runner_command,
    }
    (campaign_dir / "status.json").write_text(json.dumps(campaign_status, indent=2), encoding="utf-8")
    print(f"\nCampaign artifacts: {campaign_dir}", flush=True)
    print(f"Summary: {campaign_dir / 'summary.md'}", flush=True)
    if returncode != 0 or not summary["valid"]:
        raise SystemExit("Benchmark failed or exactness checks did not pass; inspect campaign logs/artifacts.")


if __name__ == "__main__":
    main()
