"""Profile a fresh Stage-0 precompute followed by Stage-2 quantization/eval.

This runner keeps a single NVML and host-process-tree sampling window around
both commands, so the reported maximum includes the full precompute-to-eval
workflow rather than only a run that reuses a static cache.
"""
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

PROCESS_PEAK_FOOTER = '''
# Report allocator peaks after the complete command, including final eval.
from utils.memory_utils import get_process_cuda_memory_peaks as _process_peaks
import importlib.metadata as _process_package_metadata
if torch.cuda.is_available():
    torch.cuda.synchronize()
    _allocated, _reserved = _process_peaks()
    _stage = "stage0" if "--exit_after_precompute" in sys.argv else "stage2"
    _rank = int(os.environ.get("RANK", "0"))
    _destination = Path(os.environ["RESEARCH_RUN_DIR"])
    _record = {"rank": _rank, "stage": _stage,
               "peak_allocated_bytes": _allocated, "peak_reserved_bytes": _reserved,
               "scope": "complete_process_through_evaluation"}
    (_destination / f"process_allocator_peak_{_stage}_rank{_rank}.json").write_text(json.dumps(_record, indent=2))
    _software = {"rank": _rank, "stage": _stage, "python": sys.version,
                 "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                 "packages": {name: _process_package_metadata.version(name) for name in ("transformers", "accelerate", "safetensors")}}
    (_destination / f"software_environment_{_stage}_rank{_rank}.json").write_text(json.dumps(_software, indent=2))
    _trace = _destination / f"layer_output_fingerprints_rank{_rank}.json"
    if _rank == 0 and _trace.exists():
        (_destination / "layer_output_fingerprints.json").write_bytes(_trace.read_bytes())
if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
'''

EVALUATION_PROFILE_SETUP = '''
# Profile final evaluation separately while retaining process-wide maxima.
from utils import eval_utils as _process_eval_utils
from utils.memory_utils import CudaPhaseMemoryProfiler as _EvalMemoryProfiler
_process_original_eval = _process_eval_utils.kl_ppl_eval
def _process_profiled_eval(args, analyzer, *eval_args, **eval_kwargs):
    _profile = _EvalMemoryProfiler(str(Path(args.output_dir) / "memory_profile"),
                                  torch.device("cuda", torch.cuda.current_device()),
                                  tag="evaluation")
    _profile.mark("final_kl_ppl_evaluation")
    try:
        _result = _process_original_eval(args, analyzer, *eval_args, **eval_kwargs)
    except BaseException:
        _profile.finish(status="failed")
        raise
    else:
        _profile.finish(status="completed")
        return _result
_process_eval_utils.kl_ppl_eval = _process_profiled_eval
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def option_value(command: list[str], option: str) -> int:
    try:
        return command.index(option) + 1
    except ValueError as exc:
        raise ValueError(f"manifest command is missing {option}") from exc


def set_option(command: list[str], option: str, value: str) -> None:
    try:
        command[option_value(command, option)] = value
    except ValueError:
        # Source manifests may omit optional paths such as --static_cache_path.
        # The full-pipeline runner owns the fresh cache location, so append it
        # when absent instead of requiring every source manifest to predeclare it.
        command.extend([option, value])


def run_stage(
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
    psutil,
    rss_writer,
) -> dict:
    start = time.monotonic()
    print(f"[pipeline] {name} starting; live worker log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["timeout", "--signal=TERM", "--kill-after=30s", str(timeout_seconds), *command],
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        next_progress = start + 30
        while proc.poll() is None:
            try:
                root = psutil.Process(proc.pid)
                tree = [root, *root.children(recursive=True)]
                rss = 0
                count = 0
                for child in tree:
                    try:
                        rss += child.memory_info().rss
                        count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                if count:
                    rss_writer.writerow(
                        [datetime.now(timezone.utc).isoformat(), name, proc.pid, count, rss]
                    )
                    rss_writer._file.flush()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if time.monotonic() >= next_progress:
                print(f"[pipeline] {name} running: {time.monotonic() - start:.1f}s; log={log_path}", flush=True)
                next_progress = time.monotonic() + 30
            time.sleep(0.25)
        print(f"[pipeline] {name} finished: returncode={proc.returncode}, wall={time.monotonic() - start:.2f}s", flush=True)
        return {"returncode": proc.returncode, "wall_seconds": time.monotonic() - start}


class FlushableWriter:
    def __init__(self, writer, file):
        self._writer = writer
        self._file = file

    def writerow(self, row):
        self._writer.writerow(row)


def profile_rows(exp_name: str, tag="quantization") -> list[dict]:
    rows = []
    # Model-specific output roots vary (Qwen3-0.6B, Qwen3-1.7B, etc.).
    # Search each model root by the unique experiment name instead of silently
    # omitting allocator profiles for models other than the original 1.7B run.
    for model_root in (ROOT / "outputs").iterdir():
        if not model_root.is_dir():
            continue
        profile_dir = model_root / exp_name / "memory_profile"
        if not profile_dir.is_dir():
            continue
        for path in profile_dir.glob(f"cuda_memory_{tag}_rank*_pid*.jsonl"):
            for line in path.read_text(errors="replace").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-manifest", required=True)
    parser.add_argument("--stage2-manifest", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--telemetry-interval-ms", type=int, default=100)
    parser.add_argument("--hinv-cpu-offload", action="store_true")
    parser.add_argument("--hinv-cpu-offload-modules", default="")
    parser.add_argument("--hessian-damping-inplace", action="store_true")
    parser.add_argument("--exact-phase-peaks", action="store_true", help="Diagnostic allocator interval peaks; preserves process maxima in the profiler.")
    parser.add_argument("--stage0-only", action="store_true", help="Precompute diagnostic only; does not claim quantization/evaluation completion.")
    parser.add_argument("--gpu-indices", default=None, help="Physical nvidia-smi indices to monitor; comma-separated. Does not set CUDA_VISIBLE_DEVICES.")
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("tag may contain only letters, digits, and hyphens")
    if args.timeout_seconds <= 0 or args.telemetry_interval_ms <= 0:
        raise ValueError("timeout and telemetry interval must be positive")

    stage0_source = Path(args.stage0_manifest).resolve()
    stage2_source = Path(args.stage2_manifest).resolve()
    stage0 = json.loads(stage0_source.read_text())
    stage2 = json.loads(stage2_source.read_text())
    s0 = list(stage0["command"])
    s2 = list(stage2["command"])
    def rank_count(command):
        for token in command:
            if token.startswith("--nproc_per_node="):
                return int(token.split("=", 1)[1])
        return int(command[option_value(command, "--nproc_per_node")])
    expected_ranks = {"stage0": rank_count(s0), "stage2": rank_count(s2)}
    if not args.stage0_only:
        # A fresh cache must be consumed with the calibration conditions that
        # produced it, rather than silently triggering another precompute.
        for option in ("--model", "--dataset", "--nsamples", "--seq_len", "--seed", "--num_groups"):
            if s0[option_value(s0, option)] != s2[option_value(s2, option)]:
                raise ValueError(f"Stage-0 and Stage-2 disagree on {option}")
        for option in ("--static_fisher_label_chunk_tokens",):
            values = [cmd[option_value(cmd, option)] if option in cmd else "0" for cmd in (s0, s2)]
            if values[0] != values[1]:
                raise ValueError(f"Stage-0 and Stage-2 disagree on {option}")

    # Both manifests originate from the same run configuration. Use the
    # current saved runner snapshot from Stage-2 for both commands.
    out = ROOT / "outputs" / f"full_pipeline_profile_{args.tag}"
    out.mkdir(parents=True, exist_ok=False)
    entry_copy = out / "entry.py"
    source_entry = Path(s2[next(i for i, v in enumerate(s2) if str(v).endswith("entry.py"))])
    entry_source = source_entry.read_text()
    entry_source = entry_source.replace("torch.cuda.max_memory_allocated()", "impl.memory_utils.get_process_cuda_memory_peaks()[0]")
    entry_source = entry_source.replace("torch.cuda.max_memory_reserved()", "impl.memory_utils.get_process_cuda_memory_peaks()[1]")
    metrics_log = "logging.info('OVERNIGHT_METRICS %s',json.dumps(row))"
    if metrics_log not in entry_source:
        raise ValueError("source entry must expose OVERNIGHT_METRICS row logging for rank-safe auditing")
    entry_source = entry_source.replace(metrics_log, "(root / f\"baseline-profile_metrics_rank{os.environ.get('RANK', '0')}.json\").write_text(json.dumps(row, indent=2))\n    " + metrics_log)
    entry_source = entry_source.replace('"layer_output_fingerprints.json"', 'f"layer_output_fingerprints_rank{os.environ.get(\'RANK\', \'0\')}.json"')
    run_calls = ("runpy.run_path('ptq.py', run_name='__main__')", 'runpy.run_path("ptq.py", run_name="__main__")')
    for run_call in run_calls:
        if run_call in entry_source:
            entry_source = entry_source.replace(run_call, EVALUATION_PROFILE_SETUP + "\ntry:\n    " + run_call + "\nexcept SystemExit as _stage_exit:\n    if _stage_exit.code not in (None, 0):\n        raise")
            break
    else:
        raise ValueError("source entry must contain a ptq.py runpy call")
    entry_copy.write_text(entry_source + "\n" + PROCESS_PEAK_FOOTER, encoding="utf-8")
    for command in (s0, s2):
        entry_i = next(i for i, value in enumerate(command) if str(value).endswith("entry.py"))
        command[entry_i] = str(entry_copy)

    s0[option_value(s0, "--exp")] = f"full_pipeline_{args.tag}_stage0"
    s2[option_value(s2, "--exp")] = f"full_pipeline_{args.tag}_stage2"
    cache_path = out / "static_cache"
    cache_path_str = str(cache_path)
    set_option(s0, "--static_cache_path", cache_path_str)
    set_option(s2, "--static_cache_path", cache_path_str)

    if "--fsdp_precompute" not in s0 or "--exit_after_precompute" not in s0:
        raise ValueError("Stage-0 manifest must be a FSDP precompute-only command")
    if "--fsdp_precompute" in s2 or "--exit_after_precompute" in s2:
        raise ValueError("Stage-2 manifest must perform quantization, not precompute-only")
    if args.hinv_cpu_offload and "--stage2_hinv_cpu_offload" not in s2:
        s2.append("--stage2_hinv_cpu_offload")
    if args.hinv_cpu_offload_modules:
        if "--stage2_hinv_cpu_offload_modules" in s2:
            set_option(s2, "--stage2_hinv_cpu_offload_modules", args.hinv_cpu_offload_modules)
        else:
            s2.extend(["--stage2_hinv_cpu_offload_modules", args.hinv_cpu_offload_modules])
    if args.hessian_damping_inplace and "--stage2_hessian_damping_inplace" not in s2:
        s2.append("--stage2_hessian_damping_inplace")
    if "--enable_memory_profile" not in s0:
        s0.append("--enable_memory_profile")
    if "--enable_memory_profile" not in s2:
        s2.append("--enable_memory_profile")

    tracked_paths = sorted(set([ROOT / path for path in TRACKED] + [entry_copy, Path(__file__).resolve()] + [path for folder in ("gptq_utils", "utils") for path in (ROOT / folder).rglob("*.py") if "__pycache__" not in path.parts]))
    source_hashes = {str(path): sha256(path) for path in tracked_paths}
    run_manifest = {
        "tag": args.tag,
        "stage0_source_manifest": str(stage0_source),
        "stage0_source_manifest_sha256": sha256(stage0_source),
        "stage2_source_manifest": str(stage2_source),
        "stage2_source_manifest_sha256": sha256(stage2_source),
        "stage0_command": s0,
        "stage2_command": s2,
        "fresh_static_cache": cache_path_str,
        "hinv_cpu_offload": args.hinv_cpu_offload,
        "hinv_cpu_offload_modules": args.hinv_cpu_offload_modules,
        "hessian_damping_inplace": args.hessian_damping_inplace,
        "source_sha256_at_launch": source_hashes,
        "timeout_seconds_per_stage": args.timeout_seconds,
        "telemetry_interval_ms": args.telemetry_interval_ms,
        "exact_phase_peaks": args.exact_phase_peaks,
        "stage0_only": args.stage0_only,
        "gpu_indices": args.gpu_indices,
        "expected_ranks": expected_ranks,
        "runtime_environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "NCCL_SHM_DISABLE", "REALQ_HESSIAN_SYMMETRIZE_IMPL")},
    }
    (out / "manifest.json").write_text(json.dumps(run_manifest, indent=2))

    env = os.environ.copy()
    env.update(
        RESEARCH_RUN_DIR=str(out),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        REALQ_PHASE_PEAK_MODE="exact" if args.exact_phase_peaks else "sampled",
    )
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("full pipeline memory profile requires psutil") from exc

    nvml_path = out / "nvml_telemetry.csv"
    rss_path = out / "host_rss_telemetry.csv"
    with nvml_path.open("w", encoding="utf-8") as nvml_file, rss_path.open(
        "w", newline="", encoding="utf-8"
    ) as rss_file:
        rss_writer = FlushableWriter(
            csv.writer(rss_file), rss_file
        )
        csv.writer(rss_file).writerow(
            ["timestamp_utc", "stage", "root_pid", "process_count", "aggregate_rss_bytes"]
        )
        rss_file.flush()
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,index,uuid",
                "--format=csv,noheader,nounits",
                f"--loop-ms={args.telemetry_interval_ms}",
                *([f"--id={args.gpu_indices}"] if args.gpu_indices else []),
            ],
            cwd=ROOT,
            stdout=nvml_file,
            stderr=subprocess.STDOUT,
        )
        started = time.monotonic()
        try:
            stage0_result = run_stage(
                "stage0", s0, ROOT, env, out / "stage0.log", args.timeout_seconds,
                psutil, rss_writer,
            )
            if args.stage0_only:
                stage2_result = {"returncode": None, "skipped": "Stage-0-only diagnostic"}
            elif stage0_result["returncode"] != 0:
                stage2_result = {"returncode": None, "skipped": "Stage-0 failed"}
            else:
                if not cache_path.exists() or not any(cache_path.rglob("*.pt")):
                    raise RuntimeError(f"Stage-0 exited successfully but cache is empty: {cache_path}")
                stage2_result = run_stage(
                    "stage2", s2, ROOT, env, out / "stage2.log", args.timeout_seconds,
                    psutil, rss_writer,
                )
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()

    nvml_mib = []
    with nvml_path.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) > 1:
                try:
                    nvml_mib.append(int(row[1].strip()))
                except ValueError:
                    continue
    rss_values = []
    with rss_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                rss_values.append(int(row["aggregate_rss_bytes"]))
            except (KeyError, ValueError):
                continue

    stage2_exp = f"full_pipeline_{args.tag}_stage2"
    stage2_log = (out / "stage2.log").read_text(errors="replace") if (out / "stage2.log").exists() else ""
    metrics_match = re.search(r"OVERNIGHT_METRICS (\{.*\})", stage2_log)
    klppl_match = re.search(
        r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)", stage2_log
    )
    stage0_exp = f"full_pipeline_{args.tag}_stage0"
    stage0_rows = profile_rows(stage0_exp)
    stage2_rows = profile_rows(stage2_exp)
    evaluation_rows = profile_rows(stage2_exp, tag="evaluation")
    unchanged = all(sha256(Path(path)) == digest for path, digest in source_hashes.items())
    process_peaks = {stage: [json.loads(path.read_text()) for path in sorted(out.glob(f"process_allocator_peak_{stage}_rank*.json"))] for stage in ("stage0", "stage2")}
    metrics_by_rank = [json.loads(path.read_text()) for path in sorted(out.glob("baseline-profile_metrics_rank*.json"))] if not args.stage0_only else []
    states_match = bool(metrics_by_rank) and len({(row.get("state_sha256"), row.get("state_tensors")) for row in metrics_by_rank}) == 1
    fingerprint_paths = sorted(out.glob("layer_output_fingerprints_rank*.json"))
    fingerprints_match = bool(fingerprint_paths) and len({sha256(path) for path in fingerprint_paths}) == 1
    status = {
        "stage0": stage0_result,
        "stage2": stage2_result,
        "returncode": stage0_result["returncode"] if args.stage0_only or stage0_result["returncode"] != 0 else stage2_result.get("returncode"),
        "scope": "stage0_only" if args.stage0_only else "full_pipeline",
        "state_metrics_by_rank": metrics_by_rank,
        "state_hashes_match_across_ranks": states_match if not args.stage0_only else None,
        "layer_fingerprints_match_across_ranks": fingerprints_match if not args.stage0_only else None,
        "expected_ranks": expected_ranks,
        "sources_unchanged": unchanged,
        "wall_seconds": time.monotonic() - started,
        "metrics": json.loads(metrics_match.group(1)) if metrics_match else None,
        "kl": float(klppl_match.group(1)) if klppl_match else None,
        "ppl": float(klppl_match.group(2)) if klppl_match else None,
        "max_nvml_memory_used_mib": max(nvml_mib, default=None),
        "nvml_sample_count": len(nvml_mib),
        "max_sampled_aggregate_host_rss_bytes": max(rss_values, default=None),
        "host_rss_sample_count": len(rss_values),
        "stage2_allocator_peak_allocated_bytes": max(
            (row.get("cumulative_peak_allocated_bytes", 0) for row in stage2_rows), default=None
        ),
        "stage2_allocator_peak_reserved_bytes": max(
            (row.get("cumulative_peak_reserved_bytes", 0) for row in stage2_rows), default=None
        ),
        "stage0_allocator_peak_allocated_bytes": max(
            (row.get("cumulative_peak_allocated_bytes", 0) for row in stage0_rows), default=None
        ),
        "stage0_allocator_peak_reserved_bytes": max(
            (row.get("cumulative_peak_reserved_bytes", 0) for row in stage0_rows), default=None
        ),
        "allocator_peak_allocated_bytes_any_stage": max(
            (row.get("cumulative_peak_allocated_bytes", 0) for row in [*stage0_rows, *stage2_rows]), default=None
        ),
        "allocator_peak_reserved_bytes_any_stage": max(
            (row.get("cumulative_peak_reserved_bytes", 0) for row in [*stage0_rows, *stage2_rows]), default=None
        ),
        "memory_profile_files": sorted(
            str(path.relative_to(ROOT))
            for exp_name in (stage0_exp, stage2_exp)
            for model_root in (ROOT / "outputs").iterdir()
            if model_root.is_dir()
            for path in (model_root / exp_name / "memory_profile").glob("*.jsonl")
        ),
        "evaluation_allocator_peak_allocated_bytes": max((row.get("sampled_peak_allocated_bytes", 0) for row in evaluation_rows if "phase" in row), default=None),
        "evaluation_allocator_peak_reserved_bytes": max((row.get("sampled_peak_reserved_bytes", 0) for row in evaluation_rows if "phase" in row), default=None),
        "finished": time.strftime("%F %T"),
    }
    for stage in ("stage0", "stage2"):
        status[f"{stage}_process_allocator_peaks_by_rank"] = process_peaks[stage]
        for metric in ("allocated", "reserved"):
            status[f"{stage}_process_peak_{metric}_bytes"] = max((row[f"peak_{metric}_bytes"] for row in process_peaks[stage]), default=None)
    for metric in ("allocated", "reserved"):
        status[f"full_process_peak_{metric}_bytes"] = max((row[f"peak_{metric}_bytes"] for rows in process_peaks.values() for row in rows), default=None)
    status["valid"] = not (
        status["returncode"] != 0
        or not unchanged
        or (not args.stage0_only and (status["metrics"] is None or status["kl"] is None or status["ppl"] is None))
        or not process_peaks["stage0"]
        or (not args.stage0_only and not process_peaks["stage2"])
        or (not args.stage0_only and (not states_match or not fingerprints_match))
        or len(process_peaks["stage0"]) != expected_ranks["stage0"]
        or (not args.stage0_only and (len(process_peaks["stage2"]) != expected_ranks["stage2"] or len(metrics_by_rank) != expected_ranks["stage2"] or len(fingerprint_paths) != expected_ranks["stage2"]))
    )
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2), flush=True)
    if not status["valid"]:
        raise RuntimeError("full-pipeline run failed or lacks complete metrics")


if __name__ == "__main__":
    main()
