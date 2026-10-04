"""Run matched full-model REAL-Q torch/Triton pairs with stop-on-mismatch checks."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_quant_profile_from_manifest.py"
TRACKED = (
    "gptq_utils/gptq_plus_utils.py",
    "process_args.py",
    "ptq.py",
    "utils/data_utils.py",
    "tools/overnight_ptq_probe.py",
    "tools/run_shadow_quant_mechanism_audit.py",
    "gptq_utils/triton_gptq_kernels.py",
    "utils/reproducibility.py",
)


class GpuTelemetryMonitor:
    """Sample device and compute-process memory, tagged to the active A/B arm."""

    def __init__(self, campaign_dir: Path, gpu_index: str, interval: float = 1.0):
        self.campaign_dir = campaign_dir
        self.gpu_index = gpu_index
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.telemetry_path = campaign_dir / "gpu_telemetry.csv"
        self.active_run_path = campaign_dir / "active_run.json"

    @staticmethod
    def _query(command: list[str]) -> str:
        try:
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    def start(self) -> None:
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path.write_text(
            "timestamp,active_tag,gpu_index,memory_used_mib,memory_total_mib,"
            "utilization_gpu_pct,compute_process_memory_mib,compute_process_count\n",
            encoding="utf-8",
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 6)

    def summary_for(self, tag: str) -> dict:
        rows = []
        with self.telemetry_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["active_tag"] != tag:
                    continue
                parsed = {}
                for key in (
                    "memory_used_mib", "memory_total_mib",
                    "utilization_gpu_pct", "compute_process_memory_mib",
                ):
                    try:
                        parsed[key] = float(row[key])
                    except (KeyError, ValueError):
                        parsed[key] = None
                if parsed["memory_used_mib"] is not None:
                    rows.append(parsed)
        if not rows:
            return {"samples": 0}

        def stats(key: str) -> dict:
            values = [row[key] for row in rows if row[key] is not None]
            if not values:
                return {"mean": None, "median": None, "max": None, "min": None}
            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "max": max(values),
                "min": min(values),
            }

        return {
            "samples": len(rows),
            "sample_interval_seconds": self.interval,
            "gpu_index": self.gpu_index,
            "device_memory_used_mib": stats("memory_used_mib"),
            "compute_process_memory_mib": stats("compute_process_memory_mib"),
            "gpu_utilization_pct": stats("utilization_gpu_pct"),
        }

    def _run(self) -> None:
        while not self.stop_event.is_set():
            now = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
            try:
                active = json.loads(self.active_run_path.read_text()).get("tag")
            except (OSError, json.JSONDecodeError):
                active = None
            gpu = self._query([
                "nvidia-smi", "-i", self.gpu_index,
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            process_rows = self._query([
                "nvidia-smi", "-i", self.gpu_index,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ])
            process_mem = 0.0
            process_count = 0
            for line in process_rows.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 2:
                    try:
                        process_mem += float(parts[1])
                        process_count += 1
                    except ValueError:
                        pass
            if gpu:
                with self.telemetry_path.open("a", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    for line in gpu.splitlines():
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) != 3:
                            continue
                        writer.writerow([
                            now, active or "", self.gpu_index, *parts,
                            process_mem if process_count else "", process_count,
                        ])
            self.stop_event.wait(self.interval)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_manifest(
    source: Path,
    destination: Path,
    trace_layer_outputs: bool = False,
) -> None:
    manifest = json.loads(source.read_text())
    command = manifest["command"]
    if trace_layer_outputs:
        entry_index = next(
            i for i, value in enumerate(command)
            if str(value).endswith("/entry.py") or str(value).endswith("\\entry.py")
        )
        original_entry = Path(command[entry_index])
        entry_text = original_entry.read_text()
        needle = "sys.argv[0]='ptq.py'"
        trace_marker = "_layer_trace_eval_utils"
        if trace_marker not in entry_text:
            if entry_text.count(needle) != 1:
                raise RuntimeError(f"unexpected entry wrapper layout: {original_entry}")
            trace_code = '''
from utils import eval_utils as _layer_trace_eval_utils
import hashlib as _layer_trace_hashlib
import json as _layer_trace_json
from pathlib import Path as _layer_trace_Path

_layer_trace_original_eval = _layer_trace_eval_utils.kl_ppl_eval
def _layer_trace_eval(args, analyzer, *eval_args, **eval_kwargs):
    model = analyzer.model
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None) if body is not None else None
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise RuntimeError("could not locate transformer layers for output fingerprinting")
    records = [{"digest": _layer_trace_hashlib.sha256(), "count": 0, "shapes": []}
               for _ in layers]
    handles = []
    def _hook(index):
        def _capture(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor):
                return
            value = value.detach().contiguous()
            row = records[index]
            row["digest"].update(str((tuple(value.shape), str(value.dtype))).encode())
            row["digest"].update(value.view(torch.uint8).cpu().numpy().tobytes())
            row["count"] += 1
            row["shapes"].append(list(value.shape))
        return _capture
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_hook(index)))
    try:
        return _layer_trace_original_eval(args, analyzer, *eval_args, **eval_kwargs)
    finally:
        for handle in handles:
            handle.remove()
        root = _layer_trace_Path(os.environ.get("RESEARCH_RUN_DIR", "outputs"))
        root.mkdir(parents=True, exist_ok=True)
        report = [{"layer": i, "sha256": row["digest"].hexdigest(),
                   "forward_count": row["count"], "shapes": row["shapes"]}
                  for i, row in enumerate(records)]
        (root / "layer_output_fingerprints.json").write_text(
            _layer_trace_json.dumps(report, indent=2))
_layer_trace_eval_utils.kl_ppl_eval = _layer_trace_eval
'''
            entry_text = entry_text.replace(needle, needle + "\n" + trace_code)
        trace_entry = destination.parent / destination.stem / "entry.py"
        trace_entry.parent.mkdir(parents=True, exist_ok=True)
        # Keep the two-line profiling hook contiguous: the downstream profile
        # runner replaces that exact marker when creating its instrumented entry.
        trace_entry.write_text(entry_text)
        command[entry_index] = str(trace_entry)
        manifest["source_sha256"][str(trace_entry)] = sha256(trace_entry)
    manifest["source_sha256"] = {
        **{
            rel: sha256(ROOT / rel) for rel in TRACKED if (ROOT / rel).is_file()
        },
        **{
            key: value for key, value in manifest.get("source_sha256", {}).items()
            if Path(key).is_file() and destination.parent in Path(key).parents
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2))


def run(manifest: Path, tag: str, timeout: int) -> dict:
    command = [
        sys.executable,
        str(RUNNER),
        "--manifest",
        str(manifest),
        "--tag",
        tag,
        "--stop-layer",
        "27",
        "--timeout-seconds",
        str(timeout),
    ]
    # The control/candidate exactness gate must not compare different random
    # atomic-reduction outcomes from memory-efficient SDPA backward.  Keep the
    # calibration-gradient path deterministic for both children.
    child_env = os.environ.copy()
    child_env["REALQ_DETERMINISTIC_SDPA"] = "1"
    subprocess.run(command, cwd=ROOT, env=child_env, check=True)
    status_path = ROOT / "outputs" / f"phase_profile_{tag}" / "status.json"
    status = json.loads(status_path.read_text())
    if status["returncode"] != 0 or not status["sources_unchanged"]:
        raise RuntimeError(f"failed run or source changed during {tag}: {status}")
    metrics = status["metrics"]
    required_metrics = (
        "quantization_seconds",
        "state_sha256",
        "state_tensors",
        "peak_allocated",
        "peak_reserved",
        "cuda_allocator_memory_telemetry",
    )
    missing_metrics = [key for key in required_metrics if not metrics or key not in metrics]
    if missing_metrics or int(metrics.get("state_tensors", 0)) <= 0:
        raise RuntimeError(
            f"incomplete quantization metrics for {tag}: "
            f"missing={missing_metrics}, metrics={metrics}"
        )
    log_path = ROOT / "outputs" / f"phase_profile_{tag}" / "run.log"
    tables = log_path.read_text(errors="replace").split("Wall-clock section summary")
    if len(tables) < 2:
        raise RuntimeError(f"missing detailed wall-clock section table: {log_path}")
    rank_sections = []
    import re
    row_pattern = re.compile(
        r"\s*(\S+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+%)\s+([0-9.]+)\s*$"
    )
    for table in tables[1:]:
        per_rank = {}
        for line in table.splitlines():
            match = row_pattern.match(line)
            if match:
                per_rank[match.group(1)] = float(match.group(3))
        if per_rank:
            rank_sections.append(per_rank)
    if not rank_sections:
        raise RuntimeError(f"could not parse detailed wall-clock section table: {log_path}")

    # torchrun can concatenate one profile table per rank. Sum repeated work
    # within each rank, then use the slowest rank as the distributed critical
    # path; summing the rank tables would incorrectly multiply timings by 4.
    def max_rank_total(suffix):
        return max(
            sum(value for name, value in sections.items() if name.endswith(suffix))
            for sections in rank_sections
        )

    section_suffixes = (
        "fasterquant.block.inner_column_loop",
        "fasterquant.block.writeback_inner",
        "fasterquant.block.outer_update_delta_w",
        "fasterquant.block.outer_update_ghinv",
        "fasterquant.block.true_gradient_refresh",
        "fasterquant.block.outer_update_grad_descent",
    )
    section_totals_ms = {
        suffix: max_rank_total(suffix) for suffix in section_suffixes
    }
    closed_form_ms = section_totals_ms["fasterquant.block.inner_column_loop"]
    # The outer delta-W phase is shared with block_gd's gradient correction,
    # so expose its pieces as well as the aggregate instead of claiming the
    # aggregate is a mathematically pure closed-form solve.
    compensation_path_ms = sum(section_totals_ms[key] for key in (
        "fasterquant.block.inner_column_loop",
        "fasterquant.block.writeback_inner",
        "fasterquant.block.outer_update_delta_w",
        "fasterquant.block.outer_update_ghinv",
    ))
    gradient_update_ms = (
        section_totals_ms["fasterquant.block.true_gradient_refresh"]
        + section_totals_ms["fasterquant.block.outer_update_grad_descent"]
    )
    per_layer_quantization_ms = {}
    for sections in rank_sections:
        for name, value in sections.items():
            match = re.fullmatch(r"layers\.(\d+)\.layer\.total", name)
            if match:
                layer = int(match.group(1))
                per_layer_quantization_ms[str(layer)] = max(
                    per_layer_quantization_ms.get(str(layer), 0.0), value
                )
    if set(per_layer_quantization_ms) != {str(i) for i in range(28)}:
        raise RuntimeError(
            "incomplete per-layer quantization timings; expected layers 0-27, got "
            f"{sorted(per_layer_quantization_ms, key=int)} in {log_path}"
        )
    return {
        "tag": tag,
        "wall_seconds": status["wall_seconds"],
        "quantization_seconds": metrics["quantization_seconds"],
        "state_sha256": metrics["state_sha256"],
        "state_tensors": metrics["state_tensors"],
        "peak_allocated_bytes": metrics["peak_allocated"],
        "peak_reserved_bytes": metrics["peak_reserved"],
        "mean_allocated_bytes": metrics["cuda_allocator_memory_telemetry"].get(
            "mean_allocated_bytes_across_ranks"
        ),
        "mean_reserved_bytes": metrics["cuda_allocator_memory_telemetry"].get(
            "mean_reserved_bytes_across_ranks"
        ),
        "allocator_memory_sample_count": metrics["cuda_allocator_memory_telemetry"].get(
            "sample_count_total", 0
        ),
        "kl": status["partial_prefix_kl"],
        "ppl": status["partial_prefix_ppl"],
        "profile_section_totals_ms": section_totals_ms,
        "per_layer_quantization_ms": per_layer_quantization_ms,
        "gptq_inner_column_compensation_ms": closed_form_ms,
        "gptq_compensation_path_ms": compensation_path_ms,
        "gradient_update_total_ms": gradient_update_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--tag-prefix", default="triton-final")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--trace-layer-outputs",
        action="store_true",
        help="fingerprint every Transformer layer output during final evaluation",
    )
    parser.add_argument(
        "--monitor-gpu-index",
        default="0",
        help="nvidia-smi GPU index used for per-arm memory telemetry (default: 0)",
    )
    parser.add_argument(
        "--monitor-interval-seconds",
        type=float,
        default=1.0,
        help="sampling interval for per-arm GPU memory telemetry",
    )
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")

    campaign_dir = ROOT / "outputs" / f"{args.tag_prefix}-campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    control_manifest = campaign_dir / "control-current-source.json"
    candidate_manifest = campaign_dir / "candidate-current-source.json"
    refresh_manifest(
        Path(args.control_manifest).resolve(), control_manifest,
        trace_layer_outputs=args.trace_layer_outputs,
    )
    refresh_manifest(
        Path(args.candidate_manifest).resolve(),
        candidate_manifest,
        args.trace_layer_outputs,
    )

    results = []
    active_run_path = campaign_dir / "active_run.json"
    monitor = GpuTelemetryMonitor(
        campaign_dir, args.monitor_gpu_index, args.monitor_interval_seconds
    )

    def set_active_run(tag: str | None) -> None:
        payload = {"tag": tag}
        temporary = active_run_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, active_run_path)

    monitor.start()
    try:
        for i in range(1, 1 + args.pairs):
            control_tag = f"{args.tag_prefix}-r{i}-control"
            candidate_tag = f"{args.tag_prefix}-r{i}-triton"
            set_active_run(control_tag)
            try:
                control = run(control_manifest, control_tag, args.timeout_seconds)
                control["gpu_telemetry"] = monitor.summary_for(control_tag)
            finally:
                set_active_run(None)
            set_active_run(candidate_tag)
            try:
                candidate = run(
                    candidate_manifest,
                    candidate_tag,
                    args.timeout_seconds,
                )
                candidate["gpu_telemetry"] = monitor.summary_for(candidate_tag)
            finally:
                set_active_run(None)
            check_keys = ("state_sha256", "state_tensors", "kl", "ppl")
            mismatches = {
                key: (control[key], candidate[key])
                for key in check_keys
                if control[key] != candidate[key]
            }
            if mismatches:
                raise RuntimeError(f"control/Triton mismatch; stopped campaign: {mismatches}")
            peak_memory_ok = (
                candidate["peak_allocated_bytes"] <= control["peak_allocated_bytes"]
                and candidate["peak_reserved_bytes"] <= control["peak_reserved_bytes"]
            )
            if not peak_memory_ok:
                raise RuntimeError(
                    "candidate PyTorch allocator peak exceeded control: "
                    f"allocated {candidate['peak_allocated_bytes']} > {control['peak_allocated_bytes']} or "
                    f"reserved {candidate['peak_reserved_bytes']} > {control['peak_reserved_bytes']}"
                )
            if args.trace_layer_outputs:
                layer_outputs = {}
                for side in (control, candidate):
                    trace_path = (
                        ROOT / "outputs" / f"phase_profile_{side['tag']}"
                        / "layer_output_fingerprints.json"
                    )
                    if not trace_path.is_file():
                        raise RuntimeError(f"missing per-layer output fingerprints: {trace_path}")
                    layer_outputs[side["tag"]] = json.loads(trace_path.read_text())
                control_trace = layer_outputs[control["tag"]]
                candidate_trace = layer_outputs[candidate["tag"]]
                expected_layers = list(range(28))
                if (
                    [row.get("layer") for row in control_trace] != expected_layers
                    or [row.get("layer") for row in candidate_trace] != expected_layers
                    or any(row.get("forward_count", 0) <= 0 for row in control_trace + candidate_trace)
                ):
                    raise RuntimeError("per-layer evaluation trace is incomplete; expected outputs from all 28 layers")
                if control_trace != candidate_trace:
                    first = next(
                        (i for i, (a, b) in enumerate(zip(control_trace, candidate_trace)) if a != b),
                        None,
                    )
                    raise RuntimeError(
                        f"per-layer evaluation outputs differ; first mismatch layer={first}"
                    )
                results_trace = layer_outputs
            else:
                results_trace = None
            results.append({
                "control": control,
                "candidate": candidate,
                "peak_memory_not_higher": peak_memory_ok,
                "layer_output_fingerprints": results_trace,
            })
            report = campaign_dir / "paired_repetitions.json"
            report.write_text(json.dumps(results, indent=2))
            print(json.dumps({"completed_pair": i, **results[-1]}, indent=2), flush=True)
    finally:
        monitor.stop()

    quant_speedups = [
        row["control"]["quantization_seconds"] / row["candidate"]["quantization_seconds"]
        for row in results
    ]
    wall_speedups = [
        row["control"]["wall_seconds"] / row["candidate"]["wall_seconds"]
        for row in results
    ]
    def telemetry_means(side: str, metric: str) -> list[float]:
        return [
            float(row[side]["gpu_telemetry"][metric]["mean"])
            for row in results
            if row[side].get("gpu_telemetry", {}).get(metric, {}).get("mean") is not None
        ]

    control_device_means = telemetry_means("control", "device_memory_used_mib")
    candidate_device_means = telemetry_means("candidate", "device_memory_used_mib")
    control_process_means = telemetry_means("control", "compute_process_memory_mib")
    candidate_process_means = telemetry_means("candidate", "compute_process_memory_mib")
    control_allocated_means = [
        float(row["control"]["mean_allocated_bytes"])
        for row in results if row["control"].get("mean_allocated_bytes") is not None
    ]
    candidate_allocated_means = [
        float(row["candidate"]["mean_allocated_bytes"])
        for row in results if row["candidate"].get("mean_allocated_bytes") is not None
    ]
    control_reserved_means = [
        float(row["control"]["mean_reserved_bytes"])
        for row in results if row["control"].get("mean_reserved_bytes") is not None
    ]
    candidate_reserved_means = [
        float(row["candidate"]["mean_reserved_bytes"])
        for row in results if row["candidate"].get("mean_reserved_bytes") is not None
    ]

    def reduction_percent(control_values: list[float], candidate_values: list[float]):
        if not control_values or len(control_values) != len(candidate_values):
            return None
        control_mean = statistics.mean(control_values)
        candidate_mean = statistics.mean(candidate_values)
        if control_mean <= 0:
            return None
        return 100.0 * (control_mean - candidate_mean) / control_mean

    summary = {
        "campaign": args.tag_prefix,
        "pairs": len(results),
        "trace_layer_outputs": args.trace_layer_outputs,
        "exact_state_kl_ppl_all_pairs": all(
            row["control"]["state_sha256"] == row["candidate"]["state_sha256"]
            and row["control"]["state_tensors"] == row["candidate"]["state_tensors"]
            and row["control"]["kl"] == row["candidate"]["kl"]
            and row["control"]["ppl"] == row["candidate"]["ppl"]
            for row in results
        ),
        "peak_allocated_and_reserved_never_above_control": all(
            row["peak_memory_not_higher"] for row in results
        ),
        "quantization_speedup_mean": statistics.mean(quant_speedups) if quant_speedups else None,
        "quantization_speedup_median": statistics.median(quant_speedups) if quant_speedups else None,
        "quantization_time_reduction_percent": (
            100.0 * (1.0 - statistics.mean([
                row["candidate"]["quantization_seconds"]
                for row in results
            ]) / statistics.mean([
                row["control"]["quantization_seconds"]
                for row in results
            ])) if results else None
        ),
        "end_to_end_speedup_mean": statistics.mean(wall_speedups) if wall_speedups else None,
        "mean_device_memory_reduction_percent": reduction_percent(
            control_device_means, candidate_device_means
        ),
        "mean_compute_process_memory_reduction_percent": reduction_percent(
            control_process_means, candidate_process_means
        ),
        "mean_cuda_allocated_memory_reduction_percent": reduction_percent(
            control_allocated_means, candidate_allocated_means
        ),
        "mean_cuda_reserved_memory_reduction_percent": reduction_percent(
            control_reserved_means, candidate_reserved_means
        ),
        "telemetry_pairs_with_samples": sum(
            row["control"].get("gpu_telemetry", {}).get("samples", 0) > 0
            and row["candidate"].get("gpu_telemetry", {}).get("samples", 0) > 0
            for row in results
        ),
        "pair_results": results,
    }
    (campaign_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "pair_results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
