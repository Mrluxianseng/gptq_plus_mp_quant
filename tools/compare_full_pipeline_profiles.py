"""Audit completed full-pipeline profiles without initializing CUDA.

Per-phase values are sampled maxima, not proof about every instant. Keep
global allocator peaks, NVML device occupancy and sampled phases separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_FIELDS = (
    "max_nvml_memory_used_mib",
    "stage0_allocator_peak_allocated_bytes",
    "stage0_allocator_peak_reserved_bytes",
    "stage2_allocator_peak_allocated_bytes",
    "stage2_allocator_peak_reserved_bytes",
    "allocator_peak_allocated_bytes_any_stage",
    "allocator_peak_reserved_bytes_any_stage",
)
IGNORED_COMMAND_VALUES = {"--exp", "--static_cache_path"}
SYSTEM_COMMAND_VALUES = {"--eval_ce_chunk_tokens", "--stage2_hinv_cpu_offload_modules", "--gptq_inner_kernel", "--static_saved_tensor_offload_mib"}
SYSTEM_COMMAND_FLAGS = {"--stage2_input_cache_cpu_init", "--stage2_hinv_cpu_offload", "--stage2_hinv_cpu_reuse_buffer", "--stage2_hessian_damping_inplace", "--stage2_hessian_cpu_offload", "--stage2_hessian_groupwise_accum", "--static_saved_tensor_offload_deduplicate", "--research_stage0_checkpoint", "--research_stage0_attention_checkpoint", "--research_stage0_frozen_weight_offload", "--research_stage0_no_backward_unshard"}
SYSTEM_COMMAND_FLAGS.add("--research_stage2_quantizer_metadata_cpu")
SYSTEM_COMMAND_FLAGS.add("--research_stage2_rehome_completed_layer")
SYSTEM_COMMAND_FLAGS.add("--research_stage2_cpu_saved_logits")
SYSTEM_COMMAND_FLAGS.add("--research_stage2_lazy_export_buffers")
SYSTEM_COMMAND_FLAGS.add("--research_hinv_early_release")
SYSTEM_COMMAND_FLAGS.add("--research_hinv_reuse_consumed_h")
SYSTEM_COMMAND_FLAGS.add("--research_hinv_isolated_pool")
SYSTEM_COMMAND_FLAGS.add("--research_hinv_stable_weight")
SYSTEM_COMMAND_FLAGS.add("--research_hinv_solver_workspace")


def algorithm_command(command):
    result = []
    i = 0
    while i < len(command):
        value = command[i]
        if value in IGNORED_COMMAND_VALUES | SYSTEM_COMMAND_VALUES:
            if i + 1 >= len(command) or command[i + 1].startswith("--"):
                raise ValueError(f"missing value for {value}")
            i += 2
            continue
        if value not in SYSTEM_COMMAND_FLAGS:
            result.append("<entry.py>" if value.replace("\\", "/").endswith("/entry.py") else value)
        i += 1
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path, progress=None):
    value = hashlib.sha256()
    processed = 0
    next_progress = time.monotonic() + 30
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
            processed += len(block)
            if progress is not None and time.monotonic() >= next_progress:
                progress(f"hashing {path}: {processed / 2**20:.0f} MiB read")
                next_progress = time.monotonic() + 30
    return value.hexdigest()


def source_hashes(manifest):
    hashes = {}
    for path, value in manifest["source_sha256_at_launch"].items():
        normalized = path.replace("\\", "/")
        key = "entry.py" if normalized.endswith("/entry.py") else normalized.split("gptq_plus_realq/")[-1]
        hashes[key] = value
    return hashes


def phase_peaks(status, root):
    result = {}
    occurrences = {}
    for filename in status.get("memory_profile_files", []):
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        rank = re.search(r"rank(\d+)", path.name)
        stage = "stage0" if "_stage0/" in path.as_posix() else "stage2"
        if rank is None:
            raise ValueError(f"missing rank in profile filename: {path}")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "phase" not in row:
                continue
            # A reused phase name (e.g. backward for each calibration batch)
            # must not hide one batch's regression behind another's maximum.
            base_key = (stage, int(rank.group(1)), row["phase"])
            occurrence = occurrences.get(base_key, 0)
            occurrences[base_key] = occurrence + 1
            key = (*base_key, occurrence)
            values = result.setdefault(key, {"allocated": 0, "reserved": 0, "boundary_allocated": 0, "boundary_reserved": 0, "measurement_mode": row.get("peak_measurement_mode", "sampled")})
            for metric in ("allocated", "reserved"):
                samples = [row.get(f"{point}_{metric}_bytes", 0) for point in ("start", "sampled_peak", "end")]
                values[metric] = max(values[metric], *samples)
                boundary = [row.get(f"{point}_{metric}_bytes", 0) for point in ("start", "end")]
                values[f"boundary_{metric}"] = max(values[f"boundary_{metric}"], *boundary)
    return result


def cache_hashes(directory, progress=None):
    directory = Path(directory) / "static_cache"
    if not directory.is_dir():
        raise ValueError(f"missing static cache: {directory}")
    hashes = {}
    files = [path for path in sorted(directory.rglob("*")) if path.is_file()]
    for index, path in enumerate(files, 1):
        if progress is not None:
            progress(f"cache {directory.parent.name}: file {index}/{len(files)} {path.relative_to(directory)}")
        hashes[str(path.relative_to(directory))] = digest(path, progress)
    return hashes


def telemetry_stats(directory, interval_ms):
    timestamps_by_gpu, used, uuids = {}, [], set()
    with (Path(directory) / "nvml_telemetry.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 4:
                continue
            try:
                stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                memory = int(row[1].strip())
            except ValueError:
                continue
            gpu = row[4].strip() if len(row) >= 6 else "unspecified"
            if len(row) >= 6:
                uuids.add(row[5].strip())
            timestamps_by_gpu.setdefault(gpu, []).append(stamp)
            used.append(memory)
    gaps = [(b - a).total_seconds() for timestamps in timestamps_by_gpu.values() for a, b in zip(timestamps, timestamps[1:])]
    discontinuities = sum(gap <= 0 or gap > 5 * interval_ms / 1000 for gap in gaps)
    return {"samples": len(used), "maximum_mib": max(used, default=None),
            "physical_gpu_indices": sorted(timestamps_by_gpu),
            "physical_gpu_uuids": sorted(uuids),
            "sample_mean_mib": sum(used) / len(used) if used else None,
            "sampling_discontinuities": discontinuities,
            "maximum_sample_gap_seconds": max(gaps, default=None),
            "time_average_claim_supported": False,
            "note": "Sample mean is diagnostic; startup/shutdown and sampling gaps affect it. It is not a matched-phase time-weighted memory average."}


def compare(control_dir, candidate_dir, *, root=ROOT, compare_cache=False, stage0_only=False, progress=None):
    control_dir, candidate_dir = Path(control_dir), Path(candidate_dir)
    control = read_json(control_dir / "status.json")
    candidate = read_json(candidate_dir / "status.json")
    errors = []
    stages = ("stage0",) if stage0_only else ("stage0", "stage2")
    for side, status in (("control", control), ("candidate", candidate)):
        if stage0_only and status.get("scope") != "stage0_only":
            errors.append(f"{side}: requested Stage-0 diagnostic but run has a different scope")
        if status.get("valid") is False:
            errors.append(f"{side}: run validation failed")
        if status.get("returncode") != 0 or not status.get("sources_unchanged"):
            errors.append(f"{side}: failed run or modified sources")
        for stage in stages:
            if status.get(stage, {}).get("returncode") != 0:
                errors.append(f"{side}: incomplete {stage}")
        if stage0_only:
            continue
        for metric in ("kl", "ppl"):
            value = status.get(metric)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                errors.append(f"{side}: missing or nonfinite {metric}")
        if not (status.get("metrics") or {}).get("state_sha256"):
            errors.append(f"{side}: missing state digest")
        if (status.get("metrics") or {}).get("state_tensors", 0) <= 0:
            errors.append(f"{side}: missing tensor count")
    manifests = [read_json(path / "manifest.json") for path in (control_dir, candidate_dir)]
    source_c, source_n = [source_hashes(manifest) for manifest in manifests]
    hashes_match = bool(source_c) and source_c == source_n
    if not hashes_match:
        errors.append("different source snapshots")
    commands_match = all(algorithm_command(manifests[0][f"{stage}_command"]) == algorithm_command(manifests[1][f"{stage}_command"]) for stage in stages)
    if not commands_match:
        errors.append("different algorithm/data/parallelism commands")
    environment_match = manifests[0].get("runtime_environment", {}) == manifests[1].get("runtime_environment", {})
    if not environment_match:
        errors.append("different recorded runtime/allocator/communication environments")
    telemetry = [telemetry_stats(path, manifest.get("telemetry_interval_ms", 100)) for path, manifest in zip((control_dir, candidate_dir), manifests)]
    for side, status, data in zip(("control", "candidate"), (control, candidate), telemetry):
        if not data["samples"] or data["maximum_mib"] != status.get("max_nvml_memory_used_mib"):
            errors.append(f"{side}: NVML status peak does not match raw telemetry")
    identity_match = all(telemetry[0][key] == telemetry[1][key] for key in ("physical_gpu_indices", "physical_gpu_uuids"))
    if not identity_match:
        errors.append("different monitored physical GPUs")
    traces = []
    for side, path in zip(("control", "candidate"), (control_dir, candidate_dir)):
        trace_path = path / "layer_output_fingerprints.json"
        if stage0_only:
            traces.append([])
        elif not trace_path.is_file():
            errors.append(f"{side}: missing layer fingerprints")
            traces.append([])
        else:
            traces.append(read_json(trace_path))
    indexed = [{row["layer"]: row for row in trace} for trace in traces]
    if any(len(trace) != len(index) for trace, index in zip(traces, indexed)):
        errors.append("duplicate layer IDs in fingerprints")
    if not stage0_only and (not indexed[0] or not indexed[1]):
        errors.append("empty layer fingerprints")
    quantized_prefix = False
    model_layers = None
    quantized_through_layer = None
    for side, manifest, index in zip(("control", "candidate"), manifests, indexed):
        if stage0_only:
            continue
        for layer, row in index.items():
            count = row.get("forward_count")
            shapes = row.get("shapes")
            if not isinstance(count, int) or count <= 0 or not isinstance(shapes, list) or len(shapes) != count:
                errors.append(f"{side}: layer {layer} has no complete recorded evaluation forwards")
        command = manifest["stage2_command"]
        # Evaluation forwards the whole model, including the unquantized
        # suffix of a prefix diagnostic. quant_stop_layer limits quantization;
        # it must not limit the expected evaluation fingerprint coverage.
        if "--model" in command:
            config_path = Path(command[command.index("--model") + 1]) / "config.json"
            try:
                expected_layers = int(read_json(config_path)["num_hidden_layers"])
                model_layers = expected_layers
                stop = int(command[command.index("--quant_stop_layer") + 1]) if "--quant_stop_layer" in command else -1
                quantized_through_layer = stop if stop >= 0 else expected_layers - 1
                quantized_prefix = quantized_prefix or 0 <= stop < expected_layers - 1
                if expected_layers <= 0 or set(index) != set(range(expected_layers)):
                    errors.append(f"{side}: incomplete full evaluation layer coverage")
            except (OSError, KeyError, ValueError, TypeError):
                errors.append(f"{side}: cannot verify model layer count from config")
        elif "--quant_stop_layer" in command:
            # Legacy fixtures/manifests without a model path retain their
            # minimum coverage guard; production manifests always name a model.
            stop = int(command[command.index("--quant_stop_layer") + 1])
            if stop >= 0 and set(index) != set(range(stop + 1)):
                errors.append(f"{side}: incomplete layer coverage through layer {stop}")
    mismatches = sorted(key for key in indexed[0].keys() | indexed[1].keys() if indexed[0].get(key) != indexed[1].get(key))
    for metric in ("kl", "ppl"):
        if control.get(metric) != candidate.get(metric):
            errors.append(f"{metric} differs")
    for metric in ("state_sha256", "state_tensors"):
        if (control.get("metrics") or {}).get(metric) != (candidate.get("metrics") or {}).get(metric):
            errors.append(f"{metric} differs")
    if mismatches:
        errors.append("layer output fingerprints differ")
    cache_match = None
    if compare_cache or stage0_only:
        cache_c, cache_n = cache_hashes(control_dir, progress), cache_hashes(candidate_dir, progress)
        cache_match = bool(cache_c) and cache_c == cache_n
        if not cache_match:
            errors.append("cache files differ or cache is empty")
    global_rows = []
    global_fields = tuple(key for key in GLOBAL_FIELDS if not key.startswith("stage2_")) if stage0_only else GLOBAL_FIELDS
    for key in global_fields:
        c, n = control.get(key), candidate.get(key)
        global_rows.append({"metric": key, "control": c, "candidate": n, "delta": n - c if c is not None and n is not None else None})
    global_ok = all(row["delta"] is not None and row["delta"] <= 0 for row in global_rows)
    phases = [phase_peaks(status, root) for status in (control, candidate)]
    phase_rows = []
    missing_phases = []
    for key in sorted(phases[0].keys() | phases[1].keys()):
        if key not in phases[0] or key not in phases[1]:
            missing_phases.append(list(key))
            continue
        row = dict(stage=key[0], rank=key[1], phase=key[2], occurrence=key[3],
                   control_measurement_mode=phases[0][key]["measurement_mode"],
                   candidate_measurement_mode=phases[1][key]["measurement_mode"])
        for metric in ("allocated", "reserved", "boundary_allocated", "boundary_reserved"):
            c, n = phases[0][key][metric], phases[1][key][metric]
            row.update({f"control_{metric}_bytes": c, f"candidate_{metric}_bytes": n, f"delta_{metric}_bytes": n - c})
        phase_rows.append(row)
    regressions = [row for row in phase_rows if row["delta_allocated_bytes"] > 0 or row["delta_reserved_bytes"] > 0]
    boundary_regressions = [row for row in phase_rows if row["delta_boundary_allocated_bytes"] > 0 or row["delta_boundary_reserved_bytes"] > 0]
    phase_ok = bool(phase_rows) and not missing_phases and not regressions
    full_allocator_fields = tuple(f"{stage}_process_peak_{metric}_bytes" for stage in stages for metric in ("allocated", "reserved")) + ("full_process_peak_allocated_bytes", "full_process_peak_reserved_bytes")
    complete_allocator = all(all(status.get(field) is not None for field in full_allocator_fields) for status in (control, candidate))
    if complete_allocator:
        for key in full_allocator_fields:
            c, n = control[key], candidate[key]
            global_rows.append({"metric": key, "control": c, "candidate": n, "delta": n - c})
        global_ok = global_ok and all(candidate[key] <= control[key] for key in full_allocator_fields)
    peak = next(row for row in global_rows if row["metric"] == "max_nvml_memory_used_mib")
    reduction = -100 * peak["delta"] / peak["control"] if peak["delta"] is not None and peak["control"] else None
    exact_phases = bool(phase_rows) and all(row[side + "_measurement_mode"] == "exact" for row in phase_rows for side in ("control", "candidate"))
    return {
        "scope": "stage0_only" if stage0_only else "full_pipeline",
        "quantization_and_quality_validated": not stage0_only and not errors,
        "full_model_quantization_validated": not stage0_only and not errors and not quantized_prefix and model_layers is not None,
        "quantized_prefix_diagnostic": quantized_prefix,
        "model_layer_count": model_layers,
        "quantized_through_layer": quantized_through_layer,
        "control_directory": str(control_dir.resolve()),
        "candidate_directory": str(candidate_dir.resolve()),
        "exactness_passed": not errors,
        "exactness_errors": errors,
        "tracked_source_snapshots_match": hashes_match,
        "algorithm_commands_match": commands_match,
        "recorded_runtime_environments_match": environment_match,
        "monitored_gpu_identities_match": identity_match,
        "control_layer_count": len(indexed[0]),
        "candidate_layer_count": len(indexed[1]),
        "mismatched_layers": mismatches,
        "cache_files_match": cache_match,
        "global_peaks_no_regression": global_ok,
        "sampled_phase_peaks_no_regression": phase_ok,
        "sampled_phase_regression_count": len(regressions),
        "synchronized_boundary_regression_count": len(boundary_regressions),
        "missing_phases": missing_phases,
        "nvml_peak_reduction_percent": reduction,
        "complete_process_allocator_peaks_available": complete_allocator,
        "phase_allocator_measurement": "exact_intervals" if exact_phases else "sampled_or_mixed",
        "strict_measured_memory_gate_passed": not errors and global_ok and phase_ok and complete_allocator,
        "strict_sampled_memory_gate_passed": not errors and global_ok and phase_ok and complete_allocator,
        "measurement_limit": ("Allocator interval maxima are exact; NVML is sampled." if exact_phases else "Phase maxima are sampled or mixed.") + " Neither proves candidate <= control at every instant. NVML covers the whole device, not only the run.",
        "profiling_times_are_not_speed_evidence": True,
        "control_telemetry": telemetry[0],
        "candidate_telemetry": telemetry[1],
        "global_metrics": global_rows,
        "phase_metrics": phase_rows,
        "phase_regressions": regressions,
        "boundary_regressions": boundary_regressions,
        "control_kl": control.get("kl"),
        "candidate_kl": candidate.get("kl"),
        "control_ppl": control.get("ppl"),
        "candidate_ppl": candidate.get("ppl"),
        "control_host_rss_peak_bytes": control.get("max_sampled_aggregate_host_rss_bytes"),
        "candidate_host_rss_peak_bytes": candidate.get("max_sampled_aggregate_host_rss_bytes"),
    }


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("phase_metrics", "phase_regressions", "boundary_regressions"):
        rows = report[name]
        with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = [f"# {report['scope']} paired profile audit", "",
             f"Scope-specific exactness: {report['exactness_passed']}", "",
             f"Quantization and KL/PPL validated: {report['quantization_and_quality_validated']}", "",
             f"Global peaks did not increase: {report['global_peaks_no_regression']}", "",
             f"Matched sampled phase peaks did not increase: {report['sampled_phase_peaks_no_regression']}", "",
             f"NVML peak reduction: {report['nvml_peak_reduction_percent']}%", "",
             report["measurement_limit"], "",
             "Profiling times are not speed evidence.", "",
             "| Metric | Control | Candidate | Candidate minus control |",
             "| --- | ---: | ---: | ---: |"]
    lines.extend(f"| {row['metric']} | {row['control']} | {row['candidate']} | {row['delta']} |" for row in report["global_metrics"])
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-cache", action="store_true")
    parser.add_argument("--stage0-only", action="store_true", help="Audit precompute and exact caches only; never claims quantization/KL/PPL validation.")
    args = parser.parse_args()
    report = compare(args.control, args.candidate, compare_cache=args.compare_cache, stage0_only=args.stage0_only)
    write_report(report, args.output_dir)
    print(json.dumps({key: value for key, value in report.items() if key not in ("global_metrics", "phase_metrics", "phase_regressions", "boundary_regressions")}, indent=2))
    raise SystemExit(0 if report["strict_sampled_memory_gate_passed"] else 2)


if __name__ == "__main__":
    main()
