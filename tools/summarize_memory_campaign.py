"""CPU-only manuscript tables from audited paired memory campaigns.

Reports diagnostic timing, not speed evidence. Rejects missing pairs and
preserves failed memory gates; does not remove extrema or failed repeats.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def offload_metrics(log_path):
    """Structured per-rank counters; absent legacy instrumentation is unknown.

    Tqdm may prefix a record with carriage returns/ANSI bytes, so match the
    explicit marker rather than treating each line as plain JSON.
    """
    records = []
    for line in log_path.read_text(errors="replace").splitlines():
        marker = "STAGE0_OFFLOAD_METRICS "
        if marker not in line:
            continue
        row = json.loads(line.split(marker, 1)[1])
        fields = ("requests", "requested_bytes", "copies", "copied_bytes",
                  "dedup_hits", "live_cpu_bytes", "peak_live_cpu_bytes")
        if any(not isinstance(row[key], int) or row[key] < 0 for key in fields):
            raise ValueError(f"invalid offload counters in {log_path}")
        if (row["copies"] + row["dedup_hits"] != row["requests"]
                or row["copied_bytes"] > row["requested_bytes"]
                or row["live_cpu_bytes"] > row["peak_live_cpu_bytes"]):
            raise ValueError(f"inconsistent offload counters in {log_path}")
        records.append(row)
    ranks = [row["rank"] for row in records]
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"duplicate Stage-0 offload rank records in {log_path}")
    return records


def peak_locations(status):
    rows = []
    for relative in status.get("memory_profile_files", []):
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        stage = "stage0" if "_stage0/" in path.as_posix() else "stage2"
        rank = int(re.search(r"rank(\d+)", path.name).group(1))
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "phase" in row:
                rows.append({"stage": stage, "rank": rank, "phase": row["phase"],
                             "profile_file": relative, "measurement_mode": row.get("peak_measurement_mode", "sampled"),
                             "allocated_bytes": row["sampled_peak_allocated_bytes"],
                             "reserved_bytes": row["sampled_peak_reserved_bytes"]})
    result = {}
    for metric in ("allocated", "reserved"):
        maximum = max((row[f"{metric}_bytes"] for row in rows), default=None)
        complete = status.get(f"full_process_peak_{metric}_bytes")
        result[metric] = {"largest_profiled_interval_bytes": maximum,
                          "matches_complete_process_high_watermark": maximum == complete,
                          "note": "If false, the complete peak occurred outside recorded intervals; do not assign it to these phases.",
                          "locations": [row for row in rows if row[f"{metric}_bytes"] == maximum]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.campaign / "manifest.json").read_text())
    campaign_status = json.loads((args.campaign / "status.json").read_text())
    records = json.loads((args.campaign / "paired_repetitions.json").read_text())
    if not records:
        raise ValueError("No audited pair is available. Inspect campaign status and worker logs; do not summarize a failed launch as a measured result.")
    values, locations, offload = [], {}, {}
    for record in records:
        pair = record["pair"]
        audit = json.loads((args.campaign / f"pair{pair}_audit/comparison.json").read_text())
        for side in ("control", "candidate"):
            tag = f"{meta['tag']}-r{pair}-{side}"
            directory = ROOT / f"outputs/full_pipeline_profile_{tag}"
            status = json.loads((directory / "status.json").read_text())
            counters = offload_metrics(directory / "stage0.log")
            if counters and {row["rank"] for row in counters} != set(range(status["expected_ranks"]["stage0"])):
                raise ValueError(f"incomplete offload rank coverage for {tag}")
            offload[tag] = {"instrumented": bool(counters), "records_by_rank": counters,
                            "note": "Absent counters are unknown, not zero. Cumulative copied bytes are not active or peak GPU/CPU residency."}
            values.append({"pair": pair, "side": side, "scope": status["scope"],
                           "exactness_passed": audit["exactness_passed"],
                           "measured_memory_gate_passed": audit["strict_measured_memory_gate_passed"],
                           "nvml_peak_mib": status["max_nvml_memory_used_mib"],
                           "allocated_peak_mib": status["full_process_peak_allocated_bytes"] / 2**20,
                           "reserved_peak_mib": status["full_process_peak_reserved_bytes"] / 2**20,
                           "host_rss_peak_mib": status["max_sampled_aggregate_host_rss_bytes"] / 2**20,
                           "diagnostic_stage0_seconds": status["stage0"]["wall_seconds"],
                           "diagnostic_quantization_seconds": (status.get("metrics") or {}).get("quantization_seconds"),
                           "diagnostic_total_seconds": status["wall_seconds"], "kl": status.get("kl"), "ppl": status.get("ppl"),
                           "phase_regressions": audit["sampled_phase_regression_count"],
                           "boundary_regressions": audit["synchronized_boundary_regression_count"],
                           "offload_copied_bytes_sum_ranks": sum(row["copied_bytes"] for row in counters) if counters else None,
                           "offload_requested_bytes_sum_ranks": sum(row["requested_bytes"] for row in counters) if counters else None,
                           "offload_dedup_hits_sum_ranks": sum(row["dedup_hits"] for row in counters) if counters else None,
                           "offload_live_cpu_bytes_sum_ranks_at_end": sum(row["live_cpu_bytes"] for row in counters) if counters else None,
                           "offload_peak_live_cpu_mib_max_rank": max(row["peak_live_cpu_bytes"] for row in counters) / 2**20 if counters else None})
            locations[tag] = peak_locations(status)
    summary = {"campaign": meta["tag"], "scope": meta["scope"],
               "requested_pairs": meta["pairs"], "audited_pairs": len(records),
               "all_requested_pairs_finished": len(records) == meta["pairs"],
               "campaign_valid": campaign_status["valid"],
               "all_audited_pairs_exact": bool(records) and all(row["exactness_passed"] for row in records),
               "all_audited_pairs_pass_memory_gate": bool(records) and all(row["strict_measured_memory_gate_passed"] for row in records),
               "diagnostic_times_are_not_speed_evidence": True, "unit": "MiB", "paired_statistics": {}}
    for field in ("nvml_peak_mib", "allocated_peak_mib", "reserved_peak_mib", "host_rss_peak_mib"):
        control = [row[field] for row in values if row["side"] == "control"]
        candidate = [row[field] for row in values if row["side"] == "candidate"]
        reductions = [100*(c-n)/c for c,n in zip(control, candidate)]
        summary["paired_statistics"][field] = {
            "control_mean": statistics.mean(control), "candidate_mean": statistics.mean(candidate),
            "control_min": min(control), "control_max": max(control), "candidate_min": min(candidate), "candidate_max": max(candidate),
            "paired_reduction_percent_mean": statistics.mean(reductions),
            "paired_reduction_percent_min": min(reductions), "paired_reduction_percent_max": max(reductions),
            "note": "Mean of run peaks; not a time-weighted mean memory occupancy. Single-pair ranges do not establish repeatability."}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "peak_locations.json").write_text(json.dumps(locations, indent=2))
    (args.output_dir / "offload_metrics.json").write_text(json.dumps(offload, indent=2))
    with (args.output_dir / "paired_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    lines = [f"# {meta['tag']} audited memory results", "", f"Scope: {meta['scope']}; completed {len(records)}/{meta['pairs']} pairs; campaign valid={campaign_status['valid']}.", "",
             "Diagnostic timing is not speed evidence. Reported means are means of per-run peaks, not time-averaged occupancy.", "",
             "| Metric (MiB) | Control mean | Candidate mean | Mean paired reduction |", "| --- | ---: | ---: | ---: |"]
    for field, data in summary["paired_statistics"].items():
        lines.append(f"| {field} | {data['control_mean']:.2f} | {data['candidate_mean']:.2f} | {data['paired_reduction_percent_mean']:.2f}% |")
    lines += ["", "Quality/phase gates and every raw repetition are in paired_values.csv; peak positions are in peak_locations.json. Offload counters and instrumentation coverage are in offload_metrics.json. No outliers or failed audited pairs have been removed."]
    (args.output_dir / "summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
