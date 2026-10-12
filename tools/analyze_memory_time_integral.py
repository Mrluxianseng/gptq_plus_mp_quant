"""Time-weighted NVML occupancy over the recorded workflow window.

This is device occupancy, including startup, checkpoint I/O and shutdown;
it is neither tensor-only memory nor a matched-phase average. Large gaps
or reversed timestamps reject the estimate instead of filling missing data.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


def integrate(samples, max_gap_seconds):
    if len(samples) < 2:
        raise ValueError("at least two timestamped samples are required")
    area, duration, max_gap = 0.0, 0.0, 0.0
    for (first, a), (second, b) in zip(samples, samples[1:]):
        dt = (second - first).total_seconds()
        if dt <= 0 or dt > max_gap_seconds:
            raise ValueError(f"invalid sampling interval: {dt:.6f} seconds")
        if a < 0 or b < 0:
            raise ValueError("memory samples must be nonnegative")
        area += 0.5 * (a + b) * dt
        duration += dt
        max_gap = max(max_gap, dt)
    return {"samples": len(samples), "window_seconds": duration,
            "time_weighted_device_memory_mib": area / duration,
            "device_memory_mib_seconds": area,
            "maximum_gap_seconds": max_gap,
            "peak_sampled_device_memory_mib": max(value for _, value in samples)}


def analyze(path, interval_ms=100):
    streams, uuids = {}, {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if len(row) != 6:
                raise ValueError(f"malformed NVML record at {path}:{line}")
            stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            memory = int(row[1].strip())
            gpu, uuid = row[4].strip(), row[5].strip()
            if gpu in uuids and uuids[gpu] != uuid:
                raise ValueError(f"GPU identity changed within {path}")
            uuids[gpu] = uuid
            streams.setdefault(gpu, []).append((stamp, memory))
    if not streams or interval_ms <= 0:
        raise ValueError("empty telemetry or invalid sampling interval")
    return {"scope": "recorded_complete_workflow_monitor_window",
            "method": "trapezoidal integration; reject gaps above five sampling intervals",
            "matched_phase_average": False, "tensor_only_memory": False,
            "note": "Includes startup, checkpoint I/O and shutdown. Each device is integrated separately; per-device peaks and per-device means are not summed as a simultaneous maximum.",
            "devices": {gpu: {"uuid": uuids[gpu], **integrate(samples, 5*interval_ms/1000)}
                        for gpu, samples in streams.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, right = analyze(args.control, args.interval_ms), analyze(args.candidate, args.interval_ms)
    if left["devices"].keys() != right["devices"].keys():
        raise ValueError("different monitored GPU sets")
    reductions = {}
    for gpu, old in left["devices"].items():
        new = right["devices"][gpu]
        if old["uuid"] != new["uuid"]:
            raise ValueError("different physical GPU identities")
        if old["time_weighted_device_memory_mib"] <= 0:
            raise ValueError("zero control mean cannot define a reduction")
        reductions[gpu] = 100*(1-new["time_weighted_device_memory_mib"]/old["time_weighted_device_memory_mib"])
    result = {"control": left, "candidate": right,
              "time_weighted_device_memory_reduction_percent_by_device": reductions,
              "speed_claim_supported": False}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
