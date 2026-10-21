"""Streaming allocator intervals and bounded external NVML/host sampling."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time


class Observer:
    def __init__(self, torch, path: Path, rank: int):
        self.torch, self.rank = torch, rank
        self.stream = Path(path).open("x", encoding="utf-8", buffering=1)
        self.stack, self.sequence = [], 0

    def emit(self, row):
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def peaks(self, devices):
        result = {}
        for device in devices:
            self.torch.cuda.synchronize(device)
            result[str(device)] = {
                "allocated": self.torch.cuda.memory_allocated(device),
                "reserved": self.torch.cuda.memory_reserved(device),
                "peak_allocated": self.torch.cuda.max_memory_allocated(device),
                "peak_reserved": self.torch.cuda.max_memory_reserved(device),
            }
        return result

    def accumulate(self, values):
        for frame in self.stack:
            for device, row in values.items():
                prior = frame["peaks"].setdefault(device, {"allocated": 0, "reserved": 0})
                prior["allocated"] = max(prior["allocated"], row["peak_allocated"])
                prior["reserved"] = max(prior["reserved"], row["peak_reserved"])

    @contextmanager
    def scope(self, name, *, devices=None, **metadata):
        devices = devices or [self.rank]
        before = self.peaks(devices)
        self.accumulate(before)
        for device in devices:
            self.torch.cuda.reset_peak_memory_stats(device)
        self.sequence += 1
        frame = {"sequence": self.sequence, "stage": name, "rank": self.rank,
                 "start_ns": time.perf_counter_ns(), "peaks": {}, **metadata}
        self.stack.append(frame)
        self.emit({**frame, "event": "start", "devices": before})
        completed = False
        try:
            yield
            completed = True
        finally:
            measurement_error = None
            try:
                try:
                    after = self.peaks(devices)
                    self.accumulate(after)
                    self.emit({**frame, "event": "end", "completed": completed,
                               "measurement_completed": True,
                               "end_ns": time.perf_counter_ns(), "devices": after})
                except Exception as exc:
                    measurement_error = exc
                    self.emit({**frame, "event": "end", "completed": completed,
                               "measurement_completed": False, "error_type": type(exc).__name__,
                               "end_ns": time.perf_counter_ns()})
            finally:
                self.stack.pop()
            if completed and measurement_error is not None:
                raise RuntimeError("Allocator observation failed") from measurement_error

    def close(self):
        self.stream.close()


def sample(parent: int, devices: list[str], output: Path, stop):
    """One synchronous sample at a time; no in-memory history or overlapping scans."""
    import psutil
    owner = psutil.Process(parent)
    with Path(output).open("x", encoding="utf-8", buffering=1) as stream:
        while not stop.is_set():
            start = time.perf_counter_ns()
            row = {"start_ns": start, "host_complete": True, "errors": []}
            try:
                members = [owner] + owner.children(recursive=True)
                rss = pss = 0
                seen = set()
                for process in members:
                    try:
                        key = (process.pid, process.create_time())
                        if key in seen: continue
                        seen.add(key)
                        memory = process.memory_full_info()
                        rss += memory.rss
                        if not hasattr(memory, "pss"):
                            row["host_complete"] = False
                        else:
                            pss += memory.pss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        row["host_complete"] = False
                row.update(rss_bytes=rss, pss_bytes=pss, process_count=len(seen))
                result = subprocess.run(["nvidia-smi", "--id=" + ",".join(devices),
                    "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True)
                row["nvml"] = result.stdout.strip().splitlines()
            except Exception as exc:
                row["errors"].append(type(exc).__name__)
            row["end_ns"] = time.perf_counter_ns()
            stream.write(json.dumps(row) + "\n")
            stop.wait(0.5)
