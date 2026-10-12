"""Serial, alternating fresh-cache memory pairs with one live combined log.

Uses the full-pipeline runner and scope-aware audit; never treats diagnostic
wall times as an uninstrumented speed benchmark. Stops on invalid outputs or
the strict measured-memory gate, preserving every completed artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_full_pipeline_profiles import compare, write_report


def stop_process_tree(pid):
    """Terminate only this controller's descendants, including torchrun ranks."""
    import psutil
    try:
        root = psutil.Process(pid)
        processes = list(reversed(root.children(recursive=True))) + [root]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


class CombinedLog:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            self.file.write(text)
            self.file.flush()
            print(text, end="", flush=True)

    def close(self):
        self.file.close()


def run_logged(command, output_dir, log):
    """Forward controller stdout and append both worker logs while running."""
    log.write("[campaign] command: " + json.dumps(command) + "\n")
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    positions = {stage: 0 for stage in ("stage0", "stage2")}
    stop = threading.Event()
    def forward_worker_logs():
        while True:
            for stage in positions:
                path = output_dir / f"{stage}.log"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8", errors="replace") as stream:
                    stream.seek(positions[stage])
                    text = stream.read()
                    positions[stage] = stream.tell()
                if text:
                    log.write(f"\n[worker {output_dir.name} {stage}]\n" + text)
            if stop.wait(.5):
                break
    thread = threading.Thread(target=forward_worker_logs, daemon=True)
    thread.start()
    try:
        for line in proc.stdout:
            log.write(line)
        code = proc.wait()
    except BaseException:
        stop_process_tree(proc.pid)
        proc.wait()
        raise
    finally:
        stop.set()
        thread.join()
        # Capture the last write between the thread's final read and exit.
        for stage, position in positions.items():
            path = output_dir / f"{stage}.log"
            if path.exists():
                with path.open(encoding="utf-8", errors="replace") as stream:
                    stream.seek(position)
                    text = stream.read()
                if text:
                    log.write(f"\n[worker {output_dir.name} {stage} final]\n" + text)
    if code != 0:
        raise RuntimeError(f"worker/controller failed with exit {code}: {output_dir}")


def main():
    def terminated(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, terminated)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-stage0-manifest", required=True)
    parser.add_argument("--candidate-stage0-manifest", required=True)
    parser.add_argument("--control-stage2-manifest", required=True)
    parser.add_argument("--candidate-stage2-manifest", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--gpu-indices", default="0")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--stage0-only", action="store_true")
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum() or args.pairs <= 0:
        parser.error("use a unique alphanumeric/hyphen tag and positive pairs")
    out = ROOT / "outputs" / f"memory_campaign_{args.tag}"
    out.mkdir(parents=True, exist_ok=False)
    log = CombinedLog(out / "combined.log")
    manifest = {**vars(args), "scope": "stage0_only" if args.stage0_only else "full_pipeline",
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": sys.executable, "diagnostic_times_are_not_speed_evidence": True,
                "launched_local": time.strftime("%F %T")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    records = []
    error = None
    try:
        for pair in range(1, args.pairs+1):
            order = ("control", "candidate") if pair % 2 else ("candidate", "control")
            runs = {}
            for side in order:
                tag = f"{args.tag}-r{pair}-{side}"
                run_dir = ROOT / "outputs" / f"full_pipeline_profile_{tag}"
                command = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_profile.py"),
                           "--stage0-manifest", getattr(args, f"{side}_stage0_manifest"),
                           "--stage2-manifest", getattr(args, f"{side}_stage2_manifest"),
                           "--tag", tag, "--gpu-indices", args.gpu_indices,
                           "--timeout-seconds", str(args.timeout_seconds), "--exact-phase-peaks"]
                if args.stage0_only:
                    command.append("--stage0-only")
                (out / "active_run.json").write_text(json.dumps({"pair": pair, "side": side, "tag": tag, "directory": str(run_dir)}))
                run_logged(command, run_dir, log)
                runs[side] = run_dir
            log.write(f"[campaign] auditing pair {pair}: cache/state/layer/quality/memory checks\n")
            report = compare(runs["control"], runs["candidate"], compare_cache=True, stage0_only=args.stage0_only,
                             progress=lambda message: log.write("[audit] " + message + "\n"))
            write_report(report, out / f"pair{pair}_audit")
            records.append({"pair": pair, "order": order, "audit": str(out / f"pair{pair}_audit"),
                            "exactness_passed": report["exactness_passed"],
                            "strict_measured_memory_gate_passed": report["strict_measured_memory_gate_passed"],
                            "nvml_peak_reduction_percent": report["nvml_peak_reduction_percent"]})
            (out / "paired_repetitions.json").write_text(json.dumps(records, indent=2))
            log.write("[campaign] paired result: " + json.dumps(records[-1]) + "\n")
            if not report["exactness_passed"] or not report["strict_measured_memory_gate_passed"]:
                raise RuntimeError(f"pair {pair} did not pass; inspect {out / f'pair{pair}_audit'}")
    except BaseException as exc:
        error = repr(exc)
        log.write("[campaign] stopped: " + error + "\n")
    finally:
        status = {"valid": error is None and len(records) == args.pairs, "error": error,
                  "completed_pairs": len(records), "requested_pairs": args.pairs,
                  "scope": manifest["scope"], "finished_local": time.strftime("%F %T"),
                  "diagnostic_times_are_not_speed_evidence": True}
        (out / "status.json").write_text(json.dumps(status, indent=2))
        log.write("[campaign] final status: " + json.dumps(status) + "\n")
        log.close()
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
