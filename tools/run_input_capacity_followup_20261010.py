"""Gated real-input cache capacity checks after the serial GPU experiments."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_pipeline_memory_campaign import stop_process_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/input-cache-capacity-followup-20261010")
    parser.add_argument("--wait-for", type=Path, default=ROOT / "outputs/quantizer-lifetime-followup-20261010/status.json")
    options = parser.parse_args()
    out = options.output_dir
    out.mkdir(exist_ok=False)
    jobs = []
    proc = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        print("[input-capacity] waiting for all earlier serial GPU controllers", flush=True)
        began = time.monotonic()
        while not options.wait_for.exists():
            if time.monotonic() - began > 21600:
                raise RuntimeError("prior queue not finished; no GPU probe launched")
            time.sleep(5)
        small_passed = {}
        for samples, length in ((32, 1024), (512, 2048)):
            for model, stem in (("Qwen3-0.6B", "qwen06b"), ("Qwen3-1.7B", "qwen17b")):
                if samples > 32 and not small_passed.get(stem):
                    continue
                rows = []
                for side in ("control", "candidate"):
                    label = f"{stem}-{samples}x{length}-{side}"
                    output = out / f"{label}.json"
                    command = [sys.executable, "-u", str(ROOT / "tools/probe_real_input_cache_capacity.py"),
                               "--model-path", f"/mnt/d/llamaModels/{model}", "--samples", str(samples),
                               "--seq-len", str(length), "--output", str(output)]
                    if side == "candidate":
                        command.append("--cpu-init")
                    (out / "active.json").write_text(json.dumps({"label": label, "command": command}, indent=2))
                    print(f"[input-capacity] starting {label}", flush=True)
                    with (out / f"{label}.log").open("x") as log:
                        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        try:
                            proc.wait(timeout=1800)
                        except BaseException:
                            stop_process_tree(proc.pid)
                            proc.wait()
                            raise
                    if proc.returncode:
                        raise RuntimeError(f"capacity worker failed: {label}; inspect its log")
                    rows.append(json.loads(output.read_text()))
                c, n = rows
                exact = bool(c.get("inps_sha256")) and c["inps_sha256"] == n.get("inps_sha256") == n.get("fp_inps_sha256")
                expected_control_oom = c.get("cuda_oom", False) and c.get("failed_phase") == "clone_reference_inputs"
                valid = exact and n["completed"] and (c["completed"] or expected_control_oom)
                row = {"model": model, "samples": samples, "seq_len": length, "input_bytes_exact": exact,
                       "control_completed": c["completed"], "candidate_completed": n["completed"],
                       "control_clone_cuda_oom": expected_control_oom, "valid": valid,
                       "scope": "input capture phase only; no full pipeline/quality/4-GPU assertion"}
                jobs.append(row)
                (out / "jobs.json").write_text(json.dumps(jobs, indent=2))
                print(f"[input-capacity] {json.dumps(row)}", flush=True)
                if not valid:
                    raise RuntimeError("input values differ or failure is not the expected CUDA reference-clone OOM")
                if samples == 32:
                    small_passed[stem] = True
        result = {"valid": True, "jobs": jobs, "finished": time.strftime("%F %T")}
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        (out / "status.json").write_text(json.dumps({"valid": False, "error": repr(exc), "jobs": jobs,
                                                   "finished": time.strftime("%F %T")}, indent=2))
        raise
    (out / "status.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
