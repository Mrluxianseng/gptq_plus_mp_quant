"""Serial independent saved-parameter and attention recompute pilots.

Waits for the existing repair/CE queue to exit; never overlaps GPU workers.
Numerical preflight is mandatory. Each negative resource result is retained.
"""
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
    out = ROOT / "outputs/memory_residency_followup_20261010"
    out.mkdir(exist_ok=False)
    records = []
    prior = ROOT / "outputs/memory_repair_queue_20261010"
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        print("[residency] waiting for serial repair/CE queue completion", flush=True)
        started = time.monotonic()
        while not (prior / "status.json").exists():
            if time.monotonic() - started > 7200:
                raise RuntimeError("prior queue did not finish; no GPU worker launched")
            time.sleep(5)
        for model in ("qwen3-06b", "qwen3-17b"):
            data = json.loads((prior / f"{model}-cuda-checkpoint-warmed-selective.json").read_text())
            if not data["exact_output_input_and_module_gradients"]:
                raise RuntimeError("real CUDA numerical preflight failed")
            if data["reports"]["frozen_parameter_saved_offload"]["saved_offload_stats"]["frozen_parameter_requests"] <= 0:
                raise RuntimeError("CUDA preflight did not exercise saved frozen parameters")

        for label, source in (
            ("qwen06b-frozen-weight-offload-stage0-2x1024-20261010", "frozen-weight-offload-qwen06b-manifests-20261010"),
            ("qwen17b-frozen-weight-offload-stage0-2x512-20261010", "frozen-weight-offload-qwen17b-manifests-20261010"),
            ("qwen06b-selective-attention-offload-stage0-2x1024-20261010", "selective-checkpoint-offload-stage0-manifests-20261010"),
        ):
            command = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
                       "--tag", label, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "5400", "--stage0-only"]
            for side in ("control", "candidate"):
                for stage in ("stage0", "stage2"):
                    command += [f"--{side}-{stage}-manifest", str(ROOT / "outputs" / source / f"{side}-{stage}.json")]
            (out / "active.json").write_text(json.dumps({"label": label, "command": command}, indent=2))
            print(f"[residency] launching {label}", flush=True)
            with (out / f"{label}.log").open("x") as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        time.sleep(3)
                except BaseException:
                    stop_process_tree(proc.pid)
                    proc.wait()
                    raise
            audit_path = ROOT / f"outputs/memory_campaign_{label}/pair1_audit/comparison.json"
            audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
            record = {"label": label, "returncode": proc.returncode,
                      "exact": audit.get("exactness_passed"),
                      "strict_memory": audit.get("strict_measured_memory_gate_passed"),
                      "audit": str(audit_path), "scope": "stage0_only"}
            records.append(record)
            (out / "jobs.json").write_text(json.dumps(records, indent=2))
            print(f"[residency] result {json.dumps(record)}", flush=True)
        status = {"finished": time.strftime("%F %T"), "jobs": records,
                  "all_stage0_pilots_passed": all(row["returncode"] == 0 for row in records),
                  "complete_quantization_claim": False}
    except BaseException as exc:
        (out / "status.json").write_text(json.dumps({"error": repr(exc), "jobs": records,
                         "finished": time.strftime("%F %T")}, indent=2))
        raise
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print("[residency] all independent pilots finished", flush=True)


if __name__ == "__main__":
    main()
