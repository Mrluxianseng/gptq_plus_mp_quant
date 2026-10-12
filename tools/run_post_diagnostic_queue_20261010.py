"""Serial independent experiments after the Hinv prefix diagnostic.

No GPU overlap. Numerical prefix failures stop follow-ups. A failed memory
gate is preserved and does not invalidate independent checkpoint/CE studies.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import signal

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from tools.run_full_pipeline_memory_campaign import stop_process_tree
from tools.run_full_pipeline_memory_profile import set_option


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    out = root / "outputs/post_diagnostic_queue_20261010"
    out.mkdir(exist_ok=False)
    prerequisite = root / "outputs/memory_campaign_qwen17b-layer0-hinv-reuse-fresh-2x512-20261010"
    print("[post] waiting for Hinv prefix to finish", flush=True)
    while not (prerequisite/"status.json").exists():
        time.sleep(5)
    audit = json.loads((prerequisite/"pair1_audit/comparison.json").read_text())
    if not audit["exactness_passed"]:
        raise RuntimeError("Hinv prefix numerical/incomplete failure; diagnose before further GPU work")
    records = []
    def execute(label, command):
        print("[post] launching " + label, flush=True)
        with (out/f"{label}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    time.sleep(5)
            except BaseException:
                stop_process_tree(process.pid)
                process.wait()
                raise
        records.append({"label": label, "returncode": process.returncode})
        (out/"jobs.json").write_text(json.dumps(records, indent=2))
        return process.returncode

    for model, length in (("Qwen3-0.6B", 1024), ("Qwen3-1.7B", 512)):
        label = model.lower().replace(".", "") + "-cuda-checkpoint-warmed"
        result = execute(label, [sys.executable, str(root/"tools/check_qwen_checkpoint_cuda_compatibility.py"),
                      "--model-path", f"/mnt/d/llamaModels/{model}", "--seq-len", str(length),
                      "--output", str(out/f"{label}.json")])
        if result:
            raise RuntimeError("CUDA checkpoint numerical preflight failed")

    tag = "qwen06b-checkpoint-stage0-fsdp-2x1024-20261010"
    source = root/"outputs/checkpoint-stage0-manifests-20261010"
    command = [sys.executable, "-u", str(root/"tools/run_full_pipeline_memory_campaign.py"),
               "--tag", tag, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "5400", "--stage0-only"]
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            command += [f"--{side}-{stage}-manifest", str(source/f"{side}-{stage}.json")]
    execute(tag, command)
    checkpoint_campaign = root/"outputs"/f"memory_campaign_{tag}"
    checkpoint_audit = json.loads((checkpoint_campaign/"pair1_audit/comparison.json").read_text()) if (checkpoint_campaign/"pair1_audit/comparison.json").exists() else None
    (out/"checkpoint_result.json").write_text(json.dumps({"stage0_audit_available": checkpoint_audit is not None,
                        "exact": checkpoint_audit and checkpoint_audit["exactness_passed"],
                        "strict_memory": checkpoint_audit and checkpoint_audit["strict_measured_memory_gate_passed"],
                        "full_model_validation": False}, indent=2))

    # Independent evaluation component, using the accepted 0.6B system.
    source = out/"ce256-manifests"
    source.mkdir()
    original = root/"outputs/portable-memory-pilot-06b-20261010"
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            data = json.loads((original/f"candidate-{stage}.json").read_text())
            data["scope_note"] = "Independent CE256 ablation on activation offload+dedup, fresh full-28. Not Hinv/checkpoint combination."
            if side == "candidate" and stage == "stage2":
                set_option(data["command"], "--eval_ce_chunk_tokens", "256")
            (source/f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
    tag = "qwen06b-offload-dedup-ce256-full-ablation-20261010"
    command = [sys.executable, "-u", str(root/"tools/run_full_pipeline_memory_campaign.py"),
               "--tag", tag, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "5400"]
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            command += [f"--{side}-{stage}-manifest", str(source/f"{side}-{stage}.json")]
    execute(tag, command)
    (out/"status.json").write_text(json.dumps({"finished": time.strftime("%F %T"), "jobs": records,
                    "every_job_returned_zero": all(row["returncode"] == 0 for row in records)}, indent=2))
    print("[post] completed; inspect individual audits including failures", flush=True)


if __name__ == "__main__":
    main()
