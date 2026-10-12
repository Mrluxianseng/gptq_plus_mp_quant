"""Serial fresh-cache input-residency pilots and gated full-model checkpoint work.

Never expands a failed mechanism. Keeps completed independent experiments;
diagnostic timings are not uninstrumented speed evidence.
"""
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_pipeline_memory_campaign import stop_process_tree
from tools.run_full_pipeline_memory_profile import set_option


def main():
    out = ROOT / "outputs/cpu-input-checkpoint-followup-20261010"
    out.mkdir(exist_ok=False)
    jobs = []
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))

    def execute(label, command):
        (out / "active.json").write_text(json.dumps({"label": label, "command": command}, indent=2))
        print(f"[followup] starting {label}", flush=True)
        with (out / f"{label}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    time.sleep(3)
            except BaseException:
                stop_process_tree(proc.pid)
                proc.wait()
                raise
        row = {"label": label, "returncode": proc.returncode, "finished": time.strftime("%F %T")}
        jobs.append(row)
        (out / "jobs.json").write_text(json.dumps(jobs, indent=2))
        print(f"[followup] {json.dumps(row)}", flush=True)
        return proc.returncode == 0

    def prepare(label, model, length, prefix, cpu_input, checkpoint=False):
        folder = out / f"{label}-manifests"
        folder.mkdir()
        source = ROOT / "outputs/portable-memory-pilot-06b-20261010"
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                data = json.loads((source / f"control-{stage}.json").read_text())
                command = data["command"]
                command[command.index("baseline-profile") - 1] = str(ROOT / "tools/realq_checkpoint_research_entry.py")
                set_option(command, "--model", f"/mnt/d/llamaModels/{model}")
                set_option(command, "--seq_len", str(length))
                set_option(command, "--quant_stop_layer", "0" if prefix else "27")
                # Match the documented two-sample reference gradients.
                set_option(command, "--backward_bsz", "2")
                set_option(command, "--final_layer_backward_bsz", "2")
                if prefix:
                    set_option(command, "--static_saved_tensor_offload_mib", "4")
                    command.append("--static_saved_tensor_offload_deduplicate")
                command.append("--offload_inps")
                if side == "candidate":
                    if stage == "stage2" and cpu_input:
                        command.append("--stage2_input_cache_cpu_init")
                    if stage == "stage0" and checkpoint:
                        command.append("--research_stage0_checkpoint")
                    if stage == "stage2" and not prefix:
                        set_option(command, "--eval_ce_chunk_tokens", "256")
                data["scope_note"] = ("Fresh two-sample single-GPU pilot; full evaluation outputs checked. "
                                      "Input-only prefix or gated checkpoint/CE complete-model experiment; no paper-scale/multicard claim.")
                (folder / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
        return folder

    def campaign(label, folder, pairs=1):
        command = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
                   "--tag", label, "--pairs", str(pairs), "--gpu-indices", "0", "--timeout-seconds", "7200"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                command += [f"--{side}-{stage}-manifest", str(folder / f"{side}-{stage}.json")]
        return execute(label, command)

    try:
        if not execute("production-ce-fidelity", [sys.executable, "-u", str(ROOT / "tools/check_ce_chunk_exactness_cuda_20261010.py"),
                       "--layout-preserving", "--require-production-exact", "--output", str(out / "production-ce-fidelity.json")]):
            raise RuntimeError("CE preflight failed; no model expansion")
        passed = {}
        for model, stem, length in (("Qwen3-0.6B", "qwen06b", 1024), ("Qwen3-1.7B", "qwen17b", 512)):
            label = f"{stem}-cpu-input-init-layer0-2x{length}-20261010"
            folder = prepare(label, model, length, prefix=True, cpu_input=True)
            # Separate input-policy correctness and strict memory from checkpoint.
            passed[stem] = campaign(label, folder)
        # Run a complete, repeated positive-mechanism composition even if the
        # independent input allocation option fails: failed input stays disabled.
        label = "qwen06b-checkpoint-ce-full-2x1024-threepairs-20261010"
        folder = prepare(label, "Qwen3-0.6B", 1024, prefix=False,
                         cpu_input=passed["qwen06b"], checkpoint=True)
        required = 8 * 1024 ** 3
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError("less than 8 GiB free before repeated full-model run; preserve artifacts and resolve storage")
        campaign(label, folder, pairs=3)
        result = {"finished": time.strftime("%F %T"), "input_pilot_passed": passed, "jobs": jobs}
    except BaseException as exc:
        (out / "status.json").write_text(json.dumps({"error": repr(exc), "jobs": jobs, "finished": time.strftime("%F %T")}, indent=2))
        raise
    (out / "status.json").write_text(json.dumps(result, indent=2))
    print("[followup] completed; inspect each scope-specific audit", flush=True)


if __name__ == "__main__":
    main()
