"""Serial, bounded follow-up to the frozen-residency pilots.

Does not change live production source. Independent negative results are
retained; a CUDA numerical failure prevents expansion of that mechanism.
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
from tools.run_full_pipeline_memory_profile import set_option


def main():
    out = ROOT / "outputs/memory_mechanism_followup_20261010"
    out.mkdir(exist_ok=False)
    records = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)

    def execute(label, command):
        (out / "active.json").write_text(json.dumps({"label": label, "command": command}, indent=2))
        print(f"[mechanism] launching {label}", flush=True)
        with (out / f"{label}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    time.sleep(3)
            except BaseException:
                stop_process_tree(proc.pid)
                proc.wait()
                raise
        record = {"label": label, "returncode": proc.returncode, "finished": time.strftime("%F %T")}
        records.append(record)
        (out / "jobs.json").write_text(json.dumps(records, indent=2))
        print(f"[mechanism] result {json.dumps(record)}", flush=True)
        return proc.returncode

    def prepare(label, source, configure):
        folder = out / f"{label}-manifests"
        folder.mkdir()
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                data = json.loads((source / f"{side}-{stage}.json").read_text())
                configure(data, side, stage)
                (folder / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
        return folder

    def campaign(label, folder, stage0_only):
        command = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
                   "--tag", label, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "5400"]
        if stage0_only:
            command += ["--stage0-only"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                command += [f"--{side}-{stage}-manifest", str(folder / f"{side}-{stage}.json")]
        return execute(label, command)

    try:
        print("[mechanism] waiting for residency queue; GPU jobs remain serial", flush=True)
        started = time.monotonic()
        while not (ROOT / "outputs/memory_residency_followup_20261010/status.json").exists():
            if time.monotonic() - started > 7200:
                raise RuntimeError("prior queue did not finish; no GPU worker launched")
            time.sleep(5)
        preflight = True
        for model, length in (("Qwen3-0.6B", 1024), ("Qwen3-1.7B", 512)):
            label = model.lower().replace(".", "") + "-cuda-all-frozen-saved-preflight"
            rc = execute(label, [sys.executable, str(ROOT / "tools/check_qwen_checkpoint_cuda_compatibility.py"),
                                "--model-path", f"/mnt/d/llamaModels/{model}", "--seq-len", str(length),
                                "--all-frozen-parameter-offload", "--output", str(out / f"{label}.json")])
            preflight = preflight and rc == 0
        execute("cuda-ce-chunk-fidelity", [sys.executable, str(ROOT / "tools/check_ce_chunk_exactness_cuda_20261010.py"),
                                           "--output", str(out / "ce-chunk-fidelity.json")])

        def configure_checkpoint(data, side, stage):
            set_option(data["command"], "--model", "/mnt/d/llamaModels/Qwen3-1.7B")
            set_option(data["command"], "--seq_len", "512")
            data["scope_note"] = "Independent actual FSDP Stage-0 1.7B checkpoint pilot; no full quantization/quality claim."
        folder = prepare("checkpoint17", ROOT / "outputs/checkpoint-stage0-manifests-20261010", configure_checkpoint)
        campaign("qwen17b-checkpoint-stage0-fsdp-2x512-20261010", folder, True)

        if preflight:
            def configure_unshard(data, side, stage):
                set_option(data["command"], "--seq_len", "256")
                if side == "candidate" and stage == "stage0":
                    data["command"].append("--research_stage0_no_backward_unshard")
                data["scope_note"] = "All-sized saved frozen parameters plus no backward unshard; actual FSDP Stage-0 only. Not checkpoint or full quantization."
            folder = prepare("no-backward-unshard", ROOT / "outputs/frozen-weight-offload-qwen06b-manifests-20261010", configure_unshard)
            campaign("qwen06b-all-saved-frozen-no-backward-unshard-stage0-2x256-20261010", folder, True)
        else:
            print("[mechanism] numerical preflight failed; skip backward-unshard expansion", flush=True)

        def configure_hessian(data, side, stage):
            # Start from matched 1.7B fresh-prefix manifests; remove Hinv
            # residency changes so the only Stage-2 candidate is Hessian CPU.
            command = data["command"]
            for flag in ("--stage2_hinv_cpu_offload", "--stage2_hinv_cpu_reuse_buffer"):
                while flag in command:
                    command.remove(flag)
            if "--stage2_hinv_cpu_offload_modules" in command:
                index = command.index("--stage2_hinv_cpu_offload_modules")
                del command[index:index + 2]
            set_option(command, "--static_saved_tensor_offload_mib", "4")
            if "--static_saved_tensor_offload_deduplicate" not in command:
                command.append("--static_saved_tensor_offload_deduplicate")
            if side == "candidate" and stage == "stage2":
                command.append("--stage2_hessian_cpu_offload")
            data["scope_note"] = "Independent fresh Stage-0 + layer0 prefix + full evaluation: groupwise Hessian CPU residency; no Hinv CPU or reuse."
        folder = prepare("hessian-cpu-layer0", ROOT / "outputs/hinv-reuse-layer0-manifests-20261010", configure_hessian)
        campaign("qwen17b-layer0-hessian-cpu-fresh-2x512-20261010", folder, False)
        status = {"finished": time.strftime("%F %T"), "jobs": records,
                  "scope": "independent CUDA and Stage-0/prefix diagnostics; no complete-model quantization claim"}
    except BaseException as exc:
        (out / "status.json").write_text(json.dumps({"error": repr(exc), "jobs": records,
                                                   "finished": time.strftime("%F %T")}, indent=2))
        raise
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print("[mechanism] bounded follow-up completed; inspect every independent audit", flush=True)


if __name__ == "__main__":
    main()
