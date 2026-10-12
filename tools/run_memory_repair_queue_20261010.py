"""Serial numerical repair and independent recomputation/CE experiments.

Prefix smoke is explicitly distinct from the fresh complete-pipeline audit.
No failed numerical candidate advances to a larger quantization experiment.
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
    out = ROOT / "outputs/memory_repair_queue_20261010"
    out.mkdir(exist_ok=False)
    records = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)

    def execute(label, command):
        print(f"[repair] launching {label}", flush=True)
        (out / "active.json").write_text(json.dumps({"label": label, "command": command}, indent=2))
        with (out / f"{label}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    time.sleep(3)
            except BaseException:
                stop_process_tree(proc.pid)
                proc.wait()
                raise
        records.append({"label": label, "returncode": proc.returncode, "finished": time.strftime("%F %T")})
        (out / "jobs.json").write_text(json.dumps(records, indent=2))
        print(f"[repair] {label} returned {proc.returncode}", flush=True)
        return proc.returncode

    def campaign(tag, source, stage0_only=False):
        cmd = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
               "--tag", tag, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "5400"]
        if stage0_only:
            cmd += ["--stage0-only"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                cmd += [f"--{side}-{stage}-manifest", str(source / f"{side}-{stage}.json")]
        return execute(tag, cmd)

    smoke = out / "smoke-manifests"
    smoke.mkdir()
    cache = ROOT / "outputs/full_pipeline_profile_qwen17b-layer0-hinv-reuse-fresh-2x512-20261010-r1-control/static_cache"
    if not cache.is_dir():
        raise RuntimeError("required shared completed static cache is missing")
    statuses = []
    for side in ("control", "candidate"):
        data = json.loads((ROOT / f"outputs/hinv-reuse-layer0-manifests-20261010/{side}-stage2.json").read_text())
        set_option(data["command"], "--static_cache_path", str(cache))
        data["scope_note"] = "Cached Stage-2 layer0 numerical repair smoke only; no fresh Stage-0 or complete memory claim."
        path = smoke / f"{side}.json"
        path.write_text(json.dumps(data, indent=2))
        tag = f"qwen17b-layer0-hinv-layout-repair-{side}-20261010"
        rc = execute(tag, [sys.executable, "-u", str(ROOT / "tools/run_stage2_memory_profile.py"),
                           "--manifest", str(path), "--tag", tag, "--sample-host-rss"])
        if rc:
            raise RuntimeError("layout repair smoke execution failed")
        statuses.append(json.loads((ROOT / f"outputs/phase_profile_{tag}/status.json").read_text()))
    numerical = all(s["sources_unchanged"] and s["returncode"] == 0 for s in statuses)
    numerical = numerical and all(statuses[0]["metrics"][key] == statuses[1]["metrics"][key]
                                    for key in ("state_sha256", "state_tensors"))
    numerical = numerical and all(statuses[0][key] == statuses[1][key] for key in ("kl", "ppl"))
    (out / "layout_repair_numerical_smoke.json").write_text(json.dumps({
        "passed": numerical, "scope": "cached_layer0_prefix", "statuses": statuses}, indent=2))
    if numerical:
        campaign("qwen17b-layer0-hinv-layout-repair-fresh-2x512-20261010",
                 ROOT / "outputs/hinv-reuse-layer0-manifests-20261010")
    else:
        print("[repair] numerical failure preserved; skip Hinv expansion", flush=True)

    # These are independent candidates with Hinv reuse disabled; a Hinv failure
    # must not be silently ignored or used as their baseline.
    for model, length in (("Qwen3-0.6B", 1024), ("Qwen3-1.7B", 512)):
        label = model.lower().replace(".", "") + "-cuda-checkpoint-warmed-selective"
        rc = execute(label, [sys.executable, str(ROOT / "tools/check_qwen_checkpoint_cuda_compatibility.py"),
                            "--model-path", f"/mnt/d/llamaModels/{model}", "--seq-len", str(length),
                            "--output", str(out / f"{label}.json")])
        if rc:
            raise RuntimeError("CUDA checkpoint numerical preflight failed; no collector expansion")
    campaign("qwen06b-checkpoint-stage0-fsdp-2x1024-20261010",
             ROOT / "outputs/checkpoint-stage0-manifests-20261010", stage0_only=True)

    source = out / "ce256-manifests"
    source.mkdir()
    original = ROOT / "outputs/portable-memory-pilot-06b-20261010"
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            data = json.loads((original / f"candidate-{stage}.json").read_text())
            data["scope_note"] = "Independent CE256 ablation on offload+dedup, fresh full28; Hinv and checkpoint disabled."
            if side == "candidate" and stage == "stage2":
                set_option(data["command"], "--eval_ce_chunk_tokens", "256")
            (source / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
    campaign("qwen06b-offload-dedup-ce256-full-ablation-20261010", source)
    (out / "status.json").write_text(json.dumps({"finished": time.strftime("%F %T"), "jobs": records,
                    "every_job_returned_zero": all(row["returncode"] == 0 for row in records)}, indent=2))
    print("[repair] queue completed; assess each scope and audit separately", flush=True)


if __name__ == "__main__":
    main()
