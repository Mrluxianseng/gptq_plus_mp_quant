"""Serial follow-ups after the primary complete-model memory cohort.

No simultaneous CUDA runs. First isolate evaluation CE chunking against
the accepted activation-offload system, then repeat the larger-model pair.
Every cohort must pass its numerical and measured memory audit to proceed.
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
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    out = ROOT / "outputs/memory_completion_queue_20261010"
    out.mkdir(parents=True, exist_ok=False)
    prerequisite = ROOT / "outputs/memory_followup_queue_20261010/status.json"
    print("[completion] waiting for both complete-model correctness/memory cohorts", flush=True)
    while not prerequisite.exists():
        time.sleep(5)
    if not json.loads(prerequisite.read_text())["all_cohorts_valid"]:
        raise RuntimeError("primary full-model audit failed; stop for diagnosis before adding components")

    ce_manifests = out / "qwen06b-ce256-manifests"
    ce_manifests.mkdir()
    original = ROOT / "outputs/portable-memory-pilot-06b-20261010"
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            data = json.loads((original / f"candidate-{stage}.json").read_text())
            data["followup_scope"] = "CE256 added to the accepted offload+dedup system; not its total benefit against the original control"
            if side == "candidate" and stage == "stage2":
                if "--eval_ce_chunk_tokens" in data["command"]:
                    index = data["command"].index("--eval_ce_chunk_tokens")
                    data["command"][index+1] = "256"
                else:
                    data["command"] += ["--eval_ce_chunk_tokens", "256"]
            (ce_manifests / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))

    cohorts = (
        ("qwen06b-full-offload-dedup-ce256-ablation-2x1024-20261010", ce_manifests, 1),
        ("qwen17b-full-offload-dedup-hinv-repeat-2x512-20261010", ROOT / "outputs/portable-memory-pilot-17b-2x512-20261010", 2),
    )
    records = []
    for tag, source, pairs in cohorts:
        command = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
                   "--tag", tag, "--pairs", str(pairs), "--gpu-indices", "0", "--timeout-seconds", "5400"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                command += [f"--{side}-{stage}-manifest", str(source / f"{side}-{stage}.json")]
        campaign = ROOT / "outputs" / f"memory_campaign_{tag}"
        (out / "active_campaign.json").write_text(json.dumps({"tag": tag, "directory": str(campaign), "command": command}, indent=2))
        print(f"[completion] launching {tag}; live log: {campaign / 'combined.log'}", flush=True)
        with (out / f"{tag}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            next_update = started + 30
            try:
                while process.poll() is None:
                    if time.monotonic() >= next_update:
                        active = campaign / "active_run.json"
                        print(f"[completion] elapsed={time.monotonic()-started:.0f}s; active=" + (active.read_text() if active.exists() else "starting"), flush=True)
                        next_update = time.monotonic() + 30
                    time.sleep(5)
            except BaseException:
                stop_process_tree(process.pid)
                process.wait()
                raise
        result = json.loads((campaign / "status.json").read_text())
        records.append({"tag": tag, "exit_code": process.returncode, **result})
        (out / "cohorts.json").write_text(json.dumps(records, indent=2))
        if not result["valid"]:
            raise RuntimeError(f"{tag} did not pass; inspect its saved pair audit before continuing")
        print("[completion] passed " + tag, flush=True)
    (out / "status.json").write_text(json.dumps({"finished": time.strftime("%F %T"), "cohorts": records, "all_cohorts_valid": True}, indent=2))
    print("[completion] finished; review and append results to the journal", flush=True)


if __name__ == "__main__":
    main()
