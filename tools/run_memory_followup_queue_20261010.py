"""Date-scoped serial follow-up; real output artifacts remain authoritative.

Waits for the Stage-0 handoff gate, runs three complete 0.6B pairs and one
complete 1.7B pair. Independent cohorts may continue after a memory-only
failure, but numerical failures stop the queue. No automatic tuning/push.
"""
import json
import signal
from pathlib import Path
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
    out = ROOT / "outputs/memory_followup_queue_20261010"
    out.mkdir(parents=True, exist_ok=False)
    prerequisite = ROOT / "outputs/memory_campaign_qwen17b-stage0-handoff-dedup-2x512-20261010/status.json"
    print("[queue] waiting for the Stage-0 1.7B handoff/correctness gate", flush=True)
    while not prerequisite.exists():
        time.sleep(5)
    status = json.loads(prerequisite.read_text())
    if not status.get("valid"):
        raise RuntimeError("Stage-0 handoff pair did not pass; inspect its audit before running larger follow-ups")
    records = []
    cohorts = (("qwen06b-full-offload-dedup-2x1024-20261010", "portable-memory-pilot-06b-20261010", 3),
               ("qwen17b-full-offload-dedup-hinv-2x512-20261010", "portable-memory-pilot-17b-2x512-20261010", 1))
    for tag, directory, pairs in cohorts:
        source = ROOT / "outputs" / directory
        command = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
                   "--tag", tag, "--pairs", str(pairs), "--gpu-indices", "0", "--timeout-seconds", "5400"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                command += [f"--{side}-{stage}-manifest", str(source / f"{side}-{stage}.json")]
        campaign = ROOT / "outputs" / f"memory_campaign_{tag}"
        (out / "active_campaign.json").write_text(json.dumps({"tag": tag, "directory": str(campaign), "command": command}, indent=2))
        print(f"[queue] launching {tag}; single live log: {campaign / 'combined.log'}", flush=True)
        with (out / f"{tag}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            next_update = started + 30
            try:
                while process.poll() is None:
                    if time.monotonic() >= next_update:
                        active = campaign / "active_run.json"
                        detail = active.read_text() if active.exists() else "starting"
                        print(f"[queue] {tag}: elapsed={time.monotonic()-started:.0f}s; active={detail}", flush=True)
                        next_update = time.monotonic() + 30
                    time.sleep(5)
            except BaseException:
                stop_process_tree(process.pid)
                process.wait()
                raise
        result = json.loads((campaign / "status.json").read_text())
        records.append({"tag": tag, "exit_code": process.returncode, **result})
        (out / "cohorts.json").write_text(json.dumps(records, indent=2))
        print("[queue] cohort result: " + json.dumps(records[-1]), flush=True)
        if not result["valid"]:
            audits = sorted(campaign.glob("pair*_audit/comparison.json"))
            if not audits or not all(json.loads(path.read_text())["exactness_passed"] for path in audits):
                raise RuntimeError("numerical or incomplete-run failure; stop to diagnose")
            print("[queue] measured memory gate failed; this cohort is not accepted. Continue only the independent model cohort.", flush=True)
    (out / "status.json").write_text(json.dumps({"finished": time.strftime("%F %T"), "cohorts": records,
                                                 "all_cohorts_valid": all(row["valid"] for row in records)}, indent=2))
    print("[queue] finished; review cohort audits and update the research journal", flush=True)


if __name__ == "__main__":
    main()
