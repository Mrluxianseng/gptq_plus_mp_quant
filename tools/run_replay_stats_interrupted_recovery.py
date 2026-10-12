"""Recover an interrupted prefix in new directories, then gate the next method.

Never synthesize a successful status for an incomplete worker or reuse its peaks.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_full_pipeline_profiles import digest


def main():
    os.chdir(ROOT)
    old = ROOT / "outputs/qwen17b-cached-dead-subgroup-screen-20261011-v1"
    out = ROOT / "outputs/replay-stats-interrupted-recovery-20261012-v1"
    out.mkdir(exist_ok=False)
    manifest = json.loads((old / "manifest.json").read_text())
    if (old / "status.json").exists():
        raise RuntimeError("old run completed; inspect it instead of repeating")
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if any(line.strip().isdigit() for line in gpu.splitlines()):
        raise RuntimeError("CUDA already active; recovery must be serial")
    for file, expected in manifest["source_sha256"].items():
        if digest(file) != expected:
            raise RuntimeError("interrupted bound source changed: " + file)
    report = {"valid": False, "reason": "Interrupted during evaluation; fresh completion audit required", "old_directory": str(old)}
    (out / "interrupted_source_audit.json").write_text(json.dumps({"source_unchanged": True, "count": len(manifest["source_sha256"]), "original_manifest_sha256": digest(old / "manifest.json")}, indent=2))
    common = ["--workspace-queue", "outputs/hessian-workspace-adaptive-followup-20261011-v2",
              "--reference-run", "outputs/full_pipeline_profile_qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125-r1-candidate",
              "--workspace-prefix-run", "outputs/full_pipeline_profile_qwen17b-hinv-workspace-stable-layer2-20261011-workspace-v2-r1-control"]
    dead = "outputs/qwen17b-cached-dead-subgroup-screen-20261012-recovery-v2"
    replay = "outputs/qwen17b-cached-replay-stats-screen-20261012-recovery-v2"
    commands = [
        [sys.executable, "-u", "tools/run_cached_dead_subgroup_screen.py", "--wait-for", "outputs/qwen17b-cached-bounded-finite-screen-20261011-v1/status.json", *common, "--output-directory", dead],
        [sys.executable, "-u", "tools/run_cached_replay_stats_screen.py", "--wait-for", dead + "/status.json", *common, "--output-directory", replay],
    ]
    (out / "manifest.json").write_text(json.dumps({"commands": commands, "scope": "recovery after external interruption; no resumed allocator state or successful original run claim"}, indent=2))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    try:
        for label, command in zip(("dead-completion", "replay-stats"), commands):
            print("[recovery] starting " + label, flush=True)
            (out / "active.json").write_text(json.dumps({"label": label, "command": command}))
            with (ROOT / ("logs/" + label + "-recovery-20261012-v2.log")).open("x") as log:
                code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            report[label + "_returncode"] = code
            print("[recovery] " + label + " returncode=" + str(code), flush=True)
            if label == "dead-completion":
                status = json.loads((ROOT / dead / "status.json").read_text())
                if not all(status.get(k) for k in ("exactness_passed", "cache_unchanged", "dead_input_inverse_write_coverage", "exact_interval_peaks")):
                    raise RuntimeError("recovered predecessor incomplete or incorrect; no next method")
            else:
                report["valid"] = json.loads((ROOT / replay / "status.json").read_text()).get("valid", False)
    finally:
        report["finished"] = time.strftime("%F %T")
        (out / "status.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
