"""Restart the interrupted full comparison without reusing incomplete statistics.

Completed independent prefix audits remain historical evidence. This restart
uses their existing configurations, unique run directories and fresh Stage-0
statistics for every arm. It does not diagnose the cause of a vanished WSL job.
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
    out = ROOT / "outputs/full-memory-restart-20261010-1408"
    out.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    old = ROOT / "outputs/cpu-input-checkpoint-followup-20261010/active.json"
    command = json.loads(old.read_text())["command"]
    tag = "qwen06b-checkpoint-ce-full-2x1024-threepairs-20261010-restart1408"
    command[command.index("--tag") + 1] = tag
    command[0] = sys.executable
    (out / "active.json").write_text(json.dumps({"label": tag, "command": command}, indent=2))
    proc = None
    result = {"previous_run_incomplete": True, "previous_progress_last_local": "2026-10-10 12:36:20",
              "reason": "No surviving WSL research/GPU processes at 14:03; cause undetermined.",
              "reuse_incomplete_cache": False, "started": time.strftime("%F %T")}
    try:
        print(f"[restart] fresh full comparison {tag}", flush=True)
        with (out / f"{tag}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            proc.wait()
        result.update(returncode=proc.returncode, valid=proc.returncode == 0)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        result.update(valid=False, error=repr(exc))
        raise
    finally:
        result["finished"] = time.strftime("%F %T")
        (out / "status.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
