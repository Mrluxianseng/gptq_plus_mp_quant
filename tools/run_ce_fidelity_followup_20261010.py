"""Run the revised CE fixture only after serial mechanism pilots finish."""
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
    out = ROOT / "outputs/ce_fidelity_followup_20261010"
    out.mkdir(exist_ok=False)
    proc = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        print("[ce-fidelity] waiting for serial mechanism queue completion", flush=True)
        began = time.monotonic()
        while not (ROOT / "outputs/memory_mechanism_followup_20261010/status.json").exists():
            if time.monotonic() - began > 7200:
                raise RuntimeError("prior queue did not finish; no GPU fixture started")
            time.sleep(5)
        command = [sys.executable, "-u", str(ROOT / "tools/check_ce_chunk_exactness_cuda_20261010.py"),
                   "--layout-preserving", "--output", str(out / "branch-preserving-cuda.json")]
        (out / "command.json").write_text(json.dumps(command, indent=2))
        print("[ce-fidelity] launching isolated CUDA fidelity fixture", flush=True)
        with (out / "run.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(3)
        status = {"returncode": proc.returncode, "finished": time.strftime("%F %T"),
                  "scope": "isolated synthetic CUDA fixture; not full-model validation"}
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        (out / "status.json").write_text(json.dumps({"error": repr(exc), "finished": time.strftime("%F %T")}, indent=2))
        raise
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
