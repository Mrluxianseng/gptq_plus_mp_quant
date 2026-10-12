"""Continue the same progress file after its previous reader has exited.

Read-only watcher. Launches no GPU jobs and never interrupts another reader.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    print("[progress-continuation] waiting for prior GPU queue and existing log reader", flush=True)
    while not args.wait_for.exists():
        if time.monotonic()-started > 24*3600:
            raise RuntimeError("prior queue did not complete within a day")
        time.sleep(5)
    # The old reader knows all preceding queues and will exit naturally.
    # Preserve its lifetime; never make two writers to the user entry file.
    deadline = time.monotonic()+60
    while True:
        readers = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = proc.info["cmdline"] or []
                if any(Path(value).name == "stream_memory_research_progress.py" for value in command):
                    if "--output" in command and Path(command[command.index("--output")+1]).resolve() == args.output.resolve():
                        readers.append(proc.pid)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        if not readers:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"previous reader still present: {readers}; avoid duplicate writers")
        time.sleep(1)
    with (args.queue / "progress-forwarder.log").open("x") as log:
        command = [sys.executable, "-u", str(ROOT / "tools/stream_memory_research_progress.py"),
                   "--queues", str(args.queue), "--output", str(args.output), "--append"]
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    (args.queue / "progress-forwarder-status.json").write_text(json.dumps(dict(returncode=result.returncode)))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
