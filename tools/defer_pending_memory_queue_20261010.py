"""Let the active campaign finish, then retire its superseded queue.

Only the orchestration PID is stopped; GPU workers remain running. Pending
speed/component jobs are deferred under the user's revised priority. The
campaign's own status remains authoritative, including negative results.
"""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "outputs/post-memory-v6-followup-20261010-2125"
TAG = "qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125"
CAMPAIGN = ROOT / "outputs" / f"memory_campaign_{TAG}"
PARENT, CHILD, WAITING = 290, 518, 298


def process(pid):
    folder = Path(f"/proc/{pid}")
    try:
        stat = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        return {"state": stat[0], "ppid": int(stat[1]),
                "started": stat[19],
                "command": (folder / "cmdline").read_bytes().replace(b"\0", b" ").decode()}
    except FileNotFoundError:
        return None


def same(pid, identity):
    current = process(pid)
    return current is not None and current["started"] == identity["started"]


def main():
    parent, child, waiting = process(PARENT), process(CHILD), process(WAITING)
    if not parent or "run_post_restart_memory_followup_20261010.py" not in parent["command"]:
        raise RuntimeError("expected queue controller identity changed")
    if not child or child["ppid"] != PARENT or TAG not in child["command"]:
        raise RuntimeError("active campaign identity changed")
    if not waiting or "run_lazy_export_buffer_followup_20261010.py" not in waiting["command"]:
        raise RuntimeError("waiting component controller identity changed")
    lazy = ROOT / "outputs/lazy-export-buffer-followup-v5-20261010-2125"
    if (lazy / "active.json").exists():
        raise RuntimeError("component queue already has active work; do not interrupt")
    record = {"reason": "User prioritizes only stages that limit hardware capacity",
              "active_campaign": TAG, "controller_pid": PARENT,
              "campaign_pid": CHILD, "gpu_campaign_uninterrupted": True,
              "deferred": ["Triton speed experiments", "optional export-buffer experiments"],
              "started": time.strftime("%F %T")}
    path = QUEUE / "priority_deferral.json"
    if path.exists():
        raise RuntimeError("priority decision already recorded")
    os.kill(WAITING, signal.SIGTERM)  # Verified waiting controller, no CUDA child.
    os.kill(PARENT, signal.SIGSTOP)  # Individual PID, never its process group.
    retired = False
    try:
        path.write_text(json.dumps(record, indent=2))
        print("[priority] active 1.7B campaign continues; future jobs deferred", flush=True)
        while True:
            current = process(CHILD)
            if current is None or current["started"] != child["started"] or current["state"] == "Z":
                break
            time.sleep(5)
        status_file = CAMPAIGN / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else None
        # The original controller cannot schedule another job while stopped.
        # Its child has exited; terminating this PID cannot interrupt workers.
        if same(PARENT, parent):
            os.kill(PARENT, signal.SIGKILL)
        retired = True
        result = dict(record, controller_retired=True,
                      campaign_status_present=status is not None,
                      campaign_valid=(status or {}).get("valid"),
                      jobs=[{"label": TAG, "returncode": (status or {}).get("returncode"),
                             "status_source": str(status_file)}],
                      finished=time.strftime("%F %T"),
                      note="Queue retirement is not a method result; use the original campaign audit")
        (QUEUE / "status.json").write_text(json.dumps(result, indent=2))
        print("[priority] active campaign exited; superseded queue retired", flush=True)
    finally:
        if not retired and same(PARENT, parent):
            os.kill(PARENT, signal.SIGCONT)


if __name__ == "__main__":
    main()
