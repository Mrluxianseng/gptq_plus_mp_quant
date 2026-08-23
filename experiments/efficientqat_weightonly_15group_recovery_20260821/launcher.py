#!/usr/bin/env python3
"""Launch recovery watchers for this pod's still-running EfficientQAT lanes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess

from . import worker


def main() -> int:
    plan = worker.read_json(worker.PARENT_PLAN)
    if worker.sha256_file(worker.PARENT_PLAN) != worker.EXPECTED_PARENT_PLAN_SHA256:
        raise worker.RecoveryError("parent plan changed")
    hostname = socket.gethostname()
    runs = [run for run in plan["runs"] if run["canoe_pod"] == hostname]
    if not runs:
        raise worker.RecoveryError(f"parent plan has no runs for {hostname}")
    log_root = worker.RECOVERY_ROOT / "queue_logs" / hostname
    receipt = log_root / "launcher_receipt.json"
    if receipt.exists() or receipt.is_symlink():
        raise worker.RecoveryError(f"launcher receipt already exists: {receipt}")
    log_root.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONHASHSEED="1",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=os.pathsep.join(
            value
            for value in (str(worker.REPO_ROOT), env_base.get("PYTHONPATH", ""))
            if value
        ),
    )
    rows = []
    handles = []
    python = plan["venv_python"]
    try:
        for run in runs:
            log = log_root / f"{run['run_id']}.log"
            handle = log.open("x", encoding="utf-8")
            handles.append(handle)
            command = [
                python,
                "-u",
                "-m",
                "experiments.efficientqat_weightonly_15group_recovery_20260821.worker",
                "--run-id",
                run["run_id"],
                "--physical-gpu",
                str(run["physical_gpu"]),
                "--poll-seconds",
                "15",
            ]
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = str(run["physical_gpu"])
            process = subprocess.Popen(
                command,
                cwd=worker.REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            rows.append(
                {
                    "run_id": run["run_id"],
                    "physical_gpu": run["physical_gpu"],
                    "pid": process.pid,
                    "command": command,
                    "log": str(log),
                }
            )
    finally:
        for handle in handles:
            handle.close()
    payload = {
        "schema_version": 1,
        "status": "launched",
        "hostname": hostname,
        "launched_at": worker.now(),
        "parent_plan": str(worker.PARENT_PLAN),
        "parent_plan_sha256": worker.EXPECTED_PARENT_PLAN_SHA256,
        "source_sha256": worker._source_hashes(),
        "workers": rows,
    }
    worker.write_json_atomic(receipt, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
