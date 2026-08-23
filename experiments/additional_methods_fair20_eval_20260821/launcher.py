#!/usr/bin/env python3
"""Launch one persistent unified-evaluation worker per physical GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import traceback

from . import common


def execute(gpus: list[int], *, launch_official_scorer: bool) -> dict:
    hostname = socket.gethostname()
    plans = common.load_quant_plans()
    python = plans["efficientqat"]["venv_python"]
    reference_manifest = common.load_reference_manifest()
    if len(gpus) != len(set(gpus)) or any(gpu < 0 or gpu > 7 for gpu in gpus):
        raise common.EvaluationError(f"invalid GPU list: {gpus}")
    log_root = common.OUTPUT_ROOT / "queue_logs" / hostname
    receipt_path = log_root / "launcher_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise common.EvaluationError(f"launcher receipt already exists: {receipt_path}")
    log_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONHASHSEED="1234",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join(
            value
            for value in (str(common.REPO_ROOT), env.get("PYTHONPATH", ""))
            if value
        ),
    )
    rows = []
    scorer = None
    handles = []
    try:
        for gpu in gpus:
            log_path = log_root / f"gpu{gpu}.log"
            if log_path.exists() or log_path.is_symlink():
                raise common.EvaluationError(f"queue log already exists: {log_path}")
            handle = log_path.open("x", encoding="utf-8")
            handles.append(handle)
            command = [
                python,
                "-u",
                "-m",
                "experiments.additional_methods_fair20_eval_20260821.worker",
                "--physical-gpu",
                str(gpu),
                "--poll-seconds",
                "30",
            ]
            process = subprocess.Popen(
                command,
                cwd=common.REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            rows.append(
                {
                    "physical_gpu": gpu,
                    "pid": process.pid,
                    "command": command,
                    "log": str(log_path),
                }
            )
        if launch_official_scorer:
            log_path = log_root / "official_scorer.log"
            if log_path.exists() or log_path.is_symlink():
                raise common.EvaluationError(
                    f"official scorer log already exists: {log_path}"
                )
            handle = log_path.open("x", encoding="utf-8")
            handles.append(handle)
            command = [
                python,
                "-u",
                "-m",
                "experiments.additional_methods_fair20_eval_20260821.score_worker",
                "--poll-seconds",
                "30",
            ]
            process = subprocess.Popen(
                command,
                cwd=common.REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            scorer = {
                "pid": process.pid,
                "command": command,
                "log": str(log_path),
            }
    finally:
        for handle in handles:
            handle.close()
    receipt = {
        "schema_version": 1,
        "status": "launched",
        "hostname": hostname,
        "launched_at": common.now(),
        "reference_manifest": str(common.REFERENCE_MANIFEST_PATH),
        "reference_manifest_sha256": common.sha256_file(
            common.REFERENCE_MANIFEST_PATH
        ),
        "reference_manifest_fingerprint": reference_manifest["fingerprint"],
        "workers": rows,
        "official_scorer": scorer,
    }
    common.atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="comma-separated physical GPU IDs",
    )
    parser.add_argument(
        "--launch-official-scorer",
        action="store_true",
        help="also launch the single CPU-only isolated HumanEval+ scorer",
    )
    args = parser.parse_args()
    try:
        gpus = [int(value) for value in args.gpus.split(",") if value]
        print(
            json.dumps(
                execute(
                    gpus,
                    launch_official_scorer=args.launch_official_scorer,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
