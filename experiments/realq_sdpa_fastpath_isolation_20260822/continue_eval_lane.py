#!/usr/bin/env python3
"""Run the SDPA isolation once, then restore the shared evaluation worker."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_fastpath_isolation_20260822 import runner


EVALPLUS_RUNTIME = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "additional_methods_fair20_eval_20260821_v1"
    / "_runtime_dependencies"
    / "evalplus_only_0.3.1"
)


def _evaluation_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "1234",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(base.REPO_ROOT),
                    str(base.REPO_ROOT / "YAQA_wclip"),
                    str(base.REPO_ROOT / "YAQA_wclip/hessian_llama"),
                    str(EVALPLUS_RUNTIME),
                )
            ),
        }
    )
    return environment


def _preflight_evaluation_runtime(environment: dict[str, str]) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.metadata as m; assert m.version('evalplus') == '0.3.1'",
        ],
        cwd=base.REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "unified evaluator runtime preflight failed: " + completed.stderr.strip()
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    gpu = int(args.physical_gpu)
    if gpu < 0 or gpu > 7:
        raise SystemExit("physical GPU must be in [0, 7]")
    hostname = socket.gethostname()
    directory = runner.OUTPUT_ROOT / "lane_supervisor" / hostname / f"gpu{gpu}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("lane supervisor is already running") from exc
        command = [
            sys.executable,
            "-u",
            "-m",
            "experiments.realq_sdpa_fastpath_isolation_20260822.runner",
            "--physical-gpu",
            str(gpu),
        ]
        started_at = base._utc_now()
        print(
            json.dumps(
                {
                    "event": "diagnostic_wait_or_start",
                    "hostname": hostname,
                    "physical_gpu": gpu,
                    "started_at": started_at,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        completed = subprocess.run(command, cwd=runner.REPO_ROOT, check=False)
        receipt = {
            "schema_version": 1,
            "diagnostic_returncode": completed.returncode,
            "diagnostic_started_at": started_at,
            "diagnostic_finished_at": base._utc_now(),
            "hostname": hostname,
            "physical_gpu": gpu,
            "next_action": "restore_additional_methods_eval_worker",
        }
        base._atomic_json(directory / "receipt.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        worker = [
            sys.executable,
            "-u",
            "-m",
            "experiments.additional_methods_fair20_eval_20260821.worker",
            "--physical-gpu",
            str(gpu),
            "--poll-seconds",
            "30",
        ]
        evaluation_environment = _evaluation_environment()
        _preflight_evaluation_runtime(evaluation_environment)
        os.execve(sys.executable, worker, evaluation_environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
