#!/usr/bin/env python3
"""Restore one evaluator lane after all TurboBOA Llama recoveries finish.

The frozen evaluator intentionally fails closed after three attempts.  Four
TurboBOA Llama suites exhausted those attempts on a manifest-alias bug and are
being repaired by a separately audited recovery queue.  An ordinary worker
started before all four suites exist therefore exits immediately on the next
still-exhausted suite.  This small supervisor waits without taking a physical
GPU lock, validates all four recovery terminals, and only then ``exec``s the
unchanged frozen worker.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

from . import common
from .recover_turboboa_llama import ALLOWED_EVAL_IDS


EVALPLUS_RUNTIME = (
    common.OUTPUT_ROOT / "_runtime_dependencies" / "evalplus_only_0.3.1"
)
SUPERVISOR_KIND = "wait_turboboa_llama_recovery_then_eval"


class SupervisorError(RuntimeError):
    """The delayed evaluator handoff violated its fail-closed contract."""


def _recovery_terminals() -> list[dict[str, Any]] | None:
    terminals: list[dict[str, Any]] = []
    for eval_id in ALLOWED_EVAL_IDS:
        path = common.OUTPUT_ROOT / "evals" / eval_id / "suite_success.json"
        if not path.is_file():
            return None
        value = common.read_object(path)
        if (
            value.get("eval_id") != eval_id
            or value.get("status")
            != "generation_succeeded_official_humaneval_pending"
            or value.get("contract_recovery", {}).get("kind")
            != "turboboa_llama_manifest_alias_only"
        ):
            raise SupervisorError(f"invalid recovery terminal: {path}")
        terminals.append(
            {
                "eval_id": eval_id,
                "path": str(path.resolve()),
                "sha256": common.sha256_file(path),
            }
        )
    return terminals


def _worker_environment() -> dict[str, str]:
    if not (EVALPLUS_RUNTIME / "evalplus").is_dir() or not (
        EVALPLUS_RUNTIME / "evalplus-0.3.1.dist-info"
    ).is_dir():
        raise SupervisorError(
            f"audited EvalPlus runtime is missing: {EVALPLUS_RUNTIME}"
        )
    environment = os.environ.copy()
    environment.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONHASHSEED="1234",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=os.pathsep.join(
            value
            for value in (
                str(common.REPO_ROOT),
                str(common.REPO_ROOT / "YAQA_wclip"),
                str(common.REPO_ROOT / "YAQA_wclip/hessian_llama"),
                str(EVALPLUS_RUNTIME),
            )
            if value
        ),
    )
    return environment


def _preflight(python: str, environment: dict[str, str]) -> None:
    completed = subprocess.run(
        [
            python,
            "-c",
            (
                "import importlib.metadata as m; import evalplus; "
                "assert m.version('evalplus') == '0.3.1'"
            ),
        ],
        cwd=common.REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SupervisorError(
            "EvalPlus runtime preflight failed: " + completed.stderr[-2000:]
        )


def execute(*, physical_gpu: int, poll_seconds: int) -> None:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise SupervisorError("supervisor must run on a Canoe pod")
    if physical_gpu < 0 or physical_gpu > 7:
        raise SupervisorError("physical GPU must be in [0, 7]")
    if poll_seconds < 10:
        raise SupervisorError("poll interval must be at least 10 seconds")

    plans = common.load_quant_plans()
    python = str(plans["efficientqat"]["venv_python"])
    common.load_reference_manifest()
    directory = (
        common.OUTPUT_ROOT
        / "lane_handoffs"
        / hostname
        / f"gpu{physical_gpu}__wait_turboboa_llama_recovery"
    )
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".supervisor.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorError("this delayed lane supervisor is already live") from exc

        last_report = 0.0
        while True:
            terminals = _recovery_terminals()
            if terminals is not None:
                break
            current = time.monotonic()
            if current - last_report >= 300:
                completed = sum(
                    (
                        common.OUTPUT_ROOT
                        / "evals"
                        / eval_id
                        / "suite_success.json"
                    ).is_file()
                    for eval_id in ALLOWED_EVAL_IDS
                )
                print(
                    f"{common.now()} waiting_for_turboboa_llama_recovery="
                    f"{completed}/{len(ALLOWED_EVAL_IDS)}",
                    flush=True,
                )
                last_report = current
            time.sleep(poll_seconds)

        environment = _worker_environment()
        _preflight(python, environment)
        source = Path(__file__).resolve()
        receipt = {
            "schema_version": 1,
            "status": "ready_to_exec_frozen_worker",
            "kind": SUPERVISOR_KIND,
            "hostname": hostname,
            "physical_gpu": physical_gpu,
            "finished_wait_at": common.now(),
            "recovery_terminals": terminals,
            "runtime_dependency": {
                "evalplus_version": "0.3.1",
                "path": str(EVALPLUS_RUNTIME.resolve()),
            },
            "source": {
                "path": str(source),
                "sha256": common.sha256_file(source),
            },
            "next_command": [
                python,
                "-u",
                "-m",
                "experiments.additional_methods_fair20_eval_20260821.worker",
                "--physical-gpu",
                str(physical_gpu),
                "--poll-seconds",
                "30",
            ],
        }
        common.atomic_json(directory / "ready_receipt.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
        os.execve(python, receipt["next_command"], environment)


def launch(*, physical_gpu: int, poll_seconds: int) -> dict[str, Any]:
    hostname = socket.gethostname()
    plans = common.load_quant_plans()
    python = str(plans["efficientqat"]["venv_python"])
    log_root = common.OUTPUT_ROOT / "queue_logs" / hostname
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"gpu{physical_gpu}.wait_turboboa_recovery_then_eval.log"
    launch_path = log_path.with_suffix(".launch.json")
    if log_path.exists() or launch_path.exists():
        raise SupervisorError(f"delayed lane launch already exists: {log_path}")
    command = [
        python,
        "-u",
        "-m",
        "experiments.additional_methods_fair20_eval_20260821.wait_recovery_then_eval",
        "--physical-gpu",
        str(physical_gpu),
        "--poll-seconds",
        str(poll_seconds),
        "--child",
    ]
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=common.REPO_ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {
        "schema_version": 1,
        "status": "launched",
        "kind": SUPERVISOR_KIND,
        "hostname": hostname,
        "physical_gpu": physical_gpu,
        "pid": process.pid,
        "launched_at": common.now(),
        "command": command,
        "log": str(log_path),
    }
    common.atomic_json(launch_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.child:
            execute(
                physical_gpu=args.physical_gpu,
                poll_seconds=args.poll_seconds,
            )
            return 0
        print(
            json.dumps(
                launch(
                    physical_gpu=args.physical_gpu,
                    poll_seconds=args.poll_seconds,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
