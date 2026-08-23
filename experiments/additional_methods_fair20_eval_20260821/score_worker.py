#!/usr/bin/env python3
"""Persistent CPU worker for isolated official HumanEval+ scoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from . import common


MAX_ATTEMPTS = 3


def _suite_dir(spec: common.EvalSpec) -> Path:
    return common.OUTPUT_ROOT / "evals" / spec.eval_id


def _official_dir(spec: common.EvalSpec) -> Path:
    return _suite_dir(spec) / "reasoning/humaneval_plus/official_eval"


def _worker_attempt_count(spec: common.EvalSpec) -> int:
    return sum(
        path.is_dir()
        for path in (_official_dir(spec) / "worker_attempts").glob(
            "attempt[0-9][0-9][0-9]"
        )
    )


def _ready(spec: common.EvalSpec) -> bool:
    suite_path = _suite_dir(spec) / "suite_success.json"
    if not suite_path.is_file():
        return False
    suite = common.read_object(suite_path)
    return (
        suite.get("eval_id") == spec.eval_id
        and suite.get("status")
        == "generation_succeeded_official_humaneval_pending"
    )


def _claim(spec: common.EvalSpec) -> Path | None:
    directory = _official_dir(spec)
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / ".score_claim"
    try:
        claim.mkdir()
    except FileExistsError:
        return None
    common.atomic_json(
        claim / "owner.json",
        {
            "eval_id": spec.eval_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_at": common.now(),
        },
    )
    return claim


def _release_claim(claim: Path) -> None:
    (claim / "owner.json").unlink(missing_ok=True)
    claim.rmdir()


def _next_ready(
    specs: tuple[common.EvalSpec, ...],
) -> tuple[common.EvalSpec, Path] | None:
    for spec in specs:
        if (_official_dir(spec) / "official_success.json").is_file():
            continue
        if _worker_attempt_count(spec) >= MAX_ATTEMPTS or not _ready(spec):
            continue
        claim = _claim(spec)
        if claim is not None:
            return spec, claim
    return None


def _run_one(spec: common.EvalSpec, python: str) -> bool:
    root = _official_dir(spec) / "worker_attempts"
    index = _worker_attempt_count(spec) + 1
    attempt = root / f"attempt{index:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    command = [
        python,
        "-u",
        "-m",
        "experiments.additional_methods_fair20_eval_20260821.score_humaneval",
        "--eval-id",
        spec.eval_id,
    ]
    manifest = {
        "schema_version": 1,
        "status": "running",
        "eval_id": spec.eval_id,
        "attempt": index,
        "hostname": socket.gethostname(),
        "started_at": common.now(),
        "command": command,
    }
    common.atomic_json(attempt / "manifest.json", manifest)
    env = os.environ.copy()
    env.update(
        PYTHONHASHSEED="1234",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=os.pathsep.join(
            value
            for value in (str(common.REPO_ROOT), env.get("PYTHONPATH", ""))
            if value
        ),
    )
    started = time.monotonic()
    with (attempt / "execution.log").open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=common.REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {
        **manifest,
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "finished_at": common.now(),
    }
    common.atomic_json(attempt / "result.json", result)
    return completed.returncode == 0


def execute(poll_seconds: int) -> int:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise common.EvaluationError("official scorer must run on a Canoe pod")
    reference = common.load_reference_manifest()
    specs = common.iter_specs()
    python = common.load_quant_plans()["efficientqat"]["venv_python"]
    print(
        json.dumps(
            {
                "status": "scorer_started",
                "hostname": hostname,
                "reference_fingerprint": reference["fingerprint"],
                "total": len(specs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    while True:
        common.load_reference_manifest()
        completed = sum(
            (_official_dir(spec) / "official_success.json").is_file()
            for spec in specs
        )
        if completed == len(specs):
            return 0
        exhausted = [
            spec.eval_id
            for spec in specs
            if not (_official_dir(spec) / "official_success.json").is_file()
            and _worker_attempt_count(spec) >= MAX_ATTEMPTS
        ]
        if exhausted:
            raise common.EvaluationError(
                f"official scoring retry budget exhausted: {exhausted}"
            )
        ready = _next_ready(specs)
        if ready is None:
            print(
                f"{common.now()} waiting_for_generation "
                f"official_completed={completed}/{len(specs)}",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        spec, claim = ready
        try:
            succeeded = _run_one(spec, python)
            print(
                f"{common.now()} official_end eval_id={spec.eval_id} "
                f"succeeded={succeeded}",
                flush=True,
            )
        finally:
            _release_claim(claim)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    try:
        return execute(args.poll_seconds)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
