#!/usr/bin/env python3
"""Dynamic one-GPU worker for the 55 additional-method evaluation suites."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from typing import Any, Mapping

from . import common


LOCK_ROOT = common.DATA_ROOT / "_fair20_physical_gpu_locks_20260821"
MAX_ATTEMPTS = 3


def _reference_manifest() -> dict[str, Any]:
    return common.load_reference_manifest()


def _lane_quant_dependencies(
    hostname: str,
    gpu: int,
    plans: Mapping[str, Mapping[str, Any]],
) -> list[Path]:
    paths: list[Path] = []
    tb = plans["turboboa"]
    for run in tb["runs"]:
        if run["canoe_pod"] != hostname or int(run["physical_gpu"]) != gpu:
            continue
        paths.append(
            Path(tb["output_root"])
            / "runs"
            / run["model"]
            / run["setting"].lower()
            / run["run_id"]
            / "run_receipt.json"
        )
    yaqa = plans["yaqa_wclip"]
    for stage in yaqa["stages"]:
        if (
            stage["canoe_pod"] == hostname
            and int(stage["physical_gpu"]) == gpu
        ):
            paths.append(
                Path(yaqa["output_root"])
                / "stages"
                / stage["stage_id"]
                / "stage_receipt.json"
            )
    return paths


def _lane_failure(hostname: str, gpu: int, plans) -> Path | None:
    tb = plans["turboboa"]
    for run in tb["runs"]:
        if run["canoe_pod"] == hostname and int(run["physical_gpu"]) == gpu:
            path = (
                Path(tb["output_root"])
                / "runs"
                / run["model"]
                / run["setting"].lower()
                / run["run_id"]
                / "failure.json"
            )
            if path.is_file():
                return path
    yaqa = plans["yaqa_wclip"]
    for stage in yaqa["stages"]:
        if stage["canoe_pod"] == hostname and int(stage["physical_gpu"]) == gpu:
            path = (
                Path(yaqa["output_root"])
                / "stages"
                / stage["stage_id"]
                / "failure.json"
            )
            if path.is_file():
                return path
    return None


def _suite_dir(spec: common.EvalSpec) -> Path:
    return common.OUTPUT_ROOT / "evals" / spec.eval_id


def _attempt_count(directory: Path) -> int:
    attempts = directory / "attempts"
    return sum(
        path.is_dir()
        for path in attempts.glob("attempt[0-9][0-9][0-9]")
    )


def _claim(spec: common.EvalSpec) -> Path | None:
    directory = _suite_dir(spec)
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / ".claim"
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
    owner = claim / "owner.json"
    owner.unlink(missing_ok=True)
    claim.rmdir()


def _next_ready(specs: tuple[common.EvalSpec, ...]) -> tuple[common.EvalSpec, Path] | None:
    for spec in specs:
        directory = _suite_dir(spec)
        if (directory / "suite_success.json").is_file():
            continue
        if _attempt_count(directory) >= MAX_ATTEMPTS:
            continue
        try:
            common.resolve_completed_artifact(spec)
        except common.EvaluationError:
            continue
        claim = _claim(spec)
        if claim is not None:
            return spec, claim
    return None


def _run_one(
    spec: common.EvalSpec,
    gpu: int,
    reference_manifest: Mapping[str, Any],
) -> bool:
    directory = _suite_dir(spec)
    attempt_index = _attempt_count(directory) + 1
    attempt = directory / "attempts" / f"attempt{attempt_index:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    reference = reference_manifest["references"][spec.model]
    plans = common.load_quant_plans()
    python = plans["efficientqat"]["venv_python"]
    command = [
        python,
        "-u",
        "-m",
        "experiments.additional_methods_fair20_eval_20260821.run_suite",
        "--eval-id",
        spec.eval_id,
        "--expected-reference-sha256",
        reference["sha256"],
        "--expected-reference-fingerprint",
        reference_manifest["fingerprint"],
        "--output-dir",
        str(directory),
    ]
    manifest = {
        "schema_version": 1,
        "status": "running",
        "eval_id": spec.eval_id,
        "attempt": attempt_index,
        "hostname": socket.gethostname(),
        "physical_gpu": gpu,
        "started_at": common.now(),
        "reference": reference,
        "command": command,
    }
    common.atomic_json(attempt / "manifest.json", manifest)
    env = os.environ.copy()
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
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
                env.get("PYTHONPATH", ""),
            )
            if value
        ),
    )
    started = time.monotonic()
    with (attempt / "execution.log").open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=common.REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record = {
        **manifest,
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "finished_at": common.now(),
    }
    common.atomic_json(attempt / "result.json", record)
    return completed.returncode == 0


def execute(args: argparse.Namespace) -> int:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise common.EvaluationError("evaluation worker must run on a Canoe pod")
    plans = common.load_quant_plans()
    reference_manifest = _reference_manifest()
    specs = common.iter_specs(plans)
    dependencies = _lane_quant_dependencies(hostname, args.physical_gpu, plans)
    if not dependencies:
        raise common.EvaluationError(
            f"no quantization lane exists for {hostname} GPU{args.physical_gpu}"
        )
    lock = LOCK_ROOT / hostname / f"gpu{args.physical_gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)

    while True:
        failure = _lane_failure(hostname, args.physical_gpu, plans)
        if failure is not None:
            raise common.EvaluationError(
                f"quantization lane failed; refusing evaluation: {failure}"
            )
        with lock.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if not all(path.is_file() for path in dependencies):
                # A quantization worker was already queued for this lock.  Give
                # it priority even if flock wake-up order changes.
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                time.sleep(args.poll_seconds)
                continue
            while True:
                completed = sum(
                    (_suite_dir(spec) / "suite_success.json").is_file()
                    for spec in specs
                )
                if completed == len(specs):
                    return 0
                ready = _next_ready(specs)
                if ready is None:
                    exhausted = [
                        spec.eval_id
                        for spec in specs
                        if not (_suite_dir(spec) / "suite_success.json").is_file()
                        and _attempt_count(_suite_dir(spec)) >= MAX_ATTEMPTS
                    ]
                    if exhausted:
                        raise common.EvaluationError(
                            f"evaluation retry budget exhausted: {exhausted}"
                        )
                    print(
                        f"{common.now()} waiting_for_ready_eval "
                        f"completed={completed}/{len(specs)}",
                        flush=True,
                    )
                    time.sleep(args.poll_seconds)
                    continue
                spec, claim = ready
                # Recheck every frozen source and every cache stat identity at
                # the task boundary, not only when this persistent worker was
                # first launched.
                reference_manifest = _reference_manifest()
                print(
                    f"{common.now()} eval_start eval_id={spec.eval_id} "
                    f"gpu={args.physical_gpu}",
                    flush=True,
                )
                try:
                    succeeded = _run_one(
                        spec,
                        args.physical_gpu,
                        reference_manifest,
                    )
                    print(
                        f"{common.now()} eval_end eval_id={spec.eval_id} "
                        f"succeeded={succeeded}",
                        flush=True,
                    )
                finally:
                    _release_claim(claim)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    try:
        return execute(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
