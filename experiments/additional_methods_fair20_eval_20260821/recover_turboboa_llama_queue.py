#!/usr/bin/env python3
"""Reserve one managed GPU and recover the four TurboBOA Llama suites."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from . import common
from .recover_turboboa_llama import ALLOWED_EVAL_IDS


LOCK_ROOT = common.DATA_ROOT / "_fair20_physical_gpu_locks_20260821"
EVALPLUS_RUNTIME = (
    common.OUTPUT_ROOT / "_runtime_dependencies" / "evalplus_only_0.3.1"
)


class QueueError(RuntimeError):
    """The recovery queue could not reserve or complete its exact matrix."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _claim(eval_id: str, hostname: str, physical_gpu: int) -> Path | None:
    suite_dir = common.OUTPUT_ROOT / "evals" / eval_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    if (suite_dir / "suite_success.json").is_file():
        return None
    claim = suite_dir / ".claim"
    try:
        claim.mkdir()
    except FileExistsError as exc:
        owner_path = claim / "owner.json"
        if not owner_path.is_file():
            raise QueueError(f"suite has an ownerless claim: {eval_id}") from exc
        owner = common.read_object(owner_path)
        previous_pid = int(owner.get("pid", -1))
        resumable = (
            owner.get("eval_id") == eval_id
            and owner.get("hostname") == hostname
            and int(owner.get("physical_gpu", -1)) == physical_gpu
            and owner.get("kind") == "turboboa_llama_contract_recovery_queue"
            and not Path(f"/proc/{previous_pid}").exists()
        )
        if not resumable:
            raise QueueError(f"suite is already claimed: {eval_id}") from exc
        common.atomic_json(
            owner_path,
            {
                **owner,
                "pid": os.getpid(),
                "previous_pid": previous_pid,
                "resumed_at": _now(),
            },
        )
        return claim
    common.atomic_json(
        claim / "owner.json",
        {
            "eval_id": eval_id,
            "hostname": hostname,
            "pid": os.getpid(),
            "physical_gpu": physical_gpu,
            "kind": "turboboa_llama_contract_recovery_queue",
            "claimed_at": _now(),
        },
    )
    return claim


def _runtime_environment(physical_gpu: int) -> dict[str, str]:
    if not (EVALPLUS_RUNTIME / "evalplus").is_dir() or not (
        EVALPLUS_RUNTIME / "evalplus-0.3.1.dist-info"
    ).is_dir():
        raise QueueError(f"audited EvalPlus runtime is missing: {EVALPLUS_RUNTIME}")
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(physical_gpu),
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
                env.get("PYTHONPATH", ""),
            )
            if value
        ),
    )
    return env


def _validate_runtime(python: str, env: dict[str, str]) -> None:
    probe = subprocess.run(
        [
            python,
            "-c",
            (
                "import importlib.metadata as m; import evalplus; "
                "assert m.version('evalplus') == '0.3.1'"
            ),
        ],
        cwd=common.REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise QueueError(
            "EvalPlus runtime preflight failed before claiming a GPU: "
            f"{probe.stderr[-2000:]}"
        )


def _release(claim: Path) -> None:
    (claim / "owner.json").unlink(missing_ok=True)
    claim.rmdir()


def execute(physical_gpu: int) -> dict[str, object]:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise QueueError("queue must run on a Canoe pod")
    common.load_reference_manifest()
    python = common.load_quant_plans()["efficientqat"]["venv_python"]
    env = _runtime_environment(physical_gpu)
    _validate_runtime(python, env)
    claims: dict[str, Path] = {}
    for eval_id in ALLOWED_EVAL_IDS:
        claim = _claim(eval_id, hostname, physical_gpu)
        if claim is not None:
            claims[eval_id] = claim
    if not claims:
        return {"status": "already_complete", "eval_ids": list(ALLOWED_EVAL_IDS)}

    queue_root = common.OUTPUT_ROOT / "turboboa_llama_contract_recovery"
    queue_root.mkdir(parents=True, exist_ok=True)
    receipt_path = queue_root / f"{hostname}_gpu{physical_gpu}.json"
    if receipt_path.exists():
        raise QueueError(f"queue receipt already exists: {receipt_path}")
    lock_path = LOCK_ROOT / hostname / f"gpu{physical_gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    started = time.monotonic()
    completed: list[str] = []
    try:
        with lock_path.open("a+", encoding="utf-8") as handle:
            print(
                f"{_now()} waiting_for_physical_gpu_lock={lock_path}",
                flush=True,
            )
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            print(f"{_now()} acquired_physical_gpu={physical_gpu}", flush=True)
            for eval_id in ALLOWED_EVAL_IDS:
                claim = claims.get(eval_id)
                if claim is None:
                    continue
                command = [
                    python,
                    "-u",
                    "-m",
                    (
                        "experiments.additional_methods_fair20_eval_20260821."
                        "recover_turboboa_llama"
                    ),
                    "--eval-id",
                    eval_id,
                ]
                print(f"{_now()} recovery_start eval_id={eval_id}", flush=True)
                result = subprocess.run(
                    command,
                    cwd=common.REPO_ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    check=False,
                )
                if result.returncode != 0:
                    raise QueueError(
                        f"recovery failed for {eval_id}: returncode={result.returncode}"
                    )
                completed.append(eval_id)
                _release(claim)
                del claims[eval_id]
                print(f"{_now()} recovery_end eval_id={eval_id}", flush=True)
    except BaseException:
        # Keep uncompleted claims as explicit reservations.  A monitored retry
        # must inspect the failed recovery before releasing them.
        raise

    receipt = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": "turboboa_llama_contract_recovery_queue",
        "hostname": hostname,
        "physical_gpu": physical_gpu,
        "eval_ids": list(ALLOWED_EVAL_IDS),
        "completed": completed,
        "started_at": started_at,
        "finished_at": _now(),
        "wall_seconds": time.monotonic() - started,
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": common.sha256_file(Path(__file__).resolve()),
        },
        "runtime_dependency": {
            "evalplus_version": "0.3.1",
            "path": str(EVALPLUS_RUNTIME.resolve()),
        },
    }
    common.atomic_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.physical_gpu)
    except BaseException:
        traceback.print_exc()
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
