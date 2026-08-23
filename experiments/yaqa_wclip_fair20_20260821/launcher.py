#!/usr/bin/env python3
"""Launch or inspect YAQA_wclip stage queues assigned to this pod."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = Path(__file__).with_name("plan.json")


class LaunchError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path, expected_sha: str):
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise LaunchError(f"plan SHA256 mismatch: {actual_sha} != {expected_sha}")
    return json.loads(path.read_text(encoding="utf-8")), actual_sha


def assigned_queues(plan: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    pod = socket.gethostname()
    queues: dict[int, list[dict[str, Any]]] = {}
    for stage in plan["stages"]:
        if stage["canoe_pod"] == pod:
            queues.setdefault(int(stage["physical_gpu"]), []).append(stage)
    if not queues:
        raise LaunchError(f"no YAQA queues assigned to {pod}")
    for gpu, queue in queues.items():
        queue.sort(key=lambda stage: int(stage["queue_order"]))
        if [stage["queue_order"] for stage in queue] != list(range(len(queue))):
            raise LaunchError(f"invalid queue order on GPU {gpu}")
    return dict(sorted(queues.items()))


def stage_state(plan: dict[str, Any], stage: dict[str, Any]) -> str:
    directory = Path(plan["output_root"]) / "stages" / stage["stage_id"]
    if (directory / "stage_receipt.json").is_file():
        return "succeeded"
    if (directory / "failure.json").is_file():
        return "failed"
    if (directory / "stage_manifest.json").is_file():
        return "running_or_interrupted"
    return "not_started"


def status(plan: dict[str, Any], queues: dict[int, list[dict[str, Any]]]) -> None:
    for gpu, queue in queues.items():
        print(
            json.dumps(
                {
                    "physical_gpu": gpu,
                    "stages": [
                        {
                            "stage_id": stage["stage_id"],
                            "kind": stage["kind"],
                            "state": stage_state(plan, stage),
                        }
                        for stage in queue
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def launch(
    plan_path: Path,
    plan_sha: str,
    plan: dict[str, Any],
    queues: dict[int, list[dict[str, Any]]],
) -> None:
    preflight_path = Path(plan["preflight_path"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "succeeded"
        or preflight.get("plan_sha256") != plan_sha
    ):
        raise LaunchError("preflight does not bind this plan")
    python = Path(plan["venv_python"])
    venv_root = python.parent.parent
    torch_lib = venv_root / "lib/python3.12/site-packages/torch/lib"
    if not python.is_file() or not torch_lib.is_dir():
        raise LaunchError("formal Python/torch library is missing")
    log_root = Path(plan["output_root"]) / "queue_logs" / socket.gethostname()
    log_root.mkdir(parents=True, exist_ok=True)
    os.chmod(log_root, 0o755)
    launched = []
    for gpu, queue in queues.items():
        log_path = log_root / f"gpu{gpu}.log"
        if log_path.exists() or log_path.is_symlink():
            raise LaunchError(f"queue log must be fresh: {log_path}")
        if any(stage_state(plan, stage) != "not_started" for stage in queue):
            raise LaunchError(f"GPU {gpu} queue contains claimed stage")
        command = [
            str(python), "-u", "-m",
            "experiments.yaqa_wclip_fair20_20260821.worker",
            "--plan-file", str(plan_path),
            "--expected-plan-sha256", plan_sha,
            "--physical-gpu", str(gpu),
        ]
        env = os.environ.copy()
        env.update(
            {
                "VIRTUAL_ENV": str(venv_root),
                "PATH": f"{venv_root / 'bin'}:{env.get('PATH', '')}",
                "LD_LIBRARY_PATH": f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}",
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": ":".join(
                    (
                        str(REPO_ROOT),
                        str(REPO_ROOT / "YAQA_wclip"),
                        str(REPO_ROOT / "YAQA_wclip/hessian_llama"),
                    )
                ),
                "PYTHONHASHSEED": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        handle = log_path.open("x", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
        os.chmod(log_path, 0o644)
        launched.append(
            {
                "physical_gpu": gpu,
                "pid": process.pid,
                "log": str(log_path),
                "stage_ids": [stage["stage_id"] for stage in queue],
            }
        )
    print(json.dumps(launched, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status"))
    parser.add_argument("--plan-file", default=str(DEFAULT_PLAN))
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan_file).resolve()
    plan, plan_sha = load_plan(plan_path, args.expected_plan_sha256)
    queues = assigned_queues(plan)
    if args.action == "launch":
        launch(plan_path, plan_sha, plan, queues)
    else:
        status(plan, queues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
