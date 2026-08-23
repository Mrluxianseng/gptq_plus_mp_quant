#!/usr/bin/env python3
"""Launch or inspect the EfficientQAT workers assigned to the current pod."""

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


def load_plan(path: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise LaunchError(f"plan SHA256 mismatch: {actual} != {expected_sha}")
    return json.loads(path.read_text(encoding="utf-8")), actual


def assigned_runs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    hostname = socket.gethostname()
    runs = [run for run in plan["runs"] if run["canoe_pod"] == hostname]
    if not runs:
        raise LaunchError(f"no runs assigned to current pod: {hostname}")
    gpus = [int(run["physical_gpu"]) for run in runs]
    if len(gpus) != len(set(gpus)):
        raise LaunchError("two runs are assigned to the same physical GPU")
    if any(gpu < 0 or gpu > 7 for gpu in gpus):
        raise LaunchError("physical GPU is outside 0..7")
    return sorted(runs, key=lambda run: int(run["physical_gpu"]))


def state(plan: dict[str, Any], run: dict[str, Any]) -> str:
    run_dir = Path(plan["output_root"]) / run["output_subdir"]
    if (run_dir / "quantization_result.json").is_file():
        return "quantization_succeeded"
    if (run_dir / "failure.json").is_file():
        return "failed"
    if (run_dir / "run_manifest.json").is_file():
        return "running_or_interrupted"
    return "not_started"


def print_status(plan: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    for run in runs:
        print(
            json.dumps(
                {
                    "run_id": run["run_id"],
                    "model": run["model"],
                    "w_bits": run["w_bits"],
                    "physical_gpu": run["physical_gpu"],
                    "state": state(plan, run),
                    "output_subdir": run["output_subdir"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def launch(
    plan_path: Path,
    plan_sha: str,
    plan: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    preflight_path = Path(plan["preflight_path"])
    if not preflight_path.is_file():
        raise LaunchError(f"preflight is missing: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "succeeded"
        or preflight.get("plan_sha256") != plan_sha
    ):
        raise LaunchError("preflight does not bind the current plan")
    python = Path(plan["venv_python"])
    if not python.is_file():
        raise LaunchError(f"venv Python is missing: {python}")
    venv_root = python.parent.parent
    torch_lib = (
        venv_root
        / f"lib/python{plan['runtime_contract']['versions']['python'].rsplit('.', 1)[0]}"
        / "site-packages/torch/lib"
    )
    if not torch_lib.is_dir():
        raise LaunchError(f"venv torch library directory is missing: {torch_lib}")
    nonfresh = [(run["run_id"], state(plan, run)) for run in runs if state(plan, run) != "not_started"]
    if nonfresh:
        raise LaunchError(f"refusing repeated or mixed launch: {nonfresh}")

    log_root = REPO_ROOT / "logs" / plan["campaign"] / socket.gethostname()
    log_root.mkdir(parents=True, exist_ok=True)
    launched = []
    for run in runs:
        command = [
            str(python),
            "-m",
            "experiments.efficientqat_weightonly_15group_20260821.worker",
            "--plan-file",
            str(plan_path),
            "--expected-plan-sha256",
            plan_sha,
            "--run-id",
            run["run_id"],
            "--physical-gpu",
            str(run["physical_gpu"]),
        ]
        env = os.environ.copy()
        env.update(
            {
                "VIRTUAL_ENV": str(venv_root),
                "PATH": f"{venv_root / 'bin'}:{env.get('PATH', '')}",
                "LD_LIBRARY_PATH": f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}",
                "CUDA_VISIBLE_DEVICES": str(run["physical_gpu"]),
                "PYTHONHASHSEED": str(
                    plan["runtime_contract"]["determinism"]["python_hash_seed"]
                ),
                "CUBLAS_WORKSPACE_CONFIG": plan["runtime_contract"]["determinism"][
                    "cublas_workspace_config"
                ],
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        log_path = log_root / f"{run['run_id']}.log"
        log_handle = log_path.open("x", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        launched.append(
            {
                "run_id": run["run_id"],
                "physical_gpu": run["physical_gpu"],
                "pid": process.pid,
                "log": str(log_path),
            }
        )
    print(json.dumps(launched, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "launch", "status"))
    parser.add_argument("--plan-file", default=str(DEFAULT_PLAN))
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args(argv)
    plan_path = Path(args.plan_file).resolve()
    plan, plan_sha = load_plan(plan_path, args.expected_plan_sha256)
    runs = assigned_runs(plan)
    if args.action == "launch":
        launch(plan_path, plan_sha, plan, runs)
    else:
        print_status(plan, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
