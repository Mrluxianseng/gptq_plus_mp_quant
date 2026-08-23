#!/usr/bin/env python3
"""Relaunch the 12 interrupted EfficientQAT runs with bounded CPU threads."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO_ROOT / "experiments/efficientqat_weightonly_15group_20260821/plan.json"
)
EXPECTED_PLAN_SHA256 = (
    "ceffe64189f07a5753a4feffa21553d8c802fa4c798af645ca6199b0fa271032"
)
PACK_SOURCE = REPO_ROOT / "EfficientQAT/quantize/int_linear_real.py"
EXPECTED_PACK_SOURCE_SHA256 = (
    "d259af6def9db43766ecd704a6fc03f04913c8156d81d530f62777fe5defc2dc"
)
REPLAY_RUN_IDS = frozenset(
    {
        "EQ15-Q32-W4",
        "EQ15-Q32-W3",
        "EQ15-Q8-W4",
        "EQ15-Q8-W3",
        "EQ15-Q4-W4",
        "EQ15-Q4-W3",
        "EQ15-L8-W4",
        "EQ15-Q32-W2",
        "EQ15-Q8-W2",
        "EQ15-Q4-W2",
        "EQ15-L8-W3",
        "EQ15-L8-W2",
    }
)
ARCHIVE_NAME = "_superseded_cpu_thread_oversubscription_20260821"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class ReplayError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def execute() -> dict[str, Any]:
    plan_sha = sha256_file(PLAN_PATH)
    if plan_sha != EXPECTED_PLAN_SHA256:
        raise ReplayError(f"plan SHA256 mismatch: {plan_sha}")
    pack_sha = sha256_file(PACK_SOURCE)
    if pack_sha != EXPECTED_PACK_SOURCE_SHA256:
        raise ReplayError(f"pack source SHA256 mismatch: {pack_sha}")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    hostname = socket.gethostname()
    runs = sorted(
        (
            run
            for run in plan["runs"]
            if run["run_id"] in REPLAY_RUN_IDS
            and run["canoe_pod"] == hostname
        ),
        key=lambda run: int(run["physical_gpu"]),
    )
    expected_counts = {
        "j-4mj21jb084-master-0": 7,
        "j-zogxxxduju-master-0": 5,
    }
    if hostname not in expected_counts:
        raise ReplayError(f"replay is not assigned to host: {hostname}")
    expected_count = expected_counts[hostname]
    if len(runs) != expected_count:
        raise ReplayError(
            f"unexpected replay assignment on {hostname}: {len(runs)}"
        )

    output_root = Path(plan["output_root"])
    archive_root = output_root / ARCHIVE_NAME
    for run in runs:
        run_dir = output_root / run["output_subdir"]
        archived = archive_root / run["output_subdir"]
        if run_dir.exists() or run_dir.is_symlink():
            raise ReplayError(f"fresh replay directory still exists: {run_dir}")
        if not (archived / "run_manifest.json").is_file():
            raise ReplayError(f"archived interrupted run is missing: {archived}")

    log_root = output_root / "_cpu_pack_replay" / "queue_logs" / hostname
    receipt_path = log_root / "launcher_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ReplayError(f"launcher receipt already exists: {receipt_path}")
    log_root.mkdir(parents=True, exist_ok=True)

    python = plan["venv_python"]
    determinism = plan["runtime_contract"]["determinism"]
    venv_root = Path(python).parent.parent
    torch_lib = (
        venv_root
        / "lib/python3.12/site-packages/torch/lib"
    )
    if not torch_lib.is_dir():
        raise ReplayError(f"torch library directory is missing: {torch_lib}")
    env_base = os.environ.copy()
    env_base.update(
        VIRTUAL_ENV=str(venv_root),
        PATH=f"{Path(python).parent}:{env_base.get('PATH', '')}",
        LD_LIBRARY_PATH=f"{torch_lib}:{env_base.get('LD_LIBRARY_PATH', '')}",
        PYTHONHASHSEED=str(determinism["python_hash_seed"]),
        CUBLAS_WORKSPACE_CONFIG=determinism["cublas_workspace_config"],
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        **THREAD_ENVIRONMENT,
    )
    rows = []
    handles = []
    try:
        for run in runs:
            log_path = log_root / f"{run['run_id']}.log"
            handle = log_path.open("x", encoding="utf-8")
            handles.append(handle)
            command = [
                python,
                "-u",
                "-m",
                "experiments.efficientqat_weightonly_15group_20260821.worker",
                "--plan-file",
                str(PLAN_PATH),
                "--expected-plan-sha256",
                EXPECTED_PLAN_SHA256,
                "--run-id",
                run["run_id"],
                "--physical-gpu",
                str(run["physical_gpu"]),
            ]
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = str(run["physical_gpu"])
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
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
                    "log": str(log_path),
                    "archived_interrupted_run": str(
                        archive_root / run["output_subdir"]
                    ),
                }
            )
    finally:
        for handle in handles:
            handle.close()

    receipt = {
        "schema_version": 1,
        "status": "launched",
        "reason": "replace 144-thread-per-process CPU packing oversubscription",
        "hostname": hostname,
        "launched_at": now(),
        "plan": str(PLAN_PATH),
        "plan_sha256": plan_sha,
        "pack_source": str(PACK_SOURCE),
        "pack_source_sha256": pack_sha,
        "pack_equivalence_evidence": {
            "synthetic_reference_cases": 48,
            "actual_quantlinear_pack_cases": 36,
            "w_bits": [2, 3, 4],
            "result": "byte_identical",
        },
        "thread_environment": THREAD_ENVIRONMENT,
        "workers": rows,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def main() -> int:
    print(json.dumps(execute(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
