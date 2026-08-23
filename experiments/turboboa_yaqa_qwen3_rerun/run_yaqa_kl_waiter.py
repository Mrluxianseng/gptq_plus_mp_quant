#!/usr/bin/env python3
"""Wait for a YAQA artifact and run its shared-protocol KL/PPL on one GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv/bin/python"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _gpu_free(gpu: int) -> bool:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    return not any(
        line.strip().isdigit() for line in completed.stdout.splitlines()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=("W3A16KV16", "W4A4KV4"),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--expected-hostname", required=True)
    args = parser.parse_args()

    hostname = socket.gethostname()
    if (
        hostname != args.expected_hostname
        or not re.fullmatch(r"j-[a-z0-9]+-master-0", hostname)
    ):
        raise RuntimeError(
            f"KL waiter bound to {args.expected_hostname}, got {hostname}"
        )
    if not 0 <= args.gpu <= 7:
        raise ValueError("--gpu must be in [0,7]")

    model = Path(args.model).resolve()
    run_dir = Path(args.run_dir).resolve()
    gate = run_dir / "quantization_result.json"
    hf_dir = run_dir / "hf"
    validation = run_dir / "hf_validation.json"
    output = run_dir / "kl_ppl/result.json"
    state_path = run_dir / "kl_ppl/waiter_state.json"
    log_path = run_dir / "kl_ppl/launcher.log"

    while True:
        if gate.is_file():
            payload = _read_json(gate)
            if payload.get("status") == "failed":
                raise RuntimeError("YAQA quantization pipeline failed")
            if payload.get("status") == "quantization_succeeded":
                if (
                    Path(payload.get("model", "")).resolve() != model
                    or payload.get("setting") != args.setting
                    or Path(payload.get("hf_dir", "")).resolve() != hf_dir
                    or Path(payload.get("hf_validation", "")).resolve()
                    != validation
                ):
                    raise RuntimeError("YAQA quantization gate mismatch")
                expected_validation_sha = payload.get(
                    "hf_validation_sha256"
                )
                if (
                    not isinstance(expected_validation_sha, str)
                    or len(expected_validation_sha) != 64
                    or _sha256(validation) != expected_validation_sha
                ):
                    raise RuntimeError("YAQA validation receipt hash mismatch")
                break
        _write_atomic(
            state_path,
            {
                "schema_version": 1,
                "status": "waiting_for_quantized_artifact",
                "gate": str(gate),
                "updated_at": _now(),
                "hostname": hostname,
            },
        )
        time.sleep(20)

    while not _gpu_free(args.gpu):
        _write_atomic(
            state_path,
            {
                "schema_version": 1,
                "status": "waiting_for_gpu",
                "gpu": args.gpu,
                "updated_at": _now(),
                "hostname": hostname,
            },
        )
        time.sleep(20)

    command = [
        str(PYTHON),
        "-u",
        "-m",
        "experiments.turboboa_yaqa_qwen3_rerun.eval_yaqa_kl_ppl",
        "--model",
        str(model),
        "--setting",
        args.setting,
        "--hf-dir",
        str(hf_dir),
        "--hf-validation",
        str(validation),
        "--expected-validation-sha256",
        expected_validation_sha,
        "--output",
        str(output),
    ]
    env = os.environ.copy()
    python_paths = [
        str(REPO_ROOT / "YAQA_wclip"),
        str(REPO_ROOT / "YAQA_wclip/qtip-kernels"),
        str(REPO_ROOT),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": os.pathsep.join(python_paths),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started_at = _now()
    _write_atomic(
        state_path,
        {
            "schema_version": 1,
            "status": "running",
            "command": command,
            "gpu": args.gpu,
            "started_at": started_at,
            "hostname": hostname,
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        _write_atomic(
            state_path,
            {
                "schema_version": 1,
                "status": "failed",
                "returncode": completed.returncode,
                "gpu": args.gpu,
                "finished_at": _now(),
                "hostname": hostname,
            },
        )
        raise RuntimeError(
            f"YAQA KL/PPL exited with code {completed.returncode}"
        )
    result = _read_json(output)
    evaluation = result.get("evaluation", {})
    if (
        result.get("status") != "evaluated"
        or result.get("hf_validation_sha256")
        != expected_validation_sha
        or evaluation.get("dataset") != "wikitext2"
        or evaluation.get("sequence_length") != 2048
        or evaluation.get("kl_direction") != "KL(FP||quantized)"
        or evaluation.get("kl_full_vocabulary_fp32") is not True
    ):
        raise RuntimeError("YAQA KL/PPL result audit failed")
    _write_atomic(
        state_path,
        {
            "schema_version": 1,
            "status": "succeeded",
            "result": str(output),
            "result_sha256": _sha256(output),
            "gpu": args.gpu,
            "started_at": started_at,
            "finished_at": _now(),
            "hostname": hostname,
        },
    )


if __name__ == "__main__":
    main()
