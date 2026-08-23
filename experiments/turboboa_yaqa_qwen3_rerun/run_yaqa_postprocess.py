#!/usr/bin/env python3
"""Wait for a YAQA Hessian, validate, quantize, hfize, and hand off eval."""

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


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(
    *,
    stage: str,
    command: list[str],
    log_path: Path,
    state_path: Path,
    state: dict[str, Any],
    env: dict[str, str],
) -> None:
    state.update(
        {
            "status": "running",
            "stage": stage,
            "command": command,
            "stage_started_at": _now(),
            "updated_at": _now(),
        }
    )
    _write_atomic(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as handle:
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
        state.update(
            {
                "status": "failed",
                "stage": stage,
                "returncode": int(completed.returncode),
                "failed_at": _now(),
                "updated_at": _now(),
            }
        )
        _write_atomic(state_path, state)
        raise RuntimeError(
            f"{stage} failed with code {completed.returncode}; see {log_path}"
        )


def _wait_hessian(
    timing_path: Path,
    hessian_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    while True:
        if timing_path.is_file():
            timing = _read_json(timing_path)
            if timing.get("status") == "completed":
                if (
                    timing.get("returncode") != 0
                    or not (hessian_dir / "manifest.json").is_file()
                ):
                    raise RuntimeError(
                        "completed Hessian timing has no valid artifact"
                    )
                return timing
            if timing.get("status") == "failed":
                raise RuntimeError("Hessian collection failed")
        state.update(
            {
                "status": "waiting_for_hessian",
                "stage": "hessian",
                "updated_at": _now(),
            }
        )
        _write_atomic(state_path, state)
        time.sleep(20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=("W3A16KV16", "W4A4KV4"),
    )
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--hessian-dir", required=True)
    parser.add_argument("--hessian-timing", required=True)
    parser.add_argument("--calib-path", required=True)
    parser.add_argument("--calib-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    hostname = socket.gethostname()
    if hostname != args.expected_hostname or not re.fullmatch(
        r"j-[a-z0-9]+-master-0", hostname
    ):
        raise RuntimeError(
            f"postprocess pipeline bound to {args.expected_hostname}, got {hostname}"
        )
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(
        gpu < 0 or gpu > 7 for gpu in gpus
    ):
        raise ValueError("--gpus must contain four distinct indices in [0,7]")

    model = Path(args.model).resolve()
    hessian_dir = Path(args.hessian_dir).resolve()
    hessian_timing = Path(args.hessian_timing).resolve()
    calib_path = Path(args.calib_path).resolve()
    run_dir = Path(args.run_dir).resolve()
    validation = run_dir / "hessian_validation.json"
    raw_dir = run_dir / "raw"
    raw_timing = run_dir / "raw_quantization_timing.json"
    hf_dir = run_dir / "hf"
    hf_validation = run_dir / "hf_validation.json"
    result_path = run_dir / "quantization_result.json"
    state_path = run_dir / "pipeline_state.json"
    logs = run_dir / "pipeline_logs"
    for path in (
        validation,
        raw_dir,
        raw_timing,
        hf_dir,
        hf_validation,
        result_path,
    ):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"postprocess output must be fresh: {path}")

    state: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "hostname": hostname,
        "model": str(model),
        "setting": args.setting,
        "gpus": gpus,
        "hessian_dir": str(hessian_dir),
        "hessian_timing": str(hessian_timing),
        "started_at": _now(),
    }
    _write_atomic(state_path, state)
    hessian_timing_payload = _wait_hessian(
        hessian_timing,
        hessian_dir,
        state_path,
        state,
    )

    base_env = os.environ.copy()
    python_paths = [
        str(REPO_ROOT / "YAQA_wclip"),
        str(REPO_ROOT),
        str(REPO_ROOT / "YAQA_wclip/qtip-kernels"),
    ]
    if base_env.get("PYTHONPATH"):
        python_paths.append(base_env["PYTHONPATH"])
    base_env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ALLOC_CONF": "max_split_size_mb:512",
        }
    )

    akv_mode = "aware" if args.setting == "W4A4KV4" else "n/a"
    validation_env = base_env.copy()
    validation_env["CUDA_VISIBLE_DEVICES"] = ""
    _stage(
        stage="validate_hessian",
        command=[
            str(PYTHON),
            "-u",
            "-m",
            "experiments.yaqa_compare.validate_hessian",
            "--hessian-dir",
            str(hessian_dir),
            "--model",
            str(model),
            "--output",
            str(validation),
            "--expected-samples",
            "256",
            "--expected-seq-len",
            "2048",
            "--expected-calib-path",
            str(calib_path),
            "--expected-calib-sha256",
            args.calib_sha256,
            "--expected-world-size",
            "4",
            "--expected-batch-size-per-rank",
            "2",
            "--expected-akv-mode",
            akv_mode,
        ],
        log_path=logs / "validate_hessian.log",
        state_path=state_path,
        state=state,
        env=validation_env,
    )

    w_bits = 4 if args.setting == "W4A4KV4" else 3
    akv_bits = 4 if args.setting == "W4A4KV4" else 16
    clip_ratio = "0.9" if akv_bits == 4 else "1.0"
    quant_command = [
        str(PYTHON),
        "-u",
        str(REPO_ROOT / "tools/record_gpu_stage.py"),
        "--output",
        str(raw_timing),
        "--run-id",
        args.run_id,
        "--stage",
        "yaqa_raw_quantization",
        "--gpu-count",
        "4",
        "--",
        str(PYTHON),
        "-u",
        str(
            REPO_ROOT
            / "YAQA_wclip/quantize_llama/quantize_finetune_llama.py"
        ),
        "--save_path",
        str(raw_dir),
        "--hess_path",
        str(hessian_dir),
        "--hessian_validation_path",
        str(validation),
        "--base_model",
        str(model),
        "--setting",
        args.setting,
        "--seed",
        "0",
        "--num_cpu_threads",
        "8",
        "--ctx_size",
        "2048",
        "--sigma_reg",
        "1e-2",
        "--scale_override",
        "1.0",
        "--codebook",
        "realq_wclip",
        "--ft_epochs",
        "0",
        "--td_x",
        "1",
        "--td_y",
        "128",
        "--L",
        "16",
        "--K",
        str(w_bits),
        "--V",
        "1",
        "--tlut_bits",
        "0",
        "--decode_mode",
        "dense_fake_quant",
        "--calib_tokens_path",
        str(calib_path),
        "--calib_tokens_sha256",
        args.calib_sha256,
        "--calib_n_seqs",
        "256",
        "--a_bits",
        str(akv_bits),
        "--k_bits",
        str(akv_bits),
        "--v_bits",
        str(akv_bits),
        "--akv_groupsize",
        "-1",
        "--akv_clip_ratio",
        clip_ratio,
        "--w_bits",
        str(w_bits),
        "--w_groupsize",
        "128",
    ]
    if akv_bits == 4:
        quant_command.append("--akv_aware_hessian")
    quant_env = base_env.copy()
    quant_env["CUDA_VISIBLE_DEVICES"] = args.gpus
    _stage(
        stage="raw_quantization",
        command=quant_command,
        log_path=logs / "raw_quantization.log",
        state_path=state_path,
        state=state,
        env=quant_env,
    )

    hfize_env = base_env.copy()
    hfize_env["CUDA_VISIBLE_DEVICES"] = ""
    _stage(
        stage="hfize",
        command=[
            str(PYTHON),
            "-u",
            str(REPO_ROOT / "YAQA_wclip/quantize_llama/hfize_llama.py"),
            "--quantized_path",
            str(raw_dir),
            "--hf_output_path",
            str(hf_dir),
        ],
        log_path=logs / "hfize.log",
        state_path=state_path,
        state=state,
        env=hfize_env,
    )

    hf_validate_env = base_env.copy()
    hf_validate_env["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    _stage(
        stage="validate_hfized",
        command=[
            str(PYTHON),
            "-u",
            str(REPO_ROOT / "YAQA_wclip/eval/validate_hfized_wclip.py"),
            "--model",
            str(model),
            "--setting",
            args.setting,
            "--raw-dir",
            str(raw_dir),
            "--hf-dir",
            str(hf_dir),
            "--output",
            str(hf_validation),
        ],
        log_path=logs / "validate_hfized.log",
        state_path=state_path,
        state=state,
        env=hf_validate_env,
    )

    raw_timing_payload = _read_json(raw_timing)
    if (
        raw_timing_payload.get("status") != "completed"
        or raw_timing_payload.get("returncode") != 0
    ):
        raise RuntimeError("raw quantization timing receipt is incomplete")
    hessian_gpu_hours = float(hessian_timing_payload["gpu_hours"])
    raw_gpu_hours = float(raw_timing_payload["gpu_hours"])
    result = {
        "schema_version": 1,
        "status": "quantization_succeeded",
        "run_id": args.run_id,
        "method": "YAQA_wclip",
        "hostname": hostname,
        "model": str(model),
        "setting": args.setting,
        "calibration": {
            "dataset": "wikitext2_train",
            "samples": 256,
            "sequence_length": 2048,
            "seed": 1,
            "artifact": str(calib_path),
            "sha256": args.calib_sha256,
        },
        "hessian_dir": str(hessian_dir),
        "hessian_validation": str(validation),
        "hessian_validation_sha256": _sha256(validation),
        "raw_dir": str(raw_dir),
        "hf_dir": str(hf_dir),
        "hf_validation": str(hf_validation),
        "hf_validation_sha256": _sha256(hf_validation),
        "quantization_gpu_hours": {
            "hessian_precompute": hessian_gpu_hours,
            "raw_weight_quantization": raw_gpu_hours,
            "total": hessian_gpu_hours + raw_gpu_hours,
        },
        "timing": {
            "hessian": hessian_timing_payload,
            "raw_weight_quantization": raw_timing_payload,
            "excluded": [
                "hfize",
                "artifact_validation",
                "reasoning_evaluation",
                "checkpoint_serialization",
            ],
        },
        "finished_at": _now(),
    }
    _write_atomic(result_path, result)
    state.update(
        {
            "status": "quantization_succeeded",
            "stage": "reasoning_handoff",
            "quantization_result": str(result_path),
            "quantization_result_sha256": _sha256(result_path),
            "updated_at": _now(),
        }
    )
    _write_atomic(state_path, state)

    reasoning_command = [
        str(PYTHON),
        "-u",
        "-m",
        "experiments.turboboa_yaqa_qwen3_rerun.run_reasoning_group",
        "--method",
        "yaqa_wclip",
        "--model",
        str(model),
        "--setting",
        args.setting,
        "--artifact",
        str(hf_dir),
        "--gate",
        str(result_path),
        "--hf-validation",
        str(hf_validation),
        "--gpus",
        ",".join(str(gpu) for gpu in gpus[:3]),
        "--expected-hostname",
        hostname,
        "--run-dir",
        str(run_dir),
    ]
    _stage(
        stage="reasoning",
        command=reasoning_command,
        log_path=logs / "reasoning_group.log",
        state_path=state_path,
        state=state,
        env=base_env,
    )
    state.update(
        {
            "status": "completed",
            "stage": "completed",
            "finished_at": _now(),
            "updated_at": _now(),
        }
    )
    _write_atomic(state_path, state)


if __name__ == "__main__":
    main()
