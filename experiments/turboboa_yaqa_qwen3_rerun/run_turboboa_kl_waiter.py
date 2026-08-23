#!/usr/bin/env python3
"""Evaluate the frozen paper-pure TurboBOA checkpoint on WikiText2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
WEIGHT_METHOD = "turboboa_rmsnorm_mean_jacobian_kfac"
EXACT_RE = re.compile(
    r"Exact KL&PPL on wikitext2: "
    r"([-+0-9.eE]+), ([-+0-9.eE]+)"
)


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


def _setting(setting: str) -> tuple[int, int, int, int]:
    if setting == "W3A16KV16":
        return 3, 16, 16, 16
    if setting == "W4A4KV4":
        return 4, 4, 4, 4
    raise ValueError(setting)


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


def _validate_gate(
    payload: dict[str, Any],
    *,
    model: Path,
    artifact: Path,
    setting: str,
) -> None:
    w_bits, a_bits, k_bits, v_bits = _setting(setting)
    configuration = payload.get("configuration", {})
    if (
        payload.get("status") != "quantized_only"
        or Path(payload.get("model", {}).get("path", "")).resolve()
        != model
        or Path(payload.get("quantized_checkpoint", "")).resolve()
        != artifact
        or configuration.get("w_bits") != w_bits
        or configuration.get("a_bits") != a_bits
        or configuration.get("k_bits") != k_bits
        or configuration.get("v_bits") != v_bits
        or configuration.get("group_size") != 128
        or configuration.get("nsamples") != 256
        or configuration.get("seqlen") != 2048
        or configuration.get("seed") != 1
        or configuration.get("calib_data") != "wikitext2"
        or configuration.get("weight_quantizer") != "realq_mse"
        or configuration.get("qparam_comput") != "RealQ-MSE"
        or configuration.get("w_method") != WEIGHT_METHOD
    ):
        raise RuntimeError("paper-pure TurboBOA result provenance mismatch")
    aware_expected = setting == "W4A4KV4"
    if (
        configuration.get("act_quant_aware_gptq") is not aware_expected
        or configuration.get("k_cache_quant_aware_gptq")
        is not aware_expected
    ):
        raise RuntimeError("paper-pure TurboBOA aware-mode mismatch")
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        raise RuntimeError("TurboBOA gate published without checkpoint")


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
    gate = run_dir / "result.json"
    artifact = run_dir / "qmodel.pt"
    output = run_dir / "kl_ppl/result.json"
    state_path = run_dir / "kl_ppl/waiter_state.json"
    log_path = run_dir / "kl_ppl/launcher.log"

    while True:
        if gate.is_file():
            gate_payload = _read_json(gate)
            _validate_gate(
                gate_payload,
                model=model,
                artifact=artifact,
                setting=args.setting,
            )
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

    w_bits, a_bits, k_bits, v_bits = _setting(args.setting)
    command = [
        str(PYTHON),
        "-u",
        "-m",
        "realq_benchmark.ptq",
        "--model",
        str(model),
        "--load_qmodel_path",
        str(artifact),
        "--w_bits",
        str(w_bits),
        "--w_groupsize",
        "128",
        "--a_bits",
        str(a_bits),
        "--a_groupsize",
        "-1",
        "--k_bits",
        str(k_bits),
        "--k_groupsize",
        "-1",
        "--v_bits",
        str(v_bits),
        "--v_groupsize",
        "-1",
        "--require_reference_cache_hit",
        "true",
        "--eval_datasets",
        "wikitext2",
        "--eval_seq_len",
        "2048",
        "--kl_topk",
        "-1",
        "--skip_eval",
        "false",
        "--skip_kl_ppl_eval",
        "false",
        "--lm_eval",
        "false",
        "--reasoning_eval",
        "false",
        "--output_dir",
        str(run_dir / "kl_ppl/runtime"),
        "--exp",
        f"formal_kl_ppl_turboboa_qwen3_32b_{args.setting.lower()}",
    ]
    if args.setting == "W4A4KV4":
        command.extend(
            [
                "--act_quant_aware_gptq",
                "true",
                "--k_cache_quant_aware_gptq",
                "true",
                "--a_clip_ratio",
                "0.9",
                "--k_clip_ratio",
                "0.9",
                "--v_clip_ratio",
                "0.9",
            ]
        )
    checkpoint_stat = artifact.stat()
    gate_sha = _sha256(gate)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(12920 + args.gpu),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
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
            "gate_sha256": gate_sha,
            "checkpoint_bytes": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
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
        raise RuntimeError(
            f"TurboBOA KL/PPL exited with code {completed.returncode}"
        )
    current_stat = artifact.stat()
    if (
        current_stat.st_size != checkpoint_stat.st_size
        or current_stat.st_mtime_ns != checkpoint_stat.st_mtime_ns
        or _sha256(gate) != gate_sha
    ):
        raise RuntimeError("TurboBOA artifact changed during KL/PPL")
    matches = EXACT_RE.findall(log_path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact KL/PPL line, got {matches}")
    kl_raw, ppl = (float(value) for value in matches[0])
    if (
        not math.isfinite(kl_raw)
        or not math.isfinite(ppl)
        or kl_raw < 0
        or ppl <= 0
    ):
        raise RuntimeError(f"invalid KL/PPL: kl={kl_raw}, ppl={ppl}")
    result = {
        "schema_version": 1,
        "status": "evaluated",
        "method": "TurboBOA",
        "model": str(model),
        "setting": args.setting,
        "checkpoint": str(artifact),
        "checkpoint_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "quantization_gate": str(gate),
        "quantization_gate_sha256": gate_sha,
        "hostname": hostname,
        "gpu": args.gpu,
        "started_at": started_at,
        "finished_at": _now(),
        "evaluation": {
            "dataset": "wikitext2",
            "split": "test",
            "sequence_length": 2048,
            "kl_direction": "KL(FP||quantized)",
            "kl_full_vocabulary_fp32": True,
            "kl_topk": -1,
            "kl_raw": kl_raw,
            "kl_x100": kl_raw * 100.0,
            "ppl": ppl,
            "reference_cache_required": True,
        },
    }
    _write_atomic(output, result)
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
