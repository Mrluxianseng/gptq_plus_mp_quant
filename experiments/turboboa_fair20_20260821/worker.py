#!/usr/bin/env python3
"""Wait for one physical GPU and execute its immutable TurboBOA run queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


class WorkerError(RuntimeError):
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


def load_contract(args: argparse.Namespace):
    plan_path = Path(args.plan_file).resolve()
    plan_sha = sha256_file(plan_path)
    if plan_sha != args.expected_plan_sha256:
        raise WorkerError("plan SHA256 mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if sha256_file(plan["source_plan"]) != plan["source_plan_sha256"]:
        raise WorkerError("source comparison plan SHA256 mismatch")
    source = json.loads(Path(plan["source_plan"]).read_text(encoding="utf-8"))
    pod = socket.gethostname()
    queue = sorted(
        (
            run
            for run in plan["runs"]
            if run["canoe_pod"] == pod
            and int(run["physical_gpu"]) == args.physical_gpu
        ),
        key=lambda run: int(run["queue_order"]),
    )
    if not queue:
        raise WorkerError(f"no queue for {pod} GPU {args.physical_gpu}")
    if [run["queue_order"] for run in queue] != list(range(len(queue))):
        raise WorkerError("queue order is not contiguous")
    preflight_path = Path(plan["preflight_path"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "succeeded"
        or preflight.get("plan_sha256") != plan_sha
    ):
        raise WorkerError("preflight does not bind this plan")
    return plan_path, plan_sha, plan, source, queue, preflight_path


def gpu_compute_pids(gpu: int) -> list[int]:
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
        raise WorkerError(completed.stderr.strip())
    return [int(line.strip()) for line in completed.stdout.splitlines() if line.strip().isdigit()]


def wait_until_gpu_free(gpu: int, poll_seconds: int) -> None:
    last_report = 0.0
    while True:
        pids = gpu_compute_pids(gpu)
        if not pids:
            return
        current = time.monotonic()
        if current - last_report >= 300:
            print(
                f"{now()} waiting_for_gpu={gpu} compute_pids={pids}",
                flush=True,
            )
            last_report = current
        time.sleep(poll_seconds)


def command_for_run(
    plan: dict[str, Any], source: dict[str, Any], run: dict[str, Any], run_dir: Path
) -> list[str]:
    model = source["models"][run["model"]]
    token = source["calibrations"][run["calibration"]]["token_artifact"]
    aware = run["setting"] == "W4A4KV4"
    bits = 4 if aware else 16
    clip = "0.9" if aware else "1.0"
    command = [
        plan["venv_python"],
        "-u",
        "-m",
        "turboBOA.main",
        "--llm_path", model["path"],
        "--tokenizer_path", model["path"],
        "--cache_dir", str(run_dir / "cache"),
        "--calib_tokens_file", token["path"],
        "--calib_tokens_sha256", token["sha256"],
        "--nsamples", "256",
        "--seqlen", "2048",
        "--seed", "1",
        "--global_seed", "1",
        "--deterministic",
        "--w_bits", str(run["w_bits"]),
        "--group_size", "128",
        "--weight_quantizer", "realq_mse",
        "--qparam_comput", "RealQ-MSE",
        "--w_sym",
        "--block_v",
        "--n_quant_rows", "16",
        "--consider_dX",
        "--alpha", "0.25",
        "--adaptive_qparam",
        "--refine_qparam",
        "--n_iters", "1",
        "--act_order_col",
        "--no-act_order_row",
        "--qk_rmsnorm_hessian_mode", "mean_jacobian_kfac",
        "--rotate",
        "--rotation_seed", "0",
        "--a_bits", str(bits),
        "--k_bits", str(bits),
        "--v_bits", str(bits),
        "--a_groupsize", "-1",
        "--k_groupsize", "-1",
        "--v_groupsize", "-1",
        "--a_clip_ratio", clip,
        "--k_clip_ratio", clip,
        "--v_clip_ratio", clip,
        "--act_quant_aware_gptq" if aware else "--no-act_quant_aware_gptq",
        "--k_cache_quant_aware_gptq" if aware else "--no-k_cache_quant_aware_gptq",
        "--skip_eval",
        "--results_path", str(run_dir / "result.json"),
        "--save_qmodel_path", str(run_dir / "qmodel.pt"),
    ]
    return command


def validate_result(
    payload: dict[str, Any], run: dict[str, Any], source: dict[str, Any]
) -> None:
    configuration = payload.get("configuration", {})
    token = source["calibrations"][run["calibration"]]["token_artifact"]
    aware = run["setting"] == "W4A4KV4"
    akv_bits = 4 if aware else 16
    expected = {
        "w_bits": run["w_bits"],
        "group_size": 128,
        "w_sym": True,
        "weight_quantizer": "realq_mse",
        "qparam_comput": "RealQ-MSE",
        "block_v": True,
        "n_quant_rows": 16,
        "consider_dX": True,
        "alpha": 0.25,
        "adaptive_qparam": True,
        "refine_qparam": True,
        "n_iters": 1,
        "act_order_col": True,
        "act_order_row": False,
        "rotate": True,
        "rotation_seed": 0,
        "seed": 1,
        "global_seed": 1,
        "deterministic": True,
        "nsamples": 256,
        "seqlen": 2048,
        "calib_tokens_file": str(Path(token["path"]).resolve()),
        "calib_tokens_sha256": token["sha256"],
        "a_bits": akv_bits,
        "k_bits": akv_bits,
        "v_bits": akv_bits,
        "a_groupsize": -1,
        "k_groupsize": -1,
        "v_groupsize": -1,
        "a_asym": False,
        "k_asym": False,
        "v_asym": False,
        "a_clip_ratio": 0.9 if aware else 1.0,
        "k_clip_ratio": 0.9 if aware else 1.0,
        "v_clip_ratio": 0.9 if aware else 1.0,
        "act_quant_aware_gptq": aware,
        "k_cache_quant_aware_gptq": aware,
        "quant_stop_layer": None,
        "skip_eval": True,
    }
    mismatches = {
        key: (configuration.get(key), value)
        for key, value in expected.items()
        if configuration.get(key) != value
    }
    determinism = configuration.get("determinism_contract")
    expected_determinism = {
        "enabled": True,
        "global_seed": 1,
        "python_hash_seed": 1,
        "python_random_seed": 1,
        "numpy_random_seed": 1,
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32": False,
    }
    if determinism != expected_determinism:
        mismatches["determinism_contract"] = (determinism, expected_determinism)
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "quantized_only"
        or payload.get("metrics") is not None
    ):
        mismatches["result_header"] = (
            (payload.get("schema_version"), payload.get("status"), payload.get("metrics")),
            (1, "quantized_only", None),
        )
    if mismatches:
        raise WorkerError(f"TurboBOA result contract mismatch: {mismatches!r}")


def run_one(
    plan_path: Path,
    plan_sha: str,
    plan: dict[str, Any],
    source: dict[str, Any],
    preflight_path: Path,
    run: dict[str, Any],
) -> dict[str, Any]:
    output_root = Path(plan["output_root"])
    run_dir = output_root / "runs" / run["model"] / run["setting"].lower() / run["run_id"]
    if run_dir.exists() or run_dir.is_symlink():
        raise WorkerError(f"immutable run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    os.chmod(run_dir, 0o755)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "hostname": socket.gethostname(),
        "physical_gpu": run["physical_gpu"],
        "plan_file": str(plan_path),
        "plan_sha256": plan_sha,
        "preflight": str(preflight_path),
        "run": run,
        "command": command_for_run(plan, source, run, run_dir),
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    log_path = run_dir / "run.log"
    with log_path.open("x", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            manifest["command"],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    os.chmod(log_path, 0o644)
    if completed.returncode != 0:
        raise WorkerError(f"TurboBOA exited with code {completed.returncode}")
    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "qmodel.pt"
    if (
        not result_path.is_file()
        or not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size <= 0
    ):
        raise WorkerError("TurboBOA did not atomically publish result/checkpoint")
    os.chmod(result_path, 0o644)
    os.chmod(checkpoint_path, 0o644)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_result(result, run, source)
    receipt = {
        **manifest,
        "status": "quantization_succeeded",
        "ended_at": now(),
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": None,
        "checkpoint_sha256_status": "deferred_to_post_quant_cpu_audit",
        "runtime_seconds": result["runtime_seconds"],
        "quantization_gpu_hours": result["gpu_hours"][
            "algorithm_including_precompute_excluding_eval"
        ],
        "evaluation_status": "pending",
    }
    write_json_atomic(run_dir / "run_receipt.json", receipt)
    return receipt


def execute(args: argparse.Namespace) -> int:
    plan_path, plan_sha, plan, source, queue, preflight_path = load_contract(args)
    output_root = Path(plan["output_root"])
    # This lock is deliberately shared with the YAQA campaign.  Both queue
    # workers may be installed while EfficientQAT is still occupying a card;
    # the common lock gives them a deterministic post-Efficient order and
    # prevents a simultaneous nvidia-smi-free race.
    lock_path = (
        Path("/minimax-avatar-new/zhangqian/realq/experiment_data")
        / "_fair20_physical_gpu_locks_20260821"
        / socket.gethostname()
        / f"gpu{args.physical_gpu}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        for run in queue:
            print(f"{now()} queue_wait run_id={run['run_id']} gpu={args.physical_gpu}", flush=True)
            wait_until_gpu_free(args.physical_gpu, args.poll_seconds)
            print(f"{now()} queue_start run_id={run['run_id']} gpu={args.physical_gpu}", flush=True)
            try:
                receipt = run_one(
                    plan_path, plan_sha, plan, source, preflight_path, run
                )
                print(
                    f"{now()} queue_success run_id={run['run_id']} "
                    f"gpu_hours={receipt['quantization_gpu_hours']}",
                    flush=True,
                )
            except Exception as exc:
                run_dir = output_root / "runs" / run["model"] / run["setting"].lower() / run["run_id"]
                failure = {
                    "schema_version": 1,
                    "status": "failed",
                    "ended_at": now(),
                    "run": run,
                    "error_type": type(exc).__qualname__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json_atomic(run_dir / "failure.json", failure)
                failures.append(run["run_id"])
                print(f"{now()} queue_failure run_id={run['run_id']} error={exc}", flush=True)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    try:
        return execute(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
