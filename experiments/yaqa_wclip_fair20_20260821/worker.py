#!/usr/bin/env python3
"""Execute one YAQA_wclip per-GPU stage queue with strict dependencies."""

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


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def chmod_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o755)
        elif path.is_file():
            os.chmod(path, 0o644)
    os.chmod(root, 0o755)


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
            stage
            for stage in plan["stages"]
            if stage["canoe_pod"] == pod
            and int(stage["physical_gpu"]) == args.physical_gpu
        ),
        key=lambda stage: int(stage["queue_order"]),
    )
    if not queue:
        raise WorkerError(f"no YAQA queue for {pod} GPU {args.physical_gpu}")
    if [stage["queue_order"] for stage in queue] != list(range(len(queue))):
        raise WorkerError("queue order is not contiguous")
    preflight_path = Path(plan["preflight_path"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "succeeded"
        or preflight.get("plan_sha256") != plan_sha
    ):
        raise WorkerError("preflight does not bind this plan")
    for relative, expected in preflight["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise WorkerError(f"source changed after preflight: {relative}")
    return plan_path, plan_sha, plan, source, queue, preflight_path


def gpu_compute_pids(gpu: int) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
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
            print(f"{now()} waiting_for_gpu={gpu} compute_pids={pids}", flush=True)
            last_report = current
        time.sleep(poll_seconds)


def run_command(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
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
    os.chmod(log_path, 0o644)
    if completed.returncode != 0:
        raise WorkerError(
            f"command exited with {completed.returncode}: {command!r}"
        )


def stage_dir(plan: dict[str, Any], stage_id: str) -> Path:
    return Path(plan["output_root"]) / "stages" / stage_id


def base_env(gpu: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    env["PYTHONPATH"] = ":".join(
        (
            str(REPO_ROOT),
            str(REPO_ROOT / "YAQA_wclip"),
            str(REPO_ROOT / "YAQA_wclip/hessian_llama"),
        )
    )
    return env


def write_stage_manifest(
    directory: Path,
    plan_path: Path,
    plan_sha: str,
    preflight_path: Path,
    stage: dict[str, Any],
) -> dict[str, Any]:
    if directory.exists() or directory.is_symlink():
        raise WorkerError(f"immutable stage directory exists: {directory}")
    directory.mkdir(parents=True)
    os.chmod(directory, 0o755)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "hostname": socket.gethostname(),
        "physical_gpu": stage["physical_gpu"],
        "plan_file": str(plan_path),
        "plan_sha256": plan_sha,
        "preflight": str(preflight_path),
        "stage": stage,
    }
    write_json_atomic(directory / "stage_manifest.json", manifest)
    return manifest


def run_hessian(
    plan_path: Path,
    plan_sha: str,
    plan: dict[str, Any],
    source: dict[str, Any],
    preflight_path: Path,
    stage: dict[str, Any],
) -> dict[str, Any]:
    directory = stage_dir(plan, stage["stage_id"])
    manifest = write_stage_manifest(
        directory, plan_path, plan_sha, preflight_path, stage
    )
    model = source["models"][stage["model"]]
    token = source["calibrations"][stage["calibration"]]["token_artifact"]
    hessian_dir = directory / "hessian"
    validation = directory / "validation.json"
    timing = directory / "timing.json"
    aware = stage["hessian_mode"] == "a4"
    bits = "4" if aware else "16"
    clip = "0.9" if aware else "1.0"
    env = base_env(int(stage["physical_gpu"]))
    env["YAQA_HESSIAN_TIMING_JSON"] = str(timing)
    command = [
        plan["venv_python"], "-u", "-m", "torch.distributed.run",
        "--standalone", "--nproc-per-node=1",
        str(REPO_ROOT / "YAQA_wclip/hessian_llama/get_hess_llama.py"),
        "--orig_model", model["path"],
        "--save_path", str(hessian_dir),
        "--calib_tokens_path", token["path"],
        "--calib_tokens_sha256", token["sha256"],
        "--seed", "42",
        "--global_seed", "1",
        "--deterministic",
        "--n_seqs", "256",
        "--ctx_size", "2048",
        "--batch_size", "8",
        "--power_iters", "1",
        "--hessian_sketch", "B",
        "--start_layer", "0",
        "--end_layer", str(model.get("layer_count", 100000)),
        "--cpu_offload",
        "--a_bits", bits,
        "--k_bits", bits,
        "--v_bits", bits,
        "--akv_groupsize", "-1",
        "--akv_clip_ratio", clip,
    ]
    if aware:
        command.append("--akv_aware")
    manifest["command"] = command
    write_json_atomic(directory / "stage_manifest.json", manifest)
    run_command(command, env=env, log_path=directory / "hessian.log")
    validate_command = [
        plan["venv_python"], "-u", "-m", "experiments.yaqa_compare.validate_hessian",
        "--hessian-dir", str(hessian_dir),
        "--model", model["path"],
        "--output", str(validation),
        "--expected-samples", "256",
        "--expected-seq-len", "2048",
        "--expected-calib-path", token["path"],
        "--expected-calib-sha256", token["sha256"],
        "--expected-world-size", "1",
        "--expected-batch-size-per-rank", "8",
        "--expected-deterministic",
        "--expected-global-seed", "1",
        "--expected-akv-mode", "aware" if aware else "n/a",
    ]
    run_command(
        validate_command,
        env=base_env(None),
        log_path=directory / "validate_hessian.log",
    )
    timing_payload = json.loads(timing.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    if (
        timing_payload.get("status") != "completed"
        or validation_payload.get("status") != "validated"
    ):
        raise WorkerError("Hessian timing/validation gate failed")
    chmod_tree(directory)
    receipt = {
        **manifest,
        "status": "succeeded",
        "ended_at": now(),
        "hessian_dir": str(hessian_dir),
        "hessian_validation": str(validation),
        "hessian_validation_sha256": sha256_file(validation),
        "hessian_manifest": str(hessian_dir / "manifest.json"),
        "hessian_manifest_sha256": sha256_file(hessian_dir / "manifest.json"),
        "timing": timing_payload,
        "hessian_gpu_hours": float(timing_payload["gpu_seconds"]) / 3600.0,
    }
    write_json_atomic(directory / "stage_receipt.json", receipt)
    return receipt


def wait_hessian_dependency(
    plan: dict[str, Any], stage: dict[str, Any], poll_seconds: int
) -> dict[str, Any]:
    dependency_dir = stage_dir(plan, stage["hessian_id"])
    receipt_path = dependency_dir / "stage_receipt.json"
    failure_path = dependency_dir / "failure.json"
    last_report = 0.0
    while not receipt_path.is_file():
        if failure_path.is_file():
            raise WorkerError(f"Hessian dependency failed: {stage['hessian_id']}")
        current = time.monotonic()
        if current - last_report >= 300:
            print(f"{now()} waiting_for_hessian={stage['hessian_id']}", flush=True)
            last_report = current
        time.sleep(poll_seconds)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "succeeded"
        or sha256_file(receipt["hessian_validation"])
        != receipt["hessian_validation_sha256"]
        or sha256_file(receipt["hessian_manifest"])
        != receipt["hessian_manifest_sha256"]
    ):
        raise WorkerError("Hessian dependency receipt/hash mismatch")
    return receipt


def run_quantize(
    plan_path: Path,
    plan_sha: str,
    plan: dict[str, Any],
    source: dict[str, Any],
    preflight_path: Path,
    stage: dict[str, Any],
    dependency: dict[str, Any],
) -> dict[str, Any]:
    directory = stage_dir(plan, stage["stage_id"])
    manifest = write_stage_manifest(
        directory, plan_path, plan_sha, preflight_path, stage
    )
    model = source["models"][stage["model"]]
    token = source["calibrations"][stage["calibration"]]["token_artifact"]
    raw_dir = directory / "raw"
    hf_dir = directory / "hf"
    validation = directory / "validation.json"
    timing = directory / "timing.json"
    aware = stage["setting"] == "W4A4KV4"
    bits = "4" if aware else "16"
    clip = "0.9" if aware else "1.0"
    env = base_env(int(stage["physical_gpu"]))
    env["YAQA_QUANT_TIMING_JSON"] = str(timing)
    command = [
        plan["venv_python"], "-u",
        str(REPO_ROOT / "YAQA_wclip/quantize_llama/quantize_finetune_llama.py"),
        "--seed", "0",
        "--global_seed", "1",
        "--rotation_seed", "0",
        "--deterministic",
        "--num_cpu_threads", "8",
        "--batch_size", "16",
        "--devset_size", "384",
        "--ctx_size", "2048",
        "--save_path", str(raw_dir),
        "--hess_path", dependency["hessian_dir"],
        "--hessian_validation_path", dependency["hessian_validation"],
        "--base_model", model["path"],
        "--setting", stage["setting"],
        "--sigma_reg", "0.01",
        "--scale_override", "1.0",
        "--codebook", "realq_wclip",
        "--hessian_solver_dtype", "float64",
        "--ft_epochs", "0",
        "--td_x", "1",
        "--td_y", "128",
        "--L", "16",
        "--K", str(stage["w_bits"]),
        "--V", "1",
        "--tlut_bits", "0",
        "--decode_mode", "dense_fake_quant",
        "--calib_tokens_path", token["path"],
        "--calib_tokens_sha256", token["sha256"],
        "--calib_n_seqs", "256",
        "--a_bits", bits,
        "--k_bits", bits,
        "--v_bits", bits,
        "--akv_groupsize", "-1",
        "--akv_clip_ratio", clip,
        "--w_bits", str(stage["w_bits"]),
        "--w_groupsize", "128",
    ]
    if aware:
        command.append("--akv_aware_hessian")
    manifest["command"] = command
    manifest["hessian_dependency"] = {
        "stage_id": stage["hessian_id"],
        "validation": dependency["hessian_validation"],
        "validation_sha256": dependency["hessian_validation_sha256"],
    }
    write_json_atomic(directory / "stage_manifest.json", manifest)
    run_command(command, env=env, log_path=directory / "quantize.log")

    hfize_command = [
        plan["venv_python"], "-u",
        str(REPO_ROOT / "YAQA_wclip/quantize_llama/hfize_llama.py"),
        "--quantized_path", str(raw_dir),
        "--hf_output_path", str(hf_dir),
    ]
    run_command(
        hfize_command, env=base_env(None), log_path=directory / "hfize.log"
    )
    validate_command = [
        plan["venv_python"], "-u",
        str(REPO_ROOT / "YAQA_wclip/eval/validate_hfized_wclip.py"),
        "--model", model["path"],
        "--setting", stage["setting"],
        "--raw-dir", str(raw_dir),
        "--hf-dir", str(hf_dir),
        "--output", str(validation),
    ]
    run_command(
        validate_command,
        env=env,
        log_path=directory / "validate_hfized.log",
    )
    timing_payload = json.loads(timing.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    if (
        timing_payload.get("status") != "completed"
        or validation_payload.get("status") != "validated"
        or validation_payload.get("setting") != stage["setting"]
        or validation_payload.get("weight_bits") != stage["w_bits"]
        or validation_payload.get("weight_groupsize") != 128
        or validation_payload.get("integer_reconstruction_verified") is not True
        or validation_payload.get("reload_deterministic") is not True
    ):
        raise WorkerError("quantized checkpoint validation gate failed")
    chmod_tree(directory)
    quant_gpu_hours = float(timing_payload["gpu_seconds"]) / 3600.0
    hessian_gpu_hours = float(dependency["hessian_gpu_hours"])
    divisor = 3 if stage["setting"] != "W4A4KV4" else 1
    receipt = {
        **manifest,
        "status": "succeeded",
        "ended_at": now(),
        "raw_dir": str(raw_dir),
        "hf_dir": str(hf_dir),
        "validation": str(validation),
        "validation_sha256": sha256_file(validation),
        "timing": timing_payload,
        "quantization_gpu_hours_excluding_hessian": quant_gpu_hours,
        "hessian_gpu_hours": hessian_gpu_hours,
        "standalone_complete_gpu_hours": quant_gpu_hours + hessian_gpu_hours,
        "campaign_amortized_gpu_hours": quant_gpu_hours + hessian_gpu_hours / divisor,
        "evaluation_status": "pending",
    }
    write_json_atomic(directory / "stage_receipt.json", receipt)
    return receipt


def execute(args: argparse.Namespace) -> int:
    plan_path, plan_sha, plan, source, queue, preflight_path = load_contract(args)
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
        for stage in queue:
            print(f"{now()} queue_wait stage_id={stage['stage_id']} gpu={args.physical_gpu}", flush=True)
            try:
                dependency = None
                if stage["kind"] == "quantize":
                    dependency = wait_hessian_dependency(
                        plan, stage, args.poll_seconds
                    )
                wait_until_gpu_free(args.physical_gpu, args.poll_seconds)
                print(
                    f"{now()} queue_start stage_id={stage['stage_id']} "
                    f"gpu={args.physical_gpu}",
                    flush=True,
                )
                if stage["kind"] == "hessian":
                    receipt = run_hessian(
                        plan_path, plan_sha, plan, source, preflight_path, stage
                    )
                    metric = receipt["hessian_gpu_hours"]
                else:
                    receipt = run_quantize(
                        plan_path,
                        plan_sha,
                        plan,
                        source,
                        preflight_path,
                        stage,
                        dependency,
                    )
                    metric = receipt["standalone_complete_gpu_hours"]
                print(
                    f"{now()} queue_success stage_id={stage['stage_id']} gpu_hours={metric}",
                    flush=True,
                )
            except Exception as exc:
                directory = stage_dir(plan, stage["stage_id"])
                failure = {
                    "schema_version": 1,
                    "status": "failed",
                    "ended_at": now(),
                    "stage": stage,
                    "error_type": type(exc).__qualname__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json_atomic(directory / "failure.json", failure)
                failures.append(stage["stage_id"])
                print(f"{now()} queue_failure stage_id={stage['stage_id']} error={exc}", flush=True)
                break
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--poll-seconds", default=15, type=int)
    args = parser.parse_args()
    try:
        return execute(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
