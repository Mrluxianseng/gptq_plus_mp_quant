#!/usr/bin/env python3
"""Relocate the frozen YAQA Qwen3-32B W2 stage to one free GPU.

The original node1/GPU1 worker owns a two-stage W3 -> W2 queue.  A handoff is
valid only while that queue parent is SIGSTOP'ed and its W3 quantizer child is
still running.  The numerical stage remains the exact object from the frozen
plan; this helper redirects only CUDA visibility, records the actual host/GPU,
and calls the unchanged formal worker implementation.
"""

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
from typing import Any, Callable

from . import worker


PLAN_PATH = Path(__file__).with_name("plan.json").resolve()
EXPECTED_PLAN_SHA256 = (
    "a7bbd0db640b6d87edbb286f640d744fa13c66c2adccc102d16be9253145bda1"
)
STAGE_ID = "YQ-Q32-W2"
PREVIOUS_STAGE_ID = "YQ-Q32-W3"
SOURCE_HOST = "j-zogxxxduju-master-0"
SOURCE_GPU = 1
TARGET_HOST = "j-4mj21jb084-master-0"
TARGET_GPU = 2
KIND = "yaqa_q32_w2_scheduling_relocation"


class RelocationError(RuntimeError):
    """A scheduling-relocation safety gate failed."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RelocationError(f"JSON root is not an object: {path}")
    return value


def _stage(plan: dict[str, Any], stage_id: str) -> dict[str, Any]:
    matches = [stage for stage in plan["stages"] if stage["stage_id"] == stage_id]
    if len(matches) != 1:
        raise RelocationError(f"expected exactly one frozen stage: {stage_id}")
    return matches[0]


def _load_contract() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
]:
    plan_sha = worker.sha256_file(PLAN_PATH)
    if plan_sha != EXPECTED_PLAN_SHA256:
        raise RelocationError("frozen YAQA plan SHA256 mismatch")
    plan = _read_json(PLAN_PATH)
    source_path = Path(plan["source_plan"])
    if worker.sha256_file(source_path) != plan["source_plan_sha256"]:
        raise RelocationError("source comparison plan SHA256 mismatch")
    source = _read_json(source_path)
    preflight_path = Path(plan["preflight_path"])
    preflight = _read_json(preflight_path)
    if (
        preflight.get("status") != "succeeded"
        or preflight.get("plan_sha256") != plan_sha
    ):
        raise RelocationError("preflight does not bind the frozen plan")
    for relative, expected in preflight["source_sha256"].items():
        if worker.sha256_file(worker.REPO_ROOT / relative) != expected:
            raise RelocationError(f"formal source changed after preflight: {relative}")

    stage = _stage(plan, STAGE_ID)
    previous = _stage(plan, PREVIOUS_STAGE_ID)
    expected_schedule = {
        "job_id": "j-zogxxxduju",
        "canoe_pod": SOURCE_HOST,
        "physical_gpu": SOURCE_GPU,
        "queue_order": 1,
    }
    if (
        stage.get("kind") != "quantize"
        or stage.get("model") != "qwen3-32b"
        or stage.get("calibration") != "qwen3-32b-wiki256"
        or stage.get("setting") != "W2A16KV16"
        or stage.get("w_bits") != 2
        or stage.get("hessian_id") != "YH-Q32-A16"
        or any(stage.get(key) != value for key, value in expected_schedule.items())
        or previous.get("queue_order") != 0
        or previous.get("physical_gpu") != SOURCE_GPU
        or previous.get("canoe_pod") != SOURCE_HOST
    ):
        raise RelocationError("frozen W3 -> W2 queue contract mismatch")
    return plan, source, stage, previous, preflight_path


def _process(pid: int) -> dict[str, Any]:
    if pid <= 1:
        raise RelocationError("process PID must be greater than one")
    root = Path(f"/proc/{pid}")
    try:
        status_text = (root / "status").read_text(encoding="utf-8")
        cmdline = (root / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        stat_text = (root / "stat").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RelocationError(f"process is not live: pid={pid}") from exc
    status = {}
    for line in status_text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    remainder = stat_text[stat_text.rfind(")") + 2 :].split()
    fd_targets = []
    for fd in (root / "fd").iterdir():
        try:
            fd_targets.append(str(fd.resolve()))
        except FileNotFoundError:
            continue
    return {
        "pid": pid,
        "ppid": int(status["PPid"]),
        "state": status["State"],
        "cmdline": cmdline.strip(),
        "start_ticks": int(remainder[19]),
        "fd_targets": sorted(fd_targets),
    }


def _relocation_root(plan: dict[str, Any]) -> Path:
    return Path(plan["output_root"]) / "_scheduling_relocations" / STAGE_ID


def _physical_lock(plan: dict[str, Any], hostname: str, gpu: int) -> Path:
    return (
        Path(plan["output_root"]).parent
        / "_fair20_physical_gpu_locks_20260821"
        / hostname
        / f"gpu{gpu}.lock"
    )


def prepare_handoff(*, parent_pid: int, quantizer_pid: int) -> dict[str, Any]:
    if socket.gethostname() != SOURCE_HOST:
        raise RelocationError("handoff must be prepared on the original pod")
    plan, _source, stage, previous, _preflight_path = _load_contract()
    output_root = Path(plan["output_root"])
    stage_dir = output_root / "stages" / STAGE_ID
    if stage_dir.exists() or stage_dir.is_symlink():
        raise RelocationError("W2 stage directory already exists")

    parent = _process(parent_pid)
    quantizer = _process(quantizer_pid)
    expected_parent_fragments = (
        "experiments.yaqa_wclip_fair20_20260821.worker",
        f"--plan-file {PLAN_PATH}",
        f"--expected-plan-sha256 {EXPECTED_PLAN_SHA256}",
        f"--physical-gpu {SOURCE_GPU}",
    )
    if not parent["state"].startswith("T") or any(
        fragment not in parent["cmdline"] for fragment in expected_parent_fragments
    ):
        raise RelocationError("original queue parent is not the expected stopped worker")
    expected_quantizer_fragments = (
        "YAQA_wclip/quantize_llama/quantize_finetune_llama.py",
        f"stages/{PREVIOUS_STAGE_ID}/raw",
        "--setting W3A16KV16",
        "--w_bits 3",
    )
    if (
        quantizer["ppid"] != parent_pid
        or quantizer["state"].startswith(("T", "Z"))
        or any(
            fragment not in quantizer["cmdline"]
            for fragment in expected_quantizer_fragments
        )
    ):
        raise RelocationError("W3 quantizer is not continuing under the stopped parent")

    source_lock = _physical_lock(plan, SOURCE_HOST, SOURCE_GPU).resolve()
    if str(source_lock) not in parent["fd_targets"]:
        raise RelocationError("stopped parent does not retain the source physical-GPU lock")
    previous_dir = output_root / "stages" / PREVIOUS_STAGE_ID
    previous_manifest = previous_dir / "stage_manifest.json"
    if (
        not previous_manifest.is_file()
        or (previous_dir / "stage_receipt.json").exists()
        or _read_json(previous_manifest).get("status") != "running"
    ):
        raise RelocationError("W3 stage is not the expected in-progress predecessor")
    dependency = worker.wait_hessian_dependency(plan, stage, 10)

    root = _relocation_root(plan)
    handoff_path = root / "handoff.json"
    if root.exists() or root.is_symlink():
        raise RelocationError(f"relocation root already exists: {root}")
    root.mkdir(parents=True)
    handoff = {
        "schema_version": 1,
        "status": "prepared",
        "kind": KIND,
        "prepared_at": worker.now(),
        "plan": str(PLAN_PATH),
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "helper_source": str(Path(__file__).resolve()),
        "helper_source_sha256": worker.sha256_file(Path(__file__).resolve()),
        "source": {
            "hostname": SOURCE_HOST,
            "physical_gpu": SOURCE_GPU,
            "queue_parent": parent,
            "active_quantizer": quantizer,
            "physical_gpu_lock": str(source_lock),
            "previous_stage": previous,
            "previous_stage_manifest": str(previous_manifest.resolve()),
            "previous_stage_manifest_sha256": worker.sha256_file(
                previous_manifest
            ),
        },
        "target": {"hostname": TARGET_HOST, "physical_gpu": TARGET_GPU},
        "stage": stage,
        "dependency": {
            "stage_id": stage["hessian_id"],
            "receipt": str(
                (
                    output_root
                    / "stages"
                    / stage["hessian_id"]
                    / "stage_receipt.json"
                ).resolve()
            ),
            "receipt_sha256": worker.sha256_file(
                output_root
                / "stages"
                / stage["hessian_id"]
                / "stage_receipt.json"
            ),
            "hessian_validation": dependency["hessian_validation"],
            "hessian_validation_sha256": dependency[
                "hessian_validation_sha256"
            ],
        },
        "numerical_contract_changed": False,
        "scheduling_only": True,
    }
    worker.write_json_atomic(handoff_path, handoff)
    return {
        **handoff,
        "handoff": str(handoff_path.resolve()),
        "handoff_sha256": worker.sha256_file(handoff_path),
    }


def _validate_handoff(
    handoff: dict[str, Any],
    *,
    plan: dict[str, Any],
    stage: dict[str, Any],
) -> None:
    source = handoff.get("source", {})
    target = handoff.get("target", {})
    if (
        handoff.get("status") != "prepared"
        or handoff.get("kind") != KIND
        or handoff.get("plan_sha256") != EXPECTED_PLAN_SHA256
        or handoff.get("stage") != stage
        or handoff.get("scheduling_only") is not True
        or handoff.get("numerical_contract_changed") is not False
        or handoff.get("helper_source_sha256")
        != worker.sha256_file(Path(__file__).resolve())
        or source.get("hostname") != SOURCE_HOST
        or source.get("physical_gpu") != SOURCE_GPU
        or not str(source.get("queue_parent", {}).get("state", "")).startswith("T")
        or target != {"hostname": TARGET_HOST, "physical_gpu": TARGET_GPU}
    ):
        raise RelocationError("handoff contract mismatch")
    dependency = handoff.get("dependency", {})
    dependency_path = Path(str(dependency.get("receipt", "")))
    if (
        not dependency_path.is_file()
        or worker.sha256_file(dependency_path) != dependency.get("receipt_sha256")
        or Path(plan["output_root"]) not in dependency_path.parents
    ):
        raise RelocationError("handoff Hessian dependency changed")


def _runtime_environment(plan: dict[str, Any]) -> dict[str, str]:
    python = Path(plan["venv_python"])
    venv_root = python.parent.parent
    torch_lib = venv_root / "lib/python3.12/site-packages/torch/lib"
    if not python.is_file() or not torch_lib.is_dir():
        raise RelocationError("formal Python/torch runtime is missing")
    environment = os.environ.copy()
    environment.update(
        VIRTUAL_ENV=str(venv_root),
        PATH=f"{venv_root / 'bin'}:{environment.get('PATH', '')}",
        LD_LIBRARY_PATH=(
            f"{torch_lib}:{environment.get('LD_LIBRARY_PATH', '')}"
        ),
        CUDA_VISIBLE_DEVICES=str(TARGET_GPU),
        PYTHONPATH=":".join(
            (
                str(worker.REPO_ROOT),
                str(worker.REPO_ROOT / "YAQA_wclip"),
                str(worker.REPO_ROOT / "YAQA_wclip/hessian_llama"),
            )
        ),
        PYTHONHASHSEED="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return environment


def _redirected_base_env(
    original: Callable[[int | None], dict[str, str]],
) -> Callable[[int | None], dict[str, str]]:
    def redirected(gpu: int | None) -> dict[str, str]:
        if gpu is None:
            return original(None)
        if gpu != SOURCE_GPU:
            raise RelocationError(f"unexpected numerical-stage GPU: {gpu}")
        return original(TARGET_GPU)

    return redirected


def _patch_execution_metadata(
    path: Path,
    *,
    stage: dict[str, Any],
    relocation: dict[str, Any],
) -> dict[str, Any]:
    value = _read_json(path)
    if (
        value.get("stage") != stage
        or value.get("hostname") != TARGET_HOST
        or value.get("physical_gpu") != SOURCE_GPU
    ):
        raise RelocationError(f"unexpected pre-relocation metadata: {path}")
    value["physical_gpu"] = TARGET_GPU
    value["scheduling_relocation"] = relocation
    worker.write_json_atomic(path, value)
    return value


def execute(*, expected_handoff_sha256: str) -> dict[str, Any]:
    if socket.gethostname() != TARGET_HOST:
        raise RelocationError("relocated stage must execute on the target pod")
    expected_environment = _runtime_environment(_read_json(PLAN_PATH))
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONDONTWRITEBYTECODE",
    ):
        if os.environ.get(key) != expected_environment[key]:
            raise RelocationError(f"runtime environment mismatch: {key}")

    plan, source, stage, _previous, preflight_path = _load_contract()
    root = _relocation_root(plan)
    handoff_path = root / "handoff.json"
    if worker.sha256_file(handoff_path) != expected_handoff_sha256:
        raise RelocationError("handoff SHA256 mismatch")
    handoff = _read_json(handoff_path)
    _validate_handoff(handoff, plan=plan, stage=stage)
    stage_dir = Path(plan["output_root"]) / "stages" / STAGE_ID
    if stage_dir.exists() or stage_dir.is_symlink():
        raise RelocationError("W2 stage directory was claimed before execution")
    dependency = worker.wait_hessian_dependency(plan, stage, 10)
    lock_path = _physical_lock(plan, TARGET_HOST, TARGET_GPU)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = worker.now()
    started = time.monotonic()
    print(f"{started_at} waiting_for_physical_gpu_lock={lock_path}", flush=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if worker.gpu_compute_pids(TARGET_GPU):
            raise RelocationError("target GPU has an untracked compute process")
        print(f"{worker.now()} acquired_physical_gpu={TARGET_GPU}", flush=True)
        original_base_env = worker.base_env
        worker.base_env = _redirected_base_env(original_base_env)
        try:
            worker.run_quantize(
                PLAN_PATH,
                EXPECTED_PLAN_SHA256,
                plan,
                source,
                preflight_path,
                stage,
                dependency,
            )
        finally:
            worker.base_env = original_base_env

        manifest_path = stage_dir / "stage_manifest.json"
        receipt_path = stage_dir / "stage_receipt.json"
        before = {
            "stage_manifest_sha256": worker.sha256_file(manifest_path),
            "stage_receipt_sha256": worker.sha256_file(receipt_path),
        }
        relocation = {
            "kind": KIND,
            "scheduling_only": True,
            "numerical_contract_changed": False,
            "original_hostname": SOURCE_HOST,
            "original_physical_gpu": SOURCE_GPU,
            "actual_hostname": TARGET_HOST,
            "actual_physical_gpu": TARGET_GPU,
            "handoff": str(handoff_path.resolve()),
            "handoff_sha256": expected_handoff_sha256,
            "helper_source": str(Path(__file__).resolve()),
            "helper_source_sha256": worker.sha256_file(Path(__file__).resolve()),
        }
        _patch_execution_metadata(
            manifest_path, stage=stage, relocation=relocation
        )
        receipt = _patch_execution_metadata(
            receipt_path, stage=stage, relocation=relocation
        )
        validation = Path(receipt["validation"])
        if (
            receipt.get("status") != "succeeded"
            or worker.sha256_file(validation) != receipt.get("validation_sha256")
        ):
            raise RelocationError("relocated W2 terminal validation failed")

    success = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": KIND,
        "stage_id": STAGE_ID,
        "started_at": started_at,
        "finished_at": worker.now(),
        "wall_seconds": time.monotonic() - started,
        "handoff": str(handoff_path.resolve()),
        "handoff_sha256": expected_handoff_sha256,
        "prepatch_identity": before,
        "terminal": str(receipt_path.resolve()),
        "terminal_sha256": worker.sha256_file(receipt_path),
        "validation": str(validation.resolve()),
        "validation_sha256": worker.sha256_file(validation),
        "relocation": relocation,
    }
    worker.write_json_atomic(root / "success.json", success)
    print(json.dumps(success, ensure_ascii=False, sort_keys=True), flush=True)
    return success


def launch(*, expected_handoff_sha256: str) -> dict[str, Any]:
    if socket.gethostname() != TARGET_HOST:
        raise RelocationError("relocation must be launched on the target pod")
    plan, _source, stage, _previous, _preflight_path = _load_contract()
    root = _relocation_root(plan)
    handoff_path = root / "handoff.json"
    if worker.sha256_file(handoff_path) != expected_handoff_sha256:
        raise RelocationError("handoff SHA256 mismatch before launch")
    _validate_handoff(_read_json(handoff_path), plan=plan, stage=stage)
    stage_dir = Path(plan["output_root"]) / "stages" / STAGE_ID
    if stage_dir.exists() or stage_dir.is_symlink():
        raise RelocationError("W2 stage directory is not fresh")
    log_path = root / f"{TARGET_HOST}_gpu{TARGET_GPU}.log"
    launch_path = root / "launch.json"
    if log_path.exists() or launch_path.exists():
        raise RelocationError("relocation launch artifacts already exist")
    command = [
        str(plan["venv_python"]),
        "-u",
        "-m",
        "experiments.yaqa_wclip_fair20_20260821.relocate_q32_w2",
        "--expected-handoff-sha256",
        expected_handoff_sha256,
        "--child",
    ]
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=worker.REPO_ROOT,
            env=_runtime_environment(plan),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    result = {
        "schema_version": 1,
        "status": "launched",
        "kind": KIND,
        "stage_id": STAGE_ID,
        "hostname": TARGET_HOST,
        "physical_gpu": TARGET_GPU,
        "pid": process.pid,
        "launched_at": worker.now(),
        "handoff_sha256": expected_handoff_sha256,
        "command": command,
        "log": str(log_path.resolve()),
    }
    worker.write_json_atomic(launch_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-handoff", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--quantizer-pid", type=int)
    parser.add_argument("--expected-handoff-sha256")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.prepare_handoff:
            if args.parent_pid is None or args.quantizer_pid is None:
                raise RelocationError(
                    "--parent-pid and --quantizer-pid are required"
                )
            result = prepare_handoff(
                parent_pid=args.parent_pid,
                quantizer_pid=args.quantizer_pid,
            )
        else:
            if not args.expected_handoff_sha256:
                raise RelocationError("--expected-handoff-sha256 is required")
            if args.child:
                result = execute(
                    expected_handoff_sha256=args.expected_handoff_sha256
                )
            else:
                result = launch(
                    expected_handoff_sha256=args.expected_handoff_sha256
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
