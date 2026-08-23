#!/usr/bin/env python3
"""Finalize YAQA Q32/W3 after its queue parent was stopped for W2 relocation.

The stopped formal worker keeps its physical-GPU lock while the already
spawned W3 raw quantizer continues.  Once that child becomes a successful
zombie, this supervisor validates the raw timing, retires only the stopped
parent, acquires the same GPU lock, and performs the unchanged formal hfize,
validation, and receipt publication steps.  It never enters the W2 stage.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import traceback
from typing import Any

from . import relocate_q32_w2 as relocation
from . import worker


KIND = "yaqa_q32_w3_stopped_parent_finalization"


class FinalizationError(RuntimeError):
    """The stopped-parent finalization violated its safety contract."""


def _launch_artifacts(root: Path, attempt: str) -> tuple[Path, Path]:
    """Return append-only launch paths for an initial run or named retry."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", attempt):
        raise FinalizationError(f"invalid launch attempt: {attempt!r}")
    suffix = "" if attempt == "initial" else f"_{attempt}"
    return (
        root / f"{relocation.SOURCE_HOST}_w3_finalizer{suffix}.log",
        root / f"w3_finalizer_launch{suffix}.json",
    )


def _zombie_exit_code(pid: int) -> int:
    path = Path(f"/proc/{pid}/stat")
    try:
        value = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FinalizationError(f"quantizer disappeared before audit: pid={pid}") from exc
    remainder = value[value.rfind(")") + 2 :].split()
    if not remainder or remainder[0] != "Z" or len(remainder) < 50:
        raise FinalizationError(f"quantizer is not a zombie: pid={pid}")
    return int(remainder[49])


def _validate_processes(
    handoff: dict[str, Any], *, parent_pid: int, quantizer_pid: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    recorded = handoff["source"]
    parent = relocation._process(parent_pid)
    quantizer = relocation._process(quantizer_pid)
    expected_parent = recorded["queue_parent"]
    expected_quantizer = recorded["active_quantizer"]
    quantizer_is_zombie = quantizer["state"].startswith("Z")
    if (
        parent["start_ticks"] != expected_parent["start_ticks"]
        or parent["cmdline"] != expected_parent["cmdline"]
        or not parent["state"].startswith("T")
        or quantizer["start_ticks"] != expected_quantizer["start_ticks"]
        or quantizer["ppid"] != parent_pid
        # Linux normally exposes an empty /proc/PID/cmdline for a zombie.  The
        # immutable start tick and unchanged PPID still protect against PID
        # reuse at this terminal boundary.
        or (
            not quantizer_is_zombie
            and quantizer["cmdline"] != expected_quantizer["cmdline"]
        )
    ):
        raise FinalizationError("stopped parent or W3 quantizer identity changed")
    return parent, quantizer


def _wait_parent_retired(parent_pid: int) -> bool:
    """Wait until the killed parent is absent or an inert zombie.

    File locks and CUDA resources are released when a process exits, before
    its parent necessarily reaps the zombie.  Requiring /proc/PID to disappear
    would therefore make successful finalization depend on shell reaping
    latency.  The return value records whether full reaping already occurred.
    """

    for _ in range(100):
        path = Path(f"/proc/{parent_pid}/stat")
        try:
            value = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return True
        remainder = value[value.rfind(")") + 2 :].split()
        if remainder and remainder[0] in {"Z", "X", "x"}:
            return False
        time.sleep(0.1)
    raise FinalizationError("stopped queue parent remained executable after SIGKILL")


def _runtime_environment(plan: dict[str, Any]) -> dict[str, str]:
    environment = relocation._runtime_environment(plan)
    environment["CUDA_VISIBLE_DEVICES"] = str(relocation.SOURCE_GPU)
    return environment


def _finish_stage(
    *,
    plan: dict[str, Any],
    stage: dict[str, Any],
    dependency: dict[str, Any],
    handoff_path: Path,
    handoff_sha256: str,
    parent_pid: int,
    quantizer_pid: int,
) -> dict[str, Any]:
    directory = worker.stage_dir(plan, stage["stage_id"])
    manifest_path = directory / "stage_manifest.json"
    manifest = relocation._read_json(manifest_path)
    if (
        manifest.get("status") != "running"
        or manifest.get("stage") != stage
        or manifest.get("hostname") != relocation.SOURCE_HOST
        or manifest.get("physical_gpu") != relocation.SOURCE_GPU
        or manifest.get("plan_sha256") != relocation.EXPECTED_PLAN_SHA256
        or worker.sha256_file(manifest_path)
        != relocation._read_json(handoff_path)["source"][
            "previous_stage_manifest_sha256"
        ]
    ):
        raise FinalizationError("W3 running manifest changed after handoff")
    raw_dir = directory / "raw"
    hf_dir = directory / "hf"
    validation = directory / "validation.json"
    timing = directory / "timing.json"
    for path in (
        directory / "stage_receipt.json",
        directory / "failure.json",
        directory / "hfize.log",
        directory / "validate_hfized.log",
        hf_dir,
        validation,
    ):
        if path.exists() or path.is_symlink():
            raise FinalizationError(f"W3 finalization output is not fresh: {path}")
    timing_payload = relocation._read_json(timing)
    if timing_payload.get("status") != "completed":
        raise FinalizationError("W3 raw timing is not complete")

    model = relocation._read_json(Path(plan["source_plan"]))["models"][
        stage["model"]
    ]
    hfize_command = [
        plan["venv_python"],
        "-u",
        str(worker.REPO_ROOT / "YAQA_wclip/quantize_llama/hfize_llama.py"),
        "--quantized_path",
        str(raw_dir),
        "--hf_output_path",
        str(hf_dir),
    ]
    worker.run_command(
        hfize_command,
        env=worker.base_env(None),
        log_path=directory / "hfize.log",
    )
    validate_command = [
        plan["venv_python"],
        "-u",
        str(worker.REPO_ROOT / "YAQA_wclip/eval/validate_hfized_wclip.py"),
        "--model",
        model["path"],
        "--setting",
        stage["setting"],
        "--raw-dir",
        str(raw_dir),
        "--hf-dir",
        str(hf_dir),
        "--output",
        str(validation),
    ]
    worker.run_command(
        validate_command,
        env=worker.base_env(relocation.SOURCE_GPU),
        log_path=directory / "validate_hfized.log",
    )
    validation_payload = relocation._read_json(validation)
    if (
        validation_payload.get("status") != "validated"
        or validation_payload.get("setting") != stage["setting"]
        or validation_payload.get("weight_bits") != stage["w_bits"]
        or validation_payload.get("weight_groupsize") != 128
        or validation_payload.get("integer_reconstruction_verified") is not True
        or validation_payload.get("reload_deterministic") is not True
    ):
        raise FinalizationError("W3 quantized checkpoint validation gate failed")

    worker.chmod_tree(directory)
    quant_gpu_hours = float(timing_payload["gpu_seconds"]) / 3600.0
    hessian_gpu_hours = float(dependency["hessian_gpu_hours"])
    finalization = {
        "kind": KIND,
        "reason": "prevent relocated W2 from being entered by the old queue",
        "numerical_contract_changed": False,
        "raw_quantizer_completed_before_parent_retirement": True,
        "original_queue_parent_pid": parent_pid,
        "raw_quantizer_pid": quantizer_pid,
        "handoff": str(handoff_path.resolve()),
        "handoff_sha256": handoff_sha256,
        "helper_source": str(Path(__file__).resolve()),
        "helper_source_sha256": worker.sha256_file(Path(__file__).resolve()),
        "hfize_command": hfize_command,
        "validate_command": validate_command,
    }
    receipt = {
        **manifest,
        "status": "succeeded",
        "ended_at": worker.now(),
        "raw_dir": str(raw_dir),
        "hf_dir": str(hf_dir),
        "validation": str(validation),
        "validation_sha256": worker.sha256_file(validation),
        "timing": timing_payload,
        "quantization_gpu_hours_excluding_hessian": quant_gpu_hours,
        "hessian_gpu_hours": hessian_gpu_hours,
        "standalone_complete_gpu_hours": quant_gpu_hours + hessian_gpu_hours,
        "campaign_amortized_gpu_hours": quant_gpu_hours + hessian_gpu_hours / 3,
        "evaluation_status": "pending",
        "scheduling_finalization": finalization,
    }
    worker.write_json_atomic(directory / "stage_receipt.json", receipt)
    return receipt


def execute(
    *,
    expected_handoff_sha256: str,
    parent_pid: int,
    quantizer_pid: int,
    poll_seconds: int,
) -> dict[str, Any]:
    if socket.gethostname() != relocation.SOURCE_HOST:
        raise FinalizationError("W3 finalizer must run on the original pod")
    if poll_seconds < 10:
        raise FinalizationError("poll interval must be at least 10 seconds")
    plan, _source, _w2_stage, w3_stage, _preflight_path = (
        relocation._load_contract()
    )
    root = relocation._relocation_root(plan)
    handoff_path = root / "handoff.json"
    if worker.sha256_file(handoff_path) != expected_handoff_sha256:
        raise FinalizationError("handoff SHA256 mismatch")
    handoff = relocation._read_json(handoff_path)
    relocation._validate_handoff(
        handoff,
        plan=plan,
        stage=relocation._stage(plan, relocation.STAGE_ID),
    )
    if (
        handoff["source"]["queue_parent"]["pid"] != parent_pid
        or handoff["source"]["active_quantizer"]["pid"] != quantizer_pid
    ):
        raise FinalizationError("CLI PIDs do not match the immutable handoff")
    dependency = worker.wait_hessian_dependency(plan, w3_stage, 10)
    supervisor_lock = root / ".w3_finalizer.lock"
    with supervisor_lock.open("a+", encoding="utf-8") as supervisor_handle:
        try:
            fcntl.flock(
                supervisor_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise FinalizationError("W3 finalizer is already running") from exc
        last_report = 0.0
        while True:
            _parent, quantizer = _validate_processes(
                handoff,
                parent_pid=parent_pid,
                quantizer_pid=quantizer_pid,
            )
            if quantizer["state"].startswith("Z"):
                exit_code = _zombie_exit_code(quantizer_pid)
                if exit_code != 0:
                    raise FinalizationError(
                        f"W3 raw quantizer failed with wait status {exit_code}"
                    )
                break
            current = time.monotonic()
            if current - last_report >= 300:
                print(
                    f"{worker.now()} waiting_for_w3_raw_quantizer="
                    f"{quantizer_pid} state={quantizer['state']}",
                    flush=True,
                )
                last_report = current
            time.sleep(poll_seconds)

        timing = worker.stage_dir(plan, w3_stage["stage_id"]) / "timing.json"
        if relocation._read_json(timing).get("status") != "completed":
            raise FinalizationError("W3 raw child exited without completed timing")
        os.kill(parent_pid, signal.SIGKILL)
        parent_reaped = _wait_parent_retired(parent_pid)

        physical_lock = relocation._physical_lock(
            plan, relocation.SOURCE_HOST, relocation.SOURCE_GPU
        )
        started_at = worker.now()
        started = time.monotonic()
        with physical_lock.open("a+", encoding="utf-8") as physical_handle:
            fcntl.flock(physical_handle.fileno(), fcntl.LOCK_EX)
            if worker.gpu_compute_pids(relocation.SOURCE_GPU):
                raise FinalizationError(
                    "source GPU still has compute processes after raw completion"
                )
            receipt = _finish_stage(
                plan=plan,
                stage=w3_stage,
                dependency=dependency,
                handoff_path=handoff_path,
                handoff_sha256=expected_handoff_sha256,
                parent_pid=parent_pid,
                quantizer_pid=quantizer_pid,
            )

    success = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": KIND,
        "stage_id": w3_stage["stage_id"],
        "started_finalization_at": started_at,
        "finished_at": worker.now(),
        "finalization_wall_seconds": time.monotonic() - started,
        "retired_parent_pid": parent_pid,
        "parent_reaped_at_publication": parent_reaped,
        "raw_quantizer_pid": quantizer_pid,
        "handoff_sha256": expected_handoff_sha256,
        "terminal": str(
            (
                worker.stage_dir(plan, w3_stage["stage_id"])
                / "stage_receipt.json"
            ).resolve()
        ),
        "terminal_sha256": worker.sha256_file(
            worker.stage_dir(plan, w3_stage["stage_id"])
            / "stage_receipt.json"
        ),
        "validation_sha256": receipt["validation_sha256"],
        "helper_source": str(Path(__file__).resolve()),
        "helper_source_sha256": worker.sha256_file(Path(__file__).resolve()),
    }
    worker.write_json_atomic(root / "w3_finalization_success.json", success)
    print(json.dumps(success, ensure_ascii=False, sort_keys=True), flush=True)
    return success


def launch(
    *,
    expected_handoff_sha256: str,
    parent_pid: int,
    quantizer_pid: int,
    poll_seconds: int,
    launch_attempt: str,
) -> dict[str, Any]:
    if socket.gethostname() != relocation.SOURCE_HOST:
        raise FinalizationError("W3 finalizer must be launched on the original pod")
    plan, _source, _stage, _previous, _preflight = relocation._load_contract()
    root = relocation._relocation_root(plan)
    log_path, launch_path = _launch_artifacts(root, launch_attempt)
    if log_path.exists() or launch_path.exists():
        raise FinalizationError("W3 finalizer launch artifacts already exist")
    command = [
        str(plan["venv_python"]),
        "-u",
        "-m",
        "experiments.yaqa_wclip_fair20_20260821.finalize_stopped_q32_w3",
        "--expected-handoff-sha256",
        expected_handoff_sha256,
        "--parent-pid",
        str(parent_pid),
        "--quantizer-pid",
        str(quantizer_pid),
        "--poll-seconds",
        str(poll_seconds),
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
        "launch_attempt": launch_attempt,
        "pid": process.pid,
        "parent_pid": parent_pid,
        "quantizer_pid": quantizer_pid,
        "launched_at": worker.now(),
        "handoff_sha256": expected_handoff_sha256,
        "command": command,
        "log": str(log_path.resolve()),
    }
    worker.write_json_atomic(launch_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--quantizer-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", default=30, type=int)
    parser.add_argument("--launch-attempt", default="initial")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.child:
            result = execute(
                expected_handoff_sha256=args.expected_handoff_sha256,
                parent_pid=args.parent_pid,
                quantizer_pid=args.quantizer_pid,
                poll_seconds=args.poll_seconds,
            )
        else:
            result = launch(
                expected_handoff_sha256=args.expected_handoff_sha256,
                parent_pid=args.parent_pid,
                quantizer_pid=args.quantizer_pid,
                poll_seconds=args.poll_seconds,
                launch_attempt=args.launch_attempt,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
