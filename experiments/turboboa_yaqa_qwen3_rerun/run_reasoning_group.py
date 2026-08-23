#!/usr/bin/env python3
"""Wait for one artifact, then run its three frozen reasoning tasks in parallel."""

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
TASKS = ("gsm8k", "math_500", "humaneval_plus")
TASK_COUNTS = {"gsm8k": 1319, "math_500": 500, "humaneval_plus": 164}
TASK_PROTOCOL = {
    "qwen3-4b": {
        "gsm8k": (64, 1024),
        "math_500": (32, 2048),
        "humaneval_plus": (32, 2048),
    },
    "qwen3-32b": {
        "gsm8k": (32, 1024),
        "math_500": (16, 2048),
        "humaneval_plus": (16, 2048),
    },
}


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


def _model_key(model: Path) -> str:
    name = model.name.lower()
    if name in TASK_PROTOCOL:
        return name
    raise ValueError(f"unsupported formal model: {model}")


def _setting_values(setting: str) -> tuple[int, int, int, int]:
    if setting == "W3A16KV16":
        return 3, 16, 16, 16
    if setting == "W4A4KV4":
        return 4, 4, 4, 4
    raise ValueError(f"unsupported formal setting: {setting}")


def _wait_for_gate(
    *,
    method: str,
    gate: Path,
    artifact: Path,
    model: Path,
    setting: str,
    state_path: Path,
    expected_weight_method: str | None = None,
) -> dict[str, Any]:
    while True:
        if gate.is_file():
            payload = _read_json(gate)
            if method == "turboboa":
                if payload.get("status") == "quantized_only":
                    declared = Path(
                        payload.get("quantized_checkpoint", "")
                    ).resolve()
                    if declared != artifact:
                        raise RuntimeError(
                            "TurboBOA result points at a different checkpoint"
                        )
                    declared_model = Path(
                        payload.get("model", {}).get("path", "")
                    ).resolve()
                    configuration = payload.get("configuration", {})
                    w_bits, a_bits, k_bits, v_bits = _setting_values(setting)
                    if (
                        declared_model != model
                        or configuration.get("group_size") != 128
                        or configuration.get("w_bits") != w_bits
                        or configuration.get("a_bits") != a_bits
                        or configuration.get("k_bits") != k_bits
                        or configuration.get("v_bits") != v_bits
                        or configuration.get("nsamples") != 256
                        or configuration.get("seqlen") != 2048
                        or configuration.get("calib_data") != "wikitext2"
                    ):
                        raise RuntimeError("TurboBOA result provenance mismatch")
                    if (
                        expected_weight_method is not None
                        and configuration.get("w_method")
                        != expected_weight_method
                    ):
                        raise RuntimeError(
                            "TurboBOA weight-method provenance mismatch"
                        )
                    if not artifact.is_file() or artifact.stat().st_size <= 0:
                        raise RuntimeError(
                            "TurboBOA result published without checkpoint"
                        )
                    return payload
            else:
                if payload.get("status") == "quantization_succeeded":
                    if (
                        Path(payload.get("model", "")).resolve() != model
                        or payload.get("setting") != setting
                        or Path(payload.get("hf_dir", "")).resolve()
                        != artifact
                    ):
                        raise RuntimeError("YAQA quantization result mismatch")
                    return payload
                if payload.get("status") == "failed":
                    raise RuntimeError("YAQA postprocess pipeline failed")
        _write_atomic(
            state_path,
            {
                "schema_version": 1,
                "status": "waiting_for_quantized_artifact",
                "method": method,
                "gate": str(gate),
                "artifact": str(artifact),
                "updated_at": _now(),
                "hostname": socket.gethostname(),
            },
        )
        time.sleep(20)


def _validate_yaqa_receipt(
    validation_path: Path,
    artifact: Path,
    model: Path,
    setting: str,
) -> str:
    report = _read_json(validation_path)
    if (
        report.get("schema_version") != 2
        or report.get("status") != "validated"
        or report.get("model") != str(model)
        or report.get("setting") != setting
        or Path(report.get("hf_dir", "")).resolve() != artifact
        or report.get("runtime_weight_representation")
        != "dense_fake_quant"
    ):
        raise RuntimeError("YAQA HF validation report header mismatch")
    records = report.get("hf_files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("YAQA HF validation report has no file records")
    for record in records:
        path = Path(record.get("path", "")).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or artifact not in path.parents
        ):
            raise RuntimeError(f"YAQA HF artifact changed after validation: {path}")
    return _sha256(validation_path)


def _gpu_free(gpu: int) -> bool:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed for GPU {gpu}: {completed.stderr.strip()}"
        )
    return not any(line.strip().isdigit() for line in completed.stdout.splitlines())


def _wait_for_gpus(gpus: list[int], state_path: Path) -> None:
    while True:
        busy = [gpu for gpu in gpus if not _gpu_free(gpu)]
        if not busy:
            return
        _write_atomic(
            state_path,
            {
                "schema_version": 1,
                "status": "waiting_for_gpus",
                "busy_gpus": busy,
                "updated_at": _now(),
                "hostname": socket.gethostname(),
            },
        )
        time.sleep(20)


def _task_command(
    *,
    method: str,
    model: Path,
    setting: str,
    artifact: Path,
    validation: Path | None,
    validation_sha: str | None,
    task: str,
    task_dir: Path,
) -> list[str]:
    if method == "yaqa_wclip":
        assert validation is not None and validation_sha is not None
        return [
            str(PYTHON),
            "-u",
            "-m",
            "experiments.turboboa_yaqa_qwen3_rerun.eval_yaqa_reasoning",
            "--model",
            str(model),
            "--setting",
            setting,
            "--hf-dir",
            str(artifact),
            "--hf-validation",
            str(validation),
            "--expected-validation-sha256",
            validation_sha,
            "--task",
            task,
            "--output-dir",
            str(task_dir),
        ]

    w_bits, a_bits, k_bits, v_bits = _setting_values(setting)
    batch_size, max_new_tokens = TASK_PROTOCOL[_model_key(model)][task]
    return [
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
        "--skip_kl_ppl_eval",
        "true",
        "--reasoning_eval",
        "true",
        "--reasoning_tasks",
        task,
        "--reasoning_data_dir",
        str(REPO_ROOT / "datasets/reasoning_eval"),
        "--reasoning_limit",
        "-1",
        "--reasoning_batch_size",
        str(batch_size),
        "--reasoning_num_samples",
        "1",
        "--reasoning_max_new_tokens",
        str(max_new_tokens),
        "--reasoning_apply_chat_template",
        "true",
        "--reasoning_enable_thinking",
        "true",
        "--reasoning_do_sample",
        "false",
        "--reasoning_temperature",
        "0.6",
        "--reasoning_top_p",
        "0.95",
        "--reasoning_top_k",
        "20",
        "--reasoning_seed",
        "1234",
        "--reasoning_resume",
        "true",
        "--reasoning_protocol",
        "realq_zero_shot_v1",
        "--reasoning_output_dir",
        str(task_dir / "results"),
        "--output_dir",
        str(task_dir),
        "--exp",
        f"reasoning_turboboa_{_model_key(model)}_{setting}_{task}",
    ]


def _audit_task(task: str, task_dir: Path) -> dict[str, Any]:
    manifest_path = task_dir / "results/manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != "completed"
        or manifest.get("tasks") != [task]
        or len(manifest.get("results", [])) != 1
    ):
        raise RuntimeError(f"{task}: incomplete reasoning manifest")
    summary = manifest["results"][0]
    expected = TASK_COUNTS[task]
    if (
        summary.get("num_examples") != expected
        or summary.get("num_generations") != expected
    ):
        raise RuntimeError(f"{task}: formal task count mismatch")
    result: dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "summary": summary,
    }
    if task in {"gsm8k", "math_500"}:
        score_path = task_dir / f"results/{task}/scores.json"
        score = _read_json(score_path)
        if (
            score.get("summary", {}).get("status") != "scored"
            or len(score.get("details", [])) != expected
        ):
            raise RuntimeError(f"{task}: incomplete score file")
        result.update(
            {
                "score": str(score_path),
                "score_sha256": _sha256(score_path),
            }
        )
    return result


def _audit_evalplus(run_root: Path) -> dict[str, Any]:
    matches = sorted(run_root.glob("*_eval_results.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one EvalPlus result in {run_root}, got {matches}"
        )
    path = matches[0]
    payload = _read_json(path)
    evaluations = payload.get("eval")
    if not isinstance(evaluations, dict) or len(evaluations) != 164:
        raise RuntimeError("EvalPlus result does not cover 164 tasks")
    base = plus = 0
    for task_id, candidates in evaluations.items():
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise RuntimeError(f"EvalPlus candidate count mismatch: {task_id}")
        record = candidates[0]
        base_passed = record.get("base_status") == "pass"
        # EvalPlus reports the extra-test status separately.  A candidate only
        # passes HumanEval+ when it passes both the base and extra tests.
        base += base_passed
        plus += base_passed and record.get("plus_status") == "pass"
    summary = {
        "base_pass": base,
        "base_total": 164,
        "base_pass_at_1": base / 164.0,
        "plus_pass": plus,
        "plus_total": 164,
        "plus_pass_at_1": plus / 164.0,
        "official_result": str(path),
        "official_result_sha256": _sha256(path),
    }
    _write_atomic(run_root / "official_summary.json", summary)
    return summary


def _refresh_completed_result_evalplus(result_path: Path) -> dict[str, Any]:
    """Re-audit EvalPlus and atomically refresh an already-finished group result."""
    result = _read_json(result_path)
    humaneval = result.get("tasks", {}).get("humaneval_plus")
    if not isinstance(humaneval, dict):
        raise RuntimeError("completed result has no HumanEval+ task")
    previous = humaneval.get("official")
    if not isinstance(previous, dict):
        raise RuntimeError("completed result has no HumanEval+ official result")
    official_result = previous.get("official_result")
    if not isinstance(official_result, str):
        raise RuntimeError("completed result has no EvalPlus result path")

    refreshed = _audit_evalplus(Path(official_result).parent)
    humaneval["official"] = refreshed
    refreshed_at = _now()
    result["evalplus_summary_refreshed_at"] = refreshed_at
    _write_atomic(result_path, result)
    result_sha256 = _sha256(result_path)

    group_state_path = result_path.parent / "group_state.json"
    if group_state_path.exists():
        group_state = _read_json(group_state_path)
        recorded_result = group_state.get("result")
        if (
            group_state.get("status") == "succeeded"
            and isinstance(recorded_result, str)
            and Path(recorded_result).resolve() == result_path.resolve()
        ):
            group_state["result_sha256"] = result_sha256
            group_state["evalplus_summary_refreshed_at"] = refreshed_at
            _write_atomic(group_state_path, group_state)
    return refreshed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("turboboa", "yaqa_wclip"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=("W3A16KV16", "W4A4KV4"),
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--hf-validation")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-weight-method")
    args = parser.parse_args()

    hostname = socket.gethostname()
    if hostname != args.expected_hostname or not re.fullmatch(
        r"j-[a-z0-9]+-master-0", hostname
    ):
        raise RuntimeError(
            f"reasoning group bound to {args.expected_hostname}, got {hostname}"
        )
    model = Path(args.model).resolve()
    artifact = Path(args.artifact).resolve()
    gate = Path(args.gate).resolve()
    run_dir = Path(args.run_dir).resolve()
    state_path = run_dir / "reasoning/group_state.json"
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != 3 or len(set(gpus)) != 3 or any(
        gpu < 0 or gpu > 7 for gpu in gpus
    ):
        raise ValueError("--gpus must contain three distinct indices in [0,7]")

    started_at = _now()
    gate_payload = _wait_for_gate(
        method=args.method,
        gate=gate,
        artifact=artifact,
        model=model,
        setting=args.setting,
        state_path=state_path,
        expected_weight_method=args.expected_weight_method,
    )
    validation_path = (
        Path(args.hf_validation).resolve()
        if args.hf_validation
        else None
    )
    validation_sha = None
    if args.method == "yaqa_wclip":
        if validation_path is None:
            raise ValueError("YAQA reasoning requires --hf-validation")
        validation_sha = _validate_yaqa_receipt(
            validation_path,
            artifact,
            model,
            args.setting,
        )

    _wait_for_gpus(gpus, state_path)
    base_env = os.environ.copy()
    python_paths = [
        str(REPO_ROOT / "datasets/reasoning_eval/python_packages"),
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
        }
    )

    processes: dict[str, subprocess.Popen[Any]] = {}
    handles: dict[str, Any] = {}
    task_started: dict[str, float] = {}
    commands: dict[str, list[str]] = {}
    for task, gpu in zip(TASKS, gpus, strict=True):
        task_dir = run_dir / "reasoning" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        command = _task_command(
            method=args.method,
            model=model,
            setting=args.setting,
            artifact=artifact,
            validation=validation_path,
            validation_sha=validation_sha,
            task=task,
            task_dir=task_dir,
        )
        log_path = task_dir / "launcher.log"
        handle = log_path.open("a", encoding="utf-8")
        env = base_env.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(13200 + os.getpid() % 1000 + gpu),
            }
        )
        task_started[task] = time.monotonic()
        commands[task] = command
        processes[task] = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handles[task] = handle

    _write_atomic(
        state_path,
        {
            "schema_version": 1,
            "status": "generation_running",
            "method": args.method,
            "model": str(model),
            "setting": args.setting,
            "artifact": str(artifact),
            "gate": str(gate),
            "gate_sha256": _sha256(gate),
            "hf_validation_sha256": validation_sha,
            "hostname": hostname,
            "gpus": dict(zip(TASKS, gpus, strict=True)),
            "pids": {task: process.pid for task, process in processes.items()},
            "commands": commands,
            "started_at": started_at,
            "updated_at": _now(),
        },
    )

    returncodes: dict[str, int] = {}
    scorer: subprocess.Popen[Any] | None = None
    scorer_handle = None
    scorer_root = (
        run_dir
        / "reasoning/humaneval_plus/results/humaneval_plus"
        / "official_eval_sandbox_run"
    )
    while len(returncodes) < len(processes) or scorer is not None:
        for task, process in processes.items():
            if task in returncodes:
                continue
            returncode = process.poll()
            if returncode is None:
                continue
            returncodes[task] = int(returncode)
            handles[task].close()
            if task == "humaneval_plus" and returncode == 0:
                samples = (
                    run_dir
                    / "reasoning/humaneval_plus/results/humaneval_plus"
                    / "evalplus_samples.jsonl"
                )
                if sum(1 for _ in samples.open("rb")) != 164:
                    raise RuntimeError(
                        "HumanEval+ generation did not export 164 samples"
                    )
                scorer_root.mkdir(parents=True, exist_ok=True)
                lock_root = (
                    run_dir.parent.parent
                    / "_evalplus_locks"
                )
                lock_root.mkdir(parents=True, exist_ok=True)
                scorer_handle = (
                    scorer_root / "scorer_launcher.log"
                ).open("a", encoding="utf-8")
                scorer = subprocess.Popen(
                    [
                        "/usr/bin/flock",
                        "--exclusive",
                        str(lock_root / f"{hostname}.lock"),
                        str(REPO_ROOT / "tools/lowbit_activation_evalplus_canoe.sh"),
                        str(samples),
                        str(scorer_root),
                        "32",
                    ],
                    cwd=REPO_ROOT,
                    env=base_env,
                    stdin=subprocess.DEVNULL,
                    stdout=scorer_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        if scorer is not None:
            scorer_rc = scorer.poll()
            if scorer_rc is not None:
                if scorer_handle is not None:
                    scorer_handle.close()
                if scorer_rc != 0:
                    raise RuntimeError(
                        f"EvalPlus scorer exited with code {scorer_rc}"
                    )
                scorer = None
        if len(returncodes) < len(processes) or scorer is not None:
            time.sleep(10)

    failed = {
        task: returncode
        for task, returncode in returncodes.items()
        if returncode != 0
    }
    if failed:
        raise RuntimeError(f"reasoning generation failures: {failed}")

    tasks = {
        task: {
            **_audit_task(task, run_dir / "reasoning" / task),
            "gpu": gpu,
            "wall_seconds": time.monotonic() - task_started[task],
        }
        for task, gpu in zip(TASKS, gpus, strict=True)
    }
    tasks["humaneval_plus"]["official"] = _audit_evalplus(scorer_root)
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "method": args.method,
        "model": str(model),
        "setting": args.setting,
        "artifact": str(artifact),
        "gate": str(gate),
        "gate_payload": gate_payload,
        "hf_validation": (
            str(validation_path) if validation_path is not None else None
        ),
        "hf_validation_sha256": validation_sha,
        "hostname": hostname,
        "started_at": started_at,
        "finished_at": _now(),
        "protocol": {
            "full_task": True,
            "thinking": True,
            "do_sample": False,
            "num_samples": 1,
            "seed": 1234,
            "task_protocol": TASK_PROTOCOL[_model_key(model)],
        },
        "tasks": tasks,
        "quantization_gpu_hours_included": 0.0,
    }
    result_path = run_dir / "reasoning/result.json"
    _write_atomic(result_path, result)
    _write_atomic(
        state_path,
        {
            "schema_version": 1,
            "status": "succeeded",
            "result": str(result_path),
            "result_sha256": _sha256(result_path),
            "finished_at": _now(),
            "hostname": hostname,
        },
    )


if __name__ == "__main__":
    main()
