#!/usr/bin/env python3
"""Three-task reasoning generation and scoring stage for V3 checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import campaign_v3 as v3

v3._bootstrap()

from experiments.realq_fullmodel_retune_20260817 import formal_v3 as formal  # noqa: E402


REASONING_ID = "realq-fullmodel-two-branch-reasoning-20260817-v3"
REASONING_PLAN_PATH = v3.OUTPUT_ROOT / "reasoning_plan.json"
REASONING_AUDIT_PATH = v3.OUTPUT_ROOT / "reasoning_final_audit.json"
DATA_ROOT = c.REPO_ROOT / "datasets" / "reasoning_eval"
OVERLAY = DATA_ROOT / "python_packages"
TASKS: dict[str, tuple[int, int, int]] = {
    "gsm8k": (1024, 32, 1319),
    "math_500": (2048, 16, 500),
    "humaneval_plus": (2048, 16, 164),
}
DATASET_PATHS = {
    "gsm8k": DATA_ROOT / "gsm8k" / "test.jsonl",
    "math_500": DATA_ROOT / "math_500" / "test.jsonl",
    "humaneval_plus": DATA_ROOT / "humaneval_plus" / "HumanEvalPlus.jsonl.gz",
}
CLAIM_NAME = ".generation.claim"
EVALPLUS_CLAIM_NAME = ".official.claim"
EVALPLUS_PARALLEL = 32
EVALPLUS_STABILITY_SECONDS = 30
EVALPLUS_STATUSES = {"pass", "fail", "timeout"}
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
    re.IGNORECASE,
)


def _pairs() -> list[tuple[str, str]]:
    return v3.v2._balanced_pairs()


def _work_id(branch: str, config: str, task: str) -> str:
    return f"{branch}__{config}__{task}"


def _output_dir(branch: str, config: str, task: str) -> Path:
    return v3.OUTPUT_ROOT / "reasoning" / branch / config / task


def _checkpoint(branch: str, config: str) -> Path:
    return formal._checkpoint_path(branch, config)


def _checkpoint_stat(branch: str, config: str) -> dict[str, Any]:
    path = _checkpoint(branch, config)
    if not path.is_file() or path.stat().st_size <= 0:
        raise c.CampaignError(f"reasoning checkpoint missing: {branch}/{config}")
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _dataset_snapshot(task: str) -> dict[str, Any]:
    path = DATASET_PATHS[task]
    if not path.is_file() or path.stat().st_size <= 0:
        raise c.CampaignError(f"reasoning dataset missing: {path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            count = sum(bool(line.strip()) for line in handle)
    else:
        with path.open(encoding="utf-8") as handle:
            count = sum(bool(line.strip()) for line in handle)
    if count != TASKS[task][2]:
        raise c.CampaignError(
            f"reasoning dataset row mismatch for {task}: {count}"
        )
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": c._file_sha256(path),
        "rows": count,
    }


def _formal_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = formal._load_plan()
    audit = c._read_json(formal.FORMAL_AUDIT_PATH)
    identity = dict(audit)
    fingerprint = identity.pop("audit_fingerprint", None)
    identity.pop("audited_at", None)
    if c._canonical_sha256(identity) != fingerprint:
        raise c.CampaignError("reasoning formal-audit fingerprint mismatch")
    if (
        audit.get("formal_id") != formal.FORMAL_ID
        or audit.get("formal_plan_fingerprint") != plan["formal_plan_fingerprint"]
        or audit.get("status") != "complete"
        or audit.get("counts") != {"formal_checkpoints": 40}
    ):
        raise c.CampaignError("reasoning formal audit is incomplete")
    for branch, config in _pairs():
        formal._audit_one(plan, branch, config)
    return plan, audit


def _reasoning_command(
    formal_plan: Mapping[str, Any],
    branch: str,
    config: str,
    task: str,
    pipeline_output: Path,
) -> list[str]:
    max_tokens, batch_size, _ = TASKS[task]
    row = formal._plan_rows(formal_plan)[f"{branch}/{config}"]
    command = list(row["command"])
    c._remove_arg(command, "--save_qmodel_path")
    c._remove_arg(command, "--load_qmodel_path")
    c._set_or_append_arg(command, "--load_qmodel_path", str(_checkpoint(branch, config)))
    c._set_arg(command, "--skip_eval", "false")
    c._set_arg(command, "--skip_kl_ppl_eval", "true")
    c._set_arg(command, "--lm_eval", "false")
    c._set_arg(command, "--reasoning_eval", "true")
    settings = {
        "--reasoning_tasks": task,
        "--reasoning_data_dir": str(DATA_ROOT),
        "--reasoning_output_dir": str(_output_dir(branch, config, task)),
        "--reasoning_batch_size": str(batch_size),
        "--reasoning_limit": "-1",
        "--reasoning_max_new_tokens": str(max_tokens),
        "--reasoning_num_samples": "1",
        "--reasoning_apply_chat_template": "true",
        "--reasoning_enable_thinking": "true",
        "--reasoning_do_sample": "false",
        "--reasoning_temperature": "0",
        "--reasoning_top_p": "1",
        "--reasoning_top_k": "0",
        "--reasoning_seed": "1234",
        "--reasoning_resume": "true",
        "--reasoning_protocol": "realq_zero_shot_v1",
        "--reasoning_system_prompt": (
            "You are a careful reasoning assistant. Follow the requested output format exactly."
        ),
        "--require_static_cache_hit": "false",
        "--require_reference_cache_hit": "false",
        "--output_dir": str(pipeline_output),
        "--exp": f"v3_{task}",
    }
    for flag, value in settings.items():
        c._set_or_append_arg(command, flag, value)
    c._validate_full_profile(command, branch=branch, config=config)
    return command


def _normalized_command(command: Sequence[str]) -> list[str]:
    value = list(command)
    c._set_arg(value, "--output_dir", "<REASONING_ATTEMPT_OUTPUT>")
    return value


def _model_cost(config: str) -> float:
    for slug, cost in (
        ("qwen3-32b", 32.0),
        ("llama31-8b-instruct", 8.0),
        ("qwen3-8b", 8.0),
        ("qwen3-4b", 4.0),
        ("qwen3-0.6b", 0.6),
    ):
        if config.startswith(slug + "_"):
            return cost
    raise c.CampaignError(f"unknown model cost: {config}")


def _task_cost(task: str) -> float:
    return {"gsm8k": 1.35, "math_500": 1.0, "humaneval_plus": 0.34}[task]


def _balanced_works(worker_count: int = 16) -> list[dict[str, str]]:
    works = [
        {"branch": branch, "config": config, "task": task}
        for branch, config in _pairs()
        for task in TASKS
    ]
    works.sort(
        key=lambda work: (
            _model_cost(work["config"]) * _task_cost(work["task"]),
            work["config"],
            work["branch"],
            work["task"],
        ),
        reverse=True,
    )
    capacities = [8 if lane < 8 else 7 for lane in range(worker_count)]
    lanes: list[list[dict[str, str]]] = [[] for _ in range(worker_count)]
    loads = [0.0] * worker_count
    for work in works:
        choices = [lane for lane in range(worker_count) if len(lanes[lane]) < capacities[lane]]
        lane = min(choices, key=lambda item: (loads[item], item))
        lanes[lane].append(work)
        loads[lane] += _model_cost(work["config"]) * _task_cost(work["task"])
    ordered = []
    for position in range(max(capacities)):
        for lane in range(worker_count):
            if position < len(lanes[lane]):
                ordered.append(lanes[lane][position])
    if len(ordered) != 120 or len(
        {_work_id(work["branch"], work["config"], work["task"]) for work in ordered}
    ) != 120:
        raise c.CampaignError("reasoning balanced work matrix mismatch")
    # The final short stripe consists exactly of lanes 0..7, preserving
    # index % worker_count assignment for every preceding full stripe.
    for index, work in enumerate(ordered):
        expected_lane = index % worker_count
        if work not in lanes[expected_lane]:
            raise c.CampaignError("reasoning stripe/lane assignment drift")
    return ordered


def _build_plan() -> dict[str, Any]:
    formal_plan, formal_audit = _formal_inputs()
    datasets = {task: _dataset_snapshot(task) for task in TASKS}
    rows = []
    for work in _balanced_works():
        branch, config, task = work["branch"], work["config"], work["task"]
        command = _reasoning_command(
            formal_plan,
            branch,
            config,
            task,
            Path("/reasoning/attempt/pipeline_output"),
        )
        rows.append(
            {
                **work,
                "work_id": _work_id(branch, config, task),
                "checkpoint": _checkpoint_stat(branch, config),
                "command": command,
                "command_sha256": c._canonical_sha256(command),
            }
        )
    code = Path(__file__).resolve()
    body: dict[str, Any] = {
        "reasoning_id": REASONING_ID,
        "campaign_id": v3.CAMPAIGN_ID,
        "formal_plan": {
            "path": str(formal.FORMAL_PLAN_PATH),
            "sha256": c._file_sha256(formal.FORMAL_PLAN_PATH),
            "fingerprint": formal_plan["formal_plan_fingerprint"],
        },
        "formal_audit": {
            "path": str(formal.FORMAL_AUDIT_PATH),
            "sha256": c._file_sha256(formal.FORMAL_AUDIT_PATH),
            "fingerprint": formal_audit["audit_fingerprint"],
        },
        "code": {"path": str(code), "sha256": c._file_sha256(code)},
        "datasets": datasets,
        "protocol": {
            "tasks": list(TASKS),
            "task_settings": {
                task: {
                    "max_new_tokens": values[0],
                    "batch_size": values[1],
                    "examples": values[2],
                }
                for task, values in TASKS.items()
            },
            "apply_chat_template": True,
            "enable_thinking": True,
            "do_sample": False,
            "temperature": 0,
            "top_p": 1,
            "top_k": 0,
            "seed": 1234,
            "num_samples": 1,
            "protocol": "realq_zero_shot_v1",
            "human_eval_plus_official_scorer": "EvalPlus 0.3.1",
            "worker_count": 16,
        },
        "rows": rows,
    }
    body["reasoning_plan_fingerprint"] = c._canonical_sha256(body)
    return body


def _write_plan(_: argparse.Namespace) -> int:
    value = _build_plan()
    if REASONING_PLAN_PATH.exists():
        if c._read_json(REASONING_PLAN_PATH) != value:
            raise c.CampaignError("existing reasoning plan differs")
    else:
        c._atomic_json(REASONING_PLAN_PATH, value)
    print(REASONING_PLAN_PATH)
    return 0


def _load_plan() -> dict[str, Any]:
    value = c._read_json(REASONING_PLAN_PATH)
    fingerprint = value.get("reasoning_plan_fingerprint")
    identity = dict(value)
    identity.pop("reasoning_plan_fingerprint", None)
    if c._canonical_sha256(identity) != fingerprint:
        raise c.CampaignError("reasoning plan fingerprint mismatch")
    if value != _build_plan():
        raise c.CampaignError("reasoning plan no longer matches frozen inputs")
    return value


def _plan_rows(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["work_id"]): row for row in plan["rows"]}


def _make_manifest(args: argparse.Namespace) -> int:
    plan = _load_plan()
    value = {
        "reasoning_id": REASONING_ID,
        "reasoning_plan_fingerprint": plan["reasoning_plan_fingerprint"],
        "name": args.name,
        "worker_count": 16,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "works": [
            {key: row[key] for key in ("branch", "config", "task", "work_id")}
            for row in plan["rows"]
        ],
    }
    path = v3.OUTPUT_ROOT / "manifests" / f"{args.name}.json"
    if path.exists():
        raise c.CampaignError(f"reasoning manifest already exists: {path}")
    c._atomic_json(path, value)
    print(path)
    return 0


def _validate_manifest(plan: Mapping[str, Any], value: Mapping[str, Any]) -> None:
    if value.get("reasoning_id") != REASONING_ID:
        raise c.CampaignError("reasoning manifest id mismatch")
    if value.get("reasoning_plan_fingerprint") != plan["reasoning_plan_fingerprint"]:
        raise c.CampaignError("reasoning manifest plan mismatch")
    if value.get("worker_count") != 16:
        raise c.CampaignError("reasoning manifest worker_count must be 16")
    works = value.get("works")
    if not isinstance(works, list) or len(works) != 120:
        raise c.CampaignError("reasoning manifest must contain 120 works")
    if {work.get("work_id") for work in works} != set(_plan_rows(plan)):
        raise c.CampaignError("reasoning manifest work matrix mismatch")


def _worker_env(cuda_id: str) -> dict[str, str]:
    environment = os.environ.copy()
    for key in c.TORCH_DISTRIBUTED_ENV:
        environment.pop(key, None)
    prior = environment.get("PYTHONPATH", "")
    environment.update(
        CUDA_VISIBLE_DEVICES=cuda_id,
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        REALQ_DETERMINISTIC_SDPA="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONHASHSEED="0",
        NVIDIA_TF32_OVERRIDE="0",
        PYTHONPATH=os.pathsep.join(
            item for item in (str(OVERLAY), str(c.REPO_ROOT), prior) if item
        ),
    )
    return environment


def _next_attempt(directory: Path) -> Path:
    indices = []
    if directory.is_dir():
        for path in directory.glob("attempt[0-9][0-9][0-9]"):
            try:
                indices.append(int(path.name.removeprefix("attempt")))
            except ValueError:
                pass
    return directory / f"attempt{max(indices, default=0) + 1:03d}"


def _audit_generated_output(
    plan: Mapping[str, Any], branch: str, config: str, task: str
) -> dict[str, Any]:
    output = _output_dir(branch, config, task)
    manifest_path = output / "manifest.json"
    manifest = c._read_json(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("tasks") != [task]:
        raise c.CampaignError(f"incomplete reasoning output: {branch}/{config}/{task}")
    expected = plan["datasets"][task]
    dataset = manifest.get("datasets", {}).get(task, {})
    if (
        dataset.get("path") != expected["path"]
        or dataset.get("sha256") != expected["sha256"]
        or dataset.get("selected_examples") != expected["rows"]
    ):
        raise c.CampaignError(f"reasoning dataset mismatch: {branch}/{config}/{task}")
    generation = manifest.get("generation", {})
    max_tokens, batch_size, count = TASKS[task]
    required_generation = {
        "apply_chat_template": True,
        "batch_size": batch_size,
        "do_sample": False,
        "enable_thinking": True,
        "limit": -1,
        "max_new_tokens": max_tokens,
        "num_samples": 1,
        "protocol": "realq_zero_shot_v1",
        "resume": True,
        "seed": 1234,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
    }
    for key, wanted in required_generation.items():
        if generation.get(key) != wanted:
            raise c.CampaignError(
                f"reasoning generation setting mismatch {key}: {branch}/{config}/{task}"
            )
    generations = output / task / "generations.jsonl"
    if not generations.is_file():
        raise c.CampaignError(f"reasoning generations missing: {generations}")
    with generations.open(encoding="utf-8") as handle:
        observed = sum(bool(line.strip()) for line in handle)
    if observed != count:
        raise c.CampaignError(
            f"reasoning generation count mismatch: {observed} != {count}"
        )
    results = manifest.get("results")
    if not isinstance(results, list) or len(results) != 1 or results[0].get("task") != task:
        raise c.CampaignError(f"reasoning result row mismatch: {branch}/{config}/{task}")
    result = results[0]
    metrics: dict[str, Any]
    if task in {"gsm8k", "math_500"}:
        score = result.get("pass_at_1")
        if (
            result.get("status") != "scored"
            or result.get("num_examples") != count
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
        ):
            raise c.CampaignError(f"reasoning score invalid: {branch}/{config}/{task}")
        metrics = {"pass_at_1": float(score), "correct": round(float(score) * count), "total": count}
    else:
        if result.get("status") != "generated_unscored" or result.get("num_examples") != count:
            raise c.CampaignError(f"HumanEval+ generation status invalid: {branch}/{config}")
        samples = output / task / "evalplus_samples.jsonl"
        if not samples.is_file():
            raise c.CampaignError(f"HumanEval+ samples missing: {samples}")
        metrics = {"official_score_pending": True, "total": count}
    return {
        "manifest": {"path": str(manifest_path), "sha256": c._file_sha256(manifest_path)},
        "generations": {"path": str(generations), "sha256": c._file_sha256(generations), "rows": observed},
        "metrics": metrics,
    }


def _audit_one(plan: Mapping[str, Any], branch: str, config: str, task: str) -> dict[str, Any]:
    work_id = _work_id(branch, config, task)
    row = _plan_rows(plan)[work_id]
    if _checkpoint_stat(branch, config) != row["checkpoint"]:
        raise c.CampaignError(f"reasoning checkpoint changed: {branch}/{config}")
    output = _output_dir(branch, config, task)
    marker_path = output / "generation_success.json"
    marker = c._read_json(marker_path)
    if marker.get("reasoning_id") != REASONING_ID or marker.get("work_id") != work_id:
        raise c.CampaignError(f"reasoning marker identity mismatch: {work_id}")
    result_ref = marker.get("result")
    if not isinstance(result_ref, dict):
        raise c.CampaignError(f"reasoning result reference missing: {work_id}")
    result_path = Path(str(result_ref.get("path", "")))
    if not result_path.is_file() or c._file_sha256(result_path) != result_ref.get("sha256"):
        raise c.CampaignError(f"reasoning result hash mismatch: {work_id}")
    result = c._read_json(result_path)
    if result.get("status") != "succeeded" or result.get("returncode") != 0:
        raise c.CampaignError(f"reasoning result not successful: {work_id}")
    if (
        result.get("reasoning_id") != REASONING_ID
        or result.get("reasoning_plan_fingerprint") != plan["reasoning_plan_fingerprint"]
        or result.get("work_id") != work_id
        or result.get("checkpoint") != row["checkpoint"]
    ):
        raise c.CampaignError(f"reasoning result identity mismatch: {work_id}")
    command = result.get("command")
    if not isinstance(command, list) or c._canonical_sha256(command) != result.get("command_sha256"):
        raise c.CampaignError(f"reasoning command hash mismatch: {work_id}")
    if _normalized_command(command) != _normalized_command(row["command"]):
        raise c.CampaignError(f"reasoning command drift: {work_id}")
    log_ref = result.get("log")
    log = Path(str(log_ref.get("path", ""))) if isinstance(log_ref, dict) else Path("")
    if not log.is_file() or c._file_sha256(log) != log_ref.get("sha256"):
        raise c.CampaignError(f"reasoning log hash mismatch: {work_id}")
    output_audit = _audit_generated_output(plan, branch, config, task)
    if output_audit != result.get("output"):
        raise c.CampaignError(f"reasoning output audit drift: {work_id}")
    return {
        "work_id": work_id,
        "branch": branch,
        "config": config,
        "task": task,
        "checkpoint": row["checkpoint"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output": output_audit,
        "result": result_ref,
    }


def _run_one(
    plan: Mapping[str, Any], branch: str, config: str, task: str, cuda_id: str
) -> int:
    work_id = _work_id(branch, config, task)
    output = _output_dir(branch, config, task)
    marker = output / "generation_success.json"
    if marker.is_file():
        _audit_one(plan, branch, config, task)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    claim = output / CLAIM_NAME
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise c.CampaignError(f"reasoning work already claimed: {work_id}") from exc
    attempt = _next_attempt(output)
    attempt.mkdir()
    result_path = attempt / "result.json"
    log_path = attempt / "execution.log"
    command = _reasoning_command(
        formal._load_plan(), branch, config, task, attempt / "pipeline_output"
    )
    row = _plan_rows(plan)[work_id]
    manifest = {
        "reasoning_id": REASONING_ID,
        "reasoning_plan_fingerprint": plan["reasoning_plan_fingerprint"],
        "work_id": work_id,
        "branch": branch,
        "config": config,
        "task": task,
        "checkpoint": row["checkpoint"],
        "command": command,
        "command_sha256": c._canonical_sha256(command),
        "gpu": c._gpu_inventory(cuda_id),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    c._atomic_json(attempt / "manifest.json", manifest)
    started = time.monotonic()
    try:
        with log_path.open("wb") as handle:
            handle.write(
                f"[{manifest['started_at']}] command={json.dumps(command, ensure_ascii=False)}\n".encode()
            )
            process = subprocess.Popen(
                command,
                cwd=c.REPO_ROOT,
                env=_worker_env(cuda_id),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            returncode = process.wait()
        result: dict[str, Any] = {
            **manifest,
            "returncode": returncode,
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "log": {"path": str(log_path), "sha256": c._file_sha256(log_path)},
        }
        if returncode:
            tail = log_path.read_bytes()[-4 * 1024 * 1024 :].decode(errors="replace")
            result.update(status="failed", failure_class="oom" if OOM_RE.search(tail) else "non_oom")
            c._atomic_json(result_path, result)
            return 1
        result.update(
            status="succeeded",
            output=_audit_generated_output(plan, branch, config, task),
        )
        c._atomic_json(result_path, result)
        c._atomic_json(
            marker,
            {
                "reasoning_id": REASONING_ID,
                "work_id": work_id,
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "official_code_score_pending": task == "humaneval_plus",
                "result": {"path": str(result_path), "sha256": c._file_sha256(result_path)},
            },
        )
        _audit_one(plan, branch, config, task)
        return 0
    except BaseException as exc:
        c._atomic_json(
            attempt / "failure.json",
            {
                **manifest,
                "status": "failed",
                "error_type": type(exc).__qualname__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        shutil.rmtree(claim, ignore_errors=True)


def _run_worker(args: argparse.Namespace) -> int:
    plan = _load_plan()
    manifest = c._read_json(args.manifest.expanduser().resolve())
    _validate_manifest(plan, manifest)
    if args.worker_count != 16 or not 0 <= args.worker_index < 16:
        raise c.CampaignError("reasoning workers require the frozen 16-lane layout")
    selected = [
        work
        for index, work in enumerate(manifest["works"])
        if index % 16 == args.worker_index
    ]
    failures = 0
    for work in selected:
        failures += int(
            _run_one(
                plan,
                str(work["branch"]),
                str(work["config"]),
                str(work["task"]),
                args.cuda_id,
            )
            != 0
        )
    return 1 if failures else 0


def _status(_: argparse.Namespace) -> int:
    rows = []
    for branch, config in _pairs():
        for task in TASKS:
            output = _output_dir(branch, config, task)
            rows.append(
                {
                    "work_id": _work_id(branch, config, task),
                    "status": "succeeded" if (output / "generation_success.json").is_file() else "running" if (output / CLAIM_NAME).is_dir() else "pending",
                }
            )
    print(
        json.dumps(
            {
                "reasoning_id": REASONING_ID,
                "counts": {
                    state: sum(row["status"] == state for row in rows)
                    for state in ("succeeded", "running", "pending")
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("write-plan")
    plan.set_defaults(handler=_write_plan)
    manifest = subparsers.add_parser("make-manifest")
    manifest.add_argument("--name", default="reasoning_v3_v1")
    manifest.set_defaults(handler=_make_manifest)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--cuda-id", required=True)
    worker.set_defaults(handler=_run_worker)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (c.CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"realq-reasoning-v3: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
