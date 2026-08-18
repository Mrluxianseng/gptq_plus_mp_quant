#!/usr/bin/env python3
"""Checkpoint-only WikiText2 KL/PPL and ten-task QA for merged40."""

from __future__ import annotations

import argparse
import datetime as dt
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
from experiments.realq_fullmodel_retune_20260817 import formal_merged40 as formal
from experiments.realq_fullmodel_retune_20260817 import selection_merged40 as merged


QUALITY_ID = "realq-fullmodel-authoritative-quality-merged40-20260818-v1"
QUALITY_PLAN_PATH = merged.OUTPUT_ROOT / "quality_plan.json"
QUALITY_AUDIT_PATH = merged.OUTPUT_ROOT / "quality_final_audit.json"
CLAIM_NAME = ".quality.claim"
PAPER_QA_TASKS = (
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "lambada_openai",
    "ceval-valid",
    "boolq",
    "openbookqa",
    "social_iqa",
)
LM_EVAL_VERSION = "0.4.4"
LM_EVAL_BATCH_SIZE = 32
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
    re.IGNORECASE,
)
QUALITY_OVERRIDES = {
    "--load_qmodel_path",
    "--lm_eval",
    "--lm_eval_batch_size",
    "--output_dir",
    "--reasoning_eval",
    "--require_reference_cache_hit",
    "--require_static_cache_hit",
    "--skip_eval",
    "--skip_kl_ppl_eval",
    "--exp",
}


def _pairs() -> list[tuple[str, str]]:
    return formal._balanced_pairs()


def _quality_dir(branch: str, config: str) -> Path:
    return merged.OUTPUT_ROOT / "quality" / branch / config


def _checkpoint(branch: str, config: str) -> Path:
    return formal._checkpoint_path(branch, config)


def _checkpoint_stat(branch: str, config: str) -> dict[str, Any]:
    path = _checkpoint(branch, config)
    if not path.is_file() or path.stat().st_size <= 0:
        raise c.CampaignError(f"checkpoint missing or empty: {branch}/{config}")
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _formal_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = formal._load_plan()
    audit = c._read_json(formal.FORMAL_AUDIT_PATH)
    identity = dict(audit)
    fingerprint = identity.pop("audit_fingerprint", None)
    identity.pop("audited_at", None)
    if c._canonical_sha256(identity) != fingerprint:
        raise c.CampaignError("formal audit fingerprint mismatch")
    if (
        audit.get("formal_id") != formal.FORMAL_ID
        or audit.get("formal_plan_fingerprint") != plan["formal_plan_fingerprint"]
        or audit.get("status") != "complete"
        or audit.get("counts") != {"formal_checkpoints": 40}
    ):
        raise c.CampaignError("formal audit is incomplete or has the wrong identity")
    for branch, config in _pairs():
        formal._audit_one(plan, branch, config)
    return plan, audit


def _validate_quality_command(
    source: Sequence[str],
    command: Sequence[str],
    branch: str,
    config: str,
    output: Path,
) -> None:
    if list(command[:3]) != list(source[:3]):
        raise c.CampaignError("quality command changed executable/module")
    source_flags, flags = formal._flags(source), formal._flags(command)
    added = set(flags) - set(source_flags)
    if added - QUALITY_OVERRIDES:
        raise c.CampaignError(f"quality command added unexpected flags: {added}")
    dropped = set(source_flags) - set(flags)
    if dropped != {"--save_qmodel_path"}:
        raise c.CampaignError(f"quality command dropped unexpected flags: {dropped}")
    for flag, value in source_flags.items():
        if flag not in QUALITY_OVERRIDES and flag != "--save_qmodel_path":
            if flags.get(flag) != value:
                raise c.CampaignError(f"quality command drifted source flag {flag}")
    expected = {
        "--load_qmodel_path": str(_checkpoint(branch, config)),
        "--skip_eval": "false",
        "--skip_kl_ppl_eval": "false",
        "--lm_eval": "true",
        "--lm_eval_batch_size": str(LM_EVAL_BATCH_SIZE),
        "--reasoning_eval": "false",
        "--require_static_cache_hit": "false",
        "--require_reference_cache_hit": "true",
        "--output_dir": str(output),
        "--exp": "quality_authoritative_merged40",
    }
    for flag, value in expected.items():
        if flags.get(flag) != value:
            raise c.CampaignError(f"quality override mismatch: {flag}")
    if "--save_qmodel_path" in flags:
        raise c.CampaignError("quality command may not save a checkpoint")
    if flags.get("--full_block_refresh") != c.BRANCH_VALUES[branch]:
        raise c.CampaignError("quality branch flag mismatch")
    for flag, value in merged.FROZEN_FLAGS.items():
        if flags.get(flag) != value:
            raise c.CampaignError(f"quality frozen flag mismatch: {flag}")
    if flags.get("--a_loss_ratio") not in {"1", "1.0"}:
        raise c.CampaignError("quality command must preserve a_loss_ratio=1")


def _quality_command(
    formal_plan: Mapping[str, Any], branch: str, config: str, output: Path
) -> list[str]:
    row = formal._plan_rows(formal_plan)[f"{branch}/{config}"]
    source = list(row["command"])
    command = list(source)
    c._remove_arg(command, "--save_qmodel_path")
    c._remove_arg(command, "--load_qmodel_path")
    c._set_or_append_arg(command, "--load_qmodel_path", str(_checkpoint(branch, config)))
    c._set_arg(command, "--skip_eval", "false")
    c._set_arg(command, "--skip_kl_ppl_eval", "false")
    c._set_arg(command, "--lm_eval", "true")
    c._set_or_append_arg(command, "--lm_eval_batch_size", str(LM_EVAL_BATCH_SIZE))
    c._set_arg(command, "--reasoning_eval", "false")
    c._set_arg(command, "--require_static_cache_hit", "false")
    c._set_arg(command, "--require_reference_cache_hit", "true")
    c._set_arg(command, "--output_dir", str(output))
    c._set_arg(command, "--exp", "quality_authoritative_merged40")
    _validate_quality_command(source, command, branch, config, output)
    return command


def _normalized_command(command: Sequence[str]) -> list[str]:
    value = list(command)
    c._set_arg(value, "--output_dir", "<QUALITY_ATTEMPT_OUTPUT>")
    return value


def _build_plan() -> dict[str, Any]:
    formal_plan, formal_audit = _formal_inputs()
    rows = []
    for branch, config in _pairs():
        checkpoint = _checkpoint_stat(branch, config)
        command = _quality_command(
            formal_plan, branch, config, Path("/quality/attempt/realq_output")
        )
        rows.append(
            {
                "branch": branch,
                "config": config,
                "checkpoint": checkpoint,
                "formal": formal._audit_one(formal_plan, branch, config),
                "command": command,
                "command_sha256": c._canonical_sha256(command),
            }
        )
    code = Path(__file__).resolve()
    body: dict[str, Any] = {
        "quality_id": QUALITY_ID,
        "merge_id": merged.MERGE_ID,
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
        "protocol": {
            "checkpoint_only": True,
            "checkpoint_loads_per_run": 1,
            "wikitext2_exact_full_vocabulary_kl_ppl": True,
            "paper_qa_tasks": list(PAPER_QA_TASKS),
            "lm_eval_version": LM_EVAL_VERSION,
            "lm_eval_batch_size": LM_EVAL_BATCH_SIZE,
            "reasoning_eval": False,
            "require_reference_cache_hit": True,
            "qa_average": "mean of ten task percentages after per-task rounding to 2 decimals",
        },
        "rows": rows,
    }
    body["quality_plan_fingerprint"] = c._canonical_sha256(body)
    return body


def _write_plan(_: argparse.Namespace) -> int:
    value = _build_plan()
    if QUALITY_PLAN_PATH.exists():
        if c._read_json(QUALITY_PLAN_PATH) != value:
            raise c.CampaignError("existing quality plan differs")
    else:
        c._atomic_json(QUALITY_PLAN_PATH, value)
    print(QUALITY_PLAN_PATH)
    return 0


def _load_plan() -> dict[str, Any]:
    value = c._read_json(QUALITY_PLAN_PATH)
    identity = dict(value)
    fingerprint = identity.pop("quality_plan_fingerprint", None)
    if c._canonical_sha256(identity) != fingerprint:
        raise c.CampaignError("quality plan fingerprint mismatch")
    if value != _build_plan():
        raise c.CampaignError("quality plan no longer matches frozen inputs")
    return value


def _rows(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {f"{row['branch']}/{row['config']}": row for row in plan["rows"]}


def _make_manifest(args: argparse.Namespace) -> int:
    plan = _load_plan()
    value = {
        "quality_id": QUALITY_ID,
        "quality_plan_fingerprint": plan["quality_plan_fingerprint"],
        "name": args.name,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": [
            {"branch": branch, "config": config} for branch, config in _pairs()
        ],
    }
    path = merged.OUTPUT_ROOT / "manifests" / f"{args.name}.json"
    if path.exists():
        raise c.CampaignError(f"quality manifest already exists: {path}")
    c._atomic_json(path, value)
    print(path)
    return 0


def _validate_manifest(plan: Mapping[str, Any], value: Mapping[str, Any]) -> None:
    if value.get("quality_id") != QUALITY_ID:
        raise c.CampaignError("quality manifest id mismatch")
    if value.get("quality_plan_fingerprint") != plan["quality_plan_fingerprint"]:
        raise c.CampaignError("quality manifest plan mismatch")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != 40:
        raise c.CampaignError("quality manifest must contain 40 runs")
    if {(run.get("branch"), run.get("config")) for run in runs} != set(_pairs()):
        raise c.CampaignError("quality manifest matrix mismatch")


def _worker_env(gpu: str) -> dict[str, str]:
    return formal._worker_env(gpu)


def _next_attempt(directory: Path) -> Path:
    indices = []
    if directory.is_dir():
        for path in directory.glob("attempt[0-9][0-9][0-9]"):
            try:
                indices.append(int(path.name.removeprefix("attempt")))
            except ValueError:
                pass
    return directory / f"attempt{max(indices, default=0) + 1:03d}"


def _parse_metrics(path: Path) -> dict[str, Any]:
    from tools.lowbit_activation_results import _parse_metrics_log

    text = path.read_text(encoding="utf-8", errors="strict")
    kl, ppl, tasks, average = _parse_metrics_log(text, phase="final", method="realq")
    if kl is None or not math.isfinite(kl) or kl < 0:
        raise c.CampaignError("quality Exact KL is invalid")
    if ppl is None or not math.isfinite(ppl) or ppl <= 0:
        raise c.CampaignError("quality PPL is invalid")
    if tuple(tasks) != PAPER_QA_TASKS:
        raise c.CampaignError(f"quality QA task set/order mismatch: {tuple(tasks)}")
    if average is None or not math.isfinite(average):
        raise c.CampaignError("quality QA average is invalid")
    recomputed = round(sum(float(tasks[task]) for task in PAPER_QA_TASKS) / 10, 2)
    if not math.isclose(recomputed, float(average), rel_tol=0, abs_tol=0.011):
        raise c.CampaignError("quality QA average arithmetic mismatch")
    return {
        "wikitext2": {"kl": float(kl), "ppl": float(ppl)},
        "paper_qa": {
            "tasks": {task: float(tasks[task]) for task in PAPER_QA_TASKS},
            "average": float(average),
        },
    }


def _audit_one(plan: Mapping[str, Any], branch: str, config: str) -> dict[str, Any]:
    row = _rows(plan)[f"{branch}/{config}"]
    if _checkpoint_stat(branch, config) != row["checkpoint"]:
        raise c.CampaignError(f"quality checkpoint changed: {branch}/{config}")
    directory = _quality_dir(branch, config)
    marker = c._read_json(directory / "quality_success.json")
    if (
        marker.get("quality_id") != QUALITY_ID
        or marker.get("branch") != branch
        or marker.get("config") != config
    ):
        raise c.CampaignError(f"quality marker identity mismatch: {branch}/{config}")
    result_ref = marker.get("result")
    if not isinstance(result_ref, dict):
        raise c.CampaignError(f"quality result reference missing: {branch}/{config}")
    result_path = Path(str(result_ref.get("path", "")))
    if not result_path.is_file() or c._file_sha256(result_path) != result_ref.get("sha256"):
        raise c.CampaignError(f"quality result hash mismatch: {branch}/{config}")
    result = c._read_json(result_path)
    if result.get("status") != "succeeded" or result.get("returncode") != 0:
        raise c.CampaignError(f"quality result not successful: {branch}/{config}")
    if (
        result.get("quality_id") != QUALITY_ID
        or result.get("quality_plan_fingerprint") != plan["quality_plan_fingerprint"]
        or result.get("branch") != branch
        or result.get("config") != config
        or result.get("checkpoint") != row["checkpoint"]
    ):
        raise c.CampaignError(f"quality result identity mismatch: {branch}/{config}")
    command = result.get("command")
    if not isinstance(command, list) or c._canonical_sha256(command) != result.get("command_sha256"):
        raise c.CampaignError(f"quality command hash mismatch: {branch}/{config}")
    if _normalized_command(command) != _normalized_command(row["command"]):
        raise c.CampaignError(f"quality command drift: {branch}/{config}")
    log_ref = result.get("log")
    if not isinstance(log_ref, dict):
        raise c.CampaignError(f"quality log reference missing: {branch}/{config}")
    log_path = Path(str(log_ref.get("path", "")))
    if not log_path.is_file() or c._file_sha256(log_path) != log_ref.get("sha256"):
        raise c.CampaignError(f"quality log hash mismatch: {branch}/{config}")
    metrics = _parse_metrics(log_path)
    if metrics != result.get("metrics"):
        raise c.CampaignError(f"quality metric parser drift: {branch}/{config}")
    return {
        "branch": branch,
        "config": config,
        "checkpoint": row["checkpoint"],
        "metrics": metrics,
        "elapsed_seconds": result["elapsed_seconds"],
        "result": result_ref,
    }


def _run_one(plan: Mapping[str, Any], branch: str, config: str, cuda_id: str) -> int:
    directory = _quality_dir(branch, config)
    marker = directory / "quality_success.json"
    if marker.is_file():
        _audit_one(plan, branch, config)
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / CLAIM_NAME
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise c.CampaignError(f"quality run already claimed: {branch}/{config}") from exc
    attempt = _next_attempt(directory)
    attempt.mkdir()
    result_path = attempt / "result.json"
    log_path = attempt / "execution.log"
    command = _quality_command(
        formal._load_plan(), branch, config, attempt / "realq_output"
    )
    row = _rows(plan)[f"{branch}/{config}"]
    manifest = {
        "quality_id": QUALITY_ID,
        "quality_plan_fingerprint": plan["quality_plan_fingerprint"],
        "branch": branch,
        "config": config,
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
            result.update(
                status="failed",
                failure_class="oom" if OOM_RE.search(tail) else "non_oom",
            )
            c._atomic_json(result_path, result)
            return 1
        result.update(status="succeeded", metrics=_parse_metrics(log_path))
        c._atomic_json(result_path, result)
        success = {
            "quality_id": QUALITY_ID,
            "branch": branch,
            "config": config,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "result": {"path": str(result_path), "sha256": c._file_sha256(result_path)},
        }
        c._atomic_json(marker, success)
        _audit_one(plan, branch, config)
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
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise c.CampaignError("invalid quality worker index/count")
    selected = [
        run
        for index, run in enumerate(manifest["runs"])
        if index % args.worker_count == args.worker_index
    ]
    failures = 0
    for run in selected:
        failures += int(
            _run_one(plan, str(run["branch"]), str(run["config"]), args.cuda_id)
            != 0
        )
    return 1 if failures else 0


def _last_attempt_failed(directory: Path) -> bool:
    attempts = sorted(directory.glob("attempt[0-9][0-9][0-9]")) if directory.is_dir() else []
    if not attempts:
        return False
    if (attempts[-1] / "failure.json").is_file():
        return True
    result = attempts[-1] / "result.json"
    return result.is_file() and c._read_json(result).get("status") == "failed"


def _status(_: argparse.Namespace) -> int:
    rows = []
    for branch, config in _pairs():
        directory = _quality_dir(branch, config)
        state = (
            "succeeded"
            if (directory / "quality_success.json").is_file()
            else "running"
            if (directory / CLAIM_NAME).is_dir()
            else "failed"
            if _last_attempt_failed(directory)
            else "pending"
        )
        rows.append({"branch": branch, "config": config, "status": state})
    print(
        json.dumps(
            {
                "quality_id": QUALITY_ID,
                "counts": {
                    state: sum(row["status"] == state for row in rows)
                    for state in ("succeeded", "running", "failed", "pending")
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _audit_all(_: argparse.Namespace) -> int:
    plan = _load_plan()
    rows = [_audit_one(plan, branch, config) for branch, config in _pairs()]
    value: dict[str, Any] = {
        "quality_id": QUALITY_ID,
        "quality_plan_fingerprint": plan["quality_plan_fingerprint"],
        "status": "complete",
        "counts": {
            "runs": len(rows),
            "wikitext2_ppl": len(rows),
            "paper_qa_task_runs": len(rows) * len(PAPER_QA_TASKS),
        },
        "rows": rows,
    }
    value["audit_fingerprint"] = c._canonical_sha256(value)
    value["audited_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    c._atomic_json(QUALITY_AUDIT_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("write-plan")
    plan.set_defaults(handler=_write_plan)
    manifest = subparsers.add_parser("make-manifest")
    manifest.add_argument("--name", default="quality_merged40_v1")
    manifest.set_defaults(handler=_make_manifest)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--cuda-id", required=True)
    worker.set_defaults(handler=_run_worker)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    audit = subparsers.add_parser("audit")
    audit.set_defaults(handler=_audit_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        c.CampaignError,
        merged.MergeError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"realq-quality-merged40: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
