#!/usr/bin/env python3
"""Evaluate the 20 frozen REAL-Q checkpoints on WikiText2 and paper QA.

The two existing eight-GPU Canoe sleep jobs each run their ten assigned
checkpoints.  Every checkpoint is loaded exactly once for exact full-vocabulary
WikiText2 KL/PPL followed by the paper's ten zero-shot QA tasks.  The plan,
checkpoint identity, logs, parsed metrics, and completion markers are all
validated fail closed.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_linear_refresh_20group_20260811 import campaign as c


MODULE = "experiments.realq_linear_refresh_20group_20260811.quality"
QUALITY_ID = "realq-linear-refresh-20group-wikitext2-paperqa-20260811-v1"
QUALITY_SCHEMA_VERSION = 1
LM_EVAL_VERSION = "0.4.4"
LM_EVAL_BATCH_SIZE = 32
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
EXPECTED_PLAN_FINGERPRINT = (
    "6c749049865856fcda739d07e16bbcab84ddb8349e618d83b9f0f7449ae060e1"
)
EXPECTED_FINAL_AUDIT_FINGERPRINT = (
    "9ff8b504bc75c03ee755f0dd3cf6bb09424b14f004b5b330775826570e3d291d"
)
GPU_MIN_FREE_BYTES = 150_000_000_000
CLAIM_NAME = ".quality.claim"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise c.CampaignError(message)


def read_object(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required JSON file is missing: {path}")
    try:
        return c.read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise c.CampaignError(f"invalid JSON file {path}: {exc}") from exc


def plan_path(root: Path) -> Path:
    return root / "quality_plan.json"


def quality_dir(root: Path, run: c.RunSpec) -> Path:
    return c.run_root(root, run) / "quality"


def checkpoint_path(root: Path, run: c.RunSpec) -> Path:
    return c.run_root(root, run) / "checkpoint" / "quantized.pt"


def checkpoint_stat(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"checkpoint is missing: {path}")
    stat = path.stat()
    require(stat.st_size > 0, f"checkpoint is empty: {path}")
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def verify_fingerprint(
    value: Mapping[str, Any], field: str, expected: str, path: Path
) -> None:
    actual = value.get(field)
    require(actual == expected, f"frozen fingerprint mismatch in {path}: {actual}")
    identity = dict(value)
    identity.pop(field, None)
    require(
        c.canonical_sha256(identity) == expected,
        f"{field} does not match the JSON payload in {path}",
    )


def quality_args(
    root: Path, run: c.RunSpec, output: Path, *, batch_size: int = LM_EVAL_BATCH_SIZE
) -> list[str]:
    argv = c.common_args(root, run.model, run.quant, global_loss_bsz=32)
    argv.extend(
        [
            "--load_qmodel_path",
            str(checkpoint_path(root, run)),
            "--skip_eval",
            "false",
            "--skip_kl_ppl_eval",
            "false",
            "--lm_eval",
            "true",
            "--lm_eval_batch_size",
            str(batch_size),
            "--reasoning_eval",
            "false",
            "--require_static_cache_hit",
            "false",
            "--require_reference_cache_hit",
            "true",
            "--output_dir",
            str(output),
            "--exp",
            "wikitext2_paperqa",
        ]
    )
    c.validate_args(argv)
    return argv


def semantic_config(root: Path, run: c.RunSpec) -> dict[str, Any]:
    cfg = dataclasses.asdict(
        c.validate_args(quality_args(root, run, Path("/quality/output")))
    )
    cfg.pop("output_dir", None)
    return cfg


def expected_runs_for_node(node: int) -> tuple[c.RunSpec, ...]:
    require(node in (0, 1), f"node must be 0 or 1, got {node}")
    # Largest models first minimizes the tail while preserving the frozen node
    # assignment.  Stable reverse index ordering is deterministic.
    return tuple(
        sorted(
            (run for run in c.RUNS if run.node == node),
            key=lambda run: (run.model_index, run.quant_index),
            reverse=True,
        )
    )


def build_plan(root: Path) -> dict[str, Any]:
    original_plan_path = root / "plan.json"
    original_audit_path = root / "final_audit.json"
    original_plan = read_object(original_plan_path)
    original_audit = read_object(original_audit_path)
    verify_fingerprint(
        original_plan,
        "fingerprint",
        EXPECTED_PLAN_FINGERPRINT,
        original_plan_path,
    )
    verify_fingerprint(
        original_audit,
        "audit_fingerprint",
        EXPECTED_FINAL_AUDIT_FINGERPRINT,
        original_audit_path,
    )
    require(original_audit.get("status") == "complete", "base audit is not complete")
    require(
        original_audit.get("counts")
        == {
            "runs": 20,
            "formal_success": 20,
            "generation_success": 60,
            "official_evalplus_success": 20,
        },
        "base audit counts changed",
    )

    rows: list[dict[str, Any]] = []
    for run in c.RUNS:
        formal_path = c.run_root(root, run) / "formal_success.json"
        formal = read_object(formal_path)
        require(formal.get("campaign_id") == c.CAMPAIGN_ID, f"bad formal identity: {run.run_id}")
        require(formal.get("run_id") == run.run_id, f"bad formal run id: {run.run_id}")
        checkpoint = checkpoint_stat(checkpoint_path(root, run))
        require(
            formal.get("checkpoint_stat")
            == {
                "size_bytes": checkpoint["size_bytes"],
                "mtime_ns": checkpoint["mtime_ns"],
            },
            f"checkpoint stat differs from formal marker: {run.run_id}",
        )
        rows.append(
            {
                "index": run.index,
                "run_id": run.run_id,
                "node": run.node,
                "model": dataclasses.asdict(run.model),
                "quant": dataclasses.asdict(run.quant),
                "checkpoint": checkpoint,
                "formal_success": {
                    "path": str(formal_path),
                    "sha256": c.file_sha256(formal_path),
                },
                "semantic_config": semantic_config(root, run),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "quality_id": QUALITY_ID,
        "base_campaign_id": c.CAMPAIGN_ID,
        "root": str(root),
        "base_plan": {
            "path": str(original_plan_path),
            "fingerprint": original_plan["fingerprint"],
            "sha256": c.file_sha256(original_plan_path),
        },
        "base_final_audit": {
            "path": str(original_audit_path),
            "audit_fingerprint": original_audit["audit_fingerprint"],
            "sha256": c.file_sha256(original_audit_path),
        },
        "protocol": {
            "wikitext2_exact_full_vocab_kl_ppl": True,
            "require_reference_cache_hit": True,
            "paper_qa_tasks": list(PAPER_QA_TASKS),
            "lm_eval_version": LM_EVAL_VERSION,
            "lm_eval_batch_size": LM_EVAL_BATCH_SIZE,
            "task_score": "acc_norm,none when available; otherwise acc,none",
            "qa_average": "mean of ten task percentages after per-task rounding to 2 decimals",
            "reasoning_eval": False,
            "checkpoint_loads_per_run": 1,
        },
        "runs": rows,
    }
    payload["fingerprint"] = c.canonical_sha256(payload)
    return payload


def write_plan(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    expected = build_plan(root)
    path = plan_path(root)
    if path.exists():
        current = read_object(path)
        require(current == expected, f"existing quality plan differs: {path}")
    else:
        c.atomic_json(path, expected)
    require(c.file_sha256(path) == c.file_sha256(path), "quality plan changed while hashing")
    return expected


def load_and_validate_plan(root: Path) -> dict[str, Any]:
    path = plan_path(root)
    plan = read_object(path)
    fingerprint = plan.get("fingerprint")
    identity = dict(plan)
    identity.pop("fingerprint", None)
    require(
        fingerprint == c.canonical_sha256(identity),
        f"quality plan fingerprint mismatch: {path}",
    )
    require(plan == build_plan(root), "quality plan no longer matches frozen inputs")
    return plan


def plan_row(plan: Mapping[str, Any], run: c.RunSpec) -> dict[str, Any]:
    rows = plan.get("runs")
    require(isinstance(rows, list), "quality plan runs must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("run_id") == run.run_id]
    require(len(matches) == 1, f"quality plan row is not unique: {run.run_id}")
    return matches[0]


def verify_run_inputs(root: Path, plan: Mapping[str, Any], run: c.RunSpec) -> dict[str, Any]:
    row = plan_row(plan, run)
    require(checkpoint_stat(checkpoint_path(root, run)) == row.get("checkpoint"), f"checkpoint changed: {run.run_id}")
    formal = Path(row["formal_success"]["path"])
    require(c.file_sha256(formal) == row["formal_success"]["sha256"], f"formal marker changed: {run.run_id}")
    require(semantic_config(root, run) == row.get("semantic_config"), f"quality config changed: {run.run_id}")
    return row


def validate_metrics(log_path: Path) -> dict[str, Any]:
    from tools.lowbit_activation_results import _parse_metrics_log

    text = log_path.read_text(encoding="utf-8", errors="strict")
    kl, ppl, tasks, acc_avg = _parse_metrics_log(text, phase="final", method="realq")
    require(kl is not None and math.isfinite(kl) and kl >= 0, "KL is invalid")
    require(math.isfinite(ppl) and ppl > 0, "PPL is invalid")
    require(tuple(tasks) == PAPER_QA_TASKS, "QA task set/order is invalid")
    require(acc_avg is not None and math.isfinite(acc_avg), "QA average is invalid")
    recomputed = round(sum(tasks.values()) / len(PAPER_QA_TASKS), 2)
    require(math.isclose(recomputed, acc_avg, rel_tol=0, abs_tol=0.011), "QA average arithmetic mismatch")
    return {
        "wikitext2": {"kl_raw": float(kl), "kl_x100": float(kl) * 100.0, "ppl": float(ppl)},
        "paper_qa": {"tasks": {task: float(tasks[task]) for task in PAPER_QA_TASKS}, "acc_avg": float(acc_avg)},
    }


def next_attempt(directory: Path) -> tuple[int, Path]:
    indices = []
    for item in directory.glob("attempt[0-9][0-9][0-9]"):
        if item.is_dir():
            with contextlib.suppress(ValueError):
                indices.append(int(item.name.removeprefix("attempt")))
    index = max(indices, default=0) + 1
    return index, directory / f"attempt{index:03d}"


def acquire_claim(directory: Path, run: c.RunSpec, gpu: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / CLAIM_NAME
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise c.CampaignError(f"quality writer already claimed: {run.run_id}") from exc
    c.atomic_json(
        claim / "owner.json",
        {"run_id": run.run_id, "hostname": socket.gethostname(), "pid": os.getpid(), "gpu": gpu, "claimed_at": c.now()},
    )
    return claim


def gpu_snapshot(expected_physical_gpu: int) -> dict[str, Any]:
    import torch

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(expected_physical_gpu), "CUDA_VISIBLE_DEVICES does not match worker lane")
    require(torch.cuda.device_count() == 1, "quality worker must see exactly one GPU")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    require(free_bytes >= GPU_MIN_FREE_BYTES, f"GPU is busy: free_bytes={free_bytes}")
    return {
        "physical_gpu": expected_physical_gpu,
        "visible_gpu_count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def run_worker(root: Path, run: c.RunSpec, gpu: int) -> None:
    plan = load_and_validate_plan(root)
    row = verify_run_inputs(root, plan, run)
    directory = quality_dir(root, run)
    success_path = directory / "quality_success.json"
    if success_path.is_file():
        audit_one(root, plan, run)
        return
    claim = acquire_claim(directory, run, gpu)
    attempt_index, attempt = next_attempt(directory)
    attempt.mkdir()
    log_path = attempt / "execution.log"
    manifest_path = attempt / "manifest.json"
    result_path = attempt / "quality_result.json"
    command = [str(c.PYTHON), "-m", "realq.ptq", *quality_args(root, run, attempt / "realq_output")]
    manifest: dict[str, Any] = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "running",
        "quality_id": QUALITY_ID,
        "started_at": c.now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "job_id": os.environ.get("CANOE_JOB_ID"),
        "run_id": run.run_id,
        "attempt_index": attempt_index,
        "gpu": gpu_snapshot(gpu),
        "quality_plan": {"path": str(plan_path(root)), "sha256": c.file_sha256(plan_path(root)), "fingerprint": plan["fingerprint"]},
        "checkpoint": row["checkpoint"],
        "formal_success": row["formal_success"],
        "command": command,
        "protocol": plan["protocol"],
    }
    c.atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        returncode, elapsed = c.run_logged(command, log_path, gpu)
        require(returncode == 0, f"quality subprocess failed with returncode={returncode}")
        footer = c.completed_logged_run(log_path)
        require(footer is not None and footer[0] == 0, "quality log lacks a final rc=0 footer")
        metrics = validate_metrics(log_path)
        verify_run_inputs(root, plan, run)
        require(c.file_sha256(plan_path(root)) == manifest["quality_plan"]["sha256"], "quality plan changed during execution")
        result = {
            **manifest,
            "status": "succeeded",
            "finished_at": c.now(),
            "exit_code": 0,
            "evaluation_wall_seconds": elapsed,
            "quantization_gpu_hours_included": 0.0,
            "metrics": metrics,
            "log": {"path": str(log_path), "sha256": c.file_sha256(log_path), "size_bytes": log_path.stat().st_size},
        }
        c.atomic_json(result_path, result)
        marker = {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "quality_id": QUALITY_ID,
            "run_id": run.run_id,
            "completed_at": c.now(),
            "attempt": attempt_index,
            "result": {"path": str(result_path), "sha256": c.file_sha256(result_path)},
        }
        c.atomic_json(success_path, marker)
        audit_one(root, plan, run)
    except BaseException as exc:
        failure = {
            **manifest,
            "status": "failed",
            "finished_at": c.now(),
            "evaluation_wall_seconds": time.monotonic() - started,
            "error_type": type(exc).__qualname__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        c.atomic_json(attempt / "failure.json", failure)
        raise
    finally:
        shutil.rmtree(claim, ignore_errors=True)


def audit_one(root: Path, plan: Mapping[str, Any], run: c.RunSpec) -> dict[str, Any]:
    row = verify_run_inputs(root, plan, run)
    directory = quality_dir(root, run)
    marker_path = directory / "quality_success.json"
    marker = read_object(marker_path)
    require(marker.get("schema_version") == QUALITY_SCHEMA_VERSION, f"bad success schema: {run.run_id}")
    require(marker.get("quality_id") == QUALITY_ID and marker.get("run_id") == run.run_id, f"bad success identity: {run.run_id}")
    result_ref = marker.get("result")
    require(isinstance(result_ref, dict), f"bad result reference: {run.run_id}")
    result_path = Path(result_ref.get("path", ""))
    require(result_path.is_file(), f"quality result missing: {run.run_id}")
    require(c.file_sha256(result_path) == result_ref.get("sha256"), f"quality result hash mismatch: {run.run_id}")
    result = read_object(result_path)
    require(result.get("status") == "succeeded" and result.get("exit_code") == 0, f"quality result not successful: {run.run_id}")
    require(result.get("quality_id") == QUALITY_ID and result.get("run_id") == run.run_id, f"quality result identity mismatch: {run.run_id}")
    require(result.get("checkpoint") == row["checkpoint"], f"quality checkpoint mismatch: {run.run_id}")
    require(result.get("formal_success") == row["formal_success"], f"formal binding mismatch: {run.run_id}")
    plan_ref = result.get("quality_plan")
    require(isinstance(plan_ref, dict) and plan_ref.get("fingerprint") == plan["fingerprint"], f"quality plan binding mismatch: {run.run_id}")
    require(plan_ref.get("sha256") == c.file_sha256(plan_path(root)), f"quality plan hash mismatch: {run.run_id}")
    log_ref = result.get("log")
    require(isinstance(log_ref, dict), f"quality log reference invalid: {run.run_id}")
    log_path = Path(log_ref.get("path", ""))
    require(log_path.is_file() and c.file_sha256(log_path) == log_ref.get("sha256"), f"quality log hash mismatch: {run.run_id}")
    footer = c.completed_logged_run(log_path)
    require(footer is not None and footer[0] == 0, f"quality log not complete: {run.run_id}")
    metrics = validate_metrics(log_path)
    require(metrics == result.get("metrics"), f"quality metrics/parser mismatch: {run.run_id}")
    return {
        "index": run.index,
        "run_id": run.run_id,
        "node": run.node,
        "model": run.model.slug,
        "quant": run.quant.slug,
        "checkpoint": row["checkpoint"],
        "attempt": marker.get("attempt"),
        "result": result_ref,
        "evaluation_wall_seconds": result.get("evaluation_wall_seconds"),
        "metrics": metrics,
    }


def audit_all(root: Path) -> dict[str, Any]:
    plan = load_and_validate_plan(root)
    rows = [audit_one(root, plan, run) for run in c.RUNS]
    payload: dict[str, Any] = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "complete",
        "quality_id": QUALITY_ID,
        "audited_at": c.now(),
        "quality_plan_fingerprint": plan["fingerprint"],
        "counts": {"runs": len(rows), "wikitext2_ppl": len(rows), "paper_qa_tasks": len(rows) * len(PAPER_QA_TASKS)},
        "rows": rows,
    }
    identity = dict(payload)
    identity.pop("audited_at", None)
    payload["audit_fingerprint"] = c.canonical_sha256(identity)
    c.atomic_json(root / "quality_final_audit.json", payload)
    return payload


def success_count(root: Path) -> int:
    return sum((quality_dir(root, run) / "quality_success.json").is_file() for run in c.RUNS)


def run_node(root: Path, node: int) -> None:
    plan = load_and_validate_plan(root)
    runs = expected_runs_for_node(node)
    require(len(runs) == 10, f"node {node} assignment must contain ten runs")
    pending: list[c.RunSpec] = []
    reserved_gpus: set[int] = set()
    for run in runs:
        directory = quality_dir(root, run)
        marker = directory / "quality_success.json"
        if marker.is_file():
            audit_one(root, plan, run)
        elif (directory / CLAIM_NAME).is_dir():
            owner = c.read_json(directory / CLAIM_NAME / "owner.json")
            if owner.get("hostname") == socket.gethostname():
                reserved_gpus.add(int(owner["gpu"]))
            continue
        else:
            pending.append(run)
    active: dict[subprocess.Popen[Any], tuple[c.RunSpec, int]] = {}
    require(reserved_gpus <= set(c.GPU_IDS), f"invalid claimed GPUs: {reserved_gpus}")
    free_gpus = [gpu for gpu in c.GPU_IDS if gpu not in reserved_gpus]
    failures: list[str] = []
    while pending or active:
        while pending and free_gpus and not failures:
            run = pending.pop(0)
            gpu = free_gpus.pop(0)
            env = c.worker_env(gpu)
            command = [
                str(c.PYTHON), "-m", MODULE, "_worker",
                "--root", str(root), "--run-id", run.run_id,
                "--gpu", str(gpu),
            ]
            process = subprocess.Popen(
                command,
                cwd=c.WORKSPACE,
                env=env,
            )
            active[process] = (run, gpu)
        if not active:
            break
        time.sleep(2)
        for process, (run, gpu) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            del active[process]
            free_gpus.append(gpu)
            free_gpus.sort()
            if returncode != 0:
                failures.append(f"{run.run_id}:rc={returncode}")
        if failures and active:
            # Do not launch more work after a failure, but preserve completed
            # work and let already-running lanes finish cleanly.
            continue
    require(not failures, f"node {node} quality workers failed: {failures}")
    for run in runs:
        marker = quality_dir(root, run) / "quality_success.json"
        if marker.is_file():
            audit_one(root, plan, run)
        else:
            require(
                (quality_dir(root, run) / CLAIM_NAME).is_dir(),
                f"node {node} quality work missing after scheduler: {run.run_id}",
            )


def watch(root: Path, poll_seconds: int) -> None:
    load_and_validate_plan(root)
    while True:
        count = success_count(root)
        print(f"[{c.now()}] quant_quality_success={count}/20", flush=True)
        if count == len(c.RUNS):
            result = audit_all(root)
            print(json.dumps(result["counts"], sort_keys=True), flush=True)
            return
        time.sleep(poll_seconds)


def status(root: Path) -> dict[str, Any]:
    rows = []
    for run in c.RUNS:
        directory = quality_dir(root, run)
        marker = directory / "quality_success.json"
        attempts = sorted(item.name for item in directory.glob("attempt[0-9][0-9][0-9]") if item.is_dir()) if directory.is_dir() else []
        rows.append({"run_id": run.run_id, "node": run.node, "status": "succeeded" if marker.is_file() else "running" if (directory / CLAIM_NAME).is_dir() else "pending", "attempts": attempts})
    return {"quality_id": QUALITY_ID, "success": sum(row["status"] == "succeeded" for row in rows), "rows": rows}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "audit", "status"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", default=str(c.DEFAULT_ROOT))
    child = subparsers.add_parser("run-node")
    child.add_argument("--root", default=str(c.DEFAULT_ROOT))
    child.add_argument("--node", type=int, choices=(0, 1), required=True)
    child = subparsers.add_parser("_worker")
    child.add_argument("--root", default=str(c.DEFAULT_ROOT))
    child.add_argument("--run-id", choices=tuple(c.RUN_BY_ID), required=True)
    child.add_argument("--gpu", type=int, choices=c.GPU_IDS, required=True)
    child = subparsers.add_parser("watch")
    child.add_argument("--root", default=str(c.DEFAULT_ROOT))
    child.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = c.output_root(args.root)
    if args.command == "plan":
        print(json.dumps(write_plan(root), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "run-node":
        c.python_runtime_snapshot()
        run_node(root, args.node)
    elif args.command == "_worker":
        run_worker(root, c.RUN_BY_ID[args.run_id], args.gpu)
    elif args.command == "audit":
        print(json.dumps(audit_all(root), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "watch":
        watch(root, args.poll_seconds)
    elif args.command == "status":
        print(json.dumps(status(root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
