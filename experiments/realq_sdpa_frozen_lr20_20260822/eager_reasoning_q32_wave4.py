#!/usr/bin/env python3
"""Eager reasoning wave for the four remaining Q32 SDPA checkpoints.

The immutable 93-work non-Q32 runner remains byte-stable.  This module
configures its audited row executor in a fresh process for twelve Q32 generation
works and requires later adoption against the canonical 120-work plan.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback
from typing import Any, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import eager_reasoning_nonq32 as core
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as formal
from experiments.realq_sdpa_frozen_lr20_20260822 import reasoning_v5_merged as reasoning


EAGER_ID = "realq-sdpa-v5-merged-eager-reasoning-q32-wave4-20260823-v1"
OUTPUT_ROOT = campaign.OUTPUT_ROOT / "eager_reasoning_q32_wave4_v1"
PLAN_PATH = OUTPUT_ROOT / "plan.json"
AUDIT_PATH = OUTPUT_ROOT / "final_audit.json"
ADOPTION_PATH = OUTPUT_ROOT / "canonical_adoption_receipt.json"
EXPECTED_PAIR_COUNT = 4
EXPECTED_WORK_COUNT = EXPECTED_PAIR_COUNT * len(reasoning.TASKS)
PAIRS = (
    ("full_block", "qwen3-32b_w4a4kv4"),
    ("single_linear", "qwen3-32b_w4a16"),
    ("full_block", "qwen3-32b_w3a16"),
    ("full_block", "qwen3-32b_w2a16"),
)
SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(core.__file__).resolve(),
    Path(reasoning.__file__).resolve(),
    Path(reasoning._IMPLEMENTATION_PATH).resolve(),
    Path(formal.__file__).resolve(),
    Path(campaign.__file__).resolve(),
)
_BASE_BUILD_PLAN = core._build_plan


def _pairs() -> list[tuple[str, str]]:
    pairs = list(PAIRS)
    if len(pairs) != EXPECTED_PAIR_COUNT or len(set(pairs)) != EXPECTED_PAIR_COUNT:
        raise core.EagerReasoningError("Q32 reasoning wave4 pair matrix drifted")
    return pairs


def _works() -> list[tuple[str, str, str]]:
    works = [
        (branch, config, task)
        for branch, config in _pairs()
        for task in reasoning.TASKS
    ]
    works.sort(
        key=lambda item: (
            reasoning._model_cost(item[1]) * reasoning._task_cost(item[2]),
            item[1],
            item[0],
            item[2],
        ),
        reverse=True,
    )
    if len(works) != EXPECTED_WORK_COUNT or len(set(works)) != EXPECTED_WORK_COUNT:
        raise core.EagerReasoningError("Q32 reasoning wave4 work matrix drifted")
    return works


def _build_plan() -> dict[str, Any]:
    body = _BASE_BUILD_PLAN()
    body.pop("plan_fingerprint", None)
    body["protocol"] = {
        **body["protocol"],
        "checkpoint_rows": EXPECTED_PAIR_COUNT,
        "generation_works": EXPECTED_WORK_COUNT,
        "scope": "four terminal Q32 rows outside waves 1--3",
        "source_runner_is_immutable_93_work_executor": True,
        "isolated_wave_requires_canonical_adoption": True,
    }
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def _activate() -> None:
    core.EAGER_ID = EAGER_ID
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.PLAN_PATH = PLAN_PATH
    core.AUDIT_PATH = AUDIT_PATH
    core.ADOPTION_PATH = ADOPTION_PATH
    core.SOURCE_FILES = SOURCE_FILES
    core._pairs = _pairs
    core._works = _works
    core._build_plan = _build_plan


def _run_worker(args: argparse.Namespace) -> int:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise core.EagerReasoningError("Q32 eager worker must run in a debug pod")
    if not 0 <= args.physical_gpu <= 7:
        raise core.EagerReasoningError("physical GPU must be in [0, 7]")
    plan = core._load_plan()
    lock = campaign.LOCK_ROOT / hostname / f"gpu{args.physical_gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state = core._state()
        if state["succeeded"] == EXPECTED_WORK_COUNT:
            return 0
        if state["failed"]:
            return 1
        claimed = core._claim_one(plan, hostname, args.physical_gpu)
        if claimed is None:
            time.sleep(20)
            continue
        branch, config, task, claim = claimed
        try:
            with lock.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                pids = core._gpu_compute_pids(args.physical_gpu)
                if pids:
                    raise core.EagerReasoningError(f"untracked GPU processes: {pids}")
                try:
                    status = core._run_one(plan, branch, config, task, args.physical_gpu)
                except BaseException as exc:
                    base._atomic_json(
                        core._run_dir(branch, config, task) / core.TERMINAL_FAILURE,
                        {
                            "eager_id": EAGER_ID,
                            "plan_fingerprint": plan["plan_fingerprint"],
                            "work_id": core._work_id(branch, config, task),
                            "hostname": hostname,
                            "physical_gpu": args.physical_gpu,
                            "status": "failed",
                            "failure_class": "controller_exception",
                            "error_type": type(exc).__qualname__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "failed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    )
                    raise
            if status:
                base._atomic_json(
                    core._run_dir(branch, config, task) / core.TERMINAL_FAILURE,
                    {
                        "eager_id": EAGER_ID,
                        "plan_fingerprint": plan["plan_fingerprint"],
                        "work_id": core._work_id(branch, config, task),
                        "hostname": hostname,
                        "physical_gpu": args.physical_gpu,
                        "status": "failed",
                        "failed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
        finally:
            shutil.rmtree(claim, ignore_errors=True)


def _status(_: argparse.Namespace) -> int:
    print(json.dumps({"eager_id": EAGER_ID, "counts": core._state()}, indent=2))
    return 0


def _audit(args: argparse.Namespace) -> int:
    plan = core._load_plan()
    rows = []
    for branch, config, task in _works():
        if (core._run_dir(branch, config, task) / "success.json").is_file():
            rows.append(core._audit_one(plan, branch, config, task))
    if args.require_complete and len(rows) != EXPECTED_WORK_COUNT:
        raise core.EagerReasoningError(
            f"Q32 reasoning wave4 audit incomplete: {len(rows)}/{EXPECTED_WORK_COUNT}"
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if len(rows) == EXPECTED_WORK_COUNT else "partial",
        "eager_id": EAGER_ID,
        "plan": {
            "path": str(PLAN_PATH),
            "sha256": base._file_sha256(PLAN_PATH),
            "fingerprint": plan["plan_fingerprint"],
        },
        "counts": {"works": len(rows)},
        "rows": rows,
    }
    value["audit_fingerprint"] = base._canonical_sha256(value)
    base._atomic_json(AUDIT_PATH, value)
    print(json.dumps({"status": value["status"], "counts": value["counts"]}, indent=2))
    return 0


def _adopt(args: argparse.Namespace) -> int:
    eager_plan = core._load_plan()
    if args.require_complete:
        _audit(argparse.Namespace(require_complete=True))
    official_plan = reasoning._load_plan()
    official_rows = reasoning._plan_rows(official_plan)
    candidates = []
    for branch, config, task in _works():
        work_id = core._work_id(branch, config, task)
        eager_identity = core._audit_one(eager_plan, branch, config, task)
        eager_result = base._read_json(Path(eager_identity["result"]["path"]))
        official_row = official_rows[work_id]
        if (
            eager_result["checkpoint"] != official_row["checkpoint"]
            or reasoning._normalized_command(eager_result["command"])
            != reasoning._normalized_command(official_row["command"])
            or eager_result["output"] != eager_identity["output"]
        ):
            raise core.EagerReasoningError(
                f"canonical Q32 wave4 adoption mismatch: {work_id}"
            )
        output = reasoning._output_dir(branch, config, task)
        if not (output / "manifest.json").is_file():
            raise core.EagerReasoningError(f"generated manifest missing: {work_id}")
        if (output / "generation_success.json").exists() or list(
            output.glob("attempt[0-9][0-9][0-9]")
        ):
            raise core.EagerReasoningError(
                f"canonical reasoning markers are not fresh: {work_id}"
            )
        candidates.append(
            (branch, config, task, eager_identity, eager_result, official_row, output)
        )
    if len(candidates) != EXPECTED_WORK_COUNT:
        raise core.EagerReasoningError("Q32 reasoning wave4 adoption row mismatch")

    adopted = []
    for branch, config, task, eager_identity, eager_result, official_row, output in candidates:
        work_id = core._work_id(branch, config, task)
        attempt = output / "attempt001"
        attempt.mkdir()
        adoption = {
            "kind": "eager_q32_wave4_reasoning_adoption",
            "scheduling_only": True,
            "numerical_contract_changed": False,
            "eager_plan": {
                "path": str(PLAN_PATH),
                "sha256": base._file_sha256(PLAN_PATH),
                "fingerprint": eager_plan["plan_fingerprint"],
            },
            "eager_result": eager_identity["result"],
            "eager_log": eager_identity["log"],
            "canonical_full_argv_equal_after_output_dir_normalization": True,
            "checkpoint_identity_equal": True,
            "datasets_and_generated_output_revalidated": True,
            "adopted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        manifest = {
            "reasoning_id": reasoning.REASONING_ID,
            "reasoning_plan_fingerprint": official_plan["reasoning_plan_fingerprint"],
            "work_id": work_id,
            "branch": branch,
            "config": config,
            "task": task,
            "checkpoint": official_row["checkpoint"],
            "command": eager_result["command"],
            "command_sha256": eager_result["command_sha256"],
            "gpu": eager_result["gpu"],
            "hostname": eager_result["hostname"],
            "pid": eager_result["pid"],
            "started_at": eager_result["started_at"],
            "eager_adoption": adoption,
        }
        base._atomic_json(attempt / "manifest.json", manifest)
        result = {
            **manifest,
            "returncode": 0,
            "elapsed_seconds": eager_result["elapsed_seconds"],
            "finished_at": eager_result["finished_at"],
            "log": eager_result["log"],
            "status": "succeeded",
            "output": eager_result["output"],
        }
        result_path = attempt / "result.json"
        base._atomic_json(result_path, result)
        base._atomic_json(
            output / "generation_success.json",
            {
                "reasoning_id": reasoning.REASONING_ID,
                "work_id": work_id,
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "official_code_score_pending": task == "humaneval_plus",
                "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
            },
        )
        canonical = reasoning._audit_one(official_plan, branch, config, task)
        adopted.append(
            {"work_id": work_id, "canonical": canonical, "adoption": adoption}
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "kind": "eager_q32_wave4_reasoning_canonical_adoption",
        "eager_id": EAGER_ID,
        "reasoning_id": reasoning.REASONING_ID,
        "reasoning_plan_fingerprint": official_plan["reasoning_plan_fingerprint"],
        "counts": {"adopted": len(adopted)},
        "rows": adopted,
    }
    value["receipt_fingerprint"] = base._canonical_sha256(value)
    base._atomic_json(ADOPTION_PATH, value)
    print(json.dumps({"status": "complete", "adopted": len(adopted)}, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write-plan")
    write.set_defaults(handler=core._write_plan)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--physical-gpu", required=True, type=int)
    worker.set_defaults(handler=_run_worker)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--require-complete", action="store_true")
    audit.set_defaults(handler=_audit)
    adopt = subparsers.add_parser("adopt")
    adopt.add_argument("--require-complete", action="store_true")
    adopt.set_defaults(handler=_adopt)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _activate()
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
