#!/usr/bin/env python3
"""Eager quality wave for the first two audited Q32 SDPA checkpoints.

This runner never changes the frozen 31-row eager runner: that file is already
part of an immutable plan snapshot.  Instead this module configures its proven
row executor in a fresh process, freezes exactly two completed Q32 rows, and
keeps all evidence isolated until the canonical forty-row quality plan exists.
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
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import eager_quality_nonq32 as core
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as formal
from experiments.realq_sdpa_frozen_lr20_20260822 import quality_v5_merged as quality


EAGER_ID = "realq-sdpa-v5-merged-eager-quality-q32-wave1-20260823-v1"
OUTPUT_ROOT = campaign.OUTPUT_ROOT / "eager_quality_q32_wave1_v1"
PLAN_PATH = OUTPUT_ROOT / "plan.json"
AUDIT_PATH = OUTPUT_ROOT / "final_audit.json"
ADOPTION_PATH = OUTPUT_ROOT / "canonical_adoption_receipt.json"
EXPECTED_ROW_COUNT = 2
PAIRS = (
    ("single_linear", "qwen3-32b_w2a16"),
    ("single_linear", "qwen3-32b_w3a16"),
)
SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(core.__file__).resolve(),
    Path(quality.__file__).resolve(),
    Path(quality._IMPLEMENTATION_PATH).resolve(),
    Path(formal.__file__).resolve(),
    Path(campaign.__file__).resolve(),
)
_BASE_BUILD_PLAN = core._build_plan


def _pairs() -> list[tuple[str, str]]:
    pairs = list(PAIRS)
    if len(pairs) != EXPECTED_ROW_COUNT or len(set(pairs)) != EXPECTED_ROW_COUNT:
        raise core.EagerQualityError("Q32 wave1 must contain exactly two rows")
    return pairs


def _build_plan() -> dict[str, Any]:
    body = _BASE_BUILD_PLAN()
    body.pop("plan_fingerprint", None)
    body["protocol"] = {
        **body["protocol"],
        "row_count": EXPECTED_ROW_COUNT,
        "scope": "first two terminal-and-audited Q32 SDPA checkpoints",
        "source_runner_is_immutable_31_row_executor": True,
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
    core._build_plan = _build_plan


def _run_worker(args: argparse.Namespace) -> int:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise core.EagerQualityError("Q32 eager worker must run in a debug pod")
    if not 0 <= args.physical_gpu <= 7:
        raise core.EagerQualityError("physical GPU must be in [0, 7]")
    plan = core._load_plan()
    lock = campaign.LOCK_ROOT / hostname / f"gpu{args.physical_gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state = core._state()
        if state["succeeded"] == EXPECTED_ROW_COUNT:
            return 0
        if state["failed"]:
            return 1
        claimed = core._claim_one(plan, hostname, args.physical_gpu)
        if claimed is None:
            time.sleep(20)
            continue
        branch, config, claim = claimed
        try:
            with lock.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                pids = core._gpu_compute_pids(args.physical_gpu)
                if pids:
                    raise core.EagerQualityError(f"untracked GPU processes: {pids}")
                try:
                    status = core._run_one(plan, branch, config, args.physical_gpu)
                except BaseException as exc:
                    base._atomic_json(
                        core._run_dir(branch, config) / core.TERMINAL_FAILURE,
                        {
                            "eager_id": EAGER_ID,
                            "plan_fingerprint": plan["plan_fingerprint"],
                            "branch": branch,
                            "config": config,
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
                    core._run_dir(branch, config) / core.TERMINAL_FAILURE,
                    {
                        "eager_id": EAGER_ID,
                        "plan_fingerprint": plan["plan_fingerprint"],
                        "branch": branch,
                        "config": config,
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
    for branch, config in _pairs():
        if (core._run_dir(branch, config) / "success.json").is_file():
            rows.append(core._audit_one(plan, branch, config))
    if args.require_complete and len(rows) != EXPECTED_ROW_COUNT:
        raise core.EagerQualityError(
            f"Q32 wave1 audit incomplete: {len(rows)}/{EXPECTED_ROW_COUNT}"
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if len(rows) == EXPECTED_ROW_COUNT else "partial",
        "eager_id": EAGER_ID,
        "plan": {
            "path": str(PLAN_PATH),
            "sha256": base._file_sha256(PLAN_PATH),
            "fingerprint": plan["plan_fingerprint"],
        },
        "counts": {"runs": len(rows)},
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
    official_plan = quality._load_plan()
    official_rows = quality._rows(official_plan)
    candidates = []
    for branch, config in _pairs():
        eager_identity = core._audit_one(eager_plan, branch, config)
        eager_result_path = Path(eager_identity["result"]["path"])
        eager_result = base._read_json(eager_result_path)
        official_row = official_rows[f"{branch}/{config}"]
        if (
            eager_result["checkpoint"] != official_row["checkpoint"]
            or quality._normalized_command(eager_result["command"])
            != quality._normalized_command(official_row["command"])
            or eager_result["metrics"] != eager_identity["metrics"]
        ):
            raise core.EagerQualityError(
                f"canonical Q32 wave1 adoption mismatch: {branch}/{config}"
            )
        directory = quality._quality_dir(branch, config)
        if directory.exists() or directory.is_symlink():
            raise core.EagerQualityError(
                f"canonical quality directory is not fresh: {directory}"
            )
        candidates.append(
            (branch, config, eager_identity, eager_result, official_row, directory)
        )
    if len(candidates) != EXPECTED_ROW_COUNT:
        raise core.EagerQualityError("Q32 wave1 adoption preflight row mismatch")

    adopted = []
    for branch, config, eager_identity, eager_result, official_row, directory in candidates:
        attempt = directory / "attempt001"
        attempt.mkdir(parents=True)
        adoption = {
            "kind": "eager_q32_wave1_quality_adoption",
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
            "metrics_reparsed_equal": True,
            "exact_reference_cache_hit_revalidated": True,
            "adopted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        manifest = {
            "quality_id": quality.QUALITY_ID,
            "quality_plan_fingerprint": official_plan["quality_plan_fingerprint"],
            "branch": branch,
            "config": config,
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
        result = core._adopted_result(
            eager_result=eager_result,
            official_plan=official_plan,
            official_row=official_row,
            eager_identity=adoption,
        )
        result_path = attempt / "result.json"
        base._atomic_json(result_path, result)
        marker = {
            "quality_id": quality.QUALITY_ID,
            "branch": branch,
            "config": config,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
        }
        base._atomic_json(directory / "quality_success.json", marker)
        canonical = quality._audit_one(official_plan, branch, config)
        adopted.append(
            {"branch": branch, "config": config, "canonical": canonical, "adoption": adoption}
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "kind": "eager_q32_wave1_quality_canonical_adoption",
        "eager_id": EAGER_ID,
        "quality_id": quality.QUALITY_ID,
        "quality_plan_fingerprint": official_plan["quality_plan_fingerprint"],
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
