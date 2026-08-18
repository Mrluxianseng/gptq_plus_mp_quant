#!/usr/bin/env python3
"""Formal checkpoint stage for frozen final-layer-aware V3 LR selections.

The module is intentionally dormant until ``selection_v3.py freeze`` has
published a fully audited 40-row selection artifact.  It then builds an
immutable formal plan and runs one world-size-one checkpoint producer per
branch/config.  PPL/QA/reasoning are separate checkpoint-consumer stages.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c
from experiments.realq_fullmodel_retune_20260817 import campaign_v3 as v3

v3._bootstrap()

from experiments.realq_fullmodel_retune_20260817 import selection_v3 as select  # noqa: E402


FORMAL_ID = "realq-fullmodel-two-branch-formal-20260817-v3"
FORMAL_PLAN_PATH = v3.OUTPUT_ROOT / "formal_plan.json"
FORMAL_AUDIT_PATH = v3.OUTPUT_ROOT / "formal_final_audit.json"
CHECKPOINT_STABILITY_SECONDS = 30
CLAIM_NAME = ".formal.claim"
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
    re.IGNORECASE,
)


def _selection() -> dict[str, Any]:
    value = c._read_json(select.SELECTION_PATH)
    if value.get("selection_id") != select.SELECTION_ID:
        raise c.CampaignError("wrong V3 selection id")
    if value.get("campaign_id") != v3.CAMPAIGN_ID:
        raise c.CampaignError("selection campaign id mismatch")
    identity = dict(value)
    fingerprint = identity.pop("selection_fingerprint", None)
    identity.pop("created_at", None)
    if c._canonical_sha256(identity) != fingerprint:
        raise c.CampaignError("selection fingerprint mismatch")
    plan_ref = value.get("plan")
    if not isinstance(plan_ref, dict) or plan_ref.get("sha256") != c._file_sha256(
        v3.PLAN_PATH
    ):
        raise c.CampaignError("selection is not bound to the current V3 plan")
    for item in value.get("selection_code", []):
        path = Path(str(item.get("path", "")))
        if not path.is_file() or c._file_sha256(path) != item.get("sha256"):
            raise c.CampaignError(f"selection code changed: {path}")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 40:
        raise c.CampaignError("selection must contain exactly 40 rows")
    keys = {(row.get("branch"), row.get("config")) for row in rows}
    expected = {
        (branch, config)
        for branch in c.BRANCH_VALUES
        for config in c.CONFIG_IDS
    }
    if keys != expected:
        raise c.CampaignError("selection branch/config matrix mismatch")
    return value


def _selection_by_key(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        f"{row['branch']}/{row['config']}": row for row in value["rows"]
    }


def _set_or_append(command: list[str], flag: str, value: str) -> None:
    c._set_or_append_arg(command, flag, value)


def _formal_dir(branch: str, config: str) -> Path:
    return v3.OUTPUT_ROOT / "formal" / branch / config


def _checkpoint_path(branch: str, config: str) -> Path:
    return _formal_dir(branch, config) / "checkpoint" / "quantized.pt"


def _formal_command(
    plan: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    output: Path,
) -> list[str]:
    branch = str(selection_row["branch"])
    config = str(selection_row["config"])
    source = plan["configurations"][f"{branch}/{config}"]
    command = list(source["source_command"])
    c._set_arg(command, "--grad_lr", c._stable_float(float(selection_row["selected_lr"])))
    c._set_arg(command, "--skip_eval", "true")
    c._set_arg(command, "--skip_kl_ppl_eval", "true")
    c._set_arg(command, "--lm_eval", "false")
    c._set_arg(command, "--reasoning_eval", "false")
    c._set_arg(command, "--require_static_cache_hit", "true")
    c._set_arg(command, "--require_reference_cache_hit", "false")
    c._set_arg(command, "--output_dir", str(output))
    c._set_arg(command, "--exp", "formal_final_layer_aware_v3")
    _set_or_append(
        command, "--save_qmodel_path", str(_checkpoint_path(branch, config))
    )
    c._validate_full_profile(command, branch=branch, config=config)
    return command


def _build_plan() -> dict[str, Any]:
    tuning_plan = c._read_json(v3.PLAN_PATH)
    c._verify_plan(tuning_plan)
    selections = _selection()
    selected = _selection_by_key(selections)
    rows = []
    for branch, config in v3.v2._balanced_pairs():
        key = f"{branch}/{config}"
        selection_row = selected[key]
        output = _formal_dir(branch, config) / "attempt001" / "realq_output"
        command = _formal_command(tuning_plan, selection_row, output)
        rows.append(
            {
                "branch": branch,
                "config": config,
                "selected_lr": float(selection_row["selected_lr"]),
                "selection_evidence": selection_row,
                "command": command,
                "command_sha256": c._canonical_sha256(command),
                "checkpoint": str(_checkpoint_path(branch, config)),
            }
        )
    body = {
        "formal_id": FORMAL_ID,
        "campaign_id": v3.CAMPAIGN_ID,
        "protocol_fingerprint": tuning_plan["protocol_fingerprint"],
        "selection": {
            "path": str(select.SELECTION_PATH),
            "sha256": c._file_sha256(select.SELECTION_PATH),
            "selection_fingerprint": selections["selection_fingerprint"],
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": c._file_sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "world_size": 1,
            "checkpoint_producers": 40,
            "formal_profile_matches_tuning": True,
            "skip_kl_ppl_eval": True,
            "lm_eval": False,
            "reasoning_eval": False,
            "checkpoint_stability_seconds": CHECKPOINT_STABILITY_SECONDS,
        },
        "rows": rows,
    }
    body["formal_plan_fingerprint"] = c._canonical_sha256(body)
    return body


def _write_plan(_: argparse.Namespace) -> int:
    value = _build_plan()
    if FORMAL_PLAN_PATH.exists():
        current = c._read_json(FORMAL_PLAN_PATH)
        if current != value:
            raise c.CampaignError("existing formal plan differs")
    else:
        c._atomic_json(FORMAL_PLAN_PATH, value)
    print(FORMAL_PLAN_PATH)
    return 0


def _load_plan() -> dict[str, Any]:
    value = c._read_json(FORMAL_PLAN_PATH)
    fingerprint = value.get("formal_plan_fingerprint")
    identity = dict(value)
    identity.pop("formal_plan_fingerprint", None)
    if c._canonical_sha256(identity) != fingerprint:
        raise c.CampaignError("formal plan fingerprint mismatch")
    if value != _build_plan():
        raise c.CampaignError("formal plan no longer matches frozen inputs")
    return value


def _plan_rows(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {f"{row['branch']}/{row['config']}": row for row in value["rows"]}


def _normalized_attempt_command(command: Sequence[str]) -> list[str]:
    normalized = list(command)
    c._set_arg(normalized, "--output_dir", "<FORMAL_ATTEMPT_OUTPUT>")
    return normalized


def _make_manifest(args: argparse.Namespace) -> int:
    plan = _load_plan()
    manifest = {
        "formal_id": FORMAL_ID,
        "formal_plan_fingerprint": plan["formal_plan_fingerprint"],
        "name": args.name,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": [
            {"branch": branch, "config": config}
            for branch, config in v3.v2._balanced_pairs()
        ],
    }
    path = v3.OUTPUT_ROOT / "manifests" / f"{args.name}.json"
    if path.exists():
        raise c.CampaignError(f"manifest already exists: {path}")
    c._atomic_json(path, manifest)
    print(path)
    return 0


def _validate_manifest(plan: Mapping[str, Any], value: Mapping[str, Any]) -> None:
    if value.get("formal_id") != FORMAL_ID:
        raise c.CampaignError("formal manifest id mismatch")
    if value.get("formal_plan_fingerprint") != plan["formal_plan_fingerprint"]:
        raise c.CampaignError("formal manifest fingerprint mismatch")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != 40:
        raise c.CampaignError("formal manifest must contain exactly 40 runs")
    keys = {(run.get("branch"), run.get("config")) for run in runs}
    if keys != set(v3.v2._balanced_pairs()):
        raise c.CampaignError("formal manifest matrix mismatch")


def _stable_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise c.CampaignError(f"checkpoint missing or empty: {path}")
    before = path.stat()
    time.sleep(CHECKPOINT_STABILITY_SECONDS)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise c.CampaignError(f"checkpoint did not stabilize: {path}")
    return {
        "path": str(path),
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
    }


def _worker_env(gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    for key in c.TORCH_DISTRIBUTED_ENV:
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES=gpu,
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
    )
    return environment


def _gpu_snapshot(cuda_id: str) -> dict[str, str]:
    return c._gpu_inventory(cuda_id)


def _attempt_dir(branch: str, config: str) -> Path:
    directory = _formal_dir(branch, config)
    indices = []
    if directory.is_dir():
        for path in directory.glob("attempt[0-9][0-9][0-9]"):
            if path.is_dir():
                try:
                    indices.append(int(path.name.removeprefix("attempt")))
                except ValueError:
                    pass
    return directory / f"attempt{max(indices, default=0) + 1:03d}"


def _audit_one(plan: Mapping[str, Any], branch: str, config: str) -> dict[str, Any]:
    directory = _formal_dir(branch, config)
    marker_path = directory / "formal_success.json"
    marker = c._read_json(marker_path)
    if marker.get("formal_id") != FORMAL_ID or marker.get("branch") != branch or marker.get("config") != config:
        raise c.CampaignError(f"formal marker identity mismatch: {branch}/{config}")
    result_ref = marker.get("result")
    if not isinstance(result_ref, dict):
        raise c.CampaignError("formal result reference missing")
    result_path = Path(str(result_ref.get("path", "")))
    if not result_path.is_file() or c._file_sha256(result_path) != result_ref.get("sha256"):
        raise c.CampaignError(f"formal result hash mismatch: {branch}/{config}")
    result = c._read_json(result_path)
    if result.get("status") != "succeeded" or result.get("returncode") != 0:
        raise c.CampaignError(f"formal result is not successful: {branch}/{config}")
    if (
        result.get("formal_id") != FORMAL_ID
        or result.get("formal_plan_fingerprint")
        != plan["formal_plan_fingerprint"]
        or result.get("branch") != branch
        or result.get("config") != config
    ):
        raise c.CampaignError(f"formal result identity mismatch: {branch}/{config}")
    result_command = result.get("command")
    if not isinstance(result_command, list) or not all(
        isinstance(item, str) for item in result_command
    ):
        raise c.CampaignError(f"formal result command missing: {branch}/{config}")
    if c._canonical_sha256(result_command) != result.get("command_sha256"):
        raise c.CampaignError(f"formal result command hash mismatch: {branch}/{config}")
    checkpoint = Path(str(result["checkpoint"]["path"]))
    stat = checkpoint.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
        result["checkpoint"]["size_bytes"],
        result["checkpoint"]["mtime_ns"],
    ):
        raise c.CampaignError(f"formal checkpoint changed: {branch}/{config}")
    row = _plan_rows(plan)[f"{branch}/{config}"]
    if result.get("selected_lr") != row["selected_lr"]:
        raise c.CampaignError(f"formal LR mismatch: {branch}/{config}")
    if checkpoint != Path(str(row["checkpoint"])):
        raise c.CampaignError(f"formal checkpoint path mismatch: {branch}/{config}")
    if c._canonical_sha256(row["command"]) != row.get("command_sha256"):
        raise c.CampaignError(f"formal plan command hash mismatch: {branch}/{config}")
    if _normalized_attempt_command(result_command) != _normalized_attempt_command(
        row["command"]
    ):
        raise c.CampaignError(f"formal command drift: {branch}/{config}")
    log_ref = result.get("log")
    if not isinstance(log_ref, dict):
        raise c.CampaignError(f"formal log reference missing: {branch}/{config}")
    log_path = Path(str(log_ref.get("path", "")))
    if not log_path.is_file() or c._file_sha256(log_path) != log_ref.get("sha256"):
        raise c.CampaignError(f"formal log hash mismatch: {branch}/{config}")
    return {
        "branch": branch,
        "config": config,
        "selected_lr": row["selected_lr"],
        "checkpoint": result["checkpoint"],
        "elapsed_seconds": result["elapsed_seconds"],
        "result": result_ref,
    }


def _run_one(plan: Mapping[str, Any], branch: str, config: str, cuda_id: str) -> int:
    directory = _formal_dir(branch, config)
    marker = directory / "formal_success.json"
    if marker.is_file():
        _audit_one(plan, branch, config)
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / CLAIM_NAME
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise c.CampaignError(f"formal producer already claimed: {branch}/{config}") from exc
    attempt = _attempt_dir(branch, config)
    attempt.mkdir()
    result_path = attempt / "result.json"
    log_path = attempt / "execution.log"
    row = _plan_rows(plan)[f"{branch}/{config}"]
    command = _formal_command(
        c._read_json(v3.PLAN_PATH), row["selection_evidence"], attempt / "realq_output"
    )
    manifest = {
        "formal_id": FORMAL_ID,
        "formal_plan_fingerprint": plan["formal_plan_fingerprint"],
        "branch": branch,
        "config": config,
        "selected_lr": row["selected_lr"],
        "command": command,
        "command_sha256": c._canonical_sha256(command),
        "gpu": _gpu_snapshot(cuda_id),
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
        elapsed = time.monotonic() - started
        result: dict[str, Any] = {
            **manifest,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
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
        result.update(
            status="succeeded",
            checkpoint=_stable_checkpoint(_checkpoint_path(branch, config)),
        )
        c._atomic_json(result_path, result)
        success = {
            "formal_id": FORMAL_ID,
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
        raise c.CampaignError("invalid worker index/count")
    selected = [
        run
        for index, run in enumerate(manifest["runs"])
        if index % args.worker_count == args.worker_index
    ]
    failures = 0
    for run in selected:
        status = _run_one(
            plan, str(run["branch"]), str(run["config"]), args.cuda_id
        )
        failures += int(status != 0)
    return 1 if failures else 0


def _status(_: argparse.Namespace) -> int:
    rows = []
    for branch, config in v3.v2._balanced_pairs():
        directory = _formal_dir(branch, config)
        rows.append(
            {
                "branch": branch,
                "config": config,
                "status": "succeeded" if (directory / "formal_success.json").is_file() else "running" if (directory / CLAIM_NAME).is_dir() else "pending",
            }
        )
    print(
        json.dumps(
            {
                "formal_id": FORMAL_ID,
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


def _audit_all(_: argparse.Namespace) -> int:
    plan = _load_plan()
    rows = [
        _audit_one(plan, branch, config)
        for branch, config in v3.v2._balanced_pairs()
    ]
    payload = {
        "formal_id": FORMAL_ID,
        "formal_plan_fingerprint": plan["formal_plan_fingerprint"],
        "status": "complete",
        "counts": {"formal_checkpoints": len(rows)},
        "rows": rows,
    }
    payload["audit_fingerprint"] = c._canonical_sha256(payload)
    payload["audited_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    c._atomic_json(FORMAL_AUDIT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("write-plan")
    plan.set_defaults(handler=_write_plan)
    manifest = subparsers.add_parser("make-manifest")
    manifest.add_argument("--name", default="formal_v3_v1")
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
    except (c.CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"realq-formal-v3: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
