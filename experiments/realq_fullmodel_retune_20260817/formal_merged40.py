#!/usr/bin/env python3
"""Produce one audited checkpoint for every merged authoritative LR row.

This stage stays dormant until ``selection_merged40.py freeze`` has published
the complete 28+8+4 selection.  It deliberately consumes each merged row's
exact source command, so the Qwen3-32B memory wrapper and its sole numerical
override (Hessian accumulation batch size 32) survive unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
from experiments.realq_fullmodel_retune_20260817 import selection_merged40 as merged


FORMAL_ID = "realq-fullmodel-authoritative-formal-merged40-20260818-v1"
OUTPUT_ROOT = merged.OUTPUT_ROOT
FORMAL_PLAN_PATH = OUTPUT_ROOT / "formal_plan.json"
FORMAL_AUDIT_PATH = OUTPUT_ROOT / "formal_final_audit.json"
CHECKPOINT_STABILITY_SECONDS = 30
CLAIM_NAME = ".formal.claim"
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
    re.IGNORECASE,
)
FORMAL_OVERRIDES = {
    "--grad_lr",
    "--skip_eval",
    "--skip_kl_ppl_eval",
    "--lm_eval",
    "--reasoning_eval",
    "--require_static_cache_hit",
    "--require_reference_cache_hit",
    "--output_dir",
    "--exp",
    "--save_qmodel_path",
}


def _selection() -> dict[str, Any]:
    try:
        return merged.load_selection()
    except merged.MergeError as exc:
        raise c.CampaignError(str(exc)) from exc


def _selection_by_key(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        f"{row['branch']}/{row['config']}": row for row in value["rows"]
    }


def _balanced_pairs() -> list[tuple[str, str]]:
    pairs = []
    for quant in c.QUANT_SLUGS:
        for model in c.MODEL_SLUGS:
            config = f"{model}_{quant}"
            if config not in c.CONFIG_IDS:
                raise c.CampaignError(f"balanced config missing: {config}")
            for branch in c.BRANCH_VALUES:
                pairs.append((branch, config))
    if len(pairs) != 40 or len(set(pairs)) != 40:
        raise c.CampaignError("formal matrix must contain exactly 40 rows")
    return pairs


def _flags(command: Sequence[str]) -> dict[str, str]:
    if len(command) < 3 or command[1] != "-m":
        raise c.CampaignError("formal source command must use python -m")
    if (len(command) - 3) % 2:
        raise c.CampaignError("formal command is not strict flag/value form")
    output: dict[str, str] = {}
    for index in range(3, len(command), 2):
        flag, value = str(command[index]), str(command[index + 1])
        if not flag.startswith("--") or flag in output:
            raise c.CampaignError(f"invalid or duplicate formal flag: {flag}")
        output[flag] = value
    return output


def _formal_dir(branch: str, config: str) -> Path:
    return OUTPUT_ROOT / "formal" / branch / config


def _checkpoint_path(branch: str, config: str) -> Path:
    return _formal_dir(branch, config) / "checkpoint" / "quantized.pt"


def _validate_formal_command(
    source: Sequence[str],
    formal: Sequence[str],
    row: Mapping[str, Any],
    output: Path,
) -> None:
    if list(formal[:3]) != list(source[:3]):
        raise c.CampaignError("formal command changed its executable/module")
    source_flags, formal_flags = _flags(source), _flags(formal)
    allowed_added = FORMAL_OVERRIDES | {"--save_qmodel_path"}
    if (set(formal_flags) - set(source_flags)) - allowed_added:
        raise c.CampaignError("formal command added unexpected flags")
    if "--save_qmodel_path" not in formal_flags:
        raise c.CampaignError("formal command did not add checkpoint output")
    if set(source_flags) - set(formal_flags):
        raise c.CampaignError("formal command dropped source flags")
    for flag, value in source_flags.items():
        if flag not in FORMAL_OVERRIDES and formal_flags[flag] != value:
            raise c.CampaignError(f"formal command drifted source flag {flag}")
    expected = {
        "--grad_lr": c._stable_float(float(row["selected_lr"])),
        "--skip_eval": "true",
        "--skip_kl_ppl_eval": "true",
        "--lm_eval": "false",
        "--reasoning_eval": "false",
        "--require_static_cache_hit": "true",
        "--require_reference_cache_hit": "false",
        "--output_dir": str(output),
        "--exp": "formal_authoritative_merged40",
        "--save_qmodel_path": str(
            _checkpoint_path(str(row["branch"]), str(row["config"]))
        ),
    }
    for flag, value in expected.items():
        if formal_flags.get(flag) != value:
            raise c.CampaignError(
                f"formal override mismatch: {flag}={formal_flags.get(flag)!r}"
            )
    if formal_flags.get("--full_block_refresh") != c.BRANCH_VALUES[str(row["branch"])]:
        raise c.CampaignError("formal branch flag mismatch")
    for flag, value in merged.FROZEN_FLAGS.items():
        if formal_flags.get(flag) != value:
            raise c.CampaignError(f"formal frozen flag mismatch: {flag}")
    if formal_flags.get("--a_loss_ratio") not in {"1", "1.0"}:
        raise c.CampaignError("formal command must use a_loss_ratio=1")
    audit = row.get("source_command_audit")
    if not isinstance(audit, dict):
        raise c.CampaignError("merged row is missing source command audit")
    if c._canonical_sha256(list(source)) != audit.get("command_sha256"):
        raise c.CampaignError("merged source command hash mismatch")


def _formal_command(row: Mapping[str, Any], output: Path) -> list[str]:
    source = row.get("source_command")
    if not isinstance(source, list) or not all(isinstance(item, str) for item in source):
        raise c.CampaignError("merged row has no valid source command")
    command = list(source)
    c._set_arg(command, "--grad_lr", c._stable_float(float(row["selected_lr"])))
    c._set_arg(command, "--skip_eval", "true")
    c._set_arg(command, "--skip_kl_ppl_eval", "true")
    c._set_arg(command, "--lm_eval", "false")
    c._set_arg(command, "--reasoning_eval", "false")
    c._set_arg(command, "--require_static_cache_hit", "true")
    c._set_arg(command, "--require_reference_cache_hit", "false")
    c._set_arg(command, "--output_dir", str(output))
    c._set_arg(command, "--exp", "formal_authoritative_merged40")
    c._set_or_append_arg(
        command,
        "--save_qmodel_path",
        str(_checkpoint_path(str(row["branch"]), str(row["config"]))),
    )
    _validate_formal_command(source, command, row, output)
    return command


def _code_refs() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(merged.__file__).resolve(),
        Path(c.__file__).resolve(),
    )
    return [{"path": str(path), "sha256": c._file_sha256(path)} for path in paths]


def _build_plan() -> dict[str, Any]:
    selection = _selection()
    selected = _selection_by_key(selection)
    rows = []
    for branch, config in _balanced_pairs():
        key = f"{branch}/{config}"
        selection_row = selected[key]
        output = _formal_dir(branch, config) / "attempt001" / "realq_output"
        command = _formal_command(selection_row, output)
        rows.append(
            {
                "branch": branch,
                "config": config,
                "selected_lr": float(selection_row["selected_lr"]),
                "selection_source": selection_row["selection_source"],
                "selection_evidence": selection_row,
                "command": command,
                "command_sha256": c._canonical_sha256(command),
                "checkpoint": str(_checkpoint_path(branch, config)),
            }
        )
    body: dict[str, Any] = {
        "formal_id": FORMAL_ID,
        "merge_id": merged.MERGE_ID,
        "selection": {
            "path": str(merged.SELECTION_PATH),
            "sha256": c._file_sha256(merged.SELECTION_PATH),
            "selection_fingerprint": selection["selection_fingerprint"],
        },
        "code": _code_refs(),
        "protocol": {
            "world_size": 1,
            "checkpoint_producers": 40,
            "source_commands_preserved_except_formal_io": True,
            "a_loss_ratio": 1.0,
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
        if c._read_json(FORMAL_PLAN_PATH) != value:
            raise c.CampaignError("existing merged formal plan differs")
    else:
        c._atomic_json(FORMAL_PLAN_PATH, value)
    print(FORMAL_PLAN_PATH)
    return 0


def _load_plan() -> dict[str, Any]:
    value = c._read_json(FORMAL_PLAN_PATH)
    identity = dict(value)
    fingerprint = identity.pop("formal_plan_fingerprint", None)
    if c._canonical_sha256(identity) != fingerprint:
        raise c.CampaignError("merged formal plan fingerprint mismatch")
    if value != _build_plan():
        raise c.CampaignError("merged formal plan no longer matches frozen inputs")
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
            for branch, config in _balanced_pairs()
        ],
    }
    path = OUTPUT_ROOT / "manifests" / f"{args.name}.json"
    if path.exists():
        raise c.CampaignError(f"formal manifest already exists: {path}")
    c._atomic_json(path, manifest)
    print(path)
    return 0


def _validate_manifest(plan: Mapping[str, Any], value: Mapping[str, Any]) -> None:
    if value.get("formal_id") != FORMAL_ID:
        raise c.CampaignError("merged formal manifest id mismatch")
    if value.get("formal_plan_fingerprint") != plan["formal_plan_fingerprint"]:
        raise c.CampaignError("merged formal manifest fingerprint mismatch")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != 40:
        raise c.CampaignError("merged formal manifest must contain exactly 40 runs")
    keys = {(run.get("branch"), run.get("config")) for run in runs}
    if keys != set(_balanced_pairs()):
        raise c.CampaignError("merged formal manifest matrix mismatch")


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
    if (
        marker.get("formal_id") != FORMAL_ID
        or marker.get("branch") != branch
        or marker.get("config") != config
    ):
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
        or result.get("formal_plan_fingerprint") != plan["formal_plan_fingerprint"]
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
        "selection_source": row["selection_source"],
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
    command = _formal_command(row["selection_evidence"], attempt / "realq_output")
    manifest = {
        "formal_id": FORMAL_ID,
        "formal_plan_fingerprint": plan["formal_plan_fingerprint"],
        "branch": branch,
        "config": config,
        "selected_lr": row["selected_lr"],
        "selection_source": row["selection_source"],
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
        raise c.CampaignError("invalid formal worker index/count")
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
    result = attempts[-1] / "result.json"
    failure = attempts[-1] / "failure.json"
    if failure.is_file():
        return True
    return result.is_file() and c._read_json(result).get("status") == "failed"


def _status(_: argparse.Namespace) -> int:
    rows = []
    for branch, config in _balanced_pairs():
        directory = _formal_dir(branch, config)
        state = (
            "succeeded"
            if (directory / "formal_success.json").is_file()
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
                "formal_id": FORMAL_ID,
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
    rows = [
        _audit_one(plan, branch, config) for branch, config in _balanced_pairs()
    ]
    payload: dict[str, Any] = {
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
    manifest.add_argument("--name", default="formal_merged40_v1")
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
        print(f"realq-formal-merged40: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
