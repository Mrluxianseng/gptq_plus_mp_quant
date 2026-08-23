#!/usr/bin/env python3
"""Evaluate completed non-Q32 checkpoints before the merged formal audit.

The canonical quality plan intentionally cannot be frozen until all forty
formal checkpoints exist.  Thirty-one non-Q32 checkpoints finished while the
last Llama row and the eight Q32 rows were still running.  This module freezes
exactly those 31 rows, runs the exact future quality command in an isolated
evidence root, and only adopts a result
after the canonical forty-row plan exists and proves that the checkpoint,
normalized command, metrics, log and reference-cache hit are identical.
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
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as formal
from experiments.realq_sdpa_frozen_lr20_20260822 import quality_v5_merged as quality


EAGER_ID = "realq-sdpa-v5-merged-eager-quality-nonq32-20260823-v1"
OUTPUT_ROOT = campaign.OUTPUT_ROOT / "eager_quality_nonq32_v1"
PLAN_PATH = OUTPUT_ROOT / "plan.json"
AUDIT_PATH = OUTPUT_ROOT / "final_audit.json"
ADOPTION_PATH = OUTPUT_ROOT / "canonical_adoption_receipt.json"
OUTER_CLAIM = ".eager_quality_controller_claim"
TERMINAL_FAILURE = "controller_terminal_failure.json"
SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(quality.__file__).resolve(),
    Path(quality._IMPLEMENTATION_PATH).resolve(),
    Path(formal.__file__).resolve(),
    Path(campaign.__file__).resolve(),
)
EXCLUDED_RUNNING_ROWS = {
    ("full_block", "llama31-8b-instruct_w4a16"),
}


class EagerQualityError(RuntimeError):
    """A staged quality or adoption contract failed."""


def _pairs() -> list[tuple[str, str]]:
    pairs = [
        (branch, config)
        for branch, config in formal._balanced_pairs()
        if campaign._model_for_config(config) != "qwen3-32b"
        and (branch, config) not in EXCLUDED_RUNNING_ROWS
    ]
    if len(pairs) != 31 or len(set(pairs)) != 31:
        raise EagerQualityError("eager quality must cover exactly 31 frozen rows")
    return pairs


def _run_dir(branch: str, config: str) -> Path:
    return OUTPUT_ROOT / "runs" / branch / config


def _source_snapshot() -> dict[str, Any]:
    files = [
        {"path": str(path), "sha256": base._file_sha256(path)}
        for path in SOURCE_FILES
    ]
    return {"files": files, "sha256": base._canonical_sha256(files)}


def _reference_contract(model: str) -> dict[str, Any]:
    contracts = campaign.v1._baseline_cache_contracts()
    observed = contracts[model]
    reference = observed["reference_logits"]
    token = observed["tokens"]
    return {
        "baseline_plan": {
            "path": str(campaign.v1.BASELINE_PLAN_PATH),
            "sha256": base._file_sha256(campaign.v1.BASELINE_PLAN_PATH),
        },
        "token": {
            "path": token["path"],
            "archive_sha256": token["archive_sha256"],
            "semantic_sha256": token["semantic_sha256"],
        },
        "reference_logits": {
            "path": reference["path"],
            "runtime_root": reference["runtime_root"],
            "size_bytes": reference["size_bytes"],
            "mtime_ns": reference["mtime_ns"],
            "regeneration_forbidden": True,
        },
    }


def _validate_command_contract(
    command: Sequence[str], *, model: str, contract: Mapping[str, Any]
) -> None:
    flags = formal._flags(command)
    expected = {
        "--dataset": "wikitext2",
        "--eval_datasets": "wikitext2",
        "--seed": "1",
        "--rotation_seed": "0",
        "--refresh_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--eval_seq_len": "2048",
        "--rotate": "true",
        "--attention_backend": "sdpa",
        "--skip_eval": "false",
        "--skip_kl_ppl_eval": "false",
        "--lm_eval": "true",
        "--reasoning_eval": "false",
        "--require_reference_cache_hit": "true",
        "--cache_dir": contract["reference_logits"]["runtime_root"],
        "--tokens_cache_path": str(Path(contract["token"]["path"]).parent),
    }
    for flag, wanted in expected.items():
        if flags.get(flag) != wanted:
            raise EagerQualityError(
                f"eager quality command drift for {model}: {flag}={flags.get(flag)!r}"
            )


def _build_plan() -> dict[str, Any]:
    formal_plan = formal._load_plan()
    rows = []
    contracts: dict[str, Any] = {}
    for branch, config in _pairs():
        directory = formal._formal_dir(branch, config)
        if not (directory / "formal_success.json").is_file():
            raise EagerQualityError(
                f"non-Q32 formal checkpoint is not complete: {branch}/{config}"
            )
        identity = formal._audit_one(formal_plan, branch, config)
        checkpoint = quality._checkpoint_stat(branch, config)
        model = campaign._model_for_config(config)
        if model not in contracts:
            contracts[model] = _reference_contract(model)
        contract = contracts[model]
        placeholder = Path("/eager-quality-nonq32") / branch / config / "realq_output"
        command = quality._quality_command(
            formal_plan, branch, config, placeholder
        )
        _validate_command_contract(command, model=model, contract=contract)
        rows.append(
            {
                "branch": branch,
                "config": config,
                "model": model,
                "checkpoint": checkpoint,
                "formal": identity,
                "command": command,
                "command_sha256": base._canonical_sha256(command),
            }
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "eager_id": EAGER_ID,
        "formal_id": formal.FORMAL_ID,
        "formal_plan": {
            "path": str(formal.FORMAL_PLAN_PATH),
            "sha256": base._file_sha256(formal.FORMAL_PLAN_PATH),
            "fingerprint": formal_plan["formal_plan_fingerprint"],
        },
        "source_snapshot": _source_snapshot(),
        "cache_contracts": contracts,
        "protocol": {
            "scheduling_only": True,
            "numerical_contract_changed": False,
            "row_count": 31,
            "canonical_quality_plan_required_before_adoption": True,
            "canonical_command_comparison": "output-dir-normalized full argv",
            "exact_gptaq_guidedquant_token_files": True,
            "exact_gptaq_guidedquant_sdpa_reference_files": True,
            "reference_cache_regeneration_forbidden": True,
        },
        "rows": rows,
    }
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def _write_plan(_: argparse.Namespace) -> int:
    value = _build_plan()
    if PLAN_PATH.is_file():
        if base._read_json(PLAN_PATH) != value:
            raise EagerQualityError("existing eager quality plan drifted")
    else:
        if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
            raise EagerQualityError(f"eager quality root is not fresh: {OUTPUT_ROOT}")
        OUTPUT_ROOT.mkdir(parents=True)
        base._atomic_json(PLAN_PATH, value)
    print(PLAN_PATH)
    print(value["plan_fingerprint"])
    return 0


def _load_plan() -> dict[str, Any]:
    value = base._read_json(PLAN_PATH)
    stable = dict(value)
    fingerprint = stable.pop("plan_fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise EagerQualityError("eager quality plan fingerprint mismatch")
    if value != _build_plan():
        raise EagerQualityError("eager quality inputs or source changed")
    return value


def _rows(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {f"{row['branch']}/{row['config']}": row for row in plan["rows"]}


def _reference_log_gate(text: str, reference_path: str) -> None:
    load_line = f"Loading reference logits for wikitext2 from {reference_path}"
    if load_line not in text:
        raise EagerQualityError("exact SDPA BF16 reference-cache load was not observed")
    if "Generating reference logits for wikitext2" in text:
        raise EagerQualityError("reference-cache regeneration was observed")


def _gpu_compute_pids(gpu: int) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise EagerQualityError(completed.stderr.strip())
    return [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]


def _claim_one(plan: Mapping[str, Any], hostname: str, gpu: int):
    for branch, config in _pairs():
        directory = _run_dir(branch, config)
        if (directory / "success.json").is_file() or (
            directory / TERMINAL_FAILURE
        ).is_file():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        claim = directory / OUTER_CLAIM
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        base._atomic_json(
            claim / "owner.json",
            {
                "eager_id": EAGER_ID,
                "plan_fingerprint": plan["plan_fingerprint"],
                "branch": branch,
                "config": config,
                "hostname": hostname,
                "physical_gpu": gpu,
                "pid": os.getpid(),
                "claimed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        return branch, config, claim
    return None


def _runtime_command(
    plan: Mapping[str, Any], branch: str, config: str, output: Path
) -> list[str]:
    row = _rows(plan)[f"{branch}/{config}"]
    command = quality._quality_command(
        formal._load_plan(), branch, config, output
    )
    if quality._normalized_command(command) != quality._normalized_command(
        row["command"]
    ):
        raise EagerQualityError(f"runtime quality command drift: {branch}/{config}")
    contract = plan["cache_contracts"][row["model"]]
    _validate_command_contract(command, model=row["model"], contract=contract)
    return command


def _run_one(
    plan: Mapping[str, Any], branch: str, config: str, physical_gpu: int
) -> int:
    directory = _run_dir(branch, config)
    marker = directory / "success.json"
    if marker.is_file():
        _audit_one(plan, branch, config)
        return 0
    attempt = directory / "attempt001"
    if attempt.exists() or attempt.is_symlink():
        raise EagerQualityError(f"eager attempt is not fresh: {branch}/{config}")
    attempt.mkdir()
    result_path = attempt / "result.json"
    log_path = attempt / "execution.log"
    row = _rows(plan)[f"{branch}/{config}"]
    command = _runtime_command(plan, branch, config, attempt / "realq_output")
    manifest = {
        "schema_version": 1,
        "status": "running",
        "eager_id": EAGER_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "branch": branch,
        "config": config,
        "model": row["model"],
        "checkpoint": row["checkpoint"],
        "command": command,
        "command_sha256": base._canonical_sha256(command),
        "gpu": base._gpu_inventory(str(physical_gpu)),
        "hostname": socket.gethostname(),
        "physical_gpu": physical_gpu,
        "pid": os.getpid(),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    base._atomic_json(attempt / "manifest.json", manifest)
    started = time.monotonic()
    try:
        with log_path.open("wb") as handle:
            handle.write(
                (
                    f"[{manifest['started_at']}] "
                    f"command={json.dumps(command, ensure_ascii=False)}\n"
                ).encode()
            )
            process = subprocess.Popen(
                command,
                cwd=base.REPO_ROOT,
                env=quality._worker_env(str(physical_gpu)),
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
            "log": {"path": str(log_path), "sha256": base._file_sha256(log_path)},
        }
        if returncode:
            result.update(status="failed")
            base._atomic_json(result_path, result)
            return 1
        text = log_path.read_text(encoding="utf-8", errors="strict")
        contract = plan["cache_contracts"][row["model"]]
        _reference_log_gate(text, contract["reference_logits"]["path"])
        result.update(status="succeeded", metrics=quality._parse_metrics(log_path))
        base._atomic_json(result_path, result)
        success = {
            "schema_version": 1,
            "status": "succeeded",
            "eager_id": EAGER_ID,
            "plan_fingerprint": plan["plan_fingerprint"],
            "branch": branch,
            "config": config,
            "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        base._atomic_json(marker, success)
        _audit_one(plan, branch, config)
        return 0
    except BaseException as exc:
        base._atomic_json(
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


def _audit_one(
    plan: Mapping[str, Any], branch: str, config: str
) -> dict[str, Any]:
    row = _rows(plan)[f"{branch}/{config}"]
    if quality._checkpoint_stat(branch, config) != row["checkpoint"]:
        raise EagerQualityError(f"eager checkpoint changed: {branch}/{config}")
    marker = base._read_json(_run_dir(branch, config) / "success.json")
    result_ref = marker.get("result")
    if (
        marker.get("status") != "succeeded"
        or marker.get("eager_id") != EAGER_ID
        or marker.get("plan_fingerprint") != plan["plan_fingerprint"]
        or marker.get("branch") != branch
        or marker.get("config") != config
        or not isinstance(result_ref, dict)
    ):
        raise EagerQualityError(f"eager success marker drift: {branch}/{config}")
    result_path = Path(str(result_ref.get("path", "")))
    if not result_path.is_file() or base._file_sha256(result_path) != result_ref.get(
        "sha256"
    ):
        raise EagerQualityError(f"eager result hash drift: {branch}/{config}")
    result = base._read_json(result_path)
    command = result.get("command")
    log_ref = result.get("log")
    if (
        result.get("status") != "succeeded"
        or result.get("returncode") != 0
        or result.get("eager_id") != EAGER_ID
        or result.get("plan_fingerprint") != plan["plan_fingerprint"]
        or result.get("checkpoint") != row["checkpoint"]
        or not isinstance(command, list)
        or base._canonical_sha256(command) != result.get("command_sha256")
        or quality._normalized_command(command)
        != quality._normalized_command(row["command"])
        or not isinstance(log_ref, dict)
    ):
        raise EagerQualityError(f"eager result contract drift: {branch}/{config}")
    log_path = Path(str(log_ref.get("path", "")))
    if not log_path.is_file() or base._file_sha256(log_path) != log_ref.get("sha256"):
        raise EagerQualityError(f"eager log hash drift: {branch}/{config}")
    text = log_path.read_text(encoding="utf-8", errors="strict")
    contract = plan["cache_contracts"][row["model"]]
    _reference_log_gate(text, contract["reference_logits"]["path"])
    metrics = quality._parse_metrics(log_path)
    if metrics != result.get("metrics"):
        raise EagerQualityError(f"eager metrics drift: {branch}/{config}")
    return {
        "branch": branch,
        "config": config,
        "model": row["model"],
        "checkpoint": row["checkpoint"],
        "metrics": metrics,
        "elapsed_seconds": result["elapsed_seconds"],
        "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
        "log": {"path": str(log_path), "sha256": base._file_sha256(log_path)},
    }


def _state() -> dict[str, int]:
    counts = {"succeeded": 0, "claimed": 0, "failed": 0, "pending": 0}
    for branch, config in _pairs():
        directory = _run_dir(branch, config)
        if (directory / "success.json").is_file():
            counts["succeeded"] += 1
        elif (directory / OUTER_CLAIM).is_dir():
            counts["claimed"] += 1
        elif (directory / TERMINAL_FAILURE).is_file():
            counts["failed"] += 1
        else:
            counts["pending"] += 1
    return counts


def _run_worker(args: argparse.Namespace) -> int:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise EagerQualityError("eager worker must run in a debug pod")
    if not 0 <= args.physical_gpu <= 7:
        raise EagerQualityError("physical GPU must be in [0, 7]")
    plan = _load_plan()
    lock = campaign.LOCK_ROOT / hostname / f"gpu{args.physical_gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state = _state()
        if state["succeeded"] == 31:
            return 0
        if state["failed"]:
            return 1
        claimed = _claim_one(plan, hostname, args.physical_gpu)
        if claimed is None:
            time.sleep(20)
            continue
        branch, config, claim = claimed
        try:
            with lock.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                pids = _gpu_compute_pids(args.physical_gpu)
                if pids:
                    raise EagerQualityError(f"untracked GPU processes: {pids}")
                try:
                    status = _run_one(plan, branch, config, args.physical_gpu)
                except BaseException as exc:
                    base._atomic_json(
                        _run_dir(branch, config) / TERMINAL_FAILURE,
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
                    _run_dir(branch, config) / TERMINAL_FAILURE,
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
    print(json.dumps({"eager_id": EAGER_ID, "counts": _state()}, indent=2))
    return 0


def _audit(args: argparse.Namespace) -> int:
    plan = _load_plan()
    rows = []
    for branch, config in _pairs():
        if (_run_dir(branch, config) / "success.json").is_file():
            rows.append(_audit_one(plan, branch, config))
    if args.require_complete and len(rows) != 31:
        raise EagerQualityError(f"eager quality audit incomplete: {len(rows)}/31")
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if len(rows) == 31 else "partial",
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


def _adopted_result(
    *,
    eager_result: Mapping[str, Any],
    official_plan: Mapping[str, Any],
    official_row: Mapping[str, Any],
    eager_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "quality_id": quality.QUALITY_ID,
        "quality_plan_fingerprint": official_plan["quality_plan_fingerprint"],
        "branch": eager_result["branch"],
        "config": eager_result["config"],
        "checkpoint": official_row["checkpoint"],
        "command": eager_result["command"],
        "command_sha256": eager_result["command_sha256"],
        "gpu": eager_result["gpu"],
        "hostname": eager_result["hostname"],
        "pid": eager_result["pid"],
        "started_at": eager_result["started_at"],
        "returncode": 0,
        "elapsed_seconds": eager_result["elapsed_seconds"],
        "finished_at": eager_result["finished_at"],
        "log": eager_result["log"],
        "status": "succeeded",
        "metrics": eager_result["metrics"],
        "eager_adoption": dict(eager_identity),
    }


def _adopt(args: argparse.Namespace) -> int:
    eager_plan = _load_plan()
    if args.require_complete:
        _audit(argparse.Namespace(require_complete=True))
    official_plan = quality._load_plan()
    official_rows = quality._rows(official_plan)
    candidates = []
    for branch, config in _pairs():
        eager_identity = _audit_one(eager_plan, branch, config)
        eager_result_path = Path(eager_identity["result"]["path"])
        eager_result = base._read_json(eager_result_path)
        official_row = official_rows[f"{branch}/{config}"]
        if (
            eager_result["checkpoint"] != official_row["checkpoint"]
            or quality._normalized_command(eager_result["command"])
            != quality._normalized_command(official_row["command"])
            or eager_result["metrics"] != eager_identity["metrics"]
        ):
            raise EagerQualityError(f"canonical adoption mismatch: {branch}/{config}")
        directory = quality._quality_dir(branch, config)
        if directory.exists() or directory.is_symlink():
            raise EagerQualityError(f"canonical quality directory is not fresh: {directory}")
        candidates.append(
            (branch, config, eager_identity, eager_result, official_row, directory)
        )
    if len(candidates) != 31:
        raise EagerQualityError("canonical adoption preflight is not 31 rows")

    adopted = []
    for (
        branch,
        config,
        eager_identity,
        eager_result,
        official_row,
        directory,
    ) in candidates:
        attempt = directory / "attempt001"
        attempt.mkdir(parents=True)
        adoption = {
            "kind": "eager_nonq32_quality_adoption",
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
        result = _adopted_result(
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
            {
                "branch": branch,
                "config": config,
                "canonical": canonical,
                "adoption": adoption,
            }
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "kind": "eager_nonq32_quality_canonical_adoption",
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
    write.set_defaults(handler=_write_plan)
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
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
