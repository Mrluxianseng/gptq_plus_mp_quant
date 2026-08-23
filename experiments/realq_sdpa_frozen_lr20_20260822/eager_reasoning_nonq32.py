#!/usr/bin/env python3
"""Stage reasoning for the 31 completed non-Q32 SDPA checkpoints.

The canonical 120-work reasoning plan is intentionally frozen only after all
forty formal checkpoints pass their audit.  This scheduler runs the exact
future generation commands for the already-complete 31-row subset, writes the
generated task artifacts directly into their canonical per-task directories,
and keeps all execution evidence in an isolated eager root.  Results become
canonical only through ``adopt`` after the official plan exists and proves
checkpoint, full normalized argv, datasets, generated output, and log hashes.
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
import time
import traceback
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as formal
from experiments.realq_sdpa_frozen_lr20_20260822 import reasoning_v5_merged as reasoning


EAGER_ID = "realq-sdpa-v5-merged-eager-reasoning-nonq32-20260823-v1"
OUTPUT_ROOT = campaign.OUTPUT_ROOT / "eager_reasoning_nonq32_v1"
PLAN_PATH = OUTPUT_ROOT / "plan.json"
AUDIT_PATH = OUTPUT_ROOT / "final_audit.json"
ADOPTION_PATH = OUTPUT_ROOT / "canonical_adoption_receipt.json"
OUTER_CLAIM = ".eager_reasoning_controller_claim"
TERMINAL_FAILURE = "controller_terminal_failure.json"
SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(reasoning.__file__).resolve(),
    Path(reasoning._IMPLEMENTATION_PATH).resolve(),
    Path(formal.__file__).resolve(),
    Path(campaign.__file__).resolve(),
)
EXCLUDED_RUNNING_ROWS = {
    ("full_block", "llama31-8b-instruct_w4a16"),
}


class EagerReasoningError(RuntimeError):
    """A staged reasoning or adoption contract failed."""


def _pairs() -> list[tuple[str, str]]:
    pairs = [
        (branch, config)
        for branch, config in formal._balanced_pairs()
        if campaign._model_for_config(config) != "qwen3-32b"
        and (branch, config) not in EXCLUDED_RUNNING_ROWS
    ]
    if len(pairs) != 31 or len(set(pairs)) != 31:
        raise EagerReasoningError("eager reasoning must cover exactly 31 rows")
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
    if len(works) != 93 or len(set(works)) != 93:
        raise EagerReasoningError("eager reasoning work matrix must be 93 rows")
    return works


def _work_id(branch: str, config: str, task: str) -> str:
    return reasoning._work_id(branch, config, task)


def _run_dir(branch: str, config: str, task: str) -> Path:
    return OUTPUT_ROOT / "runs" / branch / config / task


def _source_snapshot() -> dict[str, Any]:
    files = [
        {"path": str(path), "sha256": base._file_sha256(path)}
        for path in SOURCE_FILES
    ]
    return {"files": files, "sha256": base._canonical_sha256(files)}


def _cache_contract(model: str) -> dict[str, Any]:
    observed = campaign.v1._baseline_cache_contracts()[model]
    token = observed["tokens"]
    reference = observed["reference_logits"]
    return {
        "baseline_plan": {
            "path": str(campaign.v1.BASELINE_PLAN_PATH),
            "sha256": base._file_sha256(campaign.v1.BASELINE_PLAN_PATH),
        },
        "token": {
            "path": token["path"],
            "archive_sha256": token["archive_sha256"],
            "semantic_sha256": token["semantic_sha256"],
            "count": 256,
            "shape": [2048],
            "dtype": "torch.int64",
        },
        "reference_runtime_root": reference["runtime_root"],
    }


def _validate_command(
    command: Sequence[str],
    branch: str,
    config: str,
    task: str,
    contract: Mapping[str, Any],
) -> None:
    flags = formal._flags(command)
    max_tokens, batch_size, _ = reasoning.TASKS[task]
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
        "--tokens_cache_path": str(Path(contract["token"]["path"]).parent),
        "--cache_dir": contract["reference_runtime_root"],
        "--load_qmodel_path": str(formal._checkpoint_path(branch, config)),
        "--skip_eval": "false",
        "--skip_kl_ppl_eval": "true",
        "--lm_eval": "false",
        "--reasoning_eval": "true",
        "--reasoning_tasks": task,
        "--reasoning_data_dir": str(reasoning.DATA_ROOT),
        "--reasoning_output_dir": str(reasoning._output_dir(branch, config, task)),
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
        "--require_static_cache_hit": "false",
        "--require_reference_cache_hit": "false",
    }
    for flag, wanted in expected.items():
        if flags.get(flag) != wanted:
            raise EagerReasoningError(
                f"reasoning command drift {branch}/{config}/{task}: "
                f"{flag}={flags.get(flag)!r} != {wanted!r}"
            )
    if flags.get("--full_block_refresh") != base.BRANCH_VALUES[branch]:
        raise EagerReasoningError("reasoning branch flag mismatch")
    if "--save_qmodel_path" in flags:
        raise EagerReasoningError("reasoning command may not save a checkpoint")


def _build_plan() -> dict[str, Any]:
    formal_plan = formal._load_plan()
    datasets = {task: reasoning._dataset_snapshot(task) for task in reasoning.TASKS}
    contracts = {
        model: _cache_contract(model)
        for model in sorted(
            {campaign._model_for_config(config) for _, config in _pairs()}
        )
    }
    formal_identities: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    for branch, config in _pairs():
        key = f"{branch}/{config}"
        formal_identities[key] = formal._audit_one(formal_plan, branch, config)
        checkpoints[key] = reasoning._checkpoint_stat(branch, config)
    rows = []
    for branch, config, task in _works():
        placeholder = Path("/eager-reasoning-nonq32") / branch / config / task
        command = reasoning._reasoning_command(
            formal_plan, branch, config, task, placeholder
        )
        model = campaign._model_for_config(config)
        _validate_command(command, branch, config, task, contracts[model])
        key = f"{branch}/{config}"
        rows.append(
            {
                "branch": branch,
                "config": config,
                "task": task,
                "work_id": _work_id(branch, config, task),
                "checkpoint": checkpoints[key],
                "formal": formal_identities[key],
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
        "datasets": datasets,
        "protocol": {
            "scheduling_only": True,
            "numerical_contract_changed": False,
            "checkpoint_rows": 31,
            "generation_works": 93,
            "canonical_reasoning_plan_required_before_adoption": True,
            "canonical_command_comparison": "output-dir-normalized full argv",
            "canonical_generation_output_path_used_during_staging": True,
            "tasks": list(reasoning.TASKS),
            "do_sample": False,
            "generation_seed": 1234,
            "attention_backend": "sdpa",
            "exact_gptaq_guidedquant_token_files": True,
        },
        "rows": rows,
    }
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def _write_plan(_: argparse.Namespace) -> int:
    value = _build_plan()
    if PLAN_PATH.is_file():
        if base._read_json(PLAN_PATH) != value:
            raise EagerReasoningError("existing eager reasoning plan drifted")
    else:
        if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
            raise EagerReasoningError(f"eager reasoning root is not fresh: {OUTPUT_ROOT}")
        for branch, config, task in _works():
            output = reasoning._output_dir(branch, config, task)
            if output.exists() or output.is_symlink():
                raise EagerReasoningError(f"canonical reasoning output is not fresh: {output}")
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
        raise EagerReasoningError("eager reasoning plan fingerprint mismatch")
    if value != _build_plan():
        raise EagerReasoningError("eager reasoning inputs or source changed")
    return value


def _rows(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["work_id"]): row for row in plan["rows"]}


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
        raise EagerReasoningError(completed.stderr.strip())
    return [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]


def _claim_one(plan: Mapping[str, Any], hostname: str, gpu: int):
    for branch, config, task in _works():
        directory = _run_dir(branch, config, task)
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
                "work_id": _work_id(branch, config, task),
                "hostname": hostname,
                "physical_gpu": gpu,
                "pid": os.getpid(),
                "claimed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        return branch, config, task, claim
    return None


def _runtime_command(
    plan: Mapping[str, Any],
    branch: str,
    config: str,
    task: str,
    pipeline_output: Path,
) -> list[str]:
    work_id = _work_id(branch, config, task)
    row = _rows(plan)[work_id]
    command = reasoning._reasoning_command(
        formal._load_plan(), branch, config, task, pipeline_output
    )
    if reasoning._normalized_command(command) != reasoning._normalized_command(
        row["command"]
    ):
        raise EagerReasoningError(f"runtime command drift: {work_id}")
    model = campaign._model_for_config(config)
    _validate_command(
        command, branch, config, task, plan["cache_contracts"][model]
    )
    return command


def _run_one(
    plan: Mapping[str, Any],
    branch: str,
    config: str,
    task: str,
    physical_gpu: int,
) -> int:
    work_id = _work_id(branch, config, task)
    directory = _run_dir(branch, config, task)
    marker = directory / "success.json"
    if marker.is_file():
        _audit_one(plan, branch, config, task)
        return 0
    attempt = directory / "attempt001"
    if attempt.exists() or attempt.is_symlink():
        raise EagerReasoningError(f"eager attempt is not fresh: {work_id}")
    canonical = reasoning._output_dir(branch, config, task)
    if canonical.exists() or canonical.is_symlink():
        raise EagerReasoningError(f"canonical reasoning output is not fresh: {canonical}")
    attempt.mkdir()
    result_path = attempt / "result.json"
    log_path = attempt / "execution.log"
    row = _rows(plan)[work_id]
    command = _runtime_command(
        plan, branch, config, task, attempt / "pipeline_output"
    )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "eager_id": EAGER_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "work_id": work_id,
        "branch": branch,
        "config": config,
        "task": task,
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
                env=reasoning._worker_env(str(physical_gpu)),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            returncode = process.wait()
        result: dict[str, Any] = {
            **manifest,
            "returncode": returncode,
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "log": {"path": str(log_path), "sha256": base._file_sha256(log_path)},
        }
        if returncode:
            result.update(status="failed")
            base._atomic_json(result_path, result)
            return 1
        output = reasoning._audit_generated_output(
            plan, branch, config, task
        )
        result.update(status="succeeded", output=output)
        base._atomic_json(result_path, result)
        base._atomic_json(
            marker,
            {
                "schema_version": 1,
                "status": "succeeded",
                "eager_id": EAGER_ID,
                "plan_fingerprint": plan["plan_fingerprint"],
                "work_id": work_id,
                "result": {
                    "path": str(result_path),
                    "sha256": base._file_sha256(result_path),
                },
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        _audit_one(plan, branch, config, task)
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
    plan: Mapping[str, Any], branch: str, config: str, task: str
) -> dict[str, Any]:
    work_id = _work_id(branch, config, task)
    row = _rows(plan)[work_id]
    if reasoning._checkpoint_stat(branch, config) != row["checkpoint"]:
        raise EagerReasoningError(f"checkpoint changed: {branch}/{config}")
    marker = base._read_json(_run_dir(branch, config, task) / "success.json")
    result_ref = marker.get("result")
    if (
        marker.get("status") != "succeeded"
        or marker.get("eager_id") != EAGER_ID
        or marker.get("plan_fingerprint") != plan["plan_fingerprint"]
        or marker.get("work_id") != work_id
        or not isinstance(result_ref, dict)
    ):
        raise EagerReasoningError(f"success marker drift: {work_id}")
    result_path = Path(str(result_ref.get("path", "")))
    if not result_path.is_file() or base._file_sha256(result_path) != result_ref.get(
        "sha256"
    ):
        raise EagerReasoningError(f"result hash drift: {work_id}")
    result = base._read_json(result_path)
    command = result.get("command")
    log_ref = result.get("log")
    if (
        result.get("status") != "succeeded"
        or result.get("returncode") != 0
        or result.get("eager_id") != EAGER_ID
        or result.get("plan_fingerprint") != plan["plan_fingerprint"]
        or result.get("work_id") != work_id
        or result.get("checkpoint") != row["checkpoint"]
        or not isinstance(command, list)
        or base._canonical_sha256(command) != result.get("command_sha256")
        or reasoning._normalized_command(command)
        != reasoning._normalized_command(row["command"])
        or not isinstance(log_ref, dict)
    ):
        raise EagerReasoningError(f"result contract drift: {work_id}")
    log_path = Path(str(log_ref.get("path", "")))
    if not log_path.is_file() or base._file_sha256(log_path) != log_ref.get("sha256"):
        raise EagerReasoningError(f"log hash drift: {work_id}")
    output = reasoning._audit_generated_output(plan, branch, config, task)
    if output != result.get("output"):
        raise EagerReasoningError(f"generated output drift: {work_id}")
    return {
        "work_id": work_id,
        "branch": branch,
        "config": config,
        "task": task,
        "checkpoint": row["checkpoint"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output": output,
        "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
        "log": {"path": str(log_path), "sha256": base._file_sha256(log_path)},
    }


def _state() -> dict[str, int]:
    counts = {"succeeded": 0, "claimed": 0, "failed": 0, "pending": 0}
    for branch, config, task in _works():
        directory = _run_dir(branch, config, task)
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
        raise EagerReasoningError("eager worker must run in a debug pod")
    if not 0 <= args.physical_gpu <= 7:
        raise EagerReasoningError("physical GPU must be in [0, 7]")
    plan = _load_plan()
    lock = campaign.LOCK_ROOT / hostname / f"gpu{args.physical_gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state = _state()
        if state["succeeded"] == 93:
            return 0
        if state["failed"]:
            return 1
        claimed = _claim_one(plan, hostname, args.physical_gpu)
        if claimed is None:
            time.sleep(20)
            continue
        branch, config, task, claim = claimed
        try:
            with lock.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                pids = _gpu_compute_pids(args.physical_gpu)
                if pids:
                    raise EagerReasoningError(f"untracked GPU processes: {pids}")
                try:
                    status = _run_one(
                        plan, branch, config, task, args.physical_gpu
                    )
                except BaseException as exc:
                    base._atomic_json(
                        _run_dir(branch, config, task) / TERMINAL_FAILURE,
                        {
                            "eager_id": EAGER_ID,
                            "plan_fingerprint": plan["plan_fingerprint"],
                            "work_id": _work_id(branch, config, task),
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
                    _run_dir(branch, config, task) / TERMINAL_FAILURE,
                    {
                        "eager_id": EAGER_ID,
                        "plan_fingerprint": plan["plan_fingerprint"],
                        "work_id": _work_id(branch, config, task),
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
    for branch, config, task in _works():
        if (_run_dir(branch, config, task) / "success.json").is_file():
            rows.append(_audit_one(plan, branch, config, task))
    if args.require_complete and len(rows) != 93:
        raise EagerReasoningError(f"eager reasoning audit incomplete: {len(rows)}/93")
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if len(rows) == 93 else "partial",
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
    eager_plan = _load_plan()
    if args.require_complete:
        _audit(argparse.Namespace(require_complete=True))
    official_plan = reasoning._load_plan()
    official_rows = reasoning._plan_rows(official_plan)
    candidates = []
    for branch, config, task in _works():
        work_id = _work_id(branch, config, task)
        eager_identity = _audit_one(eager_plan, branch, config, task)
        eager_result = base._read_json(Path(eager_identity["result"]["path"]))
        official_row = official_rows[work_id]
        if (
            eager_result["checkpoint"] != official_row["checkpoint"]
            or reasoning._normalized_command(eager_result["command"])
            != reasoning._normalized_command(official_row["command"])
            or eager_result["output"] != eager_identity["output"]
        ):
            raise EagerReasoningError(f"canonical adoption mismatch: {work_id}")
        output = reasoning._output_dir(branch, config, task)
        if not (output / "manifest.json").is_file():
            raise EagerReasoningError(f"generated manifest missing: {work_id}")
        if (output / "generation_success.json").exists() or list(
            output.glob("attempt[0-9][0-9][0-9]")
        ):
            raise EagerReasoningError(f"canonical reasoning markers are not fresh: {work_id}")
        candidates.append(
            (branch, config, task, eager_identity, eager_result, official_row, output)
        )
    if len(candidates) != 93:
        raise EagerReasoningError("canonical adoption preflight is not 93 works")

    adopted = []
    for branch, config, task, eager_identity, eager_result, official_row, output in candidates:
        work_id = _work_id(branch, config, task)
        attempt = output / "attempt001"
        attempt.mkdir()
        adoption = {
            "kind": "eager_nonq32_reasoning_adoption",
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
                "result": {
                    "path": str(result_path),
                    "sha256": base._file_sha256(result_path),
                },
            },
        )
        canonical = reasoning._audit_one(official_plan, branch, config, task)
        adopted.append({"work_id": work_id, "canonical": canonical, "adoption": adoption})
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "kind": "eager_nonq32_reasoning_canonical_adoption",
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
        print(f"{type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
