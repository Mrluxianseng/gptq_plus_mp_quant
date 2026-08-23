#!/usr/bin/env python3
"""One-time, fail-closed handoff from manual LR review to formal timing.

The command is a dry-run unless ``--execute`` is supplied.  It accepts one
canonical 15-row selection file, proves every cited tuning result against the
old plan and campaign state, pins the formal timing controller, and commits the
formal plan through a recoverable journal.  It never launches an experiment.

The only reviewed non-exact decision is Qwen3-4B/2W16A at 1.2e-4, bracketed by
the successful 1.18e-4 and 1.25e-4 runs.  That decision is recorded as
``user_override_interpolated`` and is never represented as an exact winner.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import dataclasses
import datetime as dt
import decimal
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "lowbit_activation"
for _directory in (str(TOOLS_DIR), str(EXPERIMENT_DIR)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import lowbit_activation_campaign as base_campaign  # noqa: E402
import lowbit_activation_results as results  # noqa: E402
import lowbit_activation_runner as runner  # noqa: E402
import formal_timed_campaign as timed_campaign  # noqa: E402


DEFAULT_PLAN = runner.DEFAULT_PLAN
DEFAULT_STATE = base_campaign.DEFAULT_STATE
DEFAULT_AGENT = base_campaign.DEFAULT_AGENT
DEFAULT_READY_MARKER = base_campaign.DEFAULT_READY_MARKER

SCHEMA_VERSION = 1
SCOPE_ID = "lowbit_activation_manual_lr_selection_v1"
ARTIFACT_KIND = "manual_lr_selection_freeze_attestation_v1"
ARTIFACT_FILENAME = "manual_lr_selection_freeze_v1.json"
RECONCILIATION_POLICY_ID = (
    "retirement_filtered_results_authoritative_v1"
)
MANUAL_INVALID_LEDGER_FILENAME = "manual_invalid_results.jsonl"
MANUAL_INVALID_LEDGER_RELATIVE_PATH = (
    f"_campaign/{MANUAL_INVALID_LEDGER_FILENAME}"
)
GATE_ID = "results_parser_manual_lr_selection_v1"
SOURCE_ADOPTION_KIND = "results_parser_gate_source_adoption_v1"
JOURNAL_SUFFIX = ".manual-lr-freeze.json"
MAX_ATTEMPTS = 20
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DECISION_KINDS = {"exact", "user_override_interpolated"}
QWEN_OVERRIDE_KEY = ("qwen3-4b", "2W16A")
QWEN_OVERRIDE_LR = 1.2e-4
QWEN_OVERRIDE_EVIDENCE_LRS = (1.18e-4, 1.25e-4)
MANUAL_INVALID_KEY = QWEN_OVERRIDE_KEY
MANUAL_INVALID_LR = 1.5e-4
QWEN32_MANUAL_INVALID_KEY = ("qwen3-32b", "2W4A")
QWEN32_MANUAL_INVALID_LR = 5e-6
QWEN32_MANUAL_INVALID_SOURCE_START_SHA256 = (
    "53c916a9ababb36c4e9fc16a78d04a51d360ebe46794495f3b18c992249cb64a"
)
QWEN32_MANUAL_INVALID_SOURCE_END_SHA256 = (
    "2c4f87b78eb862526cd00b534cbac65df89f116ed730b635485998be064018a2"
)
QWEN4_MANUAL_INVALID_SOURCE_END_SHA256 = (
    "817c218d7d969a754323d5d77dc1cb6f40d833b6be2ba818831223ad67d5a63e"
)
MANUAL_INVALID_REVIEWED_TARGETS = {
    (MANUAL_INVALID_KEY[0], MANUAL_INVALID_KEY[1], MANUAL_INVALID_LR, 1): {
        "numerical_source_sha256_at_start": (
            QWEN32_MANUAL_INVALID_SOURCE_START_SHA256
        ),
        "numerical_source_sha256_at_end": (
            QWEN4_MANUAL_INVALID_SOURCE_END_SHA256
        ),
    },
    (
        QWEN32_MANUAL_INVALID_KEY[0],
        QWEN32_MANUAL_INVALID_KEY[1],
        QWEN32_MANUAL_INVALID_LR,
        1,
    ): {
        "numerical_source_sha256_at_start": (
            QWEN32_MANUAL_INVALID_SOURCE_START_SHA256
        ),
        "numerical_source_sha256_at_end": (
            QWEN32_MANUAL_INVALID_SOURCE_END_SHA256
        ),
    },
}
MANUAL_INVALID_REASON_CODE = (
    "numerical_source_changed_during_execution"
)
MANUAL_INVALID_REJECTION_ERRORS = (
    "manifest.numerical_source_tree.changed_during_execution must be false",
)
SOURCE_IDENTITY_KEYS = (
    "numerical_source_sha256",
    "runner_sha256",
    "executor_sha256",
    "campaign_sha256",
    "results_sha256",
)
INPUT_ROOT_KEYS = {"schema_version", "selections"}
INPUT_SELECTION_KEYS = {
    "model",
    "setting",
    "selected_lr",
    "decision_kind",
    "evidence",
    "rationale",
}
INPUT_EVIDENCE_KEYS = {"manifest", "grad_lr", "kl_wikitext2"}
ARTIFACT_REF_KEYS = {"filename", "sha256", "size_bytes"}
WRAPPER_KEYS = {
    "schema_version",
    "scope_id",
    "ledger_sha256",
    "artifact",
    "ledger",
}
LEDGER_KEYS = {
    "schema_version",
    "scope_id",
    "campaign_id",
    "selection_sha256",
    "expected_state_sha256",
    "expected_plan_sha256",
    "source_identity_before",
    "results_source_adoption",
    "timing_gate_sha256",
    "timing_source_set_sha256",
    "downstream_required_gate",
    "selections",
}
NORMALIZED_SELECTION_KEYS = {
    "model",
    "setting",
    "selected_lr",
    "decision_kind",
    "rationale",
    "selected_manifest",
    "evidence_set_sha256",
    "evidence",
}
NORMALIZED_EVIDENCE_KEYS = {
    "manifest",
    "manifest_sha256",
    "log",
    "log_sha256",
    "execution_id",
    "run_id",
    "attempt_index",
    "grad_lr",
    "kl_wikitext2",
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256",
}
ARTIFACT_KEYS = {
    "schema_version",
    "scope_id",
    "artifact_kind",
    "transaction_id",
    "created_utc",
    "ledger_sha256",
    "ledger",
    "pre_freeze_state",
    "timing_gate",
    "state_reconciliation",
}
PRE_FREEZE_STATE_KEYS = {"sha256", "size_bytes", "content_base64"}
HANDOFF_KEYS = {
    "schema_version",
    "scope_id",
    "transaction_id",
    "phase",
    "artifact_sha256",
    "ledger_sha256",
    "selection_sha256",
    "expected_state_sha256",
    "expected_plan_sha256",
    "formal_plan_sha256",
}
RECONCILIATION_KEYS = {
    "schema_version",
    "policy_id",
    "accepted_tune_results",
    "accepted_active_set_sha256",
    "group_attempt_counts",
    "group_attempt_counts_sha256",
    "tune_launches",
    "tune_launches_sha256",
    "invalid_result_resolutions",
    "invalid_result_resolutions_sha256",
    "raw_input_prefixes",
    "raw_input_prefixes_sha256",
    "report_snapshot_sha256",
    "terminalized_tasks",
    "terminalized_tasks_sha256",
    "imported_tasks",
    "imported_tasks_sha256",
}
RECONCILIATION_POINT_KEYS = {
    "model",
    "setting",
    "manifest",
    "manifest_sha256",
    "log",
    "log_sha256",
    "execution_id",
    "run_id",
    "attempt_index",
    "world_size",
    "grad_lr",
    "kl_wikitext2",
    "ppl_wikitext2",
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256",
    "config_sha256",
}
RECONCILIATION_COUNT_KEYS = {"model", "setting", "attempt_count"}
RECONCILIATION_TERMINALIZED_KEYS = {
    "task_id",
    "previous_status",
    "reconciled_status",
    "reason",
}
RAW_PREFIX_REF_KEYS = {
    "relative_path",
    "present",
    "prefix_sha256",
    "prefix_size_bytes",
}
RAW_INPUT_PREFIX_KEYS = {
    "retirement_ledger",
    "campaign_events",
    "manual_invalid_results",
}
FILE_REF_KEYS = {
    "absolute_path",
    "relative_path",
    "present",
    "sha256",
    "size_bytes",
}
TUNE_LAUNCH_KEYS = {
    "manifest",
    "log",
    "classification",
    "rejection_errors",
    "execution_id",
    "run_id",
    "status",
    "exit_code",
    "model",
    "setting",
    "grad_lr",
    "attempt_index",
    "context_valid",
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256_at_start",
    "numerical_source_sha256_at_end",
    "numerical_source_changed_during_execution",
}
MANUAL_INVALID_ENTRY_KEYS = {
    "schema_version",
    "seq",
    "timestamp_utc",
    "campaign_id",
    "reason_code",
    "authorization",
    "launch_event_present",
    "launch_event",
    "manifest",
    "log",
    "execution_id",
    "run_id",
    "phase",
    "method",
    "model",
    "setting",
    "grad_lr",
    "attempt_index",
    "manifest_status",
    "exit_code",
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256_at_start",
    "numerical_source_sha256_at_end",
    "numerical_source_changed_during_execution",
    "expected_rejection_errors",
    "prev_record_sha256",
    "record_sha256",
}
MANUAL_INVALID_AUTHORIZATION_KEYS = {
    "kind",
    "authorized_by",
    "rationale",
}
MANUAL_INVALID_LAUNCH_EVENT_KEYS = {
    "seq",
    "line_sha256",
    "task_id",
}
IMPORTED_TASK_KEYS = {
    "task_id",
    "spec",
    "status",
    "gpu_count",
    "priority",
    "gpus",
    "attempt_index",
    "executor_pid",
    "created_utc",
    "launched_utc",
    "manifest",
    "log",
    "output_dir",
    "exit_code",
    "failure_class",
    "cache_validation",
    "imported",
    "plan_compatible",
    "metric",
    "imported_by_manual_lr_freeze",
    "reconciliation_reason",
    "finished_utc",
}
IMPORTED_TASK_SPEC_KEYS = {
    "kind",
    "method",
    "phase",
    "target_phase",
    "model",
    "setting",
    "grad_lr",
    "overrides",
    "profile_level",
    "static_profile_level",
    "generation",
    "purpose",
}
IMPORTED_TASK_METRIC_KEYS = {"kl", "ppl", "grad_lr", "identities"}
IMPORTED_TASK_IDENTITY_KEYS = {
    "plan_sha256",
    "model_sha256",
    "runner_sha256",
    "executor_sha256",
    "numerical_source_sha256",
}
STATE_RECONCILIATION_KEYS = {
    "schema_version",
    "policy_id",
    "phase",
    "artifact_sha256",
    "accepted_active_set_sha256",
    "group_attempt_counts_sha256",
    "tune_launches_sha256",
    "invalid_result_resolutions_sha256",
    "raw_input_prefixes_sha256",
    "report_snapshot_sha256",
    "terminalized_tasks_sha256",
    "imported_tasks_sha256",
}
JOURNAL_KEYS = {
    "schema_version",
    "scope_id",
    "transaction_id",
    "phase",
    "created_utc",
    "plan_path",
    "state_path",
    "selection_path",
    "expected_campaign_id",
    "expected_plan_sha256",
    "expected_state_sha256",
    "selection_sha256",
    "ledger_sha256",
    "artifact_path",
    "artifact_sha256",
    "artifact_size_bytes",
    "new_plan_sha256",
    "frozen_state_sha256",
    "committed_state_sha256",
    "payloads",
}
JOURNAL_PAYLOAD_KEYS = {
    "old_plan_base64",
    "old_state_base64",
    "artifact_base64",
    "new_plan_base64",
    "frozen_state_base64",
    "committed_state_base64",
}
JOURNAL_PHASES = {
    "prepared",
    "artifact_written",
    "state_frozen",
    "plan_replaced",
    "committed",
}

# Aliases are intentional test seams.  Production still calls the reviewed
# campaign implementations.
agent_ready = base_campaign.agent_ready
venv_ready = base_campaign.venv_ready
query_gpus = base_campaign.query_gpus
proc_identity = base_campaign._proc_identity
proc_cmdline = base_campaign._proc_cmdline
cmdline_matches_argv = base_campaign._cmdline_matches_argv


class FreezeError(RuntimeError):
    """The one-time handoff cannot be proved safe."""


class InjectedFault(RuntimeError):
    """Test-only durable-boundary fault."""


def _fault(_point: str) -> None:
    """Fault-injection seam called after every durable transaction boundary."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FreezeError(f"value is not strict canonical JSON: {exc}") from exc


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FreezeError(f"value is not strict JSON: {exc}") from exc


def _plan_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FreezeError(f"plan is not strict JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FreezeError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise FreezeError(
        f"{name} fields mismatch: missing={missing!r}, "
        f"unexpected={unexpected!r}"
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FreezeError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_json_bytes(raw: bytes, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise FreezeError(f"{name} contains forbidden JSON constant {value!r}")

    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except FreezeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"{name} is not valid UTF-8 JSON: {exc}") from exc


def _read_stable_bytes(path: Path, name: str) -> bytes:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise FreezeError(f"cannot read {name} {path}: {exc}") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(raw) != after.st_size:
        raise FreezeError(f"{name} changed while being read: {path}")
    return raw


def _stable_file_identity(path: Path, name: str) -> tuple[bytes, str]:
    raw = _read_stable_bytes(path, name)
    return raw, _sha256_bytes(raw)


def _path_inside_root(path: Path, root: Path, name: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FreezeError(f"{name} escapes the result root: {resolved}") from exc
    return resolved


def _file_ref(path: Path, *, root: Path, name: str) -> dict[str, Any]:
    resolved = _path_inside_root(path, root, name)
    relative = resolved.relative_to(root).as_posix()
    if not resolved.exists():
        result = {
            "absolute_path": str(resolved),
            "relative_path": relative,
            "present": False,
            "sha256": None,
            "size_bytes": None,
        }
    else:
        if not resolved.is_file():
            raise FreezeError(f"{name} must be a regular file: {resolved}")
        raw, digest = _stable_file_identity(resolved, name)
        result = {
            "absolute_path": str(resolved),
            "relative_path": relative,
            "present": True,
            "sha256": digest,
            "size_bytes": len(raw),
        }
    if set(result) != FILE_REF_KEYS:
        raise AssertionError("internal file reference schema drift")
    return result


def _raw_prefix_ref(
    root: Path, relative_path: str, *, name: str
) -> dict[str, Any]:
    relative = Path(relative_path)
    if relative.is_absolute() or relative.as_posix() != relative_path:
        raise AssertionError(f"non-canonical fixed raw input path {relative_path!r}")
    path = _path_inside_root(root / relative, root, name)
    if not path.exists():
        result = {
            "relative_path": relative_path,
            "present": False,
            "prefix_sha256": None,
            "prefix_size_bytes": None,
        }
    else:
        if not path.is_file():
            raise FreezeError(f"{name} must be a regular file: {path}")
        raw, digest = _stable_file_identity(path, name)
        result = {
            "relative_path": relative_path,
            "present": True,
            "prefix_sha256": digest,
            "prefix_size_bytes": len(raw),
        }
    if set(result) != RAW_PREFIX_REF_KEYS:
        raise AssertionError("internal raw-prefix reference schema drift")
    return result


def _raw_input_prefixes(root: Path) -> dict[str, dict[str, Any]]:
    retirement_relative = str(
        getattr(
            results,
            "RETIREMENT_LEDGER_RELATIVE_PATH",
            "_campaign/retirements.jsonl",
        )
    )
    events_relative = str(
        getattr(
            results,
            "CAMPAIGN_EVENTS_RELATIVE_PATH",
            "_campaign/events.jsonl",
        )
    )
    value = {
        "retirement_ledger": _raw_prefix_ref(
            root,
            retirement_relative,
            name="retirement ledger prefix",
        ),
        "campaign_events": _raw_prefix_ref(
            root,
            events_relative,
            name="campaign events prefix",
        ),
        "manual_invalid_results": _raw_prefix_ref(
            root,
            MANUAL_INVALID_LEDGER_RELATIVE_PATH,
            name="manual invalid-result ledger prefix",
        ),
    }
    if set(value) != RAW_INPUT_PREFIX_KEYS:
        raise AssertionError("internal raw input prefix schema drift")
    return value


def _manifest_inventory(root: Path) -> list[dict[str, Any]]:
    inventory = [
        _file_ref(path, root=root, name="result manifest inventory")
        for path in sorted(root.rglob(results.MANIFEST_FILENAME))
    ]
    return inventory


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FreezeError(f"{name} must be an object")
    return value


def _optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, name)


def _manifest_launch_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    try:
        context = results._tune_attempt_context(manifest, manifest_path)
    except (results.ResultError, KeyError, TypeError, ValueError) as exc:
        raise FreezeError(
            f"cannot classify tune launch {manifest_path}: {exc}"
        ) from exc
    if context is None:
        raise FreezeError(
            f"manifest is not an identifiable REAL-Q tune launch: {manifest_path}"
        )
    model = _nonempty_string(context.get("model"), "tune launch model")
    setting = _nonempty_string(context.get("setting"), "tune launch setting")
    if (model, setting) not in _matrix_keys():
        raise FreezeError(
            f"tune launch has an unknown model/setting: {model}/{setting}"
        )
    lr = _finite_nonnegative(context.get("grad_lr"), "tune launch grad_lr")
    attempt = context.get("attempt_index")
    if type(attempt) is not int or attempt <= 0 or attempt > MAX_ATTEMPTS:
        raise FreezeError("tune launch attempt_index is outside 1..20")
    plan_wrapper = _mapping(manifest.get("plan"), "tune launch manifest.plan")
    model_wrapper = _mapping(
        manifest.get("model"), "tune launch manifest.model"
    )
    sources = _mapping(
        manifest.get("source_files"), "tune launch manifest.source_files"
    )
    runner_source = _mapping(
        sources.get("runner"), "tune launch manifest.source_files.runner"
    )
    executor_source = _mapping(
        sources.get("executor"), "tune launch manifest.source_files.executor"
    )
    numerical = _mapping(
        manifest.get("numerical_source_tree"),
        "tune launch manifest.numerical_source_tree",
    )
    status = _nonempty_string(manifest.get("status"), "tune launch status")
    exit_code = manifest.get("exit_code")
    if exit_code is not None and type(exit_code) is not int:
        raise FreezeError("tune launch exit_code must be integer or null")
    context_valid = context.get("context_valid")
    if type(context_valid) is not bool:
        raise FreezeError("tune launch context_valid must be boolean")
    identity = {
        "execution_id": _nonempty_string(
            manifest.get("execution_id"), "tune launch execution_id"
        ),
        "run_id": _nonempty_string(manifest.get("run_id"), "tune launch run_id"),
        "status": status,
        "exit_code": exit_code,
        "model": model,
        "setting": setting,
        "grad_lr": lr,
        "attempt_index": attempt,
        "context_valid": context_valid,
        "plan_sha256": _require_sha256(
            plan_wrapper.get("sha256"), "tune launch plan_sha256"
        ),
        "model_sha256": _require_sha256(
            model_wrapper.get("combined_identity_sha256"),
            "tune launch model_sha256",
        ),
        "runner_sha256": _require_sha256(
            runner_source.get("sha256"), "tune launch runner_sha256"
        ),
        "executor_sha256": _require_sha256(
            executor_source.get("sha256"), "tune launch executor_sha256"
        ),
        "numerical_source_sha256_at_start": _require_sha256(
            numerical.get("combined_sha256"),
            "tune launch numerical source start SHA256",
        ),
        "numerical_source_sha256_at_end": _optional_sha256(
            numerical.get("combined_sha256_at_end"),
            "tune launch numerical source end SHA256",
        ),
        "numerical_source_changed_during_execution": numerical.get(
            "changed_during_execution"
        ),
    }
    changed = identity["numerical_source_changed_during_execution"]
    if type(changed) is not bool:
        raise FreezeError(
            "tune launch numerical source changed flag must be boolean"
        )
    return identity


def _validate_reviewed_manual_invalid_identity(
    identity: Mapping[str, Any],
    *,
    name: str,
    plan_sha256: str,
    source_identity: Mapping[str, str],
) -> tuple[str, str, float, int]:
    reviewed_key = (
        identity.get("model"),
        identity.get("setting"),
        identity.get("grad_lr"),
        identity.get("attempt_index"),
    )
    reviewed = MANUAL_INVALID_REVIEWED_TARGETS.get(reviewed_key)
    if (
        reviewed is None
        or identity.get("status") != "succeeded"
        or identity.get("exit_code") != 0
        or identity.get("numerical_source_changed_during_execution") is not True
        or identity.get("numerical_source_sha256_at_start")
        != (reviewed or {}).get("numerical_source_sha256_at_start")
        or identity.get("numerical_source_sha256_at_end")
        != (reviewed or {}).get("numerical_source_sha256_at_end")
        or identity.get("numerical_source_sha256_at_start")
        != source_identity["numerical_source_sha256"]
        or identity.get("plan_sha256") != plan_sha256
        or identity.get("runner_sha256") != source_identity["runner_sha256"]
        or identity.get("executor_sha256") != source_identity["executor_sha256"]
    ):
        raise FreezeError(
            f"{name} is not one of the two narrowly reviewed source-drift "
            "launches"
        )
    return reviewed_key


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o600
    with contextlib.suppress(OSError):
        mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    """Atomically create a read-only artifact without an overwrite window."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FreezeError(
                f"immutable freeze artifact already exists: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        with contextlib.suppress(OSError):
            _fsync_directory(path.parent)


def _append_immutable_bytes(
    path: Path, *, expected_prefix: bytes, suffix: bytes
) -> None:
    """Atomically replace an immutable ledger with prefix + one new record."""

    if not suffix or not suffix.endswith(b"\n"):
        raise FreezeError("manual invalid-result append suffix is invalid")
    current = _read_stable_bytes(path, "manual invalid-result ledger")
    if current != expected_prefix:
        raise FreezeError(
            "manual invalid-result ledger changed before append"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".append.tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected_prefix)
            handle.write(suffix)
            handle.flush()
            os.fsync(handle.fileno())
        if _read_stable_bytes(
            path, "manual invalid-result ledger"
        ) != expected_prefix:
            raise FreezeError(
                "manual invalid-result ledger changed during append"
            )
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        with contextlib.suppress(OSError):
            _fsync_directory(path.parent)


def _assert_file_sha(path: Path, expected: str, name: str) -> bytes:
    raw, observed = _stable_file_identity(path, name)
    if observed != expected:
        raise FreezeError(
            f"{name} CAS failed: expected {expected}, observed {observed}"
        )
    return raw


@contextlib.contextmanager
def _existing_state_lock(state_path: Path) -> Iterator[None]:
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    if not lock_path.is_file():
        raise FreezeError(
            f"existing campaign lock file is required: {lock_path}"
        )
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FreezeError(
                f"another campaign controller owns {lock_path}"
            ) from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _finite_nonnegative(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise FreezeError(f"{name} must be a finite non-negative number")
    return float(value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreezeError(f"{name} must be a non-empty string")
    return value


def _matrix_keys() -> tuple[tuple[str, str], ...]:
    return tuple(
        (model, setting)
        for model in runner.MODEL_ORDER
        for setting in runner.SETTING_ORDER
    )


def _campaign_events_by_seq(
    output_root: Path,
) -> dict[int, tuple[Mapping[str, Any], str]]:
    relative = str(
        getattr(
            results,
            "CAMPAIGN_EVENTS_RELATIVE_PATH",
            "_campaign/events.jsonl",
        )
    )
    path = output_root / relative
    if not path.is_file():
        return {}
    raw = _read_stable_bytes(path, "campaign events")
    if raw and not raw.endswith(b"\n"):
        raise FreezeError("campaign events must end with LF")
    events: dict[int, tuple[Mapping[str, Any], str]] = {}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise FreezeError("campaign events must not contain blank lines")
        value = _parse_json_bytes(
            line, f"campaign events line {line_number}"
        )
        if not isinstance(value, Mapping):
            raise FreezeError(
                f"campaign events line {line_number} must be an object"
            )
        sequence = value.get("seq")
        if type(sequence) is not int or sequence <= 0 or sequence in events:
            raise FreezeError(
                f"campaign events line {line_number} has invalid/duplicate seq"
            )
        events[sequence] = (value, _sha256_bytes(line))
    return events


def _validate_manual_invalid_entry(
    entry: Mapping[str, Any],
    *,
    index: int,
    previous_hash: str | None,
    output_root: Path,
    campaign_id: str,
    plan_sha256: str,
    source_identity: Mapping[str, str],
    events: Mapping[int, tuple[Mapping[str, Any], str]],
) -> dict[str, Any]:
    name = f"manual invalid-result ledger entry {index}"
    _require_exact_keys(entry, MANUAL_INVALID_ENTRY_KEYS, name)
    if entry.get("schema_version") != SCHEMA_VERSION or entry.get("seq") != index:
        raise FreezeError(f"{name} schema_version/seq is invalid")
    timestamp = _nonempty_string(
        entry.get("timestamp_utc"), f"{name}.timestamp_utc"
    )
    try:
        parsed_timestamp = dt.datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise FreezeError(f"{name}.timestamp_utc is invalid") from exc
    if parsed_timestamp.tzinfo is None:
        raise FreezeError(f"{name}.timestamp_utc must include timezone")
    if (
        entry.get("campaign_id") != campaign_id
        or entry.get("reason_code") != MANUAL_INVALID_REASON_CODE
        or entry.get("prev_record_sha256") != previous_hash
    ):
        raise FreezeError(f"{name} campaign/reason/hash-chain identity is invalid")
    record_sha = _require_sha256(
        entry.get("record_sha256"), f"{name}.record_sha256"
    )
    unsigned = dict(entry)
    unsigned.pop("record_sha256")
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != record_sha:
        raise FreezeError(f"{name}.record_sha256 is invalid")

    authorization = entry.get("authorization")
    if not isinstance(authorization, Mapping):
        raise FreezeError(f"{name}.authorization must be an object")
    _require_exact_keys(
        authorization,
        MANUAL_INVALID_AUTHORIZATION_KEYS,
        f"{name}.authorization",
    )
    if authorization.get("kind") != "user_authorized_manual_invalid_result_v1":
        raise FreezeError(f"{name}.authorization.kind is invalid")
    _nonempty_string(
        authorization.get("authorized_by"),
        f"{name}.authorization.authorized_by",
    )
    _nonempty_string(
        authorization.get("rationale"),
        f"{name}.authorization.rationale",
    )

    manifest_reference = entry.get("manifest")
    log_reference = entry.get("log")
    if not isinstance(manifest_reference, Mapping) or not isinstance(
        log_reference, Mapping
    ):
        raise FreezeError(f"{name} manifest/log references must be objects")
    _require_exact_keys(
        manifest_reference, FILE_REF_KEYS, f"{name}.manifest"
    )
    _require_exact_keys(log_reference, FILE_REF_KEYS, f"{name}.log")
    manifest_relative = _nonempty_string(
        manifest_reference.get("relative_path"),
        f"{name}.manifest.relative_path",
    )
    log_relative = _nonempty_string(
        log_reference.get("relative_path"),
        f"{name}.log.relative_path",
    )
    manifest_path = _path_inside_root(
        output_root / manifest_relative, output_root, f"{name}.manifest"
    )
    log_path = _path_inside_root(
        output_root / log_relative, output_root, f"{name}.log"
    )
    current_manifest_ref = _file_ref(
        manifest_path, root=output_root, name=f"{name}.manifest"
    )
    current_log_ref = _file_ref(
        log_path, root=output_root, name=f"{name}.log"
    )
    if (
        dict(manifest_reference) != current_manifest_ref
        or dict(log_reference) != current_log_ref
        or current_manifest_ref["present"] is not True
        or current_log_ref["present"] is not True
        or manifest_path.name != results.MANIFEST_FILENAME
        or log_path.name != results.LOG_FILENAME
        or manifest_path.parent != log_path.parent
    ):
        raise FreezeError(f"{name} manifest/log raw references are invalid")
    manifest = _parse_json_bytes(
        _read_stable_bytes(manifest_path, f"{name}.manifest"),
        f"{name}.manifest",
    )
    if not isinstance(manifest, Mapping):
        raise FreezeError(f"{name}.manifest must contain an object")
    identity = _manifest_launch_identity(
        manifest,
        manifest_path=manifest_path,
        output_root=output_root,
    )
    exact_identity = {
        "execution_id": entry.get("execution_id"),
        "run_id": entry.get("run_id"),
        "status": entry.get("manifest_status"),
        "exit_code": entry.get("exit_code"),
        "model": entry.get("model"),
        "setting": entry.get("setting"),
        "grad_lr": entry.get("grad_lr"),
        "attempt_index": entry.get("attempt_index"),
        "plan_sha256": entry.get("plan_sha256"),
        "model_sha256": entry.get("model_sha256"),
        "runner_sha256": entry.get("runner_sha256"),
        "executor_sha256": entry.get("executor_sha256"),
        "numerical_source_sha256_at_start": entry.get(
            "numerical_source_sha256_at_start"
        ),
        "numerical_source_sha256_at_end": entry.get(
            "numerical_source_sha256_at_end"
        ),
        "numerical_source_changed_during_execution": entry.get(
            "numerical_source_changed_during_execution"
        ),
    }
    for field, expected in exact_identity.items():
        if identity.get(field) != expected:
            raise FreezeError(f"{name}.{field} disagrees with current manifest")
    if entry.get("phase") != "tune" or entry.get("method") != "realq":
        raise FreezeError(f"{name} phase/method identity is invalid")
    _validate_reviewed_manual_invalid_identity(
        identity,
        name=name,
        plan_sha256=plan_sha256,
        source_identity=source_identity,
    )
    expected_errors = entry.get("expected_rejection_errors")
    if expected_errors != list(MANUAL_INVALID_REJECTION_ERRORS):
        raise FreezeError(f"{name}.expected_rejection_errors is invalid")

    event_present = entry.get("launch_event_present")
    event_reference = entry.get("launch_event")
    if type(event_present) is not bool:
        raise FreezeError(f"{name}.launch_event_present must be boolean")
    if not event_present:
        if event_reference is not None:
            raise FreezeError(
                f"{name}.launch_event must be null when explicitly absent"
            )
    else:
        if not isinstance(event_reference, Mapping):
            raise FreezeError(f"{name}.launch_event must be an object")
        _require_exact_keys(
            event_reference,
            MANUAL_INVALID_LAUNCH_EVENT_KEYS,
            f"{name}.launch_event",
        )
        sequence = event_reference.get("seq")
        if type(sequence) is not int or sequence <= 0:
            raise FreezeError(f"{name}.launch_event.seq is invalid")
        event_pair = events.get(sequence)
        if event_pair is None:
            raise FreezeError(f"{name}.launch_event references a missing event")
        event, event_sha = event_pair
        details = event.get("details")
        if (
            event_reference.get("line_sha256") != event_sha
            or not isinstance(details, Mapping)
            or event.get("schema_version") != SCHEMA_VERSION
            or event.get("campaign_id") != campaign_id
            or event.get("category") != "task"
            or event.get("code") != "task_launching"
            or details.get("task_id") != event_reference.get("task_id")
            or details.get("model") != entry.get("model")
            or details.get("setting") != entry.get("setting")
            or details.get("grad_lr") != entry.get("grad_lr")
            or details.get("attempt_index") != entry.get("attempt_index")
            or details.get("plan_sha256") != entry.get("plan_sha256")
        ):
            raise FreezeError(f"{name}.launch_event identity is invalid")
    return copy.deepcopy(dict(entry))


def _load_manual_invalid_resolutions(
    output_root: Path,
    *,
    campaign_id: str,
    plan_sha256: str,
    source_identity: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    path = output_root / MANUAL_INVALID_LEDGER_RELATIVE_PATH
    if not path.is_file():
        return ()
    raw = _read_stable_bytes(path, "manual invalid-result ledger")
    if raw and not raw.endswith(b"\n"):
        raise FreezeError("manual invalid-result ledger must end with LF")
    lines = raw.splitlines()
    if any(not line for line in lines):
        raise FreezeError(
            "manual invalid-result ledger must not contain blank lines"
        )
    if len(lines) > len(MANUAL_INVALID_REVIEWED_TARGETS):
        raise FreezeError(
            "manual invalid-result ledger exceeds the reviewed target set"
        )
    events = _campaign_events_by_seq(output_root)
    entries: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for index, line in enumerate(lines, 1):
        entry = _parse_json_bytes(
            line, f"manual invalid-result ledger line {index}"
        )
        if not isinstance(entry, Mapping):
            raise FreezeError(
                f"manual invalid-result ledger line {index} must be an object"
            )
        if line != _canonical_json_bytes(entry):
            raise FreezeError(
                f"manual invalid-result ledger line {index} is not canonical JSON"
            )
        normalized = _validate_manual_invalid_entry(
            entry,
            index=index,
            previous_hash=previous_hash,
            output_root=output_root,
            campaign_id=campaign_id,
            plan_sha256=plan_sha256,
            source_identity=source_identity,
            events=events,
        )
        entries.append(normalized)
        previous_hash = normalized["record_sha256"]
    reviewed_keys = [
        (
            entry["model"],
            entry["setting"],
            entry["grad_lr"],
            entry["attempt_index"],
        )
        for entry in entries
    ]
    if len(reviewed_keys) != len(set(reviewed_keys)):
        raise FreezeError(
            "manual invalid-result ledger resolves a reviewed target twice"
        )
    return tuple(entries)


def _load_selection(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_stable_bytes(path, "selection")
    root = _parse_json_bytes(raw, "selection")
    if not isinstance(root, dict):
        raise FreezeError("selection root must be an object")
    _require_exact_keys(root, INPUT_ROOT_KEYS, "selection root")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise FreezeError("selection schema_version must be 1")
    canonical_file = _canonical_json_bytes(root) + b"\n"
    if raw != canonical_file:
        raise FreezeError(
            "selection file must be exact canonical JSON followed by one LF"
        )
    entries = root.get("selections")
    if not isinstance(entries, list) or len(entries) != len(_matrix_keys()):
        raise FreezeError("selection must contain exactly 15 rows")
    observed_order: list[tuple[str, str]] = []
    overrides: list[tuple[str, str]] = []
    for index, entry_value in enumerate(entries):
        name = f"selection.selections[{index}]"
        if not isinstance(entry_value, dict):
            raise FreezeError(f"{name} must be an object")
        _require_exact_keys(entry_value, INPUT_SELECTION_KEYS, name)
        model = _nonempty_string(entry_value.get("model"), f"{name}.model")
        setting = _nonempty_string(
            entry_value.get("setting"), f"{name}.setting"
        )
        observed_order.append((model, setting))
        selected_lr = _finite_nonnegative(
            entry_value.get("selected_lr"), f"{name}.selected_lr"
        )
        kind = entry_value.get("decision_kind")
        if kind not in DECISION_KINDS:
            raise FreezeError(f"{name}.decision_kind is invalid")
        _nonempty_string(entry_value.get("rationale"), f"{name}.rationale")
        evidence = entry_value.get("evidence")
        required_count = 1 if kind == "exact" else 2
        if not isinstance(evidence, list) or len(evidence) != required_count:
            raise FreezeError(
                f"{name}.evidence must contain exactly {required_count} row(s)"
            )
        evidence_lrs: list[float] = []
        evidence_paths: list[str] = []
        for evidence_index, evidence_value in enumerate(evidence):
            evidence_name = f"{name}.evidence[{evidence_index}]"
            if not isinstance(evidence_value, dict):
                raise FreezeError(f"{evidence_name} must be an object")
            _require_exact_keys(
                evidence_value, INPUT_EVIDENCE_KEYS, evidence_name
            )
            manifest = _nonempty_string(
                evidence_value.get("manifest"),
                f"{evidence_name}.manifest",
            )
            manifest_path = Path(manifest)
            if not manifest_path.is_absolute():
                raise FreezeError(
                    f"{evidence_name}.manifest must be an absolute path"
                )
            evidence_paths.append(manifest)
            evidence_lrs.append(
                _finite_nonnegative(
                    evidence_value.get("grad_lr"),
                    f"{evidence_name}.grad_lr",
                )
            )
            _finite_nonnegative(
                evidence_value.get("kl_wikitext2"),
                f"{evidence_name}.kl_wikitext2",
            )
        if len(set(evidence_paths)) != len(evidence_paths):
            raise FreezeError(f"{name}.evidence contains duplicate manifests")
        if kind == "exact":
            if selected_lr != evidence_lrs[0]:
                raise FreezeError(
                    f"{name} exact selected_lr must equal its evidence LR"
                )
        else:
            overrides.append((model, setting))
            if not evidence_lrs[0] < selected_lr < evidence_lrs[1]:
                raise FreezeError(
                    f"{name} override evidence must strictly bracket selected_lr"
                )
    if tuple(observed_order) != _matrix_keys():
        raise FreezeError(
            "selection rows must follow the canonical model/setting order"
        )
    if overrides != [QWEN_OVERRIDE_KEY]:
        raise FreezeError(
            "the only reviewed override must be qwen3-4b/2W16A"
        )
    qwen_entry = entries[_matrix_keys().index(QWEN_OVERRIDE_KEY)]
    if (
        qwen_entry["decision_kind"] != "user_override_interpolated"
        or float(qwen_entry["selected_lr"]) != QWEN_OVERRIDE_LR
        or tuple(float(item["grad_lr"]) for item in qwen_entry["evidence"])
        != QWEN_OVERRIDE_EVIDENCE_LRS
    ):
        raise FreezeError(
            "qwen3-4b/2W16A must be the reviewed 1.2e-4 interpolation "
            "between 1.18e-4 and 1.25e-4"
        )
    return root, _sha256_bytes(_canonical_json_bytes(root))


def _output_root(plan: Mapping[str, Any]) -> Path:
    value = plan.get("output_root")
    if not isinstance(value, str) or not value:
        raise FreezeError("plan.output_root must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def _expected_state_path(plan: Mapping[str, Any]) -> Path:
    return _output_root(plan) / "_campaign" / "state.json"


def _artifact_path(plan: Mapping[str, Any]) -> Path:
    return _output_root(plan) / "_campaign" / ARTIFACT_FILENAME


def _current_base_source_identity() -> dict[str, str]:
    controller = base_campaign.Campaign.__new__(base_campaign.Campaign)
    controller.repo_root = REPO_ROOT.resolve()
    try:
        identity = base_campaign.Campaign._current_source_identity(controller)
    except (OSError, KeyError, base_campaign.CampaignBlocked) as exc:
        raise FreezeError(
            f"cannot establish current base source identity: {exc}"
        ) from exc
    if set(identity) != set(SOURCE_IDENTITY_KEYS):
        raise FreezeError("current base source identity has unexpected fields")
    for key in SOURCE_IDENTITY_KEYS:
        _require_sha256(identity.get(key), f"current source_identity.{key}")
    return dict(identity)


def _stable_timing_gate() -> dict[str, Any]:
    try:
        first = timed_campaign._timing_gate_now()
        second = timed_campaign._timing_gate_now()
    except (OSError, KeyError, base_campaign.CampaignBlocked) as exc:
        raise FreezeError(f"cannot establish formal timing gate: {exc}") from exc
    if first != second:
        raise FreezeError("formal timing source set changed while being hashed")
    gate_sha = _require_sha256(
        first.get("gate_sha256"), "formal timing gate_sha256"
    )
    source_sha = _require_sha256(
        first.get("source_set_sha256"),
        "formal timing source_set_sha256",
    )
    unsigned = dict(first)
    unsigned.pop("gate_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != gate_sha:
        raise FreezeError("formal timing gate self-hash is invalid")
    sources = first.get("sources")
    expected_names = {
        "formal_timed_campaign",
        "formal_timed_execute",
        "formal_timing_adapter",
        "formal_timing_sitecustomize",
        "lowbit_activation_campaign",
        "lowbit_activation_gpu_hours",
        "validate_guided_saliency",
    }
    if (
        not isinstance(sources, list)
        or {
            item.get("name")
            for item in sources
            if isinstance(item, Mapping)
        }
        != expected_names
        or len(sources) != len(expected_names)
    ):
        raise FreezeError("formal timing gate must contain the exact seven sources")
    return copy.deepcopy(first)


def _validate_source_adoption(
    state_identity_value: Any,
    current_identity: Mapping[str, str],
    expected_old_results_sha256: str | None,
) -> dict[str, str] | None:
    if not isinstance(state_identity_value, Mapping):
        raise FreezeError("state.source_identity must be an object")
    if set(state_identity_value) != set(SOURCE_IDENTITY_KEYS):
        raise FreezeError("state.source_identity must contain exactly five hashes")
    state_identity: dict[str, str] = {}
    for key in SOURCE_IDENTITY_KEYS:
        state_identity[key] = _require_sha256(
            state_identity_value.get(key), f"state.source_identity.{key}"
        )
    differences = {
        key
        for key in SOURCE_IDENTITY_KEYS
        if state_identity[key] != current_identity[key]
    }
    if not differences:
        if expected_old_results_sha256 is not None:
            raise FreezeError(
                "--expected-old-results-sha256 is forbidden when no "
                "results-source adoption is needed"
            )
        return None
    if differences != {"results_sha256"}:
        raise FreezeError(
            "source drift is not eligible for the narrow results-parser "
            f"adoption: {sorted(differences)!r}"
        )
    if expected_old_results_sha256 is None:
        raise FreezeError(
            "results.py drift requires --expected-old-results-sha256"
        )
    expected_old = _require_sha256(
        expected_old_results_sha256, "expected old results SHA256"
    )
    if state_identity["results_sha256"] != expected_old:
        raise FreezeError(
            "expected old results SHA256 does not match the pinned state"
        )
    return {
        "kind": SOURCE_ADOPTION_KIND,
        "expected_old_results_sha256": expected_old,
        "accepted_current_results_sha256": current_identity[
            "results_sha256"
        ],
    }


def _resolve_evidence_path(
    value: str,
    *,
    output_root: Path,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise FreezeError("evidence manifest path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FreezeError(f"evidence manifest is unavailable: {path}: {exc}") from exc
    if str(resolved) != value:
        raise FreezeError(
            "evidence manifest must use its canonical resolved absolute path: "
            f"{value!r}"
        )
    if resolved.name != results.MANIFEST_FILENAME:
        raise FreezeError(
            f"evidence manifest must be named {results.MANIFEST_FILENAME}"
        )
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise FreezeError(
            f"evidence manifest escapes plan output_root: {resolved}"
        ) from exc
    return resolved


def _matching_state_task(
    state: Mapping[str, Any],
    *,
    parsed: results.ParsedResult,
    manifest_path: Path,
) -> Mapping[str, Any]:
    tasks = state.get("tasks")
    if not isinstance(tasks, Mapping):
        raise FreezeError("state.tasks must be an object")
    matches: list[Mapping[str, Any]] = []
    for task_value in tasks.values():
        if not isinstance(task_value, Mapping):
            raise FreezeError("every state task must be an object")
        manifest_value = task_value.get("manifest")
        if not isinstance(manifest_value, str):
            continue
        try:
            candidate = Path(manifest_value).resolve(strict=False)
        except OSError:
            continue
        if candidate == manifest_path:
            matches.append(task_value)
    if len(matches) != 1:
        raise FreezeError(
            f"evidence manifest must match exactly one state task: {manifest_path}"
        )
    task = matches[0]
    if task.get("status") != "SUCCEEDED" or task.get("exit_code") != 0:
        raise FreezeError("evidence state task must be SUCCEEDED with exit_code 0")
    if task.get("plan_compatible") is False:
        raise FreezeError("evidence state task is marked plan-incompatible")
    spec = task.get("spec")
    if not isinstance(spec, Mapping):
        raise FreezeError("evidence state task spec must be an object")
    expected_spec = (
        spec.get("kind"),
        spec.get("method"),
        spec.get("phase"),
        spec.get("target_phase"),
        spec.get("model"),
        spec.get("setting"),
    )
    if expected_spec != (
        "realq",
        "realq",
        "tune",
        None,
        parsed.model,
        parsed.setting,
    ):
        raise FreezeError(
            "evidence state task kind/method/phase/model/setting mismatch"
        )
    spec_lr = _finite_nonnegative(spec.get("grad_lr"), "state task grad_lr")
    if parsed.grad_lr is None or spec_lr != float(parsed.grad_lr):
        raise FreezeError("evidence state task grad_lr mismatch")
    attempt = task.get("attempt_index", spec.get("attempt_index", 1))
    if type(attempt) is not int or attempt != parsed.attempt_index:
        raise FreezeError("evidence state task attempt_index mismatch")
    log_value = task.get("log")
    if (
        isinstance(log_value, str)
        and Path(log_value).resolve(strict=False) != parsed.log_path
    ):
        raise FreezeError("evidence state task log path mismatch")
    metric = task.get("metric")
    if not isinstance(metric, Mapping):
        raise FreezeError("evidence state task must retain validated metric data")
    if (
        _finite_nonnegative(metric.get("kl"), "state task metric.kl")
        != float(parsed.kl_wikitext2)
        or _finite_nonnegative(
            metric.get("grad_lr"), "state task metric.grad_lr"
        )
        != float(parsed.grad_lr)
    ):
        raise FreezeError("evidence state task metric mismatch")
    identities = metric.get("identities")
    if not isinstance(identities, Mapping):
        raise FreezeError("evidence state task metric identities are missing")
    for key, value in parsed.identities.items():
        if identities.get(key) != value:
            raise FreezeError(
                f"evidence state task metric identity {key!r} mismatch"
            )
    return task


def _validate_evidence(
    selection: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    state: Mapping[str, Any],
    source_identity: Mapping[str, str],
    output_root: Path,
    reconciliation_point: bool = False,
) -> dict[str, Any]:
    manifest_path = _resolve_evidence_path(
        str(evidence["manifest"]), output_root=output_root
    )
    manifest_before, manifest_sha = _stable_file_identity(
        manifest_path, "evidence manifest"
    )
    expected_log_path = manifest_path.parent / results.LOG_FILENAME
    log_before, log_sha = _stable_file_identity(
        expected_log_path, "evidence log"
    )
    try:
        parsed = results.parse_result(manifest_path)
    except (results.ResultError, OSError, ValueError) as exc:
        raise FreezeError(
            f"evidence parser rejected {manifest_path}: {exc}"
        ) from exc
    manifest_after, manifest_sha_after = _stable_file_identity(
        manifest_path, "evidence manifest"
    )
    if manifest_before != manifest_after or manifest_sha != manifest_sha_after:
        raise FreezeError(f"evidence manifest changed during parsing: {manifest_path}")
    if parsed.manifest_path != manifest_path:
        raise FreezeError("result parser returned a different manifest path")
    log_path = parsed.log_path
    log_after, log_sha_after = _stable_file_identity(log_path, "evidence log")
    if (
        log_path != expected_log_path.resolve(strict=True)
        or log_before != log_after
        or log_sha != log_sha_after
    ):
        raise FreezeError(f"evidence log changed while being hashed: {log_path}")
    if (
        parsed.phase != "tune"
        or parsed.method != "realq"
        or parsed.model != selection["model"]
        or parsed.setting != selection["setting"]
    ):
        raise FreezeError("evidence parsed model/setting/method/phase mismatch")
    if (
        parsed.grad_lr is None
        or float(parsed.grad_lr) != float(evidence["grad_lr"])
    ):
        raise FreezeError("evidence parsed grad_lr mismatch")
    if (
        parsed.kl_wikitext2 is None
        or not math.isfinite(float(parsed.kl_wikitext2))
        or float(parsed.kl_wikitext2) < 0
        or float(parsed.kl_wikitext2)
        != float(evidence["kl_wikitext2"])
    ):
        raise FreezeError("evidence parsed exact WikiText-2 KL mismatch")
    if (
        not math.isfinite(float(parsed.ppl_wikitext2))
        or float(parsed.ppl_wikitext2) < 0
    ):
        raise FreezeError("evidence parsed WikiText-2 PPL is invalid")
    if (
        type(parsed.attempt_index) is not int
        or parsed.attempt_index <= 0
        or parsed.attempt_index > MAX_ATTEMPTS
        or parsed.max_attempts != MAX_ATTEMPTS
    ):
        raise FreezeError("evidence attempt is outside the reviewed <=20 budget")
    if parsed.identities.get("plan_sha256") != plan_sha256:
        raise FreezeError("evidence manifest belongs to a different plan SHA256")
    if parsed.plan_content != plan:
        raise FreezeError("evidence embedded plan differs from the old plan")
    for key in (
        "numerical_source_sha256",
        "runner_sha256",
        "executor_sha256",
    ):
        if parsed.identities.get(key) != source_identity[key]:
            raise FreezeError(f"evidence {key} disagrees with pinned source")
    normalized = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "log": str(log_path),
        "log_sha256": log_sha,
        "execution_id": parsed.execution_id,
        "run_id": parsed.run_id,
        "attempt_index": parsed.attempt_index,
        "grad_lr": float(parsed.grad_lr),
        "kl_wikitext2": float(parsed.kl_wikitext2),
        "plan_sha256": parsed.identities["plan_sha256"],
        "model_sha256": parsed.identities["model_sha256"],
        "runner_sha256": parsed.identities["runner_sha256"],
        "executor_sha256": parsed.identities["executor_sha256"],
        "numerical_source_sha256": parsed.identities[
            "numerical_source_sha256"
        ],
    }
    if not reconciliation_point:
        return normalized
    config = {
        "world_size": parsed.world_size,
        "final_layer_grad_lr": parsed.final_layer_grad_lr,
        "params": parsed.params,
        "max_attempts": parsed.max_attempts,
        "search_policy": parsed.search_policy,
    }
    point = {
        "model": parsed.model,
        "setting": parsed.setting,
        **normalized,
        "world_size": parsed.world_size,
        "ppl_wikitext2": float(parsed.ppl_wikitext2),
        "config_sha256": _sha256_bytes(_canonical_json_bytes(config)),
    }
    if set(point) != RECONCILIATION_POINT_KEYS:
        raise AssertionError("internal reconciliation point schema drift")
    return point


@dataclasses.dataclass(frozen=True)
class ReportAuthority:
    accepted_tune_results: tuple[dict[str, Any], ...]
    points_by_group: dict[tuple[str, str], tuple[dict[str, Any], ...]]
    group_attempt_counts: tuple[dict[str, Any], ...]
    attempt_count_by_group: dict[tuple[str, str], int]
    tune_launches: tuple[dict[str, Any], ...]
    invalid_result_resolutions: tuple[dict[str, Any], ...]
    raw_input_prefixes: dict[str, dict[str, Any]]
    accepted_active_set_sha256: str
    group_attempt_counts_sha256: str
    tune_launches_sha256: str
    invalid_result_resolutions_sha256: str
    raw_input_prefixes_sha256: str
    report_snapshot_sha256: str


def _report_path_set(
    values: Any, *, label: str
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(values, list):
        raise FreezeError(f"authoritative results report {label} is invalid")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise FreezeError(
                f"authoritative results report {label}[{index}] is invalid"
            )
        manifest = _nonempty_string(
            value.get("manifest"),
            f"authoritative results report {label}[{index}].manifest",
        )
        canonical = str(Path(manifest).resolve(strict=False))
        if manifest != canonical or canonical in result:
            raise FreezeError(
                f"authoritative results report {label} has a "
                "non-canonical/duplicate manifest"
            )
        result[canonical] = value
    return result


def _build_tune_launch_projection(
    output_root: Path,
    *,
    report: Mapping[str, Any],
    resolutions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    accepted = _report_path_set(report.get("records"), label="records")
    rejected = _report_path_set(report.get("rejected"), label="rejected")
    informational = _report_path_set(
        report.get("informational", []), label="informational"
    )
    retired = _report_path_set(
        report.get("retired_attempts", []), label="retired_attempts"
    )
    resolution_by_manifest: dict[str, Mapping[str, Any]] = {}
    for entry in resolutions:
        manifest = entry.get("manifest")
        if not isinstance(manifest, Mapping):
            raise FreezeError("manual invalid resolution manifest is invalid")
        path = _nonempty_string(
            manifest.get("absolute_path"),
            "manual invalid resolution manifest absolute_path",
        )
        if path in resolution_by_manifest:
            raise FreezeError("manual invalid ledger resolves a manifest twice")
        resolution_by_manifest[path] = entry
    if set(resolution_by_manifest) - set(rejected):
        raise FreezeError(
            "manual invalid ledger references a manifest not rejected by "
            "the authoritative parser"
        )
    unresolved = sorted(set(rejected) - set(resolution_by_manifest))
    if unresolved:
        raise FreezeError(
            "authoritative results report contains unretired/unresolved "
            f"rejected manifests: {unresolved!r}"
        )
    for path, entry in resolution_by_manifest.items():
        errors = rejected[path].get("errors")
        if errors != entry.get("expected_rejection_errors"):
            raise FreezeError(
                "manual invalid resolution rejection errors disagree with "
                f"the authoritative parser: {path}"
            )

    launches: list[dict[str, Any]] = []
    classified_paths: set[str] = set()
    for manifest_path in sorted(
        output_root.rglob(results.MANIFEST_FILENAME)
    ):
        manifest_raw = _read_stable_bytes(
            manifest_path, "tune launch manifest"
        )
        manifest = _parse_json_bytes(manifest_raw, "tune launch manifest")
        if not isinstance(manifest, Mapping):
            raise FreezeError(
                f"result manifest must be an object: {manifest_path}"
            )
        try:
            context = results._tune_attempt_context(
                manifest, manifest_path.resolve()
            )
        except (results.ResultError, KeyError, TypeError, ValueError) as exc:
            raise FreezeError(
                f"cannot classify result manifest {manifest_path}: {exc}"
            ) from exc
        if context is None:
            continue
        manifest_resolved = str(manifest_path.resolve())
        identity = _manifest_launch_identity(
            manifest,
            manifest_path=manifest_path.resolve(),
            output_root=output_root,
        )
        if manifest_resolved in resolution_by_manifest:
            classification = "manual_invalid_result"
            rejection_errors = list(
                rejected[manifest_resolved]["errors"]
            )
        elif manifest_resolved in retired:
            classification = "retired"
            rejection_errors = []
        elif manifest_resolved in accepted:
            record = accepted[manifest_resolved]
            if (
                record.get("method") != "realq"
                or record.get("phase") != "tune"
            ):
                raise FreezeError(
                    "a tune launch maps to a non-tune accepted record"
                )
            classification = "accepted"
            rejection_errors = []
        elif manifest_resolved in rejected:
            # All rejected paths were required to be explicitly resolved above.
            raise FreezeError(
                f"unresolved rejected tune launch: {manifest_resolved}"
            )
        elif manifest_resolved in informational:
            classification = "informational"
            rejection_errors = []
        else:
            raise FreezeError(
                "identifiable tune launch is absent from every authoritative "
                f"report classification: {manifest_resolved}"
            )
        log_path = manifest_path.parent / results.LOG_FILENAME
        launch = {
            "manifest": _file_ref(
                manifest_path,
                root=output_root,
                name="tune launch manifest",
            ),
            "log": _file_ref(
                log_path,
                root=output_root,
                name="tune launch log",
            ),
            "classification": classification,
            "rejection_errors": rejection_errors,
            **identity,
        }
        if set(launch) != TUNE_LAUNCH_KEYS:
            raise AssertionError("internal tune launch schema drift")
        launches.append(launch)
        classified_paths.add(manifest_resolved)

    for label, mapping in (
        ("accepted tune record", accepted),
        ("retired tune attempt", retired),
        ("manual invalid result", resolution_by_manifest),
    ):
        for path, value in mapping.items():
            if label == "accepted tune record" and (
                value.get("method") != "realq" or value.get("phase") != "tune"
            ):
                continue
            if label == "retired tune attempt" and (
                value.get("phase") != "tune"
            ):
                continue
            if path not in classified_paths:
                raise FreezeError(
                    f"{label} lacks an identifiable raw tune launch: {path}"
                )
    launches.sort(
        key=lambda item: (
            _matrix_keys().index(
                (str(item["model"]), str(item["setting"]))
            ),
            float(item["grad_lr"]),
            str(item["manifest"]["absolute_path"]),
        )
    )
    manifests = [item["manifest"]["absolute_path"] for item in launches]
    execution_ids = [item["execution_id"] for item in launches]
    run_ids = [item["run_id"] for item in launches]
    if (
        len(manifests) != len(set(manifests))
        or len(execution_ids) != len(set(execution_ids))
        or len(run_ids) != len(set(run_ids))
    ):
        raise FreezeError(
            "complete tune launch projection has duplicate manifest/run identity"
        )
    return tuple(launches)


def _report_authority(
    output_root: Path,
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    state: Mapping[str, Any],
    source_identity: Mapping[str, str],
) -> ReportAuthority:
    output_root = output_root.resolve(strict=True)
    campaign_id = _nonempty_string(
        state.get("campaign_id"), "state campaign_id"
    )
    prefixes_before = _raw_input_prefixes(output_root)
    inventory_before = _manifest_inventory(output_root)
    resolutions_before = _load_manual_invalid_resolutions(
        output_root,
        campaign_id=campaign_id,
        plan_sha256=plan_sha256,
        source_identity=source_identity,
    )
    try:
        report = results.build_report(output_root)
    except (results.ResultError, OSError, ValueError) as exc:
        raise FreezeError(
            f"cannot build the authoritative pre-freeze results report: {exc}"
        ) from exc
    if not isinstance(report, Mapping):
        raise FreezeError("authoritative results report must be an object")
    prefixes_after_report = _raw_input_prefixes(output_root)
    inventory_after_report = _manifest_inventory(output_root)
    if (
        prefixes_after_report != prefixes_before
        or inventory_after_report != inventory_before
    ):
        raise FreezeError(
            "result manifests or authority ledgers changed while building report"
        )
    retirement = report.get("retirement_ledger")
    if (
        not isinstance(retirement, Mapping)
        or retirement.get("valid") is not True
    ):
        raise FreezeError(
            "retirement ledger is missing or invalid before manual LR freeze"
        )
    rejected = report.get("rejected")
    if not isinstance(rejected, list):
        raise FreezeError("authoritative results report rejected list is invalid")
    records = report.get("records")
    if not isinstance(records, list):
        raise FreezeError("authoritative results report records list is invalid")
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [] for key in _matrix_keys()
    }
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise FreezeError(
                f"authoritative results report record {index} is invalid"
            )
        if record.get("method") != "realq" or record.get("phase") != "tune":
            continue
        key = (record.get("model"), record.get("setting"))
        if key not in by_group:
            raise FreezeError(
                f"authoritative report contains an unknown tune group: {key!r}"
            )
        manifest = _nonempty_string(
            record.get("manifest"),
            f"authoritative results report record {index}.manifest",
        )
        evidence = {
            "manifest": manifest,
            "grad_lr": _finite_nonnegative(
                record.get("grad_lr"),
                f"authoritative record {index}.grad_lr",
            ),
            "kl_wikitext2": _finite_nonnegative(
                record.get("kl_wikitext2"),
                f"authoritative record {index}.kl_wikitext2",
            ),
        }
        point = _validate_evidence(
            {"model": key[0], "setting": key[1]},
            evidence,
            plan=plan,
            plan_sha256=plan_sha256,
            state=state,
            source_identity=source_identity,
            output_root=output_root,
            reconciliation_point=True,
        )
        by_group[key].append(point)
    missing = [
        f"{model}/{setting}"
        for (model, setting), points in by_group.items()
        if not points
    ]
    if missing:
        raise FreezeError(
            "authoritative results report lacks accepted active tune results "
            f"for {missing!r}"
        )
    accepted: list[dict[str, Any]] = []
    normalized_groups: dict[
        tuple[str, str], tuple[dict[str, Any], ...]
    ] = {}
    seen_manifests: set[str] = set()
    seen_execution_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    for key in _matrix_keys():
        ordered = sorted(
            by_group[key],
            key=lambda point: (float(point["grad_lr"]), point["manifest"]),
        )
        lrs = [float(point["grad_lr"]) for point in ordered]
        if len(lrs) != len(set(lrs)):
            raise FreezeError(
                f"authoritative accepted set has duplicate LR for "
                f"{key[0]}/{key[1]}"
            )
        for point in ordered:
            for value, seen, label in (
                (point["manifest"], seen_manifests, "manifest"),
                (
                    point["execution_id"],
                    seen_execution_ids,
                    "execution_id",
                ),
                (point["run_id"], seen_run_ids, "run_id"),
            ):
                if value in seen:
                    raise FreezeError(
                        f"authoritative accepted set duplicates {label}: {value}"
                    )
                seen.add(value)
        normalized_groups[key] = tuple(ordered)
        accepted.extend(ordered)

    tune_selections = report.get("realq_tune_selection")
    if not isinstance(tune_selections, list):
        raise FreezeError(
            "authoritative results report tune selections are invalid"
        )
    counts_by_group: dict[tuple[str, str], int] = {}
    for index, selection in enumerate(tune_selections):
        if not isinstance(selection, Mapping):
            raise FreezeError(
                f"authoritative tune selection {index} must be an object"
            )
        key = (selection.get("model"), selection.get("setting"))
        if key not in normalized_groups or key in counts_by_group:
            raise FreezeError(
                "authoritative tune selections have an unknown/duplicate group"
            )
        count = selection.get("attempt_count")
        if (
            type(count) is not int
            or count < len(normalized_groups[key])
            or count > MAX_ATTEMPTS
        ):
            raise FreezeError(
                f"authoritative launch count is invalid for {key!r}: {count!r}"
            )
        counts_by_group[key] = count
    if tuple(counts_by_group) != _matrix_keys():
        raise FreezeError(
            "authoritative tune selections must cover the canonical 15 groups"
        )
    group_counts = tuple(
        {
            "model": model,
            "setting": setting,
            "attempt_count": counts_by_group[(model, setting)],
        }
        for model, setting in _matrix_keys()
    )
    tune_launches = _build_tune_launch_projection(
        output_root,
        report=report,
        resolutions=resolutions_before,
    )
    derived_counts = {
        key: sum(
            1
            for launch in tune_launches
            if (launch["model"], launch["setting"]) == key
        )
        for key in _matrix_keys()
    }
    if derived_counts != counts_by_group:
        raise FreezeError(
            "authoritative report launch counts disagree with the complete "
            "raw tune launch projection"
        )
    prefixes_after = _raw_input_prefixes(output_root)
    inventory_after = _manifest_inventory(output_root)
    resolutions_after = _load_manual_invalid_resolutions(
        output_root,
        campaign_id=campaign_id,
        plan_sha256=plan_sha256,
        source_identity=source_identity,
    )
    if (
        prefixes_after != prefixes_before
        or inventory_after != inventory_before
        or resolutions_after != resolutions_before
    ):
        raise FreezeError(
            "result manifests or authority ledgers changed during reconciliation"
        )
    accepted_sha = _sha256_bytes(_canonical_json_bytes(accepted))
    counts_sha = _sha256_bytes(_canonical_json_bytes(group_counts))
    launches_sha = _sha256_bytes(_canonical_json_bytes(tune_launches))
    resolutions_sha = _sha256_bytes(
        _canonical_json_bytes(resolutions_before)
    )
    prefixes_sha = _sha256_bytes(
        _canonical_json_bytes(prefixes_before)
    )
    snapshot = {
        "accepted_active_set_sha256": accepted_sha,
        "group_attempt_counts_sha256": counts_sha,
        "tune_launches_sha256": launches_sha,
        "invalid_result_resolutions_sha256": resolutions_sha,
        "raw_input_prefixes_sha256": prefixes_sha,
    }
    return ReportAuthority(
        accepted_tune_results=tuple(accepted),
        points_by_group=normalized_groups,
        group_attempt_counts=group_counts,
        attempt_count_by_group=counts_by_group,
        tune_launches=tune_launches,
        invalid_result_resolutions=resolutions_before,
        raw_input_prefixes=prefixes_before,
        accepted_active_set_sha256=accepted_sha,
        group_attempt_counts_sha256=counts_sha,
        tune_launches_sha256=launches_sha,
        invalid_result_resolutions_sha256=resolutions_sha,
        raw_input_prefixes_sha256=prefixes_sha,
        report_snapshot_sha256=_sha256_bytes(
            _canonical_json_bytes(snapshot)
        ),
    )


def _validate_and_normalize_selections(
    selection_root: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    state: Mapping[str, Any],
    source_identity: Mapping[str, str],
) -> tuple[list[dict[str, Any]], ReportAuthority]:
    output_root = _output_root(plan)
    authority = _report_authority(
        output_root,
        plan=plan,
        plan_sha256=plan_sha256,
        state=state,
        source_identity=source_identity,
    )
    normalized: list[dict[str, Any]] = []
    seen_manifests: set[str] = set()
    seen_execution_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    tuning = state.get("tuning")
    if not isinstance(tuning, Mapping):
        raise FreezeError("state.tuning must be an object")
    for selection_value in selection_root["selections"]:
        selection = dict(selection_value)
        key = f"{selection['model']}/{selection['setting']}"
        group = tuning.get(key)
        if not isinstance(group, Mapping):
            raise FreezeError(f"state tuning group is missing: {key}")
        if (
            group.get("model") != selection["model"]
            or group.get("setting") != selection["setting"]
        ):
            raise FreezeError(f"state tuning group identity mismatch: {key}")
        attempt_count = authority.attempt_count_by_group[
            (selection["model"], selection["setting"])
        ]
        normalized_evidence = [
            _validate_evidence(
                selection,
                evidence,
                plan=plan,
                plan_sha256=plan_sha256,
                state=state,
                source_identity=source_identity,
                output_root=output_root,
            )
            for evidence in selection["evidence"]
        ]
        accepted_points = list(
            authority.points_by_group[
                (selection["model"], selection["setting"])
            ]
        )
        accepted_manifests = {
            point["manifest"] for point in accepted_points
        }
        if not {
            item["manifest"] for item in normalized_evidence
        }.issubset(accepted_manifests):
            raise FreezeError(
                f"selected evidence is not in the accepted active set for {key}"
            )
        if selection["decision_kind"] == "exact":
            minimum = min(point["kl_wikitext2"] for point in accepted_points)
            winners = [
                point
                for point in accepted_points
                if point["kl_wikitext2"] == minimum
            ]
            if len(winners) != 1:
                raise FreezeError(
                    f"exact selection for {key} has no unique Exact KL minimum"
                )
            winner = winners[0]
            if (
                normalized_evidence[0]["manifest"] != winner["manifest"]
                or float(selection["selected_lr"]) != winner["grad_lr"]
            ):
                raise FreezeError(
                    f"exact selection for {key} is not the unique accepted "
                    "active Exact KL minimum"
                )
            if selection["model"] == "qwen3-32b":
                ordered_points = sorted(
                    accepted_points,
                    key=lambda point: float(point["grad_lr"]),
                )
                winner_index = ordered_points.index(winner)
                if (
                    winner_index == 0
                    or winner_index == len(ordered_points) - 1
                ):
                    raise FreezeError(
                        f"qwen3-32b exact selection for {key} lacks a "
                        "strictly worse accepted KL point on both LR sides"
                    )
                lower = ordered_points[winner_index - 1]
                higher = ordered_points[winner_index + 1]
                best_kl = decimal.Decimal(
                    str(winner["kl_wikitext2"])
                )
                side_kls = (
                    decimal.Decimal(str(lower["kl_wikitext2"])),
                    decimal.Decimal(str(higher["kl_wikitext2"])),
                )
                if any(side_kl <= best_kl for side_kl in side_kls):
                    raise FreezeError(
                        f"qwen3-32b nearest LR neighbors for {key} must "
                        "both have strictly worse Exact KL"
                    )
                gaps = (
                    tuple(
                        (side_kl - best_kl) / best_kl
                        for side_kl in side_kls
                    )
                    if best_kl > 0
                    else (
                        decimal.Decimal("Infinity"),
                        decimal.Decimal("Infinity"),
                    )
                )
                if int(attempt_count) < 8 and any(
                    gap > decimal.Decimal("0.02") for gap in gaps
                ):
                    raise FreezeError(
                        f"qwen3-32b exact selection for {key} has fewer "
                        "than 8 launches and a nearest-side KL gap above 2%"
                    )
        for item in normalized_evidence:
            manifest = item["manifest"]
            execution_id = item["execution_id"]
            run_id = item["run_id"]
            if manifest in seen_manifests:
                raise FreezeError(
                    f"evidence manifest is reused across selections: {manifest}"
                )
            if execution_id in seen_execution_ids:
                raise FreezeError(
                    f"evidence execution_id is duplicated: {execution_id}"
                )
            if run_id in seen_run_ids:
                raise FreezeError(f"evidence run_id is duplicated: {run_id}")
            seen_manifests.add(manifest)
            seen_execution_ids.add(execution_id)
            seen_run_ids.add(run_id)
        kind = selection["decision_kind"]
        selected_manifest = (
            normalized_evidence[0]["manifest"] if kind == "exact" else None
        )
        normalized_selection = {
            "model": selection["model"],
            "setting": selection["setting"],
            "selected_lr": float(selection["selected_lr"]),
            "decision_kind": kind,
            "rationale": selection["rationale"],
            "selected_manifest": selected_manifest,
            "evidence_set_sha256": _sha256_bytes(
                _canonical_json_bytes(normalized_evidence)
            ),
            "evidence": normalized_evidence,
        }
        if set(normalized_selection) != NORMALIZED_SELECTION_KEYS:
            raise AssertionError("internal normalized selection schema drift")
        normalized.append(normalized_selection)
    # Re-establish source stability after parsing every external artifact.
    if _current_base_source_identity() != dict(source_identity):
        raise FreezeError("base source identity changed during evidence validation")
    return normalized, authority


def _task_has_live_process(task: Mapping[str, Any]) -> bool:
    pid = task.get("executor_pid")
    if type(pid) is int and pid > 0:
        identity = proc_identity(pid)
        if identity is not None:
            expected_start = task.get("executor_start_ticks")
            expected_session = task.get("executor_session_id")
            expected_group = task.get("executor_process_group_id")
            matches = True
            if type(expected_start) is int:
                matches = matches and identity.get("start_ticks") == expected_start
            if type(expected_session) is int:
                matches = matches and identity.get("session_id") == expected_session
            if type(expected_group) is int:
                matches = (
                    matches
                    and identity.get("process_group_id") == expected_group
                )
            if (
                type(expected_start) is int
                or type(expected_session) is int
                or type(expected_group) is int
            ) and matches:
                return True
    manifest_value = task.get("manifest")
    if not isinstance(manifest_value, str):
        return False
    manifest_path = Path(manifest_value)
    if not manifest_path.is_file():
        return False
    try:
        manifest = _parse_json_bytes(
            _read_stable_bytes(manifest_path, "task manifest"),
            "task manifest",
        )
    except FreezeError:
        return False
    if not isinstance(manifest, Mapping):
        return False
    process = manifest.get("process")
    command = manifest.get("command")
    manifest_pid = process.get("pid") if isinstance(process, Mapping) else None
    argv = command.get("argv") if isinstance(command, Mapping) else None
    if (
        type(manifest_pid) is not int
        or manifest_pid <= 0
        or not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) for item in argv)
    ):
        return False
    return (
        proc_identity(manifest_pid) is not None
        and cmdline_matches_argv(proc_cmdline(manifest_pid), argv)
    )


def _assert_no_running_work(
    state: Mapping[str, Any],
    *,
    output_root: Path,
) -> None:
    tasks = state.get("tasks")
    if not isinstance(tasks, Mapping):
        raise FreezeError("state.tasks must be an object")
    related: list[str] = []
    for task_id, task_value in tasks.items():
        if not isinstance(task_value, Mapping):
            raise FreezeError(f"state task {task_id!r} must be an object")
        if _task_has_live_process(task_value):
            related.append(str(task_id))
    if related:
        raise FreezeError(
            "related executor processes are still alive: "
            f"{sorted(related)!r}"
        )

    for manifest_path in sorted(
        output_root.rglob(results.MANIFEST_FILENAME)
    ):
        manifest = _parse_json_bytes(
            _read_stable_bytes(manifest_path, "output manifest"),
            "output manifest",
        )
        if not isinstance(manifest, Mapping):
            raise FreezeError(f"output manifest is not an object: {manifest_path}")
        process = manifest.get("process")
        command = manifest.get("command")
        pid = process.get("pid") if isinstance(process, Mapping) else None
        argv = command.get("argv") if isinstance(command, Mapping) else None
        if (
            type(pid) is int
            and pid > 0
            and isinstance(argv, list)
            and argv
            and all(isinstance(item, str) for item in argv)
            and proc_identity(pid) is not None
            and cmdline_matches_argv(proc_cmdline(pid), argv)
        ):
            raise FreezeError(
                f"output manifest still owns a live process: {manifest_path}"
            )

    lock_filename = getattr(
        base_campaign.executor, "LOCK_FILENAME", ".execution.lock"
    )
    for lock_path in sorted(output_root.rglob(lock_filename)):
        try:
            descriptor = os.open(lock_path, os.O_RDWR)
        except OSError as exc:
            raise FreezeError(
                f"cannot inspect execution lock {lock_path}: {exc}"
            ) from exc
        try:
            try:
                fcntl.flock(
                    descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError as exc:
                raise FreezeError(
                    f"an experiment still owns execution lock {lock_path}"
                ) from exc
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    snapshot = query_gpus()
    if snapshot.error:
        raise FreezeError(
            "cannot prove all GPUs idle before handoff: "
            f"{snapshot.error}"
        )
    if snapshot.compute_processes:
        raise FreezeError(
            "GPU compute processes are visible; refusing the handoff"
        )


def _validate_environment(
    *,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    agent_path: Path,
    ready_marker: str,
) -> None:
    agent_state = state.get("agent")
    if not isinstance(agent_state, Mapping):
        raise FreezeError("state.agent must be an object")
    state_agent_path = agent_state.get("path")
    if (
        not isinstance(state_agent_path, str)
        or Path(state_agent_path).resolve(strict=False)
        != agent_path.resolve(strict=False)
        or agent_state.get("ready_marker") != ready_marker
    ):
        raise FreezeError("CLI agent path/marker disagrees with campaign state")
    ready, reason = agent_ready(agent_path, ready_marker)
    if not ready:
        raise FreezeError(reason)
    environment_ok, environment = venv_ready(plan)
    if not environment_ok:
        raise FreezeError(
            "source the exact repository .venv before the manual LR handoff: "
            f"{environment}"
        )


def _validate_old_plan(
    plan: Mapping[str, Any],
    *,
    plan_sha256: str,
) -> None:
    try:
        runner.validate_structure(dict(plan))
    except runner.PlanError as exc:
        raise FreezeError(f"old plan is invalid: {exc}") from exc
    selected = plan.get("selected_grad_lr_by_model_setting")
    if not isinstance(selected, Mapping):
        raise FreezeError("old plan selected LR map is missing")
    for model, setting in _matrix_keys():
        model_values = selected.get(model)
        if (
            not isinstance(model_values, Mapping)
            or model_values.get(setting) is not None
        ):
            raise FreezeError(
                f"old plan selected LR must still be null: {model}/{setting}"
            )
    if plan.get("manual_lr_selection_freeze") is not None:
        raise FreezeError("old plan already contains a manual LR freeze")
    _require_sha256(plan_sha256, "old plan SHA256")


def _validate_old_state(
    state: Mapping[str, Any],
    *,
    state_path: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    plan_sha256: str,
    expected_campaign_id: str,
) -> None:
    if state.get("schema_version") != base_campaign.STATE_SCHEMA_VERSION:
        raise FreezeError("campaign state schema is invalid")
    if state.get("campaign_id") != expected_campaign_id:
        raise FreezeError("campaign_id CAS failed")
    if state_path != _expected_state_path(plan).resolve(strict=False):
        raise FreezeError(
            "state must be the fixed plan output_root/_campaign/state.json"
        )
    plan_state = state.get("plan")
    if not isinstance(plan_state, Mapping):
        raise FreezeError("state.plan must be an object")
    state_plan_path = plan_state.get("path")
    if (
        not isinstance(state_plan_path, str)
        or Path(state_plan_path).resolve(strict=False)
        != plan_path
    ):
        raise FreezeError("state.plan.path is invalid")
    expected_protocol = base_campaign._protocol_sha256(plan)
    if (
        plan_state.get("tune_sha256") != plan_sha256
        or plan_state.get("active_sha256") != plan_sha256
        or plan_state.get("tune_protocol_sha256") != expected_protocol
        or plan_state.get("formal_sha256") is not None
        or plan_state.get("formal_protocol_sha256") is not None
        or plan_state.get("pending_transition_sha256") is not None
    ):
        raise FreezeError("state plan identity is not an untouched tune snapshot")
    if state.get("pending_controller_source_transition") is not None:
        raise FreezeError("a controller source transition is already pending")
    if state.get("selected_lr_patch") is not None:
        raise FreezeError("state already contains a selected-LR patch")
    if (
        state.get("manual_lr_selection_freeze") is not None
        or state.get("manual_lr_handoff") is not None
        or state.get("manual_lr_state_reconciliation") is not None
    ):
        raise FreezeError("state already contains a manual LR handoff")
    formal = state.get("formal")
    if (
        not isinstance(formal, Mapping)
        or formal.get("initialized") is not False
        or formal.get("models") != {}
        or formal.get("timing_gate") is not None
    ):
        raise FreezeError("formal state must be completely uninitialized")
    tasks = state.get("tasks")
    if not isinstance(tasks, Mapping):
        raise FreezeError("state.tasks must be an object")
    for task in tasks.values():
        if not isinstance(task, Mapping):
            raise FreezeError("every state task must be an object")
        spec = task.get("spec")
        if not isinstance(spec, Mapping):
            raise FreezeError("every state task spec must be an object")
        if spec.get("phase") == "final" or spec.get("target_phase") == "final":
            raise FreezeError("pre-freeze state must not contain formal tasks")


def _selection_patch(
    normalized: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = {
        model: {setting: None for setting in runner.SETTING_ORDER}
        for model in runner.MODEL_ORDER
    }
    for item in normalized:
        selected[str(item["model"])][str(item["setting"])] = float(
            item["selected_lr"]
        )
    if any(
        value is None
        for model_values in selected.values()
        for value in model_values.values()
    ):
        raise AssertionError("internal selected LR matrix is incomplete")
    return {"selected_grad_lr_by_model_setting": selected}


def _transaction_id(
    *,
    campaign_id: str,
    expected_plan_sha256: str,
    expected_state_sha256: str,
    selection_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "campaign_id": campaign_id,
        "expected_plan_sha256": expected_plan_sha256,
        "expected_state_sha256": expected_state_sha256,
        "selection_sha256": selection_sha256,
    }
    return "manual-lr-freeze-" + _sha256_bytes(
        _canonical_json_bytes(payload)
    )[:24]


@contextlib.contextmanager
def _fixed_campaign_time(timestamp: str) -> Iterator[None]:
    previous = base_campaign._utc_now
    base_campaign._utc_now = lambda: timestamp
    try:
        yield
    finally:
        base_campaign._utc_now = previous


def _initialize_timed_formal(
    state: dict[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_raw: bytes,
    plan_path: Path,
    state_path: Path,
    agent_path: Path,
    timestamp: str,
    expected_gate: Mapping[str, Any],
) -> dict[str, Any]:
    controller = timed_campaign.FormalTimedCampaign.__new__(
        timed_campaign.FormalTimedCampaign
    )
    controller.repo_root = REPO_ROOT.resolve()
    controller.plan_path = plan_path
    controller.state_path = state_path
    controller.events_path = state_path.with_name("events.jsonl")
    controller.agent_path = agent_path
    controller.ready_marker = DEFAULT_READY_MARKER
    controller.execute = False
    controller.poll_seconds = 30.0
    controller.processes = {}
    controller.controller_logs = {}
    controller.preview_events = []
    controller.plan = dict(plan)
    controller.plan_raw = plan_raw
    controller.plan_sha256 = _sha256_bytes(plan_raw)
    controller.output_root = _output_root(plan)
    controller.retirements_path = (
        controller.output_root
        / "_campaign"
        / base_campaign.RETIREMENT_LEDGER_FILENAME
    )
    controller.retirement_records = []
    controller.state = state
    with _fixed_campaign_time(timestamp):
        try:
            timed_campaign.FormalTimedCampaign._initialize_formal(controller)
        except (base_campaign.CampaignError, OSError, KeyError) as exc:
            raise FreezeError(
                f"formal timed initialization failed closed: {exc}"
            ) from exc
    formal = controller.state.get("formal")
    if (
        not isinstance(formal, Mapping)
        or formal.get("initialized") is not True
        or formal.get("timing_gate") != expected_gate
    ):
        raise FreezeError("timed initialization did not pin the expected gate")
    try:
        timed_identity = (
            timed_campaign.FormalTimedCampaign._current_source_identity(
                controller
            )
        )
    except (base_campaign.CampaignError, OSError, KeyError) as exc:
        raise FreezeError(
            f"cannot revalidate timed controller identity: {exc}"
        ) from exc
    if controller.state.get("source_identity") != timed_identity:
        raise FreezeError("timed source identity was not pinned exactly")
    base_identity = _current_base_source_identity()
    if controller.state.get("source_identity") == base_identity:
        raise FreezeError("base controller would not fail closed after handoff")
    return controller.state


@dataclasses.dataclass(frozen=True)
class FreezeTransaction:
    transaction_id: str
    created_utc: str
    plan_path: Path
    state_path: Path
    selection_path: Path
    artifact_path: Path
    campaign_id: str
    expected_plan_sha256: str
    expected_state_sha256: str
    selection_sha256: str
    ledger_sha256: str
    artifact_sha256: str
    old_plan_raw: bytes
    old_state_raw: bytes
    artifact_raw: bytes
    new_plan_raw: bytes
    frozen_state_raw: bytes
    committed_state_raw: bytes
    wrapper: dict[str, Any]
    downstream_required_gate: dict[str, Any]

    @property
    def new_plan_sha256(self) -> str:
        return _sha256_bytes(self.new_plan_raw)

    @property
    def frozen_state_sha256(self) -> str:
        return _sha256_bytes(self.frozen_state_raw)

    @property
    def committed_state_sha256(self) -> str:
        return _sha256_bytes(self.committed_state_raw)

    def journal(self, phase: str) -> dict[str, Any]:
        if phase not in JOURNAL_PHASES:
            raise AssertionError(f"invalid journal phase {phase!r}")
        payloads = {
            "old_plan_base64": base64.b64encode(
                self.old_plan_raw
            ).decode("ascii"),
            "old_state_base64": base64.b64encode(
                self.old_state_raw
            ).decode("ascii"),
            "artifact_base64": base64.b64encode(
                self.artifact_raw
            ).decode("ascii"),
            "new_plan_base64": base64.b64encode(
                self.new_plan_raw
            ).decode("ascii"),
            "frozen_state_base64": base64.b64encode(
                self.frozen_state_raw
            ).decode("ascii"),
            "committed_state_base64": base64.b64encode(
                self.committed_state_raw
            ).decode("ascii"),
        }
        journal = {
            "schema_version": SCHEMA_VERSION,
            "scope_id": SCOPE_ID,
            "transaction_id": self.transaction_id,
            "phase": phase,
            "created_utc": self.created_utc,
            "plan_path": str(self.plan_path),
            "state_path": str(self.state_path),
            "selection_path": str(self.selection_path),
            "expected_campaign_id": self.campaign_id,
            "expected_plan_sha256": self.expected_plan_sha256,
            "expected_state_sha256": self.expected_state_sha256,
            "selection_sha256": self.selection_sha256,
            "ledger_sha256": self.ledger_sha256,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": len(self.artifact_raw),
            "new_plan_sha256": self.new_plan_sha256,
            "frozen_state_sha256": self.frozen_state_sha256,
            "committed_state_sha256": self.committed_state_sha256,
            "payloads": payloads,
        }
        if set(journal) != JOURNAL_KEYS:
            raise AssertionError("internal journal schema drift")
        return journal


def _selected_parameters(
    selection: Mapping[str, Any],
    *,
    ledger_sha256: str,
    selection_sha256: str,
) -> dict[str, Any]:
    return {
        "grad_lr": float(selection["selected_lr"]),
        "decision_kind": selection["decision_kind"],
        "rationale": selection["rationale"],
        "selected_manifest": selection["selected_manifest"],
        "evidence_set_sha256": selection["evidence_set_sha256"],
        "evidence": copy.deepcopy(selection["evidence"]),
        "manual_lr_selection_scope_id": SCOPE_ID,
        "selection_sha256": selection_sha256,
        "ledger_sha256": ledger_sha256,
    }


def _invalidate_preflight(
    plan: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    path = base_campaign._resolve_preflight_path(plan, repo_root=REPO_ROOT)
    return {
        "path": str(path) if path is not None else None,
        "valid": False,
        "errors": [reason],
        "sha256": None,
    }


def _task_manifest_key(task: Mapping[str, Any]) -> str | None:
    value = task.get("manifest")
    if not isinstance(value, str) or not value:
        return None
    return str(Path(value).resolve(strict=False))


def _derive_terminalized_tasks(
    state: Mapping[str, Any],
    authority: ReportAuthority,
) -> list[dict[str, Any]]:
    tasks = state.get("tasks")
    if not isinstance(tasks, Mapping):
        raise FreezeError("state.tasks must be an object")
    del authority
    transitions: list[dict[str, Any]] = []
    for task_id, task_value in sorted(
        tasks.items(), key=lambda item: str(item[0])
    ):
        if not isinstance(task_value, Mapping):
            raise FreezeError(f"state task {task_id!r} must be an object")
        previous = task_value.get("status")
        if not isinstance(previous, str):
            raise FreezeError(f"state task {task_id!r} status is invalid")
        spec = task_value.get("spec")
        tune_realq = (
            isinstance(spec, Mapping)
            and spec.get("kind") == "realq"
            and spec.get("phase") == "tune"
        )
        if not (
            (tune_realq and previous != "SUPERSEDED")
            or previous not in base_campaign.TERMINAL_TASK_STATES
        ):
            continue
        transition = {
            "task_id": str(task_id),
            "previous_status": previous,
            "reconciled_status": "SUPERSEDED",
            "reason": (
                "old_tune_task_superseded_by_authoritative_import"
                if tune_realq
                else "stale_runnable_task_closed_before_formal_handoff"
            ),
        }
        if set(transition) != RECONCILIATION_TERMINALIZED_KEYS:
            raise AssertionError("internal terminalized task schema drift")
        transitions.append(transition)
    return transitions


def _derive_imported_tasks(
    authority: ReportAuthority,
    *,
    timestamp: str,
) -> list[dict[str, Any]]:
    imported: list[dict[str, Any]] = []
    for point in authority.accepted_tune_results:
        task_id = "reconciled_tune:" + point["manifest_sha256"]
        task = {
            "task_id": task_id,
            "spec": {
                # This terminal evidence task is intentionally outside the
                # scheduler's runnable ``kind == realq`` task set.  Memory
                # profile/generation metadata is not reconstructed from stale
                # controller state.
                "kind": "reconciled_realq_tune_result",
                "method": "realq",
                "phase": "tune",
                "target_phase": None,
                "model": point["model"],
                "setting": point["setting"],
                "grad_lr": point["grad_lr"],
                "overrides": {},
                "profile_level": None,
                "static_profile_level": None,
                "generation": None,
                "purpose": "manual_lr_freeze_authoritative_import",
            },
            "status": "SUCCEEDED",
            "gpu_count": point["world_size"],
            "priority": 0,
            "gpus": [],
            "attempt_index": point["attempt_index"],
            "executor_pid": None,
            "created_utc": timestamp,
            "launched_utc": None,
            "finished_utc": timestamp,
            "output_dir": str(Path(point["manifest"]).parent),
            "manifest": point["manifest"],
            "log": point["log"],
            "exit_code": 0,
            "failure_class": None,
            "cache_validation": None,
            # Keep the base controller's legacy ``imported`` resume path off:
            # it expects runnable REAL-Q profile integers.  The dedicated
            # marker below is the authority for this terminal evidence task.
            "imported": False,
            "plan_compatible": True,
            "metric": {
                "kl": point["kl_wikitext2"],
                "ppl": point["ppl_wikitext2"],
                "grad_lr": point["grad_lr"],
                "identities": {
                    key: point[key]
                    for key in (
                        "plan_sha256",
                        "model_sha256",
                        "runner_sha256",
                        "executor_sha256",
                        "numerical_source_sha256",
                    )
                },
            },
            "imported_by_manual_lr_freeze": True,
            "reconciliation_reason": (
                "retirement_filtered_results_authoritative_import"
            ),
        }
        if set(task) != IMPORTED_TASK_KEYS:
            raise AssertionError("internal imported task schema drift")
        imported.append(task)
    task_ids = [item["task_id"] for item in imported]
    if len(task_ids) != len(set(task_ids)):
        raise FreezeError("authoritative imports produce duplicate task IDs")
    return imported


def _state_reconciliation_ref(
    reconciliation: Mapping[str, Any],
    *,
    artifact_sha256: str,
    phase: str,
) -> dict[str, Any]:
    result = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": RECONCILIATION_POLICY_ID,
        "phase": phase,
        "artifact_sha256": artifact_sha256,
        "accepted_active_set_sha256": reconciliation[
            "accepted_active_set_sha256"
        ],
        "group_attempt_counts_sha256": reconciliation[
            "group_attempt_counts_sha256"
        ],
        "tune_launches_sha256": reconciliation[
            "tune_launches_sha256"
        ],
        "invalid_result_resolutions_sha256": reconciliation[
            "invalid_result_resolutions_sha256"
        ],
        "raw_input_prefixes_sha256": reconciliation[
            "raw_input_prefixes_sha256"
        ],
        "report_snapshot_sha256": reconciliation[
            "report_snapshot_sha256"
        ],
        "terminalized_tasks_sha256": reconciliation[
            "terminalized_tasks_sha256"
        ],
        "imported_tasks_sha256": reconciliation["imported_tasks_sha256"],
    }
    if set(result) != STATE_RECONCILIATION_KEYS:
        raise AssertionError("internal state reconciliation ref schema drift")
    return result


def _apply_state_reconciliation(
    state: dict[str, Any],
    *,
    authority: ReportAuthority,
    reconciliation: Mapping[str, Any],
    artifact_sha256: str,
    timestamp: str,
) -> None:
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise FreezeError("state.tasks must be a mutable object")
    del authority
    transitions = {
        item["task_id"]: item
        for item in reconciliation["terminalized_tasks"]
    }
    for task_id, transition in transitions.items():
        task = tasks.get(task_id)
        if not isinstance(task, dict):
            raise FreezeError(
                f"reconciliation task disappeared from state: {task_id}"
            )
        task["reconciled_previous_status"] = transition["previous_status"]
        task["status"] = transition["reconciled_status"]
        task["reconciliation_reason"] = transition["reason"]
        task["gpus"] = []
        task["executor_pid"] = None
        for field in (
            "executor_start_ticks",
            "executor_session_id",
            "executor_process_group_id",
            "executor_cli",
            "rendezvous",
        ):
            task.pop(field, None)
        task["finished_utc"] = task.get("finished_utc") or timestamp
    for imported_value in reconciliation["imported_tasks"]:
        imported = copy.deepcopy(imported_value)
        task_id = imported["task_id"]
        if task_id in tasks:
            raise FreezeError(
                f"authoritative imported task ID collides with old state: {task_id}"
            )
        tasks[task_id] = imported
    state["manual_lr_state_reconciliation"] = _state_reconciliation_ref(
        reconciliation,
        artifact_sha256=artifact_sha256,
        phase="state_frozen",
    )


def _build_transaction(
    *,
    plan_path: Path,
    state_path: Path,
    selection_path: Path,
    agent_path: Path,
    ready_marker: str,
    expected_plan_sha256: str,
    expected_state_sha256: str,
    expected_campaign_id: str,
    expected_old_results_sha256: str | None,
) -> FreezeTransaction:
    old_plan_raw = _assert_file_sha(
        plan_path, expected_plan_sha256, "old plan"
    )
    old_state_raw = _assert_file_sha(
        state_path, expected_state_sha256, "old state"
    )
    plan_value = _parse_json_bytes(old_plan_raw, "old plan")
    state_value = _parse_json_bytes(old_state_raw, "old state")
    if not isinstance(plan_value, dict) or not isinstance(state_value, dict):
        raise FreezeError("old plan and state must both be JSON objects")
    plan = plan_value
    state = state_value
    _validate_old_plan(plan, plan_sha256=expected_plan_sha256)
    _validate_old_state(
        state,
        state_path=state_path,
        plan_path=plan_path,
        plan=plan,
        plan_sha256=expected_plan_sha256,
        expected_campaign_id=expected_campaign_id,
    )
    _validate_environment(
        plan=plan,
        state=state,
        agent_path=agent_path,
        ready_marker=ready_marker,
    )
    _assert_no_running_work(state, output_root=_output_root(plan))

    selection_root, selection_sha256 = _load_selection(selection_path)
    current_source_identity = _current_base_source_identity()
    source_adoption = _validate_source_adoption(
        state.get("source_identity"),
        current_source_identity,
        expected_old_results_sha256,
    )
    timing_gate = _stable_timing_gate()
    normalized, authority = _validate_and_normalize_selections(
        selection_root,
        plan=plan,
        plan_sha256=expected_plan_sha256,
        state=state,
        source_identity=current_source_identity,
    )
    if _stable_timing_gate() != timing_gate:
        raise FreezeError("formal timing gate changed during evidence validation")

    transaction_id = _transaction_id(
        campaign_id=expected_campaign_id,
        expected_plan_sha256=expected_plan_sha256,
        expected_state_sha256=expected_state_sha256,
        selection_sha256=selection_sha256,
    )
    created_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    downstream_gate = {
        "gate_id": GATE_ID,
        "required": True,
        "status": "required_before_formal_results_acceptance",
    }
    terminalized_tasks = _derive_terminalized_tasks(state, authority)
    imported_tasks = _derive_imported_tasks(
        authority, timestamp=created_utc
    )
    reconciliation = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": RECONCILIATION_POLICY_ID,
        "accepted_tune_results": [
            copy.deepcopy(point)
            for point in authority.accepted_tune_results
        ],
        "accepted_active_set_sha256": authority.accepted_active_set_sha256,
        "group_attempt_counts": [
            dict(item) for item in authority.group_attempt_counts
        ],
        "group_attempt_counts_sha256": (
            authority.group_attempt_counts_sha256
        ),
        "tune_launches": [
            copy.deepcopy(item) for item in authority.tune_launches
        ],
        "tune_launches_sha256": authority.tune_launches_sha256,
        "invalid_result_resolutions": [
            copy.deepcopy(item)
            for item in authority.invalid_result_resolutions
        ],
        "invalid_result_resolutions_sha256": (
            authority.invalid_result_resolutions_sha256
        ),
        "raw_input_prefixes": copy.deepcopy(
            authority.raw_input_prefixes
        ),
        "raw_input_prefixes_sha256": (
            authority.raw_input_prefixes_sha256
        ),
        "report_snapshot_sha256": authority.report_snapshot_sha256,
        "terminalized_tasks": terminalized_tasks,
        "terminalized_tasks_sha256": _sha256_bytes(
            _canonical_json_bytes(terminalized_tasks)
        ),
        "imported_tasks": imported_tasks,
        "imported_tasks_sha256": _sha256_bytes(
            _canonical_json_bytes(imported_tasks)
        ),
    }
    if set(reconciliation) != RECONCILIATION_KEYS:
        raise AssertionError("internal reconciliation schema drift")
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "campaign_id": expected_campaign_id,
        "selection_sha256": selection_sha256,
        "expected_state_sha256": expected_state_sha256,
        "expected_plan_sha256": expected_plan_sha256,
        "source_identity_before": dict(current_source_identity),
        "results_source_adoption": source_adoption,
        "timing_gate_sha256": timing_gate["gate_sha256"],
        "timing_source_set_sha256": timing_gate["source_set_sha256"],
        "downstream_required_gate": downstream_gate,
        "selections": normalized,
    }
    if set(ledger) != LEDGER_KEYS:
        raise AssertionError("internal ledger schema drift")
    ledger_sha256 = _sha256_bytes(_canonical_json_bytes(ledger))
    artifact_payload = {
        "schema_version": SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "artifact_kind": ARTIFACT_KIND,
        "transaction_id": transaction_id,
        "created_utc": created_utc,
        "ledger_sha256": ledger_sha256,
        "ledger": ledger,
        "pre_freeze_state": {
            "sha256": expected_state_sha256,
            "size_bytes": len(old_state_raw),
            "content_base64": base64.b64encode(old_state_raw).decode("ascii"),
        },
        "timing_gate": timing_gate,
        "state_reconciliation": reconciliation,
    }
    if set(artifact_payload) != ARTIFACT_KEYS:
        raise AssertionError("internal artifact schema drift")
    artifact_raw = _canonical_json_bytes(artifact_payload) + b"\n"
    artifact_sha256 = _sha256_bytes(artifact_raw)
    artifact_ref = {
        "filename": ARTIFACT_FILENAME,
        "sha256": artifact_sha256,
        "size_bytes": len(artifact_raw),
    }
    wrapper = {
        "schema_version": SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "ledger_sha256": ledger_sha256,
        "artifact": artifact_ref,
        "ledger": ledger,
    }
    if set(wrapper) != WRAPPER_KEYS:
        raise AssertionError("internal wrapper schema drift")

    patch = _selection_patch(normalized)
    new_plan = copy.deepcopy(plan)
    new_plan["selected_grad_lr_by_model_setting"] = patch[
        "selected_grad_lr_by_model_setting"
    ]
    new_plan["manual_lr_selection_freeze"] = wrapper
    try:
        runner.validate_structure(new_plan)
    except runner.PlanError as exc:
        raise FreezeError(
            f"formal plan with frozen LRs is invalid: {exc}"
        ) from exc
    new_plan_raw = _plan_json_bytes(new_plan)
    new_plan_sha256 = _sha256_bytes(new_plan_raw)

    frozen_state = copy.deepcopy(state)
    frozen_state["source_identity"] = dict(current_source_identity)
    _apply_state_reconciliation(
        frozen_state,
        authority=authority,
        reconciliation=reconciliation,
        artifact_sha256=artifact_sha256,
        timestamp=created_utc,
    )
    for selection in normalized:
        key = f"{selection['model']}/{selection['setting']}"
        group = frozen_state["tuning"][key]
        group["status"] = "SELECTED"
        group["selected_lr"] = float(selection["selected_lr"])
        group["selected_manifest"] = selection["selected_manifest"]
        group["selected_parameters"] = _selected_parameters(
            selection,
            ledger_sha256=ledger_sha256,
            selection_sha256=selection_sha256,
        )
        group["needs_user_action_reason"] = None
        group["manual_selection_decision_kind"] = selection["decision_kind"]
        group["manual_selection_evidence_set_sha256"] = selection[
            "evidence_set_sha256"
        ]
        group["attempt_count"] = authority.attempt_count_by_group[
            (selection["model"], selection["setting"])
        ]
    frozen_state["selected_lr_patch"] = patch
    frozen_state["manual_lr_selection_freeze"] = wrapper
    frozen_state["downstream_required_gate"] = downstream_gate
    frozen_state = _initialize_timed_formal(
        frozen_state,
        plan=plan,
        plan_raw=old_plan_raw,
        plan_path=plan_path,
        state_path=state_path,
        agent_path=agent_path,
        timestamp=created_utc,
        expected_gate=timing_gate,
    )
    # _initialize_formal is deliberately invoked while the old plan is still
    # the on-disk CAS target.
    _assert_file_sha(plan_path, expected_plan_sha256, "old plan")
    handoff = {
        "schema_version": SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "transaction_id": transaction_id,
        "phase": "state_frozen",
        "artifact_sha256": artifact_sha256,
        "ledger_sha256": ledger_sha256,
        "selection_sha256": selection_sha256,
        "expected_state_sha256": expected_state_sha256,
        "expected_plan_sha256": expected_plan_sha256,
        "formal_plan_sha256": new_plan_sha256,
    }
    if set(handoff) != HANDOFF_KEYS:
        raise AssertionError("internal handoff schema drift")
    frozen_state["manual_lr_handoff"] = handoff
    frozen_state["plan"]["pending_transition_sha256"] = new_plan_sha256
    frozen_state["status"] = "MANUAL_LR_HANDOFF_PENDING_PLAN"
    frozen_state["preflight"] = _invalidate_preflight(
        new_plan,
        reason=(
            "manual LR freeze changed the formal plan SHA; rerun the exact "
            "formal preflight before any launch"
        ),
    )
    frozen_state["blockers"] = [
        "manual LR plan transition is pending; resume only with the freeze tool"
    ]
    frozen_state["updated_utc"] = created_utc
    frozen_state_raw = _pretty_json_bytes(frozen_state)

    committed_state = copy.deepcopy(frozen_state)
    committed_state["plan"]["active_sha256"] = new_plan_sha256
    committed_state["plan"]["formal_sha256"] = new_plan_sha256
    committed_state["plan"]["formal_protocol_sha256"] = (
        base_campaign._protocol_sha256(new_plan)
    )
    committed_state["plan"]["pending_transition_sha256"] = None
    committed_state["manual_lr_handoff"]["phase"] = "committed"
    committed_state["manual_lr_state_reconciliation"]["phase"] = "committed"
    committed_state["status"] = "WAIT_PREFLIGHT"
    committed_state["preflight"] = _invalidate_preflight(
        new_plan,
        reason=(
            "manual LR freeze committed a new formal plan SHA; regenerate "
            "preflight before formal experiments"
        ),
    )
    committed_state["blockers"] = [
        "formal preflight is invalid for the newly frozen plan"
    ]
    committed_state["updated_utc"] = created_utc
    committed_state_raw = _pretty_json_bytes(committed_state)

    transaction = FreezeTransaction(
        transaction_id=transaction_id,
        created_utc=created_utc,
        plan_path=plan_path,
        state_path=state_path,
        selection_path=selection_path,
        artifact_path=_artifact_path(plan),
        campaign_id=expected_campaign_id,
        expected_plan_sha256=expected_plan_sha256,
        expected_state_sha256=expected_state_sha256,
        selection_sha256=selection_sha256,
        ledger_sha256=ledger_sha256,
        artifact_sha256=artifact_sha256,
        old_plan_raw=old_plan_raw,
        old_state_raw=old_state_raw,
        artifact_raw=artifact_raw,
        new_plan_raw=new_plan_raw,
        frozen_state_raw=frozen_state_raw,
        committed_state_raw=committed_state_raw,
        wrapper=wrapper,
        downstream_required_gate=downstream_gate,
    )
    _validate_transaction_payloads(transaction)
    return transaction


def _projection_from_ledger(
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    projected: list[dict[str, Any]] = []
    for selection in selections:
        evidence_value = selection.get("evidence")
        if not isinstance(evidence_value, list):
            raise FreezeError("ledger selection evidence must be a list")
        projected.append(
            {
                "model": selection.get("model"),
                "setting": selection.get("setting"),
                "selected_lr": selection.get("selected_lr"),
                "decision_kind": selection.get("decision_kind"),
                "evidence": [
                    {
                        "manifest": evidence.get("manifest"),
                        "grad_lr": evidence.get("grad_lr"),
                        "kl_wikitext2": evidence.get("kl_wikitext2"),
                    }
                    for evidence in evidence_value
                    if isinstance(evidence, Mapping)
                ],
                "rationale": selection.get("rationale"),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "selections": projected}


def _validate_ledger(ledger: Mapping[str, Any]) -> None:
    _require_exact_keys(ledger, LEDGER_KEYS, "freeze ledger")
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("scope_id") != SCOPE_ID
    ):
        raise FreezeError("freeze ledger schema/scope is invalid")
    _nonempty_string(ledger.get("campaign_id"), "freeze ledger campaign_id")
    for field in (
        "selection_sha256",
        "expected_state_sha256",
        "expected_plan_sha256",
        "timing_gate_sha256",
        "timing_source_set_sha256",
    ):
        _require_sha256(ledger.get(field), f"freeze ledger {field}")
    source = ledger.get("source_identity_before")
    if not isinstance(source, Mapping) or set(source) != set(
        SOURCE_IDENTITY_KEYS
    ):
        raise FreezeError("freeze ledger source_identity_before is invalid")
    for key in SOURCE_IDENTITY_KEYS:
        _require_sha256(source.get(key), f"freeze ledger source {key}")
    adoption = ledger.get("results_source_adoption")
    if adoption is not None:
        if not isinstance(adoption, Mapping):
            raise FreezeError("results_source_adoption must be null or an object")
        _require_exact_keys(
            adoption,
            {
                "kind",
                "expected_old_results_sha256",
                "accepted_current_results_sha256",
            },
            "results_source_adoption",
        )
        if adoption.get("kind") != SOURCE_ADOPTION_KIND:
            raise FreezeError("results_source_adoption kind is invalid")
        _require_sha256(
            adoption.get("expected_old_results_sha256"),
            "results_source_adoption old hash",
        )
        accepted = _require_sha256(
            adoption.get("accepted_current_results_sha256"),
            "results_source_adoption current hash",
        )
        if accepted != source["results_sha256"]:
            raise FreezeError(
                "results_source_adoption current hash disagrees with ledger source"
            )
    gate = ledger.get("downstream_required_gate")
    if gate != {
        "gate_id": GATE_ID,
        "required": True,
        "status": "required_before_formal_results_acceptance",
    }:
        raise FreezeError("freeze ledger downstream gate is invalid")
    selections = ledger.get("selections")
    if not isinstance(selections, list) or len(selections) != 15:
        raise FreezeError("freeze ledger must contain exactly 15 selections")
    observed: list[tuple[str, str]] = []
    override_keys: list[tuple[str, str]] = []
    for index, selection_value in enumerate(selections):
        name = f"freeze ledger selections[{index}]"
        if not isinstance(selection_value, Mapping):
            raise FreezeError(f"{name} must be an object")
        _require_exact_keys(
            selection_value, NORMALIZED_SELECTION_KEYS, name
        )
        model = _nonempty_string(selection_value.get("model"), f"{name}.model")
        setting = _nonempty_string(
            selection_value.get("setting"), f"{name}.setting"
        )
        observed.append((model, setting))
        selected_lr = _finite_nonnegative(
            selection_value.get("selected_lr"), f"{name}.selected_lr"
        )
        kind = selection_value.get("decision_kind")
        if kind not in DECISION_KINDS:
            raise FreezeError(f"{name}.decision_kind is invalid")
        _nonempty_string(selection_value.get("rationale"), f"{name}.rationale")
        evidence = selection_value.get("evidence")
        count = 1 if kind == "exact" else 2
        if not isinstance(evidence, list) or len(evidence) != count:
            raise FreezeError(
                f"{name}.evidence must contain exactly {count} item(s)"
            )
        lrs: list[float] = []
        for evidence_index, evidence_value in enumerate(evidence):
            evidence_name = f"{name}.evidence[{evidence_index}]"
            if not isinstance(evidence_value, Mapping):
                raise FreezeError(f"{evidence_name} must be an object")
            _require_exact_keys(
                evidence_value, NORMALIZED_EVIDENCE_KEYS, evidence_name
            )
            _nonempty_string(
                evidence_value.get("manifest"), f"{evidence_name}.manifest"
            )
            _nonempty_string(
                evidence_value.get("log"), f"{evidence_name}.log"
            )
            for hash_field in (
                "manifest_sha256",
                "log_sha256",
                "plan_sha256",
                "model_sha256",
                "runner_sha256",
                "executor_sha256",
                "numerical_source_sha256",
            ):
                _require_sha256(
                    evidence_value.get(hash_field),
                    f"{evidence_name}.{hash_field}",
                )
            _nonempty_string(
                evidence_value.get("execution_id"),
                f"{evidence_name}.execution_id",
            )
            _nonempty_string(
                evidence_value.get("run_id"), f"{evidence_name}.run_id"
            )
            attempt = evidence_value.get("attempt_index")
            if (
                type(attempt) is not int
                or attempt <= 0
                or attempt > MAX_ATTEMPTS
            ):
                raise FreezeError(f"{evidence_name}.attempt_index is invalid")
            lr = _finite_nonnegative(
                evidence_value.get("grad_lr"), f"{evidence_name}.grad_lr"
            )
            _finite_nonnegative(
                evidence_value.get("kl_wikitext2"),
                f"{evidence_name}.kl_wikitext2",
            )
            lrs.append(lr)
        if (
            selection_value.get("evidence_set_sha256")
            != _sha256_bytes(_canonical_json_bytes(evidence))
        ):
            raise FreezeError(f"{name}.evidence_set_sha256 mismatch")
        if kind == "exact":
            if (
                selection_value.get("selected_manifest")
                != evidence[0]["manifest"]
                or selected_lr != lrs[0]
            ):
                raise FreezeError(f"{name} exact decision linkage is invalid")
        else:
            override_keys.append((model, setting))
            if (
                selection_value.get("selected_manifest") is not None
                or not lrs[0] < selected_lr < lrs[1]
            ):
                raise FreezeError(f"{name} override linkage is invalid")
    if tuple(observed) != _matrix_keys():
        raise FreezeError("freeze ledger matrix order is invalid")
    if override_keys != [QWEN_OVERRIDE_KEY]:
        raise FreezeError("freeze ledger has an unreviewed override set")
    override = selections[_matrix_keys().index(QWEN_OVERRIDE_KEY)]
    if (
        float(override["selected_lr"]) != QWEN_OVERRIDE_LR
        or tuple(float(item["grad_lr"]) for item in override["evidence"])
        != QWEN_OVERRIDE_EVIDENCE_LRS
    ):
        raise FreezeError("freeze ledger reviewed Qwen override is invalid")
    projection = _projection_from_ledger(selections)
    if (
        _sha256_bytes(_canonical_json_bytes(projection))
        != ledger["selection_sha256"]
    ):
        raise FreezeError("freeze ledger selection_sha256 mismatch")


def _decode_exact_base64(value: Any, name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise FreezeError(f"{name} must be a non-empty base64 string")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise FreezeError(f"{name} is invalid base64") from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise FreezeError(f"{name} is not canonical base64")
    return decoded


def _validate_file_ref_payload(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise FreezeError(f"{name} must be an object")
    _require_exact_keys(value, FILE_REF_KEYS, name)
    absolute = _nonempty_string(
        value.get("absolute_path"), f"{name}.absolute_path"
    )
    relative_text = _nonempty_string(
        value.get("relative_path"), f"{name}.relative_path"
    )
    relative = Path(relative_text)
    if (
        not Path(absolute).is_absolute()
        or relative.is_absolute()
        or relative.as_posix() != relative_text
        or ".." in relative.parts
    ):
        raise FreezeError(f"{name} path identity is invalid")
    present = value.get("present")
    if type(present) is not bool:
        raise FreezeError(f"{name}.present must be boolean")
    if present:
        _require_sha256(value.get("sha256"), f"{name}.sha256")
        size = value.get("size_bytes")
        if type(size) is not int or size < 0:
            raise FreezeError(f"{name}.size_bytes is invalid")
    elif value.get("sha256") is not None or value.get("size_bytes") is not None:
        raise FreezeError(f"{name} absent file hash/size must be null")


def _validate_manual_resolution_payloads(
    values: Any,
) -> list[Mapping[str, Any]]:
    if (
        not isinstance(values, list)
        or len(values) > len(MANUAL_INVALID_REVIEWED_TARGETS)
    ):
        raise FreezeError(
            "freeze reconciliation exceeds the reviewed invalid resolutions"
        )
    previous_hash: str | None = None
    normalized: list[Mapping[str, Any]] = []
    for index, value in enumerate(values, 1):
        name = f"freeze reconciliation invalid resolution {index}"
        if not isinstance(value, Mapping):
            raise FreezeError(f"{name} must be an object")
        _require_exact_keys(value, MANUAL_INVALID_ENTRY_KEYS, name)
        reviewed = MANUAL_INVALID_REVIEWED_TARGETS.get(
            (
                value.get("model"),
                value.get("setting"),
                value.get("grad_lr"),
                value.get("attempt_index"),
            )
        )
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("seq") != index
            or value.get("prev_record_sha256") != previous_hash
            or value.get("reason_code") != MANUAL_INVALID_REASON_CODE
            or reviewed is None
            or value.get("phase") != "tune"
            or value.get("method") != "realq"
            or value.get("manifest_status") != "succeeded"
            or value.get("exit_code") != 0
            or value.get("numerical_source_changed_during_execution")
            is not True
            or value.get("numerical_source_sha256_at_start")
            != (
                reviewed or {}
            ).get("numerical_source_sha256_at_start")
            or value.get("numerical_source_sha256_at_end")
            != (
                reviewed or {}
            ).get("numerical_source_sha256_at_end")
            or value.get("expected_rejection_errors")
            != list(MANUAL_INVALID_REJECTION_ERRORS)
        ):
            raise FreezeError(f"{name} narrow authorization is invalid")
        for field in (
            "plan_sha256",
            "model_sha256",
            "runner_sha256",
            "executor_sha256",
            "numerical_source_sha256_at_start",
            "numerical_source_sha256_at_end",
        ):
            _require_sha256(value.get(field), f"{name}.{field}")
        _validate_file_ref_payload(value.get("manifest"), f"{name}.manifest")
        _validate_file_ref_payload(value.get("log"), f"{name}.log")
        authorization = value.get("authorization")
        if not isinstance(authorization, Mapping):
            raise FreezeError(f"{name}.authorization must be an object")
        _require_exact_keys(
            authorization,
            MANUAL_INVALID_AUTHORIZATION_KEYS,
            f"{name}.authorization",
        )
        if authorization.get("kind") != (
            "user_authorized_manual_invalid_result_v1"
        ):
            raise FreezeError(f"{name}.authorization.kind is invalid")
        _nonempty_string(
            authorization.get("authorized_by"),
            f"{name}.authorization.authorized_by",
        )
        _nonempty_string(
            authorization.get("rationale"),
            f"{name}.authorization.rationale",
        )
        event_present = value.get("launch_event_present")
        event = value.get("launch_event")
        if event_present is False:
            if event is not None:
                raise FreezeError(f"{name}.launch_event must be null")
        elif event_present is True:
            if not isinstance(event, Mapping):
                raise FreezeError(f"{name}.launch_event must be an object")
            _require_exact_keys(
                event,
                MANUAL_INVALID_LAUNCH_EVENT_KEYS,
                f"{name}.launch_event",
            )
            if type(event.get("seq")) is not int or event["seq"] <= 0:
                raise FreezeError(f"{name}.launch_event.seq is invalid")
            _require_sha256(
                event.get("line_sha256"),
                f"{name}.launch_event.line_sha256",
            )
            _nonempty_string(
                event.get("task_id"), f"{name}.launch_event.task_id"
            )
        else:
            raise FreezeError(f"{name}.launch_event_present is invalid")
        record_sha = _require_sha256(
            value.get("record_sha256"), f"{name}.record_sha256"
        )
        unsigned = dict(value)
        unsigned.pop("record_sha256")
        if _sha256_bytes(_canonical_json_bytes(unsigned)) != record_sha:
            raise FreezeError(f"{name}.record_sha256 is invalid")
        previous_hash = record_sha
        normalized.append(value)
    reviewed_keys = [
        (
            value["model"],
            value["setting"],
            value["grad_lr"],
            value["attempt_index"],
        )
        for value in normalized
    ]
    if len(reviewed_keys) != len(set(reviewed_keys)):
        raise FreezeError(
            "freeze reconciliation repeats a reviewed invalid target"
        )
    return normalized


def _validate_reconciliation_payload(
    value: Any,
    *,
    old_state: Mapping[str, Any],
    created_utc: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FreezeError("freeze artifact state_reconciliation must be an object")
    _require_exact_keys(
        value, RECONCILIATION_KEYS, "freeze artifact state_reconciliation"
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("policy_id") != RECONCILIATION_POLICY_ID
    ):
        raise FreezeError("freeze reconciliation identity is invalid")
    accepted = value.get("accepted_tune_results")
    if not isinstance(accepted, list) or not accepted:
        raise FreezeError("freeze reconciliation accepted set is invalid")
    previous_order: tuple[int, int, float, str] | None = None
    accepted_manifests: set[str] = set()
    accepted_execution_ids: set[str] = set()
    accepted_run_ids: set[str] = set()
    point_by_manifest: dict[str, Mapping[str, Any]] = {}
    for index, point in enumerate(accepted):
        name = f"freeze reconciliation accepted_tune_results[{index}]"
        if not isinstance(point, Mapping):
            raise FreezeError(f"{name} must be an object")
        _require_exact_keys(point, RECONCILIATION_POINT_KEYS, name)
        key = (point.get("model"), point.get("setting"))
        if key not in _matrix_keys():
            raise FreezeError(f"{name} has unknown model/setting")
        matrix_index = _matrix_keys().index(key)
        lr = _finite_nonnegative(point.get("grad_lr"), f"{name}.grad_lr")
        manifest = _nonempty_string(
            point.get("manifest"), f"{name}.manifest"
        )
        order = (
            matrix_index,
            0,
            lr,
            manifest,
        )
        if previous_order is not None and order <= previous_order:
            raise FreezeError("freeze reconciliation accepted order is invalid")
        previous_order = order
        for field in (
            "manifest_sha256",
            "log_sha256",
            "plan_sha256",
            "model_sha256",
            "runner_sha256",
            "executor_sha256",
            "numerical_source_sha256",
            "config_sha256",
        ):
            _require_sha256(point.get(field), f"{name}.{field}")
        world_size = point.get("world_size")
        if type(world_size) is not int or world_size <= 0:
            raise FreezeError(f"{name}.world_size is invalid")
        attempt = point.get("attempt_index")
        if type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
            raise FreezeError(f"{name}.attempt_index is invalid")
        _finite_nonnegative(
            point.get("kl_wikitext2"), f"{name}.kl_wikitext2"
        )
        _finite_nonnegative(
            point.get("ppl_wikitext2"), f"{name}.ppl_wikitext2"
        )
        execution_id = _nonempty_string(
            point.get("execution_id"), f"{name}.execution_id"
        )
        run_id = _nonempty_string(point.get("run_id"), f"{name}.run_id")
        if (
            manifest in accepted_manifests
            or execution_id in accepted_execution_ids
            or run_id in accepted_run_ids
        ):
            raise FreezeError("freeze reconciliation accepted identity duplicates")
        accepted_manifests.add(manifest)
        accepted_execution_ids.add(execution_id)
        accepted_run_ids.add(run_id)
        point_by_manifest[manifest] = point
    accepted_sha = _sha256_bytes(_canonical_json_bytes(accepted))
    if value.get("accepted_active_set_sha256") != accepted_sha:
        raise FreezeError("freeze reconciliation accepted set hash mismatch")

    counts = value.get("group_attempt_counts")
    if not isinstance(counts, list) or len(counts) != len(_matrix_keys()):
        raise FreezeError("freeze reconciliation group counts are invalid")
    counts_by_group: dict[tuple[str, str], int] = {}
    for index, (item, expected_key) in enumerate(zip(counts, _matrix_keys())):
        name = f"freeze reconciliation group_attempt_counts[{index}]"
        if not isinstance(item, Mapping):
            raise FreezeError(f"{name} must be an object")
        _require_exact_keys(item, RECONCILIATION_COUNT_KEYS, name)
        if (item.get("model"), item.get("setting")) != expected_key:
            raise FreezeError(f"{name} order is invalid")
        count = item.get("attempt_count")
        if type(count) is not int or not 1 <= count <= MAX_ATTEMPTS:
            raise FreezeError(f"{name}.attempt_count is invalid")
        counts_by_group[expected_key] = count
    counts_sha = _sha256_bytes(_canonical_json_bytes(counts))
    if value.get("group_attempt_counts_sha256") != counts_sha:
        raise FreezeError("freeze reconciliation group count hash mismatch")

    launches = value.get("tune_launches")
    if not isinstance(launches, list):
        raise FreezeError("freeze reconciliation tune_launches must be a list")
    previous_launch_order: tuple[int, float, str] | None = None
    launch_count_by_group = {key: 0 for key in _matrix_keys()}
    manual_invalid_manifests: set[str] = set()
    for index, launch in enumerate(launches):
        name = f"freeze reconciliation tune_launches[{index}]"
        if not isinstance(launch, Mapping):
            raise FreezeError(f"{name} must be an object")
        _require_exact_keys(launch, TUNE_LAUNCH_KEYS, name)
        _validate_file_ref_payload(launch.get("manifest"), f"{name}.manifest")
        _validate_file_ref_payload(launch.get("log"), f"{name}.log")
        manifest_ref = launch["manifest"]
        if manifest_ref.get("present") is not True:
            raise FreezeError(f"{name}.manifest must be present")
        key = (launch.get("model"), launch.get("setting"))
        if key not in launch_count_by_group:
            raise FreezeError(f"{name} has an unknown group")
        lr = _finite_nonnegative(launch.get("grad_lr"), f"{name}.grad_lr")
        order = (
            _matrix_keys().index(key),
            lr,
            str(manifest_ref["absolute_path"]),
        )
        if previous_launch_order is not None and order <= previous_launch_order:
            raise FreezeError("freeze reconciliation tune launch order is invalid")
        previous_launch_order = order
        launch_count_by_group[key] += 1
        classification = launch.get("classification")
        if classification not in {
            "accepted",
            "informational",
            "retired",
            "manual_invalid_result",
        }:
            raise FreezeError(f"{name}.classification is invalid")
        errors = launch.get("rejection_errors")
        if not isinstance(errors, list) or not all(
            isinstance(item, str) and item for item in errors
        ):
            raise FreezeError(f"{name}.rejection_errors is invalid")
        if classification == "manual_invalid_result":
            if errors != list(MANUAL_INVALID_REJECTION_ERRORS):
                raise FreezeError(f"{name} manual-invalid errors are invalid")
            manual_invalid_manifests.add(manifest_ref["absolute_path"])
        elif errors:
            raise FreezeError(f"{name} non-invalid launch has rejection errors")
        for field in (
            "plan_sha256",
            "model_sha256",
            "runner_sha256",
            "executor_sha256",
            "numerical_source_sha256_at_start",
        ):
            _require_sha256(launch.get(field), f"{name}.{field}")
        _optional_sha256(
            launch.get("numerical_source_sha256_at_end"),
            f"{name}.numerical_source_sha256_at_end",
        )
        if type(launch.get("context_valid")) is not bool or type(
            launch.get("numerical_source_changed_during_execution")
        ) is not bool:
            raise FreezeError(f"{name} boolean identity fields are invalid")
        attempt = launch.get("attempt_index")
        if type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
            raise FreezeError(f"{name}.attempt_index is invalid")
    if launch_count_by_group != counts_by_group:
        raise FreezeError("freeze reconciliation launch/count projection mismatch")
    launches_sha = _sha256_bytes(_canonical_json_bytes(launches))
    if value.get("tune_launches_sha256") != launches_sha:
        raise FreezeError("freeze reconciliation tune launch hash mismatch")

    resolutions = _validate_manual_resolution_payloads(
        value.get("invalid_result_resolutions")
    )
    resolution_manifests = {
        item["manifest"]["absolute_path"] for item in resolutions
    }
    if resolution_manifests != manual_invalid_manifests:
        raise FreezeError(
            "freeze reconciliation manual invalid launch/resolution mismatch"
        )
    resolutions_sha = _sha256_bytes(_canonical_json_bytes(resolutions))
    if value.get("invalid_result_resolutions_sha256") != resolutions_sha:
        raise FreezeError("freeze reconciliation invalid resolution hash mismatch")

    prefixes = value.get("raw_input_prefixes")
    if not isinstance(prefixes, Mapping):
        raise FreezeError("freeze reconciliation raw prefixes must be an object")
    _require_exact_keys(prefixes, RAW_INPUT_PREFIX_KEYS, "raw input prefixes")
    fixed_paths = {
        "retirement_ledger": str(
            getattr(
                results,
                "RETIREMENT_LEDGER_RELATIVE_PATH",
                "_campaign/retirements.jsonl",
            )
        ),
        "campaign_events": str(
            getattr(
                results,
                "CAMPAIGN_EVENTS_RELATIVE_PATH",
                "_campaign/events.jsonl",
            )
        ),
        "manual_invalid_results": MANUAL_INVALID_LEDGER_RELATIVE_PATH,
    }
    for key, expected_path in fixed_paths.items():
        ref = prefixes.get(key)
        if not isinstance(ref, Mapping):
            raise FreezeError(f"raw input prefix {key} must be an object")
        _require_exact_keys(ref, RAW_PREFIX_REF_KEYS, f"raw prefix {key}")
        if ref.get("relative_path") != expected_path:
            raise FreezeError(f"raw input prefix {key} path is invalid")
        present = ref.get("present")
        if present is True:
            _require_sha256(
                ref.get("prefix_sha256"), f"raw prefix {key}.prefix_sha256"
            )
            size = ref.get("prefix_size_bytes")
            if type(size) is not int or size < 0:
                raise FreezeError(f"raw prefix {key} size is invalid")
        elif present is False:
            if (
                ref.get("prefix_sha256") is not None
                or ref.get("prefix_size_bytes") is not None
            ):
                raise FreezeError(f"absent raw prefix {key} hash/size is invalid")
        else:
            raise FreezeError(f"raw prefix {key}.present is invalid")
    prefixes_sha = _sha256_bytes(_canonical_json_bytes(prefixes))
    if value.get("raw_input_prefixes_sha256") != prefixes_sha:
        raise FreezeError("freeze reconciliation raw prefix hash mismatch")
    snapshot = {
        "accepted_active_set_sha256": accepted_sha,
        "group_attempt_counts_sha256": counts_sha,
        "tune_launches_sha256": launches_sha,
        "invalid_result_resolutions_sha256": resolutions_sha,
        "raw_input_prefixes_sha256": prefixes_sha,
    }
    if value.get("report_snapshot_sha256") != _sha256_bytes(
        _canonical_json_bytes(snapshot)
    ):
        raise FreezeError("freeze reconciliation report snapshot hash mismatch")

    terminalized = value.get("terminalized_tasks")
    if not isinstance(terminalized, list):
        raise FreezeError("freeze reconciliation terminalized tasks are invalid")
    expected_terminalized = _derive_terminalized_tasks(old_state, None)  # type: ignore[arg-type]
    if terminalized != expected_terminalized:
        raise FreezeError(
            "freeze reconciliation terminalized tasks differ from old state"
        )
    if value.get("terminalized_tasks_sha256") != _sha256_bytes(
        _canonical_json_bytes(terminalized)
    ):
        raise FreezeError("freeze reconciliation terminalized task hash mismatch")

    imported = value.get("imported_tasks")
    if not isinstance(imported, list) or len(imported) != len(accepted):
        raise FreezeError("freeze reconciliation imported task set is invalid")
    imported_manifests: set[str] = set()
    for index, task in enumerate(imported):
        name = f"freeze reconciliation imported_tasks[{index}]"
        if not isinstance(task, Mapping):
            raise FreezeError(f"{name} must be an object")
        _require_exact_keys(task, IMPORTED_TASK_KEYS, name)
        manifest = _nonempty_string(task.get("manifest"), f"{name}.manifest")
        point = point_by_manifest.get(manifest)
        if point is None or manifest in imported_manifests:
            raise FreezeError(f"{name} does not map one-to-one to accepted result")
        imported_manifests.add(manifest)
        spec = task.get("spec")
        metric = task.get("metric")
        if not isinstance(spec, Mapping) or not isinstance(metric, Mapping):
            raise FreezeError(f"{name} spec/metric must be objects")
        _require_exact_keys(spec, IMPORTED_TASK_SPEC_KEYS, f"{name}.spec")
        _require_exact_keys(metric, IMPORTED_TASK_METRIC_KEYS, f"{name}.metric")
        identities = metric.get("identities")
        if not isinstance(identities, Mapping):
            raise FreezeError(f"{name}.metric.identities must be an object")
        _require_exact_keys(
            identities,
            IMPORTED_TASK_IDENTITY_KEYS,
            f"{name}.metric.identities",
        )
        expected_identity = {
            key: point[key] for key in IMPORTED_TASK_IDENTITY_KEYS
        }
        if (
            task.get("task_id")
            != "reconciled_tune:" + point["manifest_sha256"]
            or task.get("status") != "SUCCEEDED"
            or task.get("manifest") != point["manifest"]
            or task.get("log") != point["log"]
            or task.get("output_dir") != str(Path(point["manifest"]).parent)
            or task.get("attempt_index") != point["attempt_index"]
            or task.get("gpu_count") != point["world_size"]
            or task.get("exit_code") != 0
            or task.get("executor_pid") is not None
            or task.get("gpus") != []
            or task.get("plan_compatible") is not True
            or task.get("imported") is not False
            or task.get("imported_by_manual_lr_freeze") is not True
            or task.get("created_utc") != created_utc
            or task.get("finished_utc") != created_utc
            or spec
            != {
                "kind": "reconciled_realq_tune_result",
                "method": "realq",
                "phase": "tune",
                "target_phase": None,
                "model": point["model"],
                "setting": point["setting"],
                "grad_lr": point["grad_lr"],
                "overrides": {},
                "profile_level": None,
                "static_profile_level": None,
                "generation": None,
                "purpose": "manual_lr_freeze_authoritative_import",
            }
            or metric.get("kl") != point["kl_wikitext2"]
            or metric.get("ppl") != point["ppl_wikitext2"]
            or metric.get("grad_lr") != point["grad_lr"]
            or dict(identities) != expected_identity
        ):
            raise FreezeError(f"{name} authoritative content is invalid")
    if imported_manifests != accepted_manifests:
        raise FreezeError("freeze reconciliation imported coverage is incomplete")
    if value.get("imported_tasks_sha256") != _sha256_bytes(
        _canonical_json_bytes(imported)
    ):
        raise FreezeError("freeze reconciliation imported task hash mismatch")
    return value


def _validate_transaction_payloads(transaction: FreezeTransaction) -> None:
    if transaction.expected_plan_sha256 != _sha256_bytes(
        transaction.old_plan_raw
    ):
        raise FreezeError("transaction old plan payload hash mismatch")
    if transaction.expected_state_sha256 != _sha256_bytes(
        transaction.old_state_raw
    ):
        raise FreezeError("transaction old state payload hash mismatch")
    artifact_value = _parse_json_bytes(
        transaction.artifact_raw, "freeze artifact"
    )
    if not isinstance(artifact_value, Mapping):
        raise FreezeError("freeze artifact must be an object")
    if transaction.artifact_raw != _canonical_json_bytes(artifact_value) + b"\n":
        raise FreezeError("freeze artifact bytes are not canonical JSON + LF")
    _require_exact_keys(artifact_value, ARTIFACT_KEYS, "freeze artifact")
    if (
        artifact_value.get("schema_version") != SCHEMA_VERSION
        or artifact_value.get("scope_id") != SCOPE_ID
        or artifact_value.get("artifact_kind") != ARTIFACT_KIND
        or artifact_value.get("transaction_id") != transaction.transaction_id
        or artifact_value.get("created_utc") != transaction.created_utc
    ):
        raise FreezeError("freeze artifact identity is invalid")
    ledger = artifact_value.get("ledger")
    if not isinstance(ledger, Mapping):
        raise FreezeError("freeze artifact ledger must be an object")
    _validate_ledger(ledger)
    ledger_sha = _sha256_bytes(_canonical_json_bytes(ledger))
    if (
        artifact_value.get("ledger_sha256") != ledger_sha
        or transaction.ledger_sha256 != ledger_sha
        or ledger.get("campaign_id") != transaction.campaign_id
        or ledger.get("selection_sha256") != transaction.selection_sha256
        or ledger.get("expected_plan_sha256")
        != transaction.expected_plan_sha256
        or ledger.get("expected_state_sha256")
        != transaction.expected_state_sha256
    ):
        raise FreezeError("freeze artifact ledger SHA256 mismatch")
    if (
        transaction.artifact_sha256
        != _sha256_bytes(transaction.artifact_raw)
    ):
        raise FreezeError("freeze artifact external SHA256 mismatch")
    pre_state = artifact_value.get("pre_freeze_state")
    if not isinstance(pre_state, Mapping):
        raise FreezeError("freeze artifact pre_freeze_state must be an object")
    _require_exact_keys(
        pre_state, PRE_FREEZE_STATE_KEYS, "freeze artifact pre_freeze_state"
    )
    decoded_state = _decode_exact_base64(
        pre_state.get("content_base64"),
        "freeze artifact pre_freeze_state.content_base64",
    )
    if (
        decoded_state != transaction.old_state_raw
        or pre_state.get("size_bytes") != len(decoded_state)
        or pre_state.get("sha256") != _sha256_bytes(decoded_state)
        or pre_state.get("sha256") != ledger["expected_state_sha256"]
    ):
        raise FreezeError("freeze artifact pre-freeze state linkage is invalid")
    old_state = _parse_json_bytes(decoded_state, "freeze artifact old state")
    if not isinstance(old_state, Mapping):
        raise FreezeError("freeze artifact old state must be an object")
    if old_state.get("campaign_id") != ledger["campaign_id"]:
        raise FreezeError("freeze artifact campaign_id linkage is invalid")
    reconciliation = _validate_reconciliation_payload(
        artifact_value.get("state_reconciliation"),
        old_state=old_state,
        created_utc=transaction.created_utc,
    )
    old_state_plan = old_state.get("plan")
    if (
        not isinstance(old_state_plan, Mapping)
        or old_state_plan.get("tune_sha256")
        != ledger["expected_plan_sha256"]
        or old_state_plan.get("active_sha256")
        != ledger["expected_plan_sha256"]
        or old_state_plan.get("pending_transition_sha256") is not None
        or old_state_plan.get("formal_sha256") is not None
    ):
        raise FreezeError("freeze artifact old state plan linkage is invalid")
    old_source = old_state.get("source_identity")
    new_source = ledger["source_identity_before"]
    adoption = ledger["results_source_adoption"]
    if not isinstance(old_source, Mapping):
        raise FreezeError("freeze artifact old source identity is missing")
    if adoption is None:
        if dict(old_source) != dict(new_source):
            raise FreezeError("unrecorded source adoption in freeze artifact")
    else:
        for key in SOURCE_IDENTITY_KEYS:
            if key == "results_sha256":
                if (
                    old_source.get(key)
                    != adoption["expected_old_results_sha256"]
                    or new_source.get(key)
                    != adoption["accepted_current_results_sha256"]
                ):
                    raise FreezeError(
                        "results source adoption linkage is invalid"
                    )
            elif old_source.get(key) != new_source.get(key):
                raise FreezeError(
                    "results adoption changed a non-results source hash"
                )
    timing_gate = artifact_value.get("timing_gate")
    if not isinstance(timing_gate, Mapping):
        raise FreezeError("freeze artifact timing_gate must be an object")
    unsigned_gate = dict(timing_gate)
    observed_gate_sha = unsigned_gate.pop("gate_sha256", None)
    if (
        observed_gate_sha
        != _sha256_bytes(_canonical_json_bytes(unsigned_gate))
        or observed_gate_sha != ledger["timing_gate_sha256"]
        or timing_gate.get("source_set_sha256")
        != ledger["timing_source_set_sha256"]
    ):
        raise FreezeError("freeze artifact timing gate linkage is invalid")

    new_plan = _parse_json_bytes(transaction.new_plan_raw, "formal plan")
    old_plan = _parse_json_bytes(transaction.old_plan_raw, "old plan")
    if not isinstance(new_plan, dict) or not isinstance(old_plan, dict):
        raise FreezeError("transaction plans must be objects")
    wrapper = new_plan.get("manual_lr_selection_freeze")
    if not isinstance(wrapper, Mapping):
        raise FreezeError("formal plan freeze wrapper is missing")
    _require_exact_keys(wrapper, WRAPPER_KEYS, "formal plan freeze wrapper")
    if (
        wrapper.get("schema_version") != SCHEMA_VERSION
        or wrapper.get("scope_id") != SCOPE_ID
        or wrapper.get("ledger_sha256") != ledger_sha
        or wrapper.get("ledger") != ledger
        or wrapper.get("artifact")
        != {
            "filename": ARTIFACT_FILENAME,
            "sha256": transaction.artifact_sha256,
            "size_bytes": len(transaction.artifact_raw),
        }
    ):
        raise FreezeError("formal plan freeze wrapper linkage is invalid")
    derived_plan = copy.deepcopy(old_plan)
    derived_plan["selected_grad_lr_by_model_setting"] = _selection_patch(
        ledger["selections"]
    )["selected_grad_lr_by_model_setting"]
    derived_plan["manual_lr_selection_freeze"] = dict(wrapper)
    if _plan_json_bytes(derived_plan) != transaction.new_plan_raw:
        raise FreezeError("formal plan is not the deterministic freeze patch")
    try:
        runner.validate_structure(new_plan)
    except runner.PlanError as exc:
        raise FreezeError(f"transaction formal plan is invalid: {exc}") from exc

    frozen = _parse_json_bytes(
        transaction.frozen_state_raw, "frozen state"
    )
    committed = _parse_json_bytes(
        transaction.committed_state_raw, "committed state"
    )
    if not isinstance(frozen, Mapping) or not isinstance(committed, Mapping):
        raise FreezeError("transaction states must be objects")
    expected_timed_source = timed_campaign._formal_source_identity(
        new_source, timing_gate
    )
    for label, state_value, phase in (
        ("frozen", frozen, "state_frozen"),
        ("committed", committed, "committed"),
    ):
        if (
            state_value.get("campaign_id") != ledger["campaign_id"]
            or state_value.get("manual_lr_selection_freeze") != wrapper
            or state_value.get("downstream_required_gate")
            != ledger["downstream_required_gate"]
            or state_value.get("selected_lr_patch")
            != _selection_patch(ledger["selections"])
            or state_value.get("source_identity") != expected_timed_source
        ):
            raise FreezeError(f"{label} state freeze linkage is invalid")
        state_reconciliation = state_value.get(
            "manual_lr_state_reconciliation"
        )
        if not isinstance(state_reconciliation, Mapping):
            raise FreezeError(
                f"{label} state reconciliation reference is missing"
            )
        expected_reconciliation_ref = _state_reconciliation_ref(
            reconciliation,
            artifact_sha256=transaction.artifact_sha256,
            phase=phase,
        )
        if state_reconciliation != expected_reconciliation_ref:
            raise FreezeError(
                f"{label} state reconciliation reference is invalid"
            )
        formal = state_value.get("formal")
        if (
            not isinstance(formal, Mapping)
            or formal.get("initialized") is not True
            or formal.get("timing_gate") != timing_gate
        ):
            raise FreezeError(f"{label} state formal timing gate is invalid")
        handoff = state_value.get("manual_lr_handoff")
        if not isinstance(handoff, Mapping):
            raise FreezeError(f"{label} state handoff is missing")
        _require_exact_keys(handoff, HANDOFF_KEYS, f"{label} state handoff")
        if handoff != {
            "schema_version": SCHEMA_VERSION,
            "scope_id": SCOPE_ID,
            "transaction_id": transaction.transaction_id,
            "phase": phase,
            "artifact_sha256": transaction.artifact_sha256,
            "ledger_sha256": ledger_sha,
            "selection_sha256": ledger["selection_sha256"],
            "expected_state_sha256": ledger["expected_state_sha256"],
            "expected_plan_sha256": ledger["expected_plan_sha256"],
            "formal_plan_sha256": transaction.new_plan_sha256,
        }:
            raise FreezeError(f"{label} state handoff fields are invalid")
        preflight = state_value.get("preflight")
        if not isinstance(preflight, Mapping) or preflight.get("valid") is not False:
            raise FreezeError(f"{label} state preflight must be invalid")
        for selection in ledger["selections"]:
            key = f"{selection['model']}/{selection['setting']}"
            group = state_value.get("tuning", {}).get(key)
            if (
                not isinstance(group, Mapping)
                or group.get("status") != "SELECTED"
                or float(group.get("selected_lr", -1))
                != float(selection["selected_lr"])
                or group.get("selected_manifest")
                != selection["selected_manifest"]
                or group.get("attempt_count")
                != next(
                    item["attempt_count"]
                    for item in reconciliation["group_attempt_counts"]
                    if item["model"] == selection["model"]
                    and item["setting"] == selection["setting"]
                )
            ):
                raise FreezeError(f"{label} state tuning selection mismatch: {key}")
        tasks = state_value.get("tasks")
        if not isinstance(tasks, Mapping):
            raise FreezeError(f"{label} state tasks must be an object")
        for transition in reconciliation["terminalized_tasks"]:
            task = tasks.get(transition["task_id"])
            if (
                not isinstance(task, Mapping)
                or task.get("status") != transition["reconciled_status"]
                or task.get("reconciled_previous_status")
                != transition["previous_status"]
                or task.get("reconciliation_reason")
                != transition["reason"]
                or task.get("executor_pid") is not None
                or task.get("gpus") != []
                or any(
                    field in task
                    for field in (
                        "executor_start_ticks",
                        "executor_session_id",
                        "executor_process_group_id",
                        "executor_cli",
                        "rendezvous",
                    )
                )
            ):
                raise FreezeError(
                    f"{label} state terminalized task is invalid: "
                    f"{transition['task_id']}"
                )
        imported_ids = {
            task["task_id"] for task in reconciliation["imported_tasks"]
        }
        for imported_task in reconciliation["imported_tasks"]:
            if tasks.get(imported_task["task_id"]) != imported_task:
                raise FreezeError(
                    f"{label} state imported task differs from artifact: "
                    f"{imported_task['task_id']}"
                )
        accepted_manifests = {
            point["manifest"]
            for point in reconciliation["accepted_tune_results"]
        }
        succeeded_by_manifest = {
            manifest: [
                task_id
                for task_id, task in tasks.items()
                if isinstance(task, Mapping)
                and task.get("status") == "SUCCEEDED"
                and _task_manifest_key(task) == manifest
            ]
            for manifest in accepted_manifests
        }
        if any(
            len(task_ids) != 1 or task_ids[0] not in imported_ids
            for task_ids in succeeded_by_manifest.values()
        ):
            raise FreezeError(
                f"{label} state accepted manifests are not covered by exactly "
                "one authoritative imported task"
            )
    frozen_plan = frozen.get("plan")
    committed_plan = committed.get("plan")
    if (
        not isinstance(frozen_plan, Mapping)
        or frozen_plan.get("active_sha256")
        != ledger["expected_plan_sha256"]
        or frozen_plan.get("formal_sha256") is not None
        or frozen_plan.get("pending_transition_sha256")
        != transaction.new_plan_sha256
    ):
        raise FreezeError("frozen state pending plan transition is invalid")
    if (
        not isinstance(committed_plan, Mapping)
        or committed_plan.get("tune_sha256")
        != ledger["expected_plan_sha256"]
        or committed_plan.get("active_sha256")
        != transaction.new_plan_sha256
        or committed_plan.get("formal_sha256")
        != transaction.new_plan_sha256
        or committed_plan.get("pending_transition_sha256") is not None
        or committed.get("status") != "WAIT_PREFLIGHT"
    ):
        raise FreezeError("committed state plan transition is invalid")


def _journal_path(state_path: Path) -> Path:
    return state_path.with_name(state_path.name + JOURNAL_SUFFIX)


def _transaction_from_journal(
    journal_value: Mapping[str, Any],
    *,
    plan_path: Path,
    state_path: Path,
    selection_path: Path,
    expected_plan_sha256: str,
    expected_state_sha256: str,
    expected_campaign_id: str,
    expected_old_results_sha256: str | None,
) -> FreezeTransaction:
    _require_exact_keys(journal_value, JOURNAL_KEYS, "freeze journal")
    if (
        journal_value.get("schema_version") != SCHEMA_VERSION
        or journal_value.get("scope_id") != SCOPE_ID
        or journal_value.get("phase") not in JOURNAL_PHASES
        or journal_value.get("plan_path") != str(plan_path)
        or journal_value.get("state_path") != str(state_path)
        or journal_value.get("selection_path") != str(selection_path)
        or journal_value.get("expected_plan_sha256")
        != expected_plan_sha256
        or journal_value.get("expected_state_sha256")
        != expected_state_sha256
        or journal_value.get("expected_campaign_id")
        != expected_campaign_id
    ):
        raise FreezeError("freeze journal does not match the requested CAS")
    payloads = journal_value.get("payloads")
    if not isinstance(payloads, Mapping):
        raise FreezeError("freeze journal payloads must be an object")
    _require_exact_keys(payloads, JOURNAL_PAYLOAD_KEYS, "freeze journal payloads")
    decoded = {
        key: _decode_exact_base64(value, f"freeze journal payloads.{key}")
        for key, value in payloads.items()
    }
    new_plan_value = _parse_json_bytes(
        decoded["new_plan_base64"], "journal formal plan"
    )
    if not isinstance(new_plan_value, Mapping):
        raise FreezeError("journal formal plan must be an object")
    wrapper = new_plan_value.get("manual_lr_selection_freeze")
    if not isinstance(wrapper, dict):
        raise FreezeError("journal formal plan freeze wrapper is missing")
    ledger = wrapper.get("ledger")
    if not isinstance(ledger, Mapping):
        raise FreezeError("journal freeze ledger is missing")
    downstream = ledger.get("downstream_required_gate")
    if not isinstance(downstream, dict):
        raise FreezeError("journal downstream gate is missing")
    transaction = FreezeTransaction(
        transaction_id=str(journal_value.get("transaction_id")),
        created_utc=str(journal_value.get("created_utc")),
        plan_path=plan_path,
        state_path=state_path,
        selection_path=selection_path,
        artifact_path=Path(str(journal_value.get("artifact_path"))),
        campaign_id=expected_campaign_id,
        expected_plan_sha256=expected_plan_sha256,
        expected_state_sha256=expected_state_sha256,
        selection_sha256=str(journal_value.get("selection_sha256")),
        ledger_sha256=str(journal_value.get("ledger_sha256")),
        artifact_sha256=str(journal_value.get("artifact_sha256")),
        old_plan_raw=decoded["old_plan_base64"],
        old_state_raw=decoded["old_state_base64"],
        artifact_raw=decoded["artifact_base64"],
        new_plan_raw=decoded["new_plan_base64"],
        frozen_state_raw=decoded["frozen_state_base64"],
        committed_state_raw=decoded["committed_state_base64"],
        wrapper=wrapper,
        downstream_required_gate=downstream,
    )
    expected_transaction_id = _transaction_id(
        campaign_id=expected_campaign_id,
        expected_plan_sha256=expected_plan_sha256,
        expected_state_sha256=expected_state_sha256,
        selection_sha256=transaction.selection_sha256,
    )
    if transaction.transaction_id != expected_transaction_id:
        raise FreezeError("freeze journal transaction_id mismatch")
    if (
        journal_value.get("artifact_path") != str(transaction.artifact_path)
        or transaction.artifact_path
        != _artifact_path(
            _parse_json_bytes(transaction.old_plan_raw, "journal old plan")
        )
        or journal_value.get("artifact_size_bytes")
        != len(transaction.artifact_raw)
        or journal_value.get("new_plan_sha256")
        != transaction.new_plan_sha256
        or journal_value.get("frozen_state_sha256")
        != transaction.frozen_state_sha256
        or journal_value.get("committed_state_sha256")
        != transaction.committed_state_sha256
        or journal_value.get("artifact_sha256")
        != _sha256_bytes(transaction.artifact_raw)
    ):
        raise FreezeError("freeze journal payload hash/path linkage is invalid")
    _validate_transaction_payloads(transaction)
    _, current_selection_sha = _load_selection(selection_path)
    if (
        current_selection_sha != transaction.selection_sha256
        or journal_value.get("selection_sha256")
        != transaction.selection_sha256
        or journal_value.get("ledger_sha256")
        != transaction.ledger_sha256
    ):
        raise FreezeError("freeze journal selection/ledger hash mismatch")
    adoption = transaction.wrapper["ledger"]["results_source_adoption"]
    if adoption is None:
        if expected_old_results_sha256 is not None:
            raise FreezeError(
                "unexpected --expected-old-results-sha256 for this journal"
            )
    elif (
        expected_old_results_sha256
        != adoption["expected_old_results_sha256"]
    ):
        raise FreezeError(
            "expected old results SHA256 disagrees with the freeze journal"
        )
    return transaction


def _artifact_payload(transaction: FreezeTransaction) -> Mapping[str, Any]:
    value = _parse_json_bytes(transaction.artifact_raw, "freeze artifact")
    if not isinstance(value, Mapping):
        raise FreezeError("freeze artifact must be an object")
    return value


def _assert_frozen_evidence_artifacts(
    transaction: FreezeTransaction,
) -> None:
    ledger = transaction.wrapper["ledger"]
    for selection in ledger["selections"]:
        for evidence in selection["evidence"]:
            _assert_file_sha(
                Path(evidence["manifest"]),
                evidence["manifest_sha256"],
                "frozen evidence manifest",
            )
            _assert_file_sha(
                Path(evidence["log"]),
                evidence["log_sha256"],
                "frozen evidence log",
            )


def _assert_reconciliation_authority(
    transaction: FreezeTransaction,
    *,
    current_state: Mapping[str, Any],
) -> None:
    old_plan = _parse_json_bytes(
        transaction.old_plan_raw, "journal old plan"
    )
    if not isinstance(old_plan, Mapping):
        raise FreezeError("journal old plan must be an object")
    ledger = transaction.wrapper["ledger"]
    authority = _report_authority(
        _output_root(old_plan),
        plan=old_plan,
        plan_sha256=transaction.expected_plan_sha256,
        state=current_state,
        source_identity=ledger["source_identity_before"],
    )
    artifact = _artifact_payload(transaction)
    reconciliation = artifact.get("state_reconciliation")
    if not isinstance(reconciliation, Mapping):
        raise FreezeError("freeze artifact state reconciliation is missing")
    expected = {
        "accepted_tune_results": list(authority.accepted_tune_results),
        "accepted_active_set_sha256": authority.accepted_active_set_sha256,
        "group_attempt_counts": list(authority.group_attempt_counts),
        "group_attempt_counts_sha256": (
            authority.group_attempt_counts_sha256
        ),
        "tune_launches": list(authority.tune_launches),
        "tune_launches_sha256": authority.tune_launches_sha256,
        "invalid_result_resolutions": list(
            authority.invalid_result_resolutions
        ),
        "invalid_result_resolutions_sha256": (
            authority.invalid_result_resolutions_sha256
        ),
        "raw_input_prefixes": authority.raw_input_prefixes,
        "raw_input_prefixes_sha256": (
            authority.raw_input_prefixes_sha256
        ),
        "report_snapshot_sha256": authority.report_snapshot_sha256,
    }
    for field, value in expected.items():
        if reconciliation.get(field) != value:
            raise FreezeError(
                "current authoritative tune snapshot differs from the "
                f"immutable freeze artifact: {field}"
            )


def _ensure_runtime_safe_for_recovery(
    transaction: FreezeTransaction,
    *,
    current_state_raw: bytes,
    agent_path: Path,
    ready_marker: str,
) -> None:
    old_plan = _parse_json_bytes(transaction.old_plan_raw, "journal old plan")
    current_state = _parse_json_bytes(current_state_raw, "current state")
    if not isinstance(old_plan, Mapping) or not isinstance(
        current_state, Mapping
    ):
        raise FreezeError("journal old plan/current state must be objects")
    _validate_environment(
        plan=old_plan,
        state=current_state,
        agent_path=agent_path,
        ready_marker=ready_marker,
    )
    _assert_no_running_work(
        current_state, output_root=_output_root(old_plan)
    )
    ledger = transaction.wrapper["ledger"]
    if _current_base_source_identity() != ledger["source_identity_before"]:
        raise FreezeError("current base source differs from the freeze ledger")
    artifact = _artifact_payload(transaction)
    if _stable_timing_gate() != artifact["timing_gate"]:
        raise FreezeError("current formal timing gate differs from the artifact")
    _assert_frozen_evidence_artifacts(transaction)
    _assert_reconciliation_authority(
        transaction, current_state=current_state
    )


def _transaction_position(
    transaction: FreezeTransaction,
) -> tuple[str, str, bytes]:
    plan_raw, plan_sha = _stable_file_identity(
        transaction.plan_path, "current plan"
    )
    state_raw, state_sha = _stable_file_identity(
        transaction.state_path, "current state"
    )
    if plan_sha not in {
        transaction.expected_plan_sha256,
        transaction.new_plan_sha256,
    }:
        raise FreezeError(
            "current plan has a third, unauthorized SHA256: "
            f"{plan_sha}"
        )
    if state_sha not in {
        transaction.expected_state_sha256,
        transaction.frozen_state_sha256,
        transaction.committed_state_sha256,
    }:
        raise FreezeError(
            "current state has a third, unauthorized SHA256: "
            f"{state_sha}"
        )
    if (plan_sha, state_sha) not in {
        (
            transaction.expected_plan_sha256,
            transaction.expected_state_sha256,
        ),
        (
            transaction.expected_plan_sha256,
            transaction.frozen_state_sha256,
        ),
        (
            transaction.new_plan_sha256,
            transaction.frozen_state_sha256,
        ),
        (
            transaction.new_plan_sha256,
            transaction.committed_state_sha256,
        ),
    }:
        raise FreezeError(
            "plan/state transaction position is impossible or externally "
            f"modified: plan={plan_sha}, state={state_sha}"
        )
    # The raw reads above are also an exact byte check, not only a hash check.
    expected_plan_raw = (
        transaction.old_plan_raw
        if plan_sha == transaction.expected_plan_sha256
        else transaction.new_plan_raw
    )
    expected_state_raw = {
        transaction.expected_state_sha256: transaction.old_state_raw,
        transaction.frozen_state_sha256: transaction.frozen_state_raw,
        transaction.committed_state_sha256: transaction.committed_state_raw,
    }[state_sha]
    if plan_raw != expected_plan_raw or state_raw != expected_state_raw:
        raise FreezeError("transaction file bytes disagree despite matching hash")
    return plan_sha, state_sha, state_raw


def _write_journal(
    transaction: FreezeTransaction,
    journal_path: Path,
    phase: str,
) -> None:
    _atomic_write_bytes(
        journal_path, _pretty_json_bytes(transaction.journal(phase))
    )


def _ensure_artifact(transaction: FreezeTransaction) -> None:
    if transaction.artifact_path.exists():
        raw = _assert_file_sha(
            transaction.artifact_path,
            transaction.artifact_sha256,
            "immutable freeze artifact",
        )
        if raw != transaction.artifact_raw:
            raise FreezeError("immutable freeze artifact byte mismatch")
        return
    _write_immutable_bytes(
        transaction.artifact_path, transaction.artifact_raw
    )
    _assert_file_sha(
        transaction.artifact_path,
        transaction.artifact_sha256,
        "immutable freeze artifact",
    )


def _advance_transaction(
    transaction: FreezeTransaction,
    *,
    journal_path: Path,
) -> dict[str, Any]:
    plan_sha, state_sha, state_raw = _transaction_position(transaction)
    if (
        plan_sha == transaction.new_plan_sha256
        and state_sha == transaction.committed_state_sha256
    ):
        _ensure_artifact(transaction)
        _write_journal(transaction, journal_path, "committed")
        return _result_payload(transaction, status="IDEMPOTENT")

    _ensure_runtime_safe_for_recovery(
        transaction,
        current_state_raw=state_raw,
        agent_path=Path(
            _parse_json_bytes(
                transaction.old_state_raw, "journal old state"
            )["agent"]["path"]
        ).resolve(strict=False),
        ready_marker=str(
            _parse_json_bytes(
                transaction.old_state_raw, "journal old state"
            )["agent"]["ready_marker"]
        ),
    )

    _ensure_artifact(transaction)
    _fault("after_artifact_written")
    _write_journal(transaction, journal_path, "artifact_written")
    _fault("after_journal_artifact_written")

    plan_sha, state_sha, current_state_raw = _transaction_position(transaction)
    if (
        plan_sha == transaction.expected_plan_sha256
        and state_sha == transaction.expected_state_sha256
    ):
        current_state = _parse_json_bytes(
            current_state_raw, "state before freeze boundary"
        )
        if not isinstance(current_state, Mapping):
            raise FreezeError("state before freeze boundary must be an object")
        _assert_reconciliation_authority(
            transaction, current_state=current_state
        )
        _assert_file_sha(
            transaction.plan_path,
            transaction.expected_plan_sha256,
            "old plan before state freeze",
        )
        _assert_file_sha(
            transaction.state_path,
            transaction.expected_state_sha256,
            "old state before state freeze",
        )
        _atomic_write_bytes(
            transaction.state_path, transaction.frozen_state_raw
        )
        _fault("after_state_frozen")
    _write_journal(transaction, journal_path, "state_frozen")
    _fault("after_journal_state_frozen")

    plan_sha, state_sha, current_state_raw = _transaction_position(transaction)
    if (
        plan_sha == transaction.expected_plan_sha256
        and state_sha == transaction.frozen_state_sha256
    ):
        current_state = _parse_json_bytes(
            current_state_raw, "state before plan boundary"
        )
        if not isinstance(current_state, Mapping):
            raise FreezeError("state before plan boundary must be an object")
        _assert_reconciliation_authority(
            transaction, current_state=current_state
        )
        _assert_file_sha(
            transaction.plan_path,
            transaction.expected_plan_sha256,
            "old plan before replacement",
        )
        _atomic_write_bytes(
            transaction.plan_path, transaction.new_plan_raw
        )
        _fault("after_plan_replaced")
    _write_journal(transaction, journal_path, "plan_replaced")
    _fault("after_journal_plan_replaced")

    plan_sha, state_sha, current_state_raw = _transaction_position(transaction)
    if (
        plan_sha == transaction.new_plan_sha256
        and state_sha == transaction.frozen_state_sha256
    ):
        current_state = _parse_json_bytes(
            current_state_raw, "state before commit boundary"
        )
        if not isinstance(current_state, Mapping):
            raise FreezeError("state before commit boundary must be an object")
        _assert_reconciliation_authority(
            transaction, current_state=current_state
        )
        _assert_file_sha(
            transaction.plan_path,
            transaction.new_plan_sha256,
            "formal plan before state commit",
        )
        _assert_file_sha(
            transaction.state_path,
            transaction.frozen_state_sha256,
            "frozen state before commit",
        )
        _atomic_write_bytes(
            transaction.state_path, transaction.committed_state_raw
        )
        _fault("after_state_committed")
    _write_journal(transaction, journal_path, "committed")
    _fault("after_journal_committed")

    plan_sha, state_sha, _ = _transaction_position(transaction)
    if (
        plan_sha != transaction.new_plan_sha256
        or state_sha != transaction.committed_state_sha256
    ):
        raise FreezeError("manual LR handoff did not reach committed state")
    return _result_payload(transaction, status="WAIT_PREFLIGHT")


def _result_payload(
    transaction: FreezeTransaction,
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "status": status,
        "transaction_id": transaction.transaction_id,
        "campaign_id": transaction.campaign_id,
        "selection_sha256": transaction.selection_sha256,
        "ledger_sha256": transaction.ledger_sha256,
        "artifact": {
            "path": str(transaction.artifact_path),
            "sha256": transaction.artifact_sha256,
            "size_bytes": len(transaction.artifact_raw),
        },
        "plan": {
            "path": str(transaction.plan_path),
            "old_sha256": transaction.expected_plan_sha256,
            "formal_sha256": transaction.new_plan_sha256,
        },
        "state": {
            "path": str(transaction.state_path),
            "old_sha256": transaction.expected_state_sha256,
            "frozen_sha256": transaction.frozen_state_sha256,
            "committed_sha256": transaction.committed_state_sha256,
        },
        "downstream_required_gate": copy.deepcopy(
            transaction.downstream_required_gate
        ),
        "preflight_valid": False,
    }


def _manual_invalid_report_target(
    report: Any,
    *,
    manifest_path: Path,
    output_root: Path,
    plan_sha256: str,
    source_identity: Mapping[str, str],
    existing_resolutions: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    if not isinstance(report, Mapping):
        raise FreezeError("manual-invalid preparation report must be an object")
    retirement = report.get("retirement_ledger")
    if (
        not isinstance(retirement, Mapping)
        or retirement.get("valid") is not True
    ):
        raise FreezeError(
            "retirement ledger is invalid during manual-invalid preparation"
        )
    rejected = report.get("rejected")
    if not isinstance(rejected, list):
        raise FreezeError(
            "manual-invalid preparation rejected results are invalid"
        )
    rejected_by_manifest = {
        item.get("manifest"): item
        for item in rejected
        if isinstance(item, Mapping)
        and isinstance(item.get("manifest"), str)
    }
    if len(rejected_by_manifest) != len(rejected):
        raise FreezeError(
            "manual-invalid preparation rejected manifests are invalid"
        )
    resolved_manifests = {
        entry["manifest"]["absolute_path"]
        for entry in existing_resolutions
    }
    target = rejected_by_manifest.get(str(manifest_path))
    if (
        not isinstance(target, Mapping)
        or target.get("errors") != list(MANUAL_INVALID_REJECTION_ERRORS)
        or not resolved_manifests.issubset(rejected_by_manifest)
    ):
        raise FreezeError(
            "requested/resolved reviewed source-drift manifests disagree "
            "with report"
        )
    reviewed_keys: set[tuple[str, str, float, int]] = set()
    for rejected_path_text, rejected_item in rejected_by_manifest.items():
        if rejected_item.get("errors") != list(MANUAL_INVALID_REJECTION_ERRORS):
            raise FreezeError(
                "manual-invalid preparation found an unreviewed rejection"
            )
        rejected_path = _path_inside_root(
            Path(rejected_path_text),
            output_root,
            "manual-invalid rejected manifest",
        )
        if rejected_path.name != results.MANIFEST_FILENAME:
            raise FreezeError(
                "manual-invalid rejected path is not an execution manifest"
            )
        rejected_manifest = _parse_json_bytes(
            _read_stable_bytes(
                rejected_path, "manual-invalid rejected manifest"
            ),
            "manual-invalid rejected manifest",
        )
        if not isinstance(rejected_manifest, Mapping):
            raise FreezeError(
                "manual-invalid rejected manifest must contain an object"
            )
        identity = _manifest_launch_identity(
            rejected_manifest,
            manifest_path=rejected_path,
            output_root=output_root,
        )
        reviewed_key = _validate_reviewed_manual_invalid_identity(
            identity,
            name="manual-invalid rejected manifest",
            plan_sha256=plan_sha256,
            source_identity=source_identity,
        )
        if reviewed_key in reviewed_keys:
            raise FreezeError(
                "manual-invalid report contains duplicate reviewed targets"
            )
        reviewed_keys.add(reviewed_key)
    for entry in existing_resolutions:
        existing_path = entry["manifest"]["absolute_path"]
        if rejected_by_manifest[existing_path].get("errors") != entry.get(
            "expected_rejection_errors"
        ):
            raise FreezeError(
                "existing manual-invalid resolution disagrees with report"
            )
    return target


def _matching_launch_event_reference(
    output_root: Path,
    *,
    manifest_path: Path,
    identity: Mapping[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    matches: list[tuple[int, Mapping[str, Any], str]] = []
    for sequence, (event, line_sha) in _campaign_events_by_seq(
        output_root
    ).items():
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        output_dir = details.get("output_dir")
        if not isinstance(output_dir, str):
            continue
        if (
            event.get("category") == "task"
            and event.get("code") == "task_launching"
            and Path(output_dir).resolve(strict=False) == manifest_path.parent
            and details.get("phase") == "tune"
            and details.get("method") == "realq"
            and details.get("model") == identity["model"]
            and details.get("setting") == identity["setting"]
            and details.get("grad_lr") == identity["grad_lr"]
            and details.get("attempt_index") == identity["attempt_index"]
            and details.get("plan_sha256") == identity["plan_sha256"]
        ):
            matches.append((sequence, details, line_sha))
    if len(matches) > 1:
        raise FreezeError(
            "reviewed invalid launch matches multiple campaign launch events"
        )
    if not matches:
        return False, None
    sequence, details, line_sha = matches[0]
    task_id = _nonempty_string(
        details.get("task_id"), "matched launch event task_id"
    )
    return True, {
        "seq": sequence,
        "line_sha256": line_sha,
        "task_id": task_id,
    }


def prepare_manual_invalid_result(
    *,
    manifest_path: Path,
    authorized_by: str,
    authorization_rationale: str,
    expected_state_sha256: str,
    expected_plan_sha256: str,
    expected_campaign_id: str,
    plan_path: Path = DEFAULT_PLAN,
    state_path: Path = DEFAULT_STATE,
    agent_path: Path = DEFAULT_AGENT,
    ready_marker: str = DEFAULT_READY_MARKER,
    expected_old_results_sha256: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Prepare one of the two narrowly authorized invalid-result entries.

    This mode never touches the experiment plan or campaign state.  Its
    default is a read-only preview; ``execute`` atomically creates or appends
    exactly one hash-chained record.
    """

    expected_state_sha256 = _require_sha256(
        expected_state_sha256, "expected state SHA256"
    )
    expected_plan_sha256 = _require_sha256(
        expected_plan_sha256, "expected plan SHA256"
    )
    expected_campaign_id = _nonempty_string(
        expected_campaign_id, "expected campaign_id"
    )
    authorized_by = _nonempty_string(authorized_by, "authorized_by")
    authorization_rationale = _nonempty_string(
        authorization_rationale, "authorization_rationale"
    )
    if expected_old_results_sha256 is not None:
        expected_old_results_sha256 = _require_sha256(
            expected_old_results_sha256,
            "expected old results SHA256",
        )
    try:
        plan_path = plan_path.resolve(strict=True)
        state_path = state_path.resolve(strict=True)
        agent_path = agent_path.resolve(strict=True)
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise FreezeError(
            f"required manual-invalid preparation path is unavailable: {exc}"
        ) from exc
    with _existing_state_lock(state_path):
        plan_raw = _assert_file_sha(
            plan_path, expected_plan_sha256, "old plan"
        )
        state_raw = _assert_file_sha(
            state_path, expected_state_sha256, "old state"
        )
        plan = _parse_json_bytes(plan_raw, "old plan")
        state = _parse_json_bytes(state_raw, "old state")
        if not isinstance(plan, dict) or not isinstance(state, dict):
            raise FreezeError("old plan/state must be objects")
        _validate_old_plan(plan, plan_sha256=expected_plan_sha256)
        _validate_old_state(
            state,
            state_path=state_path,
            plan_path=plan_path,
            plan=plan,
            plan_sha256=expected_plan_sha256,
            expected_campaign_id=expected_campaign_id,
        )
        _validate_environment(
            plan=plan,
            state=state,
            agent_path=agent_path,
            ready_marker=ready_marker,
        )
        output_root = _output_root(plan).resolve(strict=True)
        manifest_path = _path_inside_root(
            manifest_path, output_root, "reviewed invalid manifest"
        )
        if manifest_path.name != results.MANIFEST_FILENAME:
            raise FreezeError(
                "reviewed invalid manifest must be execution_manifest.json"
            )
        _assert_no_running_work(state, output_root=output_root)
        current_source = _current_base_source_identity()
        _validate_source_adoption(
            state.get("source_identity"),
            current_source,
            expected_old_results_sha256,
        )
        ledger_path = output_root / MANUAL_INVALID_LEDGER_RELATIVE_PATH
        existing_entries: tuple[dict[str, Any], ...] = ()
        ledger_prefix = b""
        if ledger_path.exists():
            existing_entries = _load_manual_invalid_resolutions(
                output_root,
                campaign_id=expected_campaign_id,
                plan_sha256=expected_plan_sha256,
                source_identity=current_source,
            )
            ledger_prefix = _read_stable_bytes(
                ledger_path, "manual invalid-result ledger"
            )
            matching_entries = [
                entry
                for entry in existing_entries
                if entry["manifest"]["absolute_path"] == str(manifest_path)
            ]
            if len(matching_entries) > 1:
                raise FreezeError(
                    "manual-invalid ledger resolves a manifest twice"
                )
            if matching_entries:
                entry = matching_entries[0]
                if (
                    entry["authorization"]["authorized_by"] != authorized_by
                    or entry["authorization"]["rationale"]
                    != authorization_rationale
                ):
                    raise FreezeError(
                        "existing manual-invalid ledger disagrees with this "
                        "request"
                    )
                return {
                    "schema_version": SCHEMA_VERSION,
                    "scope_id": SCOPE_ID,
                    "status": "IDEMPOTENT",
                    "path": str(ledger_path),
                    "sha256": _sha256_bytes(ledger_prefix),
                    "entry": copy.deepcopy(entry),
                    "plan_state_untouched": True,
                }
            if len(existing_entries) >= len(
                MANUAL_INVALID_REVIEWED_TARGETS
            ):
                raise FreezeError(
                    "manual-invalid ledger already contains the complete "
                    "reviewed target set"
                )
        if _journal_path(state_path).exists() or _artifact_path(plan).exists():
            raise FreezeError(
                "manual-invalid ledger must be prepared before freeze starts"
            )
        prefixes_before = _raw_input_prefixes(output_root)
        inventory_before = _manifest_inventory(output_root)
        try:
            report = results.build_report(output_root)
        except (results.ResultError, OSError, ValueError) as exc:
            raise FreezeError(
                f"cannot build report for manual-invalid preparation: {exc}"
            ) from exc
        _manual_invalid_report_target(
            report,
            manifest_path=manifest_path,
            output_root=output_root,
            plan_sha256=expected_plan_sha256,
            source_identity=current_source,
            existing_resolutions=existing_entries,
        )
        manifest = _parse_json_bytes(
            _read_stable_bytes(manifest_path, "reviewed invalid manifest"),
            "reviewed invalid manifest",
        )
        if not isinstance(manifest, Mapping):
            raise FreezeError("reviewed invalid manifest must be an object")
        identity = _manifest_launch_identity(
            manifest,
            manifest_path=manifest_path,
            output_root=output_root,
        )
        event_present, event_reference = _matching_launch_event_reference(
            output_root,
            manifest_path=manifest_path,
            identity=identity,
        )
        entry: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "seq": len(existing_entries) + 1,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "campaign_id": expected_campaign_id,
            "reason_code": MANUAL_INVALID_REASON_CODE,
            "authorization": {
                "kind": "user_authorized_manual_invalid_result_v1",
                "authorized_by": authorized_by,
                "rationale": authorization_rationale,
            },
            "launch_event_present": event_present,
            "launch_event": event_reference,
            "manifest": _file_ref(
                manifest_path,
                root=output_root,
                name="reviewed invalid manifest",
            ),
            "log": _file_ref(
                manifest_path.parent / results.LOG_FILENAME,
                root=output_root,
                name="reviewed invalid log",
            ),
            "execution_id": identity["execution_id"],
            "run_id": identity["run_id"],
            "phase": "tune",
            "method": "realq",
            "model": identity["model"],
            "setting": identity["setting"],
            "grad_lr": identity["grad_lr"],
            "attempt_index": identity["attempt_index"],
            "manifest_status": identity["status"],
            "exit_code": identity["exit_code"],
            "plan_sha256": identity["plan_sha256"],
            "model_sha256": identity["model_sha256"],
            "runner_sha256": identity["runner_sha256"],
            "executor_sha256": identity["executor_sha256"],
            "numerical_source_sha256_at_start": identity[
                "numerical_source_sha256_at_start"
            ],
            "numerical_source_sha256_at_end": identity[
                "numerical_source_sha256_at_end"
            ],
            "numerical_source_changed_during_execution": identity[
                "numerical_source_changed_during_execution"
            ],
            "expected_rejection_errors": list(
                MANUAL_INVALID_REJECTION_ERRORS
            ),
            "prev_record_sha256": (
                existing_entries[-1]["record_sha256"]
                if existing_entries
                else None
            ),
            "record_sha256": "",
        }
        unsigned = dict(entry)
        unsigned.pop("record_sha256")
        entry["record_sha256"] = _sha256_bytes(
            _canonical_json_bytes(unsigned)
        )
        _validate_manual_invalid_entry(
            entry,
            index=len(existing_entries) + 1,
            previous_hash=(
                existing_entries[-1]["record_sha256"]
                if existing_entries
                else None
            ),
            output_root=output_root,
            campaign_id=expected_campaign_id,
            plan_sha256=expected_plan_sha256,
            source_identity=current_source,
            events=_campaign_events_by_seq(output_root),
        )
        entry_raw = _canonical_json_bytes(entry) + b"\n"
        raw = ledger_prefix + entry_raw
        prefixes_after = _raw_input_prefixes(output_root)
        inventory_after = _manifest_inventory(output_root)
        if (
            prefixes_after != prefixes_before
            or inventory_after != inventory_before
            or _current_base_source_identity() != current_source
        ):
            raise FreezeError(
                "authority inputs changed during manual-invalid preparation"
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "scope_id": SCOPE_ID,
            "status": "PREVIEW" if not execute else "PREPARED",
            "path": str(ledger_path),
            "sha256": _sha256_bytes(raw),
            "size_bytes": len(raw),
            "entry": copy.deepcopy(entry),
            "plan_state_untouched": True,
        }
        if not execute:
            return result
        _assert_file_sha(plan_path, expected_plan_sha256, "old plan")
        _assert_file_sha(state_path, expected_state_sha256, "old state")
        _assert_no_running_work(state, output_root=output_root)
        try:
            report_now = results.build_report(output_root)
        except (results.ResultError, OSError, ValueError) as exc:
            raise FreezeError(
                f"cannot rebuild report before manual-invalid commit: {exc}"
            ) from exc
        _manual_invalid_report_target(
            report_now,
            manifest_path=manifest_path,
            output_root=output_root,
            plan_sha256=expected_plan_sha256,
            source_identity=current_source,
            existing_resolutions=existing_entries,
        )
        if (
            _manifest_inventory(output_root) != inventory_before
            or _raw_input_prefixes(output_root) != prefixes_before
            or _current_base_source_identity() != current_source
        ):
            raise FreezeError(
                "authority inputs changed before manual-invalid commit"
            )
        if existing_entries:
            _append_immutable_bytes(
                ledger_path,
                expected_prefix=ledger_prefix,
                suffix=entry_raw,
            )
        else:
            _write_immutable_bytes(ledger_path, raw)
        validated = _load_manual_invalid_resolutions(
            output_root,
            campaign_id=expected_campaign_id,
            plan_sha256=expected_plan_sha256,
            source_identity=current_source,
        )
        if list(validated) != [*existing_entries, entry]:
            raise FreezeError(
                "committed manual-invalid ledger failed exact validation"
            )
        return result


def freeze_manual_lrs(
    *,
    selection_path: Path,
    expected_state_sha256: str,
    expected_plan_sha256: str,
    expected_campaign_id: str,
    plan_path: Path = DEFAULT_PLAN,
    state_path: Path = DEFAULT_STATE,
    agent_path: Path = DEFAULT_AGENT,
    ready_marker: str = DEFAULT_READY_MARKER,
    expected_old_results_sha256: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    expected_state_sha256 = _require_sha256(
        expected_state_sha256, "expected state SHA256"
    )
    expected_plan_sha256 = _require_sha256(
        expected_plan_sha256, "expected plan SHA256"
    )
    expected_campaign_id = _nonempty_string(
        expected_campaign_id, "expected campaign_id"
    )
    if expected_old_results_sha256 is not None:
        expected_old_results_sha256 = _require_sha256(
            expected_old_results_sha256,
            "expected old results SHA256",
        )
    try:
        plan_path = plan_path.resolve(strict=True)
        state_path = state_path.resolve(strict=True)
        selection_path = selection_path.resolve(strict=True)
        agent_path = agent_path.resolve(strict=True)
    except OSError as exc:
        raise FreezeError(f"required handoff path is unavailable: {exc}") from exc
    journal_path = _journal_path(state_path)

    with _existing_state_lock(state_path):
        if journal_path.exists():
            journal_raw = _read_stable_bytes(journal_path, "freeze journal")
            journal_value = _parse_json_bytes(journal_raw, "freeze journal")
            if not isinstance(journal_value, Mapping):
                raise FreezeError("freeze journal must be an object")
            transaction = _transaction_from_journal(
                journal_value,
                plan_path=plan_path,
                state_path=state_path,
                selection_path=selection_path,
                expected_plan_sha256=expected_plan_sha256,
                expected_state_sha256=expected_state_sha256,
                expected_campaign_id=expected_campaign_id,
                expected_old_results_sha256=expected_old_results_sha256,
            )
            plan_sha, state_sha, _ = _transaction_position(transaction)
            if not transaction.artifact_path.is_file():
                if (
                    plan_sha == transaction.new_plan_sha256
                    or state_sha
                    in {
                        transaction.frozen_state_sha256,
                        transaction.committed_state_sha256,
                    }
                ):
                    raise FreezeError(
                        "transaction advanced without its immutable artifact"
                    )
            else:
                _ensure_artifact(transaction)
            if (
                plan_sha == transaction.new_plan_sha256
                and state_sha == transaction.committed_state_sha256
            ):
                if execute and journal_value.get("phase") != "committed":
                    _write_journal(transaction, journal_path, "committed")
                    _fault("after_journal_committed")
                elif not execute and journal_value.get("phase") != "committed":
                    return _result_payload(
                        transaction, status="DRY_RUN_RECOVERY"
                    )
                return _result_payload(transaction, status="IDEMPOTENT")
            if not execute:
                return _result_payload(transaction, status="DRY_RUN_RECOVERY")
            # Recovery uses the caller-selected AGENT marker/path rather than
            # silently changing authorization.  They were already pinned in
            # the old state and are rechecked here.
            old_state = _parse_json_bytes(
                transaction.old_state_raw, "journal old state"
            )
            if (
                not isinstance(old_state, Mapping)
                or old_state.get("agent", {}).get("path") != str(agent_path)
                or old_state.get("agent", {}).get("ready_marker")
                != ready_marker
            ):
                raise FreezeError(
                    "recovery agent path/marker disagrees with old state"
                )
            return _advance_transaction(
                transaction, journal_path=journal_path
            )

        transaction = _build_transaction(
            plan_path=plan_path,
            state_path=state_path,
            selection_path=selection_path,
            agent_path=agent_path,
            ready_marker=ready_marker,
            expected_plan_sha256=expected_plan_sha256,
            expected_state_sha256=expected_state_sha256,
            expected_campaign_id=expected_campaign_id,
            expected_old_results_sha256=expected_old_results_sha256,
        )
        if transaction.artifact_path.exists():
            raise FreezeError(
                "immutable freeze artifact exists without a transaction journal"
            )
        if not execute:
            return _result_payload(transaction, status="DRY_RUN")
        _assert_file_sha(
            plan_path, expected_plan_sha256, "old plan before journal"
        )
        _assert_file_sha(
            state_path, expected_state_sha256, "old state before journal"
        )
        _, selection_sha_now = _load_selection(selection_path)
        if selection_sha_now != transaction.selection_sha256:
            raise FreezeError("selection changed before journal preparation")
        _assert_frozen_evidence_artifacts(transaction)
        if (
            _current_base_source_identity()
            != transaction.wrapper["ledger"]["source_identity_before"]
        ):
            raise FreezeError("source identity changed before journal preparation")
        if (
            _stable_timing_gate()
            != _artifact_payload(transaction)["timing_gate"]
        ):
            raise FreezeError("timing gate changed before journal preparation")
        old_state_value = _parse_json_bytes(
            transaction.old_state_raw, "transaction old state"
        )
        if not isinstance(old_state_value, Mapping):
            raise FreezeError("transaction old state must be an object")
        _assert_reconciliation_authority(
            transaction, current_state=old_state_value
        )
        _write_journal(transaction, journal_path, "prepared")
        _fault("after_journal_prepared")
        return _advance_transaction(transaction, journal_path=journal_path)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--prepare-manual-invalid-result",
        type=Path,
        metavar="MANIFEST",
        help=(
            "preview/create or append one of the two reviewed source-drift "
            "resolutions (qwen3-4b/2W16A@1.5e-4 or "
            "qwen3-32b/2W4A@5e-6 attempt1); this mode never changes "
            "plan/state"
        ),
    )
    parser.add_argument("--authorized-by")
    parser.add_argument("--authorization-rationale")
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-campaign-id", required=True)
    parser.add_argument(
        "--expected-old-results-sha256",
        help=(
            "required only for the explicit audited results.py-only "
            "old-to-current source adoption"
        ),
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument(
        "--agent-ready-marker", default=DEFAULT_READY_MARKER
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the one-time transaction; omission is read-only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    prepare_mode = args.prepare_manual_invalid_result is not None
    if prepare_mode:
        if args.selection is not None:
            parser.error(
                "--selection and --prepare-manual-invalid-result are "
                "mutually exclusive"
            )
        if not args.authorized_by or not args.authorization_rationale:
            parser.error(
                "manual-invalid preparation requires --authorized-by and "
                "--authorization-rationale"
            )
    else:
        if args.selection is None:
            parser.error("--selection is required for the freeze transaction")
        if args.authorized_by or args.authorization_rationale:
            parser.error(
                "authorization flags are only valid with "
                "--prepare-manual-invalid-result"
            )
    try:
        common = {
            "expected_state_sha256": args.expected_state_sha256,
            "expected_plan_sha256": args.expected_plan_sha256,
            "expected_campaign_id": args.expected_campaign_id,
            "plan_path": args.plan,
            "state_path": args.state,
            "agent_path": args.agent,
            "ready_marker": args.agent_ready_marker,
            "expected_old_results_sha256": (
                args.expected_old_results_sha256
            ),
            "execute": args.execute,
        }
        if prepare_mode:
            payload = prepare_manual_invalid_result(
                manifest_path=args.prepare_manual_invalid_result,
                authorized_by=args.authorized_by,
                authorization_rationale=args.authorization_rationale,
                **common,
            )
        else:
            payload = freeze_manual_lrs(
                selection_path=args.selection,
                **common,
            )
    except (FreezeError, base_campaign.CampaignError, runner.PlanError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scope_id": SCOPE_ID,
                    "status": "BLOCKED",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
