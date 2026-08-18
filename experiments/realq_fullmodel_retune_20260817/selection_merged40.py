#!/usr/bin/env python3
"""Merge the three disjoint authoritative LR selections into 40 rows.

The merger reads JSON artifacts instead of importing their selector modules;
those modules intentionally bootstrap different campaign globals and must not
share one Python process.  Every source plan, selector implementation, command,
seed, calibration path, and row is hash-bound before the merged artifact is
published.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as c


MERGE_ID = "realq-fullmodel-authoritative-selection-merged40-20260818-v1"
DATA_ROOT = c.REPO_ROOT.parent / "experiment_data"
OUTPUT_ROOT = DATA_ROOT / "realq_fullmodel_retune_20260818_merged40"
SELECTION_PATH = OUTPUT_ROOT / "selection.json"
SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "v4_subset28": {
        "path": DATA_ROOT
        / "realq_fullmodel_retune_20260817_v4"
        / "selections_subset28.json",
        "selection_id": "realq-fullmodel-v4-subset28-selection-20260818-v1",
        "rows": 28,
    },
    "v5_q32": {
        "path": DATA_ROOT
        / "realq_fullmodel_retune_20260818_v5_q32_memory"
        / "selections.json",
        "selection_id": (
            "realq-fullmodel-two-branch-lr-selection-20260818-v5-q32-memory"
        ),
        "rows": 8,
    },
    "q4_a1": {
        "path": DATA_ROOT
        / "realq_qwen3_4b_aloss_clip_independent_lr_20260817"
        / "selections_a1_only.json",
        "selection_id": "realq-qwen3-4b-aloss-one-selection-20260818-v1",
        "rows": 4,
    },
}
FROZEN_FLAGS = {
    "--dataset": "wikitext2",
    "--eval_datasets": "wikitext2",
    "--seed": "1",
    "--rotation_seed": "0",
    "--refresh_seed": "0",
    "--nsamples": "256",
    "--seq_len": "2048",
    "--eval_seq_len": "2048",
    "--a_loss_clip_scope": "local_backward_chunk",
}


class MergeError(RuntimeError):
    pass


def _expected_keys() -> set[tuple[str, str]]:
    keys = {
        (branch, config)
        for branch in c.BRANCH_VALUES
        for config in c.CONFIG_IDS
    }
    if len(keys) != 40:
        raise MergeError(f"expected 40 branch/config rows, got {len(keys)}")
    return keys


def _flags(command: Sequence[str]) -> dict[str, str]:
    if len(command) < 3 or command[1] != "-m":
        raise MergeError("source command must use python -m")
    if (len(command) - 3) % 2:
        raise MergeError("source command is not strict flag/value form")
    output: dict[str, str] = {}
    for index in range(3, len(command), 2):
        flag, value = str(command[index]), str(command[index + 1])
        if not flag.startswith("--") or flag in output:
            raise MergeError(f"invalid or duplicate command flag: {flag}")
        output[flag] = value
    return output


def _verify_selection(source: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.is_file():
        raise MergeError(f"source selection is missing: {source}: {path}")
    value = c._read_json(path)
    if value.get("selection_id") != spec["selection_id"]:
        raise MergeError(f"wrong source selection id: {source}")
    comparable = dict(value)
    fingerprint = comparable.pop("selection_fingerprint", None)
    comparable.pop("created_at", None)
    if c._canonical_sha256(comparable) != fingerprint:
        raise MergeError(f"source selection fingerprint mismatch: {source}")
    plan_ref = value.get("plan")
    if not isinstance(plan_ref, dict):
        raise MergeError(f"source selection has no plan reference: {source}")
    plan_path = Path(str(plan_ref.get("path", "")))
    if not plan_path.is_file() or c._file_sha256(plan_path) != plan_ref.get("sha256"):
        raise MergeError(f"source plan changed: {source}: {plan_path}")
    code_refs = value.get("selection_code")
    if not isinstance(code_refs, list) or not code_refs:
        raise MergeError(f"source selection has no code references: {source}")
    for item in code_refs:
        code_path = Path(str(item.get("path", "")))
        if not code_path.is_file() or c._file_sha256(code_path) != item.get("sha256"):
            raise MergeError(f"source selection code changed: {source}: {code_path}")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != int(spec["rows"]):
        raise MergeError(f"source row count mismatch: {source}")
    return value


def _source_command(
    source: str, selection: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    plan_path = Path(str(selection["plan"]["path"]))
    plan = c._read_json(plan_path)
    branch, config = str(row["branch"]), str(row["config"])
    if source == "q4_a1":
        group = str(row.get("group", ""))
        plan_row = plan.get("groups", {}).get(group)
    else:
        plan_row = plan.get("configurations", {}).get(f"{branch}/{config}")
    if not isinstance(plan_row, dict):
        raise MergeError(f"source plan row is missing: {source}/{branch}/{config}")
    command = plan_row.get("source_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise MergeError(f"source command is invalid: {source}/{branch}/{config}")
    return list(command), plan


def _audit_command(
    source: str,
    row: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    flags = _flags(command)
    for flag, expected in FROZEN_FLAGS.items():
        if flags.get(flag) != expected:
            raise MergeError(
                f"frozen flag mismatch for {source}/{row['branch']}/{row['config']}: "
                f"{flag}={flags.get(flag)!r}, expected {expected!r}"
            )
    if flags.get("--a_loss_ratio") not in {"1", "1.0"}:
        raise MergeError(f"non-one a-loss ratio in authoritative row: {source}")
    expected_branch = c.BRANCH_VALUES[str(row["branch"])]
    if flags.get("--full_block_refresh") != expected_branch:
        raise MergeError(f"branch flag mismatch in authoritative row: {source}")
    token_path = flags.get("--tokens_cache_path")
    static_path = flags.get("--static_cache_path")
    if not token_path or not static_path:
        raise MergeError(f"calibration cache path missing: {source}")
    return {
        "command_sha256": c._canonical_sha256(list(command)),
        "python_module": command[2],
        "seed": flags["--seed"],
        "rotation_seed": flags["--rotation_seed"],
        "refresh_seed": flags["--refresh_seed"],
        "tokens_cache_path": token_path,
        "static_cache_path": static_path,
        "hessian_accum_bsz": flags.get("--hessian_accum_bsz"),
    }


def _row(
    source: str,
    source_value: Mapping[str, Any],
    source_path: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    branch, config = str(row["branch"]), str(row["config"])
    selected_lr = float(row["selected_lr"])
    if not math.isfinite(selected_lr) or selected_lr < 0:
        raise MergeError(f"invalid selected LR: {source}/{branch}/{config}")
    command, _ = _source_command(source, source_value, row)
    return {
        "branch": branch,
        "config": config,
        "selected_lr": selected_lr,
        "selection_source": source,
        "source_selection": {
            "path": str(source_path),
            "sha256": c._file_sha256(source_path),
            "fingerprint": source_value["selection_fingerprint"],
        },
        "source_plan": dict(source_value["plan"]),
        "source_command": command,
        "source_command_audit": _audit_command(source, row, command),
        "selection_evidence": dict(row),
    }


def _build() -> dict[str, Any]:
    source_values = {
        source: _verify_selection(source, spec)
        for source, spec in SOURCE_SPECS.items()
    }
    rows = []
    for source, value in source_values.items():
        source_path = Path(SOURCE_SPECS[source]["path"])
        rows.extend(_row(source, value, source_path, row) for row in value["rows"])
    keys = [(row["branch"], row["config"]) for row in rows]
    if len(keys) != 40 or len(set(keys)) != 40 or set(keys) != _expected_keys():
        raise MergeError("merged source rows are not a disjoint complete 40-row matrix")
    rows.sort(key=lambda row: (c.CONFIG_IDS.index(row["config"]), row["branch"]))
    code_path = Path(__file__).resolve()
    body: dict[str, Any] = {
        "merge_id": MERGE_ID,
        "code": {"path": str(code_path), "sha256": c._file_sha256(code_path)},
        "sources": {
            source: {
                "path": str(spec["path"]),
                "sha256": c._file_sha256(Path(spec["path"])),
                "selection_id": spec["selection_id"],
                "selection_fingerprint": source_values[source][
                    "selection_fingerprint"
                ],
                "rows": spec["rows"],
            }
            for source, spec in SOURCE_SPECS.items()
        },
        "protocol": {
            "rows": 40,
            "source_partition": "28 V4 + 8 Q32 V5 + 4 Qwen3-4B a=1",
            "primary_metric": "WikiText2 Exact KL",
            "a_loss_ratio": 1.0,
            "seed": 1,
            "rotation_seed": 0,
            "refresh_seed": 0,
            "calibration_samples": 256,
            "calibration_sequence_length": 2048,
            "source_modules_are_not_imported_together": True,
        },
        "rows": rows,
    }
    body["selection_fingerprint"] = c._canonical_sha256(body)
    return body


def _status(_: argparse.Namespace) -> int:
    missing = [
        {"source": source, "path": str(spec["path"])}
        for source, spec in SOURCE_SPECS.items()
        if not Path(spec["path"]).is_file()
    ]
    value = {
        "merge_id": MERGE_ID,
        "ready": not missing,
        "missing_sources": missing,
        "expected_rows": {source: spec["rows"] for source, spec in SOURCE_SPECS.items()},
    }
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _freeze(_: argparse.Namespace) -> int:
    value = _build()
    if SELECTION_PATH.exists():
        current = c._read_json(SELECTION_PATH)
        comparable = dict(current)
        comparable.pop("created_at", None)
        if comparable != value:
            raise MergeError(f"existing merged selection differs: {SELECTION_PATH}")
    else:
        c._atomic_json(
            SELECTION_PATH,
            {**value, "created_at": dt.datetime.now(dt.timezone.utc).isoformat()},
        )
    print(SELECTION_PATH)
    return 0


def load_selection() -> dict[str, Any]:
    value = c._read_json(SELECTION_PATH)
    if value.get("merge_id") != MERGE_ID:
        raise MergeError("wrong merged selection id")
    identity = dict(value)
    fingerprint = identity.pop("selection_fingerprint", None)
    identity.pop("created_at", None)
    if c._canonical_sha256(identity) != fingerprint:
        raise MergeError("merged selection fingerprint mismatch")
    if len(value.get("rows", [])) != 40:
        raise MergeError("merged selection row count mismatch")
    current = dict(value)
    current.pop("created_at", None)
    if current != _build():
        raise MergeError("merged selection no longer matches its audited sources")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    freeze = subparsers.add_parser("freeze")
    freeze.set_defaults(handler=_freeze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        MergeError,
        c.CampaignError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"realq-merged40-selection: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
