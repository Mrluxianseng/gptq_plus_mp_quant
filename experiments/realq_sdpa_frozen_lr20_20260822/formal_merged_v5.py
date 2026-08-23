#!/usr/bin/env python3
"""Merge 32 V2b checkpoints with eight exact tiled-SDPA Q32 checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign_v2_memory as campaign
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v2 as v2
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_v5_q32_sdpa_tiled as q32


FORMAL_ID = "realq-sdpa-frozen-run15-lr20-formal-20260822-v5-merged"
MERGE_ID = FORMAL_ID
FORMAL_PLAN_PATH = campaign.OUTPUT_ROOT / "formal_plan_v5_merged.json"
FORMAL_AUDIT_PATH = campaign.OUTPUT_ROOT / "formal_final_audit_v5_merged.json"


def _activate() -> None:
    campaign._bootstrap()


def _balanced_pairs() -> list[tuple[str, str]]:
    return campaign.v1.v2._balanced_pairs()


def _flags(command: Sequence[str]) -> dict[str, str]:
    return v2._flags(command)


def _is_q32(config: str) -> bool:
    return config.startswith("qwen3-32b_")


def _formal_dir(branch: str, config: str) -> Path:
    root = q32.OUTPUT_ROOT if _is_q32(config) else campaign.OUTPUT_ROOT
    return root / "formal" / branch / config


def _checkpoint_path(branch: str, config: str) -> Path:
    return _formal_dir(branch, config) / "checkpoint" / "quantized.pt"


def _worker_env(gpu: str) -> dict[str, str]:
    return v2._worker_env(gpu)


def _load_source_plans() -> tuple[dict[str, Any], dict[str, Any]]:
    v2._activate()
    v2_plan = v2.core._load_plan()
    q32._activate()
    q32_plan = q32.core._load_plan()
    return v2_plan, q32_plan


def _plan_rows(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {f"{row['branch']}/{row['config']}": row for row in value["rows"]}


def _code_snapshot() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(v2.__file__).resolve(),
        Path(q32.__file__).resolve(),
        Path(campaign.__file__).resolve(),
    )
    files = [{"path": str(path), "sha256": base._file_sha256(path)} for path in paths]
    return {"files": files, "sha256": base._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    _activate()
    v2_plan, q32_plan = _load_source_plans()
    sources = {
        "v2b_non_q32": {
            "path": str(v2.FORMAL_PLAN_PATH),
            "sha256": base._file_sha256(v2.FORMAL_PLAN_PATH),
            "formal_id": v2.FORMAL_ID,
            "fingerprint": v2_plan["formal_plan_fingerprint"],
            "rows": 32,
        },
        "v5_q32_sdpa_tiled": {
            "path": str(q32.FORMAL_PLAN_PATH),
            "sha256": base._file_sha256(q32.FORMAL_PLAN_PATH),
            "formal_id": q32.FORMAL_ID,
            "fingerprint": q32_plan["formal_plan_fingerprint"],
            "rows": 8,
        },
    }
    source_rows = {
        "v2b_non_q32": _plan_rows(v2_plan),
        "v5_q32_sdpa_tiled": _plan_rows(q32_plan),
    }
    rows = []
    for branch, config in _balanced_pairs():
        source = "v5_q32_sdpa_tiled" if _is_q32(config) else "v2b_non_q32"
        key = f"{branch}/{config}"
        row = source_rows[source][key]
        checkpoint = str(_checkpoint_path(branch, config))
        if row.get("checkpoint") != checkpoint:
            raise base.CampaignError(f"merged checkpoint path drift: {key}")
        command = list(row["command"])
        flags = _flags(command)
        if (
            flags.get("--attention_backend") != "sdpa"
            or flags.get("--hessian_accum_bsz")
            != ("32" if _is_q32(config) else "64")
            or flags.get("--backward_bsz") != "32"
            or flags.get("--require_static_cache_hit") != "true"
            or flags.get("--require_reference_cache_hit") != "true"
        ):
            raise base.CampaignError(f"merged source protocol drift: {key}")
        if _is_q32(config) and command[2] != q32.ENTRY_MODULE:
            raise base.CampaignError(f"merged Q32 memory entry drift: {key}")
        rows.append(
            {
                "branch": branch,
                "config": config,
                "selected_lr": float(row["selected_lr"]),
                "command": command,
                "command_sha256": base._canonical_sha256(command),
                "checkpoint": checkpoint,
                "source": source,
                "source_formal_id": sources[source]["formal_id"],
                "source_formal_plan_fingerprint": sources[source]["fingerprint"],
            }
        )
    if len(rows) != 40 or sum(_is_q32(str(row["config"])) for row in rows) != 8:
        raise base.CampaignError("merged formal matrix is not 32+8 rows")
    body: dict[str, Any] = {
        "formal_id": FORMAL_ID,
        "campaign_id": campaign.CAMPAIGN_ID,
        "protocol_fingerprint": base._read_json(campaign.PLAN_PATH)[
            "protocol_fingerprint"
        ],
        "sources": sources,
        "code": _code_snapshot(),
        "protocol": {
            "checkpoint_producers": 40,
            "non_q32_source": "V2b deterministic-SDPA formal",
            "q32_source": "V5 exact batch-tiled deterministic-SDPA formal",
            "q32_numerical_flags_changed": False,
            "q32_logical_backward_bsz": 32,
            "q32_hessian_accum_bsz": 32,
            "learning_rates_retuned_for_sdpa": False,
            "exact_gptaq_guidedquant_token_files": True,
            "exact_gptaq_guidedquant_reference_cache_files": True,
            "source_commands_and_checkpoint_paths_preserved": True,
        },
        "rows": rows,
    }
    body["formal_plan_fingerprint"] = base._canonical_sha256(body)
    return body


def _write_plan(_: argparse.Namespace) -> int:
    value = _build_plan()
    if FORMAL_PLAN_PATH.is_file():
        if base._read_json(FORMAL_PLAN_PATH) != value:
            raise base.CampaignError("existing merged formal plan differs")
    else:
        base._atomic_json(FORMAL_PLAN_PATH, value)
    print(FORMAL_PLAN_PATH)
    print(value["formal_plan_fingerprint"])
    return 0


def _load_plan() -> dict[str, Any]:
    value = base._read_json(FORMAL_PLAN_PATH)
    identity = dict(value)
    fingerprint = identity.pop("formal_plan_fingerprint", None)
    if base._canonical_sha256(identity) != fingerprint:
        raise base.CampaignError("merged formal plan fingerprint mismatch")
    if value != _build_plan():
        raise base.CampaignError("merged formal plan no longer matches frozen inputs")
    return value


def _source_audit(branch: str, config: str) -> dict[str, Any]:
    if _is_q32(config):
        q32._activate()
        plan = q32.core._load_plan()
        return q32.core._audit_one(plan, branch, config)
    v2._activate()
    plan = v2.core._load_plan()
    return v2.core._audit_one(plan, branch, config)


def _audit_one(plan: Mapping[str, Any], branch: str, config: str) -> dict[str, Any]:
    key = f"{branch}/{config}"
    row = _plan_rows(plan)[key]
    source = _source_audit(branch, config)
    if (
        source.get("checkpoint", {}).get("path") != row["checkpoint"]
        or float(source["selected_lr"]) != float(row["selected_lr"])
    ):
        raise base.CampaignError(f"merged formal source audit drift: {key}")
    return {
        **source,
        "source": row["source"],
        "source_formal_id": row["source_formal_id"],
        "source_formal_plan_fingerprint": row["source_formal_plan_fingerprint"],
    }


def _state() -> dict[str, int]:
    state = {"succeeded": 0, "running": 0, "pending": 0}
    for branch, config in _balanced_pairs():
        directory = _formal_dir(branch, config)
        if (directory / "formal_success.json").is_file():
            state["succeeded"] += 1
        elif any(path.is_dir() for path in directory.glob(".*claim*")):
            state["running"] += 1
        else:
            state["pending"] += 1
    return state


def _status(_: argparse.Namespace) -> int:
    print(json.dumps({"formal_id": FORMAL_ID, "formal": _state()}, indent=2))
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
    payload["audit_fingerprint"] = base._canonical_sha256(payload)
    payload["audited_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    base._atomic_json(FORMAL_AUDIT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write-plan")
    write.set_defaults(handler=_write_plan)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    audit = subparsers.add_parser("audit")
    audit.set_defaults(handler=_audit_all)
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
