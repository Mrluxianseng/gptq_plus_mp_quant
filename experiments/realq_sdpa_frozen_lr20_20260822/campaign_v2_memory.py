#!/usr/bin/env python3
"""OOM-closed V2 of the deterministic-SDPA frozen-LR campaign.

V1 proved that the run15 global-loss batches do not all fit math-SDPA on one
B100.  V2 freezes the first lower rung of the already documented formal
capacity ladder for only the three failed models.  The global-loss batch is
part of the static-cache identity, so V2 uses a new immutable plan/root and
never relabels a V1 artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import campaign as v1


CAMPAIGN_ID = "realq-sdpa-frozen-run15-lr20-20260822-v2-memory"
OUTPUT_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_sdpa_frozen_lr20_20260822_v2_memory"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
CACHE_ROOT = OUTPUT_ROOT / "shared_cache"
SOURCE_PLAN_PATH = v1.PLAN_PATH
SOURCE_SELECTION_PATH = v1.OUTPUT_ROOT / "selections.json"
LOCK_ROOT = v1.LOCK_ROOT
MODULE_PATH = "experiments/realq_sdpa_frozen_lr20_20260822/campaign_v2_memory.py"
SELECTION_MODULE_PATH = "experiments/realq_sdpa_frozen_lr20_20260822/selection_v2.py"

GLOBAL_LOSS_BSZ = {
    "qwen3-0.6b": 1,
    "llama31-8b-instruct": 4,
    "qwen3-4b": 4,
    "qwen3-8b": 4,
    "qwen3-32b": 1,
}
ADOPT_V1_MODELS = ("qwen3-0.6b", "qwen3-4b")
OOM_V1_MODELS = ("llama31-8b-instruct", "qwen3-8b", "qwen3-32b")
RUN_FLAGS = dict(v1.RUN_FLAGS)
WORKER_ENV_OVERRIDES = dict(v1.WORKER_ENV_OVERRIDES)
WORKER_ENV_UNSET = tuple(v1.WORKER_ENV_UNSET)
CODE_INPUTS = tuple(
    dict.fromkeys((*v1.CODE_INPUTS, MODULE_PATH, SELECTION_MODULE_PATH))
)


def _source_plan() -> dict[str, Any]:
    v1._bootstrap()
    try:
        value = base._read_json(SOURCE_PLAN_PATH)
        v1._verify_plan(value)
        return value
    finally:
        _bootstrap()


def _source_selection() -> dict[str, Any]:
    value = base._read_json(SOURCE_SELECTION_PATH)
    stable = dict(value)
    fingerprint = stable.pop("selection_fingerprint", None)
    stable.pop("created_at", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise base.CampaignError("V1 selection fingerprint mismatch")
    if value.get("campaign_id") != v1.CAMPAIGN_ID or len(value.get("rows", [])) != 40:
        raise base.CampaignError("V1 selection identity/matrix mismatch")
    if value.get("plan", {}).get("sha256") != base._file_sha256(SOURCE_PLAN_PATH):
        raise base.CampaignError("V1 selection is not bound to the V1 plan")
    return value


def _model_for_config(config: str) -> str:
    return v1._model_for_config(config)


def _static_root(model: str) -> Path:
    return CACHE_ROOT / model / "static"


def _cache_marker(model: str) -> Path:
    return CACHE_ROOT / model / "producer_success.json"


def _cache_attempt_count(model: str) -> int:
    root = CACHE_ROOT / model / "attempts"
    return len(list(root.glob("attempt[0-9][0-9][0-9]"))) if root.is_dir() else 0


def _v1_result(model: str) -> dict[str, Any]:
    path = v1.CACHE_ROOT / model / "attempts" / "attempt001" / "result.json"
    value = base._read_json(path)
    if value.get("model") != model:
        raise base.CampaignError(f"V1 result identity mismatch: {path}")
    return value


def _v1_evidence() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for model in base.MODEL_SLUGS:
        result_path = v1.CACHE_ROOT / model / "attempts" / "attempt001" / "result.json"
        result = _v1_result(model)
        if model in ADOPT_V1_MODELS:
            if result.get("status") != "succeeded" or result.get("returncode") != 0:
                raise base.CampaignError(f"V1 adoption source did not succeed: {model}")
            static_path = Path(str(result.get("static_cache", {}).get("path", "")))
            if not static_path.is_file() or static_path.stat().st_size <= 0:
                raise base.CampaignError(f"V1 adoption cache missing: {static_path}")
            rows[model] = {
                "action": "adopt exact V1 deterministic-SDPA static cache",
                "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
                "source_static": {
                    "path": str(static_path),
                    "size_bytes": static_path.stat().st_size,
                    "mtime_ns": static_path.stat().st_mtime_ns,
                },
            }
        else:
            log_path = Path(str(result.get("log", {}).get("path", "")))
            tail = log_path.read_bytes()[-4 * 1024 * 1024 :].decode(errors="replace")
            if (
                result.get("status") != "failed"
                or result.get("returncode") != 1
                or "out of memory" not in tail.lower()
            ):
                raise base.CampaignError(f"V1 fallback source is not an observed OOM: {model}")
            rows[model] = {
                "action": "lower only global_loss_bsz by one frozen capacity rung",
                "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
                "log": {"path": str(log_path), "sha256": base._file_sha256(log_path)},
                "v1_global_loss_bsz": int(
                    v1._flags(result["command"])["--global_loss_bsz"]
                ),
                "v2_global_loss_bsz": GLOBAL_LOSS_BSZ[model],
            }
    return rows


def _configure_command(command: list[str], *, branch: str, config: str) -> list[str]:
    model = _model_for_config(config)
    base._set_or_append_arg(command, "--full_block_refresh", base.BRANCH_VALUES[branch])
    base._set_or_append_arg(command, "--global_loss_bsz", str(GLOBAL_LOSS_BSZ[model]))
    base._set_or_append_arg(command, "--static_cache_path", str(_static_root(model)))
    return command


def _validate_full_profile(
    command: Sequence[str], *, branch: str, config: str
) -> None:
    model = _model_for_config(config)
    expected = {
        "--dataset": "wikitext2",
        "--eval_datasets": "wikitext2",
        "--seed": "1",
        "--rotation_seed": "0",
        "--refresh_seed": "0",
        "--nsamples": "256",
        "--seq_len": "2048",
        "--eval_seq_len": "2048",
        "--w_groupsize": "128",
        "--w_asym": "false",
        "--w_clip": "true",
        "--blocksize": "128",
        "--act_order": "true",
        "--backward_samples": "32",
        "--backward_bsz": "32",
        "--loss_slide_window": "true",
        "--full_block_refresh": base.BRANCH_VALUES[branch],
        "--grad_lr_layer_schedule": base._schedule_for(config),
        "--a_loss_ratio": "1",
        "--rotate": "true",
        "--require_static_cache_hit": "true",
        "--require_reference_cache_hit": "true",
        "--global_loss_bsz": str(GLOBAL_LOSS_BSZ[model]),
        **RUN_FLAGS,
    }
    values = v1._flags(command)
    for flag, wanted in expected.items():
        if values.get(flag) != wanted:
            raise base.CampaignError(
                f"V2 {branch}/{config} {flag}: expected {wanted}, got {values.get(flag)}"
            )
    wanted_hessian = "32" if model == "qwen3-32b" else "64"
    if values.get("--hessian_accum_bsz") != wanted_hessian:
        raise base.CampaignError(f"V2 hessian_accum_bsz changed: {branch}/{config}")
    if values.get("--static_cache_path") != str(_static_root(model)):
        raise base.CampaignError("V2 run points at wrong static cache")
    contract = v1._baseline_cache_contracts()[model]
    if values.get("--cache_dir") != contract["reference_logits"]["runtime_root"]:
        raise base.CampaignError("V2 run is not using baseline reference cache")
    if sorted(Path(values["--tokens_cache_path"]).glob("*.pt")) != [
        Path(contract["tokens"]["path"])
    ]:
        raise base.CampaignError("V2 run is not using baseline calibration tokens")
    if base._arg_indices(command, "--quant_stop_layer"):
        raise base.CampaignError("formal V2 run must cover the full model")


def _build_plan() -> dict[str, Any]:
    source = _source_plan()
    selection = _source_selection()
    contracts = copy.deepcopy(v1._baseline_cache_contracts())
    evidence = _v1_evidence()
    configurations: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    for config in base.CONFIG_IDS:
        normalized: dict[str, dict[str, str]] = {}
        for branch in base.BRANCH_VALUES:
            key = f"{branch}/{config}"
            source_row = source["configurations"][key]
            source_command = list(map(str, source_row["source_command"]))
            command = _configure_command(source_command.copy(), branch=branch, config=config)
            _validate_full_profile(command, branch=branch, config=config)
            before, after = v1._flags(source_command), v1._flags(command)
            changed = {
                flag: {"v1": before.get(flag), "v2": after.get(flag)}
                for flag in sorted(set(before) | set(after))
                if before.get(flag) != after.get(flag)
            }
            expected_delta = {"--static_cache_path"}
            if before["--global_loss_bsz"] != after["--global_loss_bsz"]:
                expected_delta.add("--global_loss_bsz")
            if set(changed) != expected_delta:
                raise base.CampaignError(f"unexpected V1→V2 delta {key}: {changed}")
            row = copy.deepcopy(source_row)
            row.update(
                source_command=command,
                source_v1_command_sha256=base._canonical_sha256(source_command),
                v2_command_sha256=base._canonical_sha256(command),
                v1_to_v2_delta=changed,
                frozen_run15_lr_only=True,
            )
            configurations[key] = row
            flags = dict(after)
            flags.pop("--full_block_refresh")
            normalized[branch] = flags
        if normalized["full_block"] != normalized["single_linear"]:
            raise base.CampaignError(f"V2 branch parity failed: {config}")
        parity[config] = {
            "only_branch_difference": "--full_block_refresh=true/false",
            "normalized_command_sha256": base._canonical_sha256(normalized["full_block"]),
        }
    selected = {
        f"{row['branch']}/{row['config']}": float(row["selected_lr"])
        for row in selection["rows"]
    }
    if set(selected) != set(configurations):
        raise base.CampaignError("V2 selection matrix differs from commands")
    body: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "output_root": str(OUTPUT_ROOT),
        "source_v1": {
            "plan": {"path": str(SOURCE_PLAN_PATH), "sha256": base._file_sha256(SOURCE_PLAN_PATH)},
            "protocol_fingerprint": source["protocol_fingerprint"],
            "selection": {"path": str(SOURCE_SELECTION_PATH), "sha256": base._file_sha256(SOURCE_SELECTION_PATH), "selection_fingerprint": selection["selection_fingerprint"]},
        },
        "source_run15": copy.deepcopy(source["source_run15"]),
        "baseline_gptaq_guidedquant": copy.deepcopy(source["baseline_gptaq_guidedquant"]),
        "branches": dict(base.BRANCH_VALUES),
        "config_ids": list(base.CONFIG_IDS),
        "selected_lrs": selected,
        "global_loss_bsz_by_model": dict(GLOBAL_LOSS_BSZ),
        "calibration_and_reference_contracts": contracts,
        "v1_capacity_evidence": evidence,
        "determinism": copy.deepcopy(source["determinism"]),
        "optimization_profile": {
            **copy.deepcopy(source["optimization_profile"]),
            "capacity_fallback": (
                "only global_loss_bsz: llama31-8b-instruct 8→4, "
                "qwen3-8b 8→4, qwen3-32b 2→1"
            ),
            "learning_rates_retuned": False,
            "all_other_algorithm_flags_unchanged": True,
        },
        "paired_profile_audit": parity,
        "code_snapshot": base._code_snapshot(),
        "configurations": configurations,
    }
    body["protocol_fingerprint"] = base._canonical_sha256(body)
    body["created_at"] = base._utc_now()
    return body


def _selection_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    source = _source_selection()
    rows = [
        {
            "branch": str(row["branch"]),
            "config": str(row["config"]),
            "selected_lr": float(row["selected_lr"]),
            "reason": "verbatim V1/run15 LR; no SDPA or memory-fallback retune",
            "source_selection_row": row,
        }
        for row in source["rows"]
    ]
    module = base.REPO_ROOT / SELECTION_MODULE_PATH
    value: dict[str, Any] = {
        "selection_id": "realq-sdpa-frozen-run15-lr-selection-20260822-v2-memory",
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "plan": {"path": str(PLAN_PATH), "sha256": base._file_sha256(PLAN_PATH)},
        "source_selection": plan["source_v1"]["selection"],
        "selection_code": [{"path": str(module), "sha256": base._file_sha256(module)}],
        "rows": rows,
    }
    value["selection_fingerprint"] = base._canonical_sha256(value)
    value["created_at"] = base._utc_now()
    return value


def _verify_plan(plan: Mapping[str, Any]) -> None:
    base._verify_plan(plan)
    if plan.get("source_v1", {}).get("plan", {}).get("sha256") != base._file_sha256(
        SOURCE_PLAN_PATH
    ):
        raise base.CampaignError("source V1 plan changed")
    if plan.get("source_v1", {}).get("selection", {}).get("sha256") != base._file_sha256(
        SOURCE_SELECTION_PATH
    ):
        raise base.CampaignError("source V1 selection changed")
    v1._baseline_cache_contracts()


def _adopt_v1_cache(plan: Mapping[str, Any], model: str) -> None:
    evidence = plan["v1_capacity_evidence"][model]
    source = Path(str(evidence["source_static"]["path"]))
    target = _static_root(model) / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise base.CampaignError(f"V2 cache symlink drift: {target}")
    elif target.exists():
        raise base.CampaignError(f"unexpected non-symlink V2 adopted cache: {target}")
    else:
        target.symlink_to(source)
    stat = target.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
        int(evidence["source_static"]["size_bytes"]),
        int(evidence["source_static"]["mtime_ns"]),
    ):
        raise base.CampaignError(f"V2 adopted cache metadata mismatch: {target}")
    marker = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "status": "succeeded",
        "stage": "adopted-exact-v1-deterministic-sdpa-static-cache",
        "model": model,
        "static_cache": {
            "path": str(target),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "symlink_source": str(source),
        },
        "token_contract": plan["calibration_and_reference_contracts"][model]["tokens"],
        "baseline_reference_unchanged": True,
        "source_result": evidence["result"],
        "adopted_at": base._utc_now(),
    }
    marker_path = _cache_marker(model)
    if marker_path.is_file():
        current = base._read_json(marker_path)
        stable_current, stable_marker = dict(current), dict(marker)
        stable_current.pop("adopted_at", None)
        stable_marker.pop("adopted_at", None)
        if stable_current != stable_marker:
            raise base.CampaignError(f"V2 adopted marker drift: {marker_path}")
    else:
        base._atomic_json(marker_path, marker)


def _write_plan(_: argparse.Namespace) -> int:
    candidate = _build_plan()
    if PLAN_PATH.is_file():
        existing = base._read_json(PLAN_PATH)
        _verify_plan(existing)
        if existing["protocol_fingerprint"] != candidate["protocol_fingerprint"]:
            raise base.CampaignError("existing V2 plan differs from current inputs")
    else:
        base._atomic_json(PLAN_PATH, candidate)
    plan = base._read_json(PLAN_PATH)
    selection_path = OUTPUT_ROOT / "selections.json"
    candidate_selection = _selection_payload(plan)
    if selection_path.is_file():
        existing = base._read_json(selection_path)
        old, new = dict(existing), dict(candidate_selection)
        old.pop("created_at", None)
        new.pop("created_at", None)
        if old != new:
            raise base.CampaignError("existing V2 selection differs")
    else:
        base._atomic_json(selection_path, candidate_selection)
    for model in ADOPT_V1_MODELS:
        _adopt_v1_cache(plan, model)
    print(PLAN_PATH)
    print(selection_path)
    return 0


@contextlib.contextmanager
def _patched_v1_runtime() -> Iterator[None]:
    names = ("CAMPAIGN_ID", "OUTPUT_ROOT", "PLAN_PATH", "CACHE_ROOT")
    saved = {name: getattr(v1, name) for name in names}
    try:
        v1.CAMPAIGN_ID = CAMPAIGN_ID
        v1.OUTPUT_ROOT = OUTPUT_ROOT
        v1.PLAN_PATH = PLAN_PATH
        v1.CACHE_ROOT = CACHE_ROOT
        _bootstrap()
        yield
    finally:
        for name, value in saved.items():
            setattr(v1, name, value)
        _bootstrap()


def _prepare_cache(args: argparse.Namespace) -> int:
    with _patched_v1_runtime():
        return int(v1._prepare_cache(args))


def _status(_: argparse.Namespace) -> int:
    rows = [
        {
            "model": model,
            "global_loss_bsz": GLOBAL_LOSS_BSZ[model],
            "status": "succeeded" if _cache_marker(model).is_file() else "pending",
            "attempts": _cache_attempt_count(model),
        }
        for model in base.MODEL_SLUGS
    ]
    print(json.dumps({"campaign_id": CAMPAIGN_ID, "cache_producers": rows}, indent=2, sort_keys=True))
    return 0


def _bootstrap() -> None:
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PLAN_PATH = PLAN_PATH
    base.WORKER_ENV_OVERRIDES = dict(WORKER_ENV_OVERRIDES)
    base.WORKER_ENV_UNSET = WORKER_ENV_UNSET
    base.CODE_INPUTS = CODE_INPUTS
    base._validate_full_profile = _validate_full_profile
    base._build_plan = _build_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.set_defaults(handler=_write_plan)
    producer = subparsers.add_parser("prepare-cache")
    producer.add_argument("--model", required=True, choices=base.MODEL_SLUGS)
    producer.add_argument("--physical-gpu", required=True, type=int)
    producer.set_defaults(handler=_prepare_cache)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap()
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception:
        traceback.print_exc()
        return 1


_bootstrap()


if __name__ == "__main__":
    raise SystemExit(main())
