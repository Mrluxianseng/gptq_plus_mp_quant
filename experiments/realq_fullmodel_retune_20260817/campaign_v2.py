#!/usr/bin/env python3
"""Paired-cache V2 of the 2026-08-17 two-branch full-model retune.

V1 correctly shared calibration *tokens*, but retained each historical
branch's ``global_loss_bsz`` and static Fisher cache.  The first paired zero-LR
controls proved that this was not a numerically paired baseline.  V2 freezes a
single physical token/static/reference cache and one conservative common
``global_loss_bsz`` for both branches of every config.  A complete 40-control
zero-LR gate must pass before positive LR candidates can be released.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base


CAMPAIGN_ID = "realq-fullmodel-two-branch-retune-20260817-v2-paired-cache"
OUTPUT_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_fullmodel_retune_20260817_v2"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
ZERO_AUDIT_PATH = OUTPUT_ROOT / "paired_zero_control_audit.json"
MODULE_PATH = "experiments/realq_fullmodel_retune_20260817/campaign_v2.py"
POSITIVE_LRS = (1e-7, 1e-6, 1e-5, 1e-4)
BALANCED_MODELS = (
    "qwen3-32b",
    "qwen3-0.6b",
    "qwen3-8b",
    "qwen3-4b",
    "llama31-8b-instruct",
)


def _bootstrap() -> None:
    """Point the audited V1 execution engine at a distinct immutable V2 root."""

    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PLAN_PATH = PLAN_PATH
    base.INITIAL_LRS = (0.0, *POSITIVE_LRS)
    inputs = tuple(path for path in base.CODE_INPUTS if path != MODULE_PATH)
    base.CODE_INPUTS = (*inputs, MODULE_PATH)
    base._build_plan = _build_plan


def _old_result(branch: str, config_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source_root = base.SOURCE_ROOTS[branch]
    log_path, success = base._command_source_from_success(source_root, config_id)
    result = {
        "selected_lr": float(success["selected_lr"]),
        "selected_kl": success.get("selected_kl"),
        "formal_global_loss_bsz": int(success["formal_global_loss_bsz"]),
        "source_success": str(source_root / "runs" / config_id / "formal_success.json"),
        "source_log": str(log_path),
    }
    return log_path, success, result


def _flag_map(command: Sequence[str]) -> dict[str, str]:
    if len(command) < 3 or (len(command) - 3) % 2:
        raise base.CampaignError("source command is not a strict flag/value command")
    values: dict[str, str] = {}
    for index in range(3, len(command), 2):
        flag, value = command[index], command[index + 1]
        if not flag.startswith("--") or flag in values:
            raise base.CampaignError(f"invalid or duplicate source flag: {flag}")
        values[flag] = value
    return values


def _build_plan() -> dict[str, Any]:
    configurations: dict[str, Any] = {}
    cache_snapshots: dict[str, Any] = {}
    calibration_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    paired_profile_audit: dict[str, Any] = {}

    for config_id in base.CONFIG_IDS:
        model_slug = next(
            model for model in base.MODEL_SLUGS if config_id.startswith(f"{model}_")
        )
        histories: dict[str, dict[str, Any]] = {}
        logs: dict[str, Path] = {}
        for branch in base.BRANCH_VALUES:
            log_path, _success, result = _old_result(branch, config_id)
            histories[branch] = result
            logs[branch] = log_path

        common_global_loss_bsz = min(
            int(histories[branch]["formal_global_loss_bsz"])
            for branch in base.BRANCH_VALUES
        )
        # Choose a historical successful source whose static cache key exactly
        # matches the conservative paired batch.  Prefer full_block on ties so
        # the same rule is stable for all configs.
        source_branch = next(
            branch
            for branch in ("full_block", "single_linear")
            if int(histories[branch]["formal_global_loss_bsz"])
            == common_global_loss_bsz
        )
        shared_log = logs[source_branch]
        shared_command = base._source_command(shared_log)
        base._set_arg(
            shared_command, "--global_loss_bsz", str(common_global_loss_bsz)
        )
        base._set_arg(
            shared_command, "--grad_lr_layer_schedule", base._schedule_for(config_id)
        )
        base._set_arg(
            shared_command, "--a_loss_ratio", base._a_loss_ratio_text(config_id)
        )

        cache_key = f"shared/{model_slug}"
        # Qwen3-4B has two formally used global-loss batches across its four
        # configs, but both cache files live in the same immutable directory.
        if cache_key not in cache_snapshots:
            cache_snapshots[cache_key] = base._cache_snapshot(shared_command)
        else:
            current = base._cache_snapshot(shared_command)
            if current != cache_snapshots[cache_key]:
                raise base.CampaignError(
                    f"shared cache inventory drift within model {model_slug}"
                )
        token_hash = cache_snapshots[cache_key]["tokens"][0]["sha256"]

        paired_commands: dict[str, list[str]] = {}
        for branch in base.BRANCH_VALUES:
            command = list(shared_command)
            base._set_or_append_arg(
                command, "--full_block_refresh", base.BRANCH_VALUES[branch]
            )
            base._validate_full_profile(
                command, branch=branch, config=config_id
            )
            paired_commands[branch] = command
            calibration_hashes[model_slug][branch] = token_hash
            # The base execution engine indexes cache snapshots by
            # branch/model.  Both entries intentionally reference an exactly
            # equal snapshot of the same physical files.
            cache_snapshots[f"{branch}/{model_slug}"] = copy.deepcopy(
                cache_snapshots[cache_key]
            )
            configurations[f"{branch}/{config_id}"] = {
                "branch": branch,
                "config": config_id,
                "model": model_slug,
                "schedule": base._schedule_for(config_id),
                "a_loss_ratio": base._a_loss_ratio_for(config_id),
                "global_loss_bsz": common_global_loss_bsz,
                "old_proxy_result": histories[branch],
                "paired_cache_source_branch": source_branch,
                "paired_cache_key": cache_key,
                "source_log_snapshot": base._snapshot_path(
                    shared_log, content_hash=True
                ),
                "source_command": command,
            }

        left = _flag_map(paired_commands["full_block"])
        right = _flag_map(paired_commands["single_linear"])
        left_flag = left.pop("--full_block_refresh")
        right_flag = right.pop("--full_block_refresh")
        if left != right or left_flag != "true" or right_flag != "false":
            raise base.CampaignError(
                f"paired command gate failed for {config_id}"
            )
        paired_profile_audit[config_id] = {
            "common_global_loss_bsz": common_global_loss_bsz,
            "shared_source_branch": source_branch,
            "shared_source_log": str(shared_log),
            "shared_static_cache_path": left["--static_cache_path"],
            "shared_reference_cache_dir": left["--cache_dir"],
            "shared_tokens_cache_path": left["--tokens_cache_path"],
            "only_command_difference": "--full_block_refresh=true/false",
            "normalized_command_sha256": base._canonical_sha256(left),
        }

    # Remove internal aliases; branch/model entries remain for the base
    # verifier and are exact deep copies of the shared inventory.
    for key in [key for key in cache_snapshots if key.startswith("shared/")]:
        del cache_snapshots[key]
    for model_slug, hashes in calibration_hashes.items():
        if set(hashes) != set(base.BRANCH_VALUES) or len(set(hashes.values())) != 1:
            raise base.CampaignError(
                f"paired calibration hash gate failed for {model_slug}: {hashes}"
            )

    body = {
        "campaign_id": CAMPAIGN_ID,
        "supersedes_campaign_id": "realq-fullmodel-two-branch-retune-20260817-v1",
        "supersedes_reason": "V1 shared calibration tokens but not global_loss_bsz/static Fisher cache; paired lr=0 controls diverged",
        "output_root": str(OUTPUT_ROOT),
        "branches": base.BRANCH_VALUES,
        "config_ids": list(base.CONFIG_IDS),
        "initial_lrs": [0.0, *POSITIVE_LRS],
        "max_launches_per_branch_config": base.MAX_LAUNCHES_PER_BRANCH_CONFIG,
        "determinism": {
            "seed": 1,
            "rotation_seed": 0,
            "refresh_seed": 0,
            "realq_deterministic_sdpa": True,
            "cublas_workspace_config": ":4096:8",
            "pythonhashseed": "0",
            "nvidia_tf32_override": "0",
        },
        "selection_protocol": {
            "primary_metric": "wikitext2_exact_kl",
            "paired_zero_control_gate": "all 20 configs require exact KL and PPL equality between branches before positive LR release",
            "positive_initial_log_grid": list(POSITIVE_LRS),
            "high_side_requirement": "two successful higher-LR points worse than the incumbent; OOM/infra failures do not count",
            "local_bracket_max_dex": 0.30,
            "top_candidate_min_replicates": 2,
            "tie_rule": "median/MAD or range noise gate; if tied, select the lower LR",
            "two_percent_rule": "reporting-only near-optimal plateau, never a convergence gate",
            "formal_profile_only": True,
        },
        "paired_profile_audit": paired_profile_audit,
        "code_snapshot": base._code_snapshot(),
        "calibration_token_hashes": calibration_hashes,
        "cache_snapshots": cache_snapshots,
        "configurations": configurations,
    }
    body["protocol_fingerprint"] = base._canonical_sha256(body)
    body["created_at"] = base._utc_now()
    return body


def _balanced_pairs() -> list[tuple[str, str]]:
    pairs = []
    for quant in base.QUANT_SLUGS:
        for model in BALANCED_MODELS:
            config = f"{model}_{quant}"
            if config not in base.CONFIG_IDS:
                raise base.CampaignError(f"balanced config missing: {config}")
            for branch in base.BRANCH_VALUES:
                pairs.append((branch, config))
    if len(pairs) != 40 or len(set(pairs)) != 40:
        raise base.CampaignError("balanced paired matrix must contain exactly 40 rows")
    return pairs


def _write_manifest(
    *, name: str, stage: str, rationale: str, lrs: Sequence[float]
) -> Path:
    plan = base._read_json(PLAN_PATH)
    base._verify_plan(plan)
    trials = []
    for lr in lrs:
        for branch, config in _balanced_pairs():
            trials.append(
                {
                    "branch": branch,
                    "config": config,
                    "schedule": base._schedule_for(config),
                    "lr": float(lr),
                    "stage": stage,
                }
            )
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "name": name,
        "rationale": rationale,
        "created_at": base._utc_now(),
        "trials": trials,
    }
    base._validate_manifest(plan, manifest)
    path = OUTPUT_ROOT / "manifests" / f"{name}.json"
    if path.exists():
        raise base.CampaignError(f"manifest already exists: {path}")
    base._atomic_json(path, manifest)
    return path


def _make_zero_manifest(args: argparse.Namespace) -> int:
    path = _write_manifest(
        name=args.name,
        stage="F2-paired-zero-control",
        rationale="paired physical cache/global-loss profile; positive LR remains gated until exact KL/PPL equality passes for all 20 config pairs",
        lrs=(0.0,),
    )
    print(path)
    return 0


def _zero_results() -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    trial_root = OUTPUT_ROOT / "trials"
    if not trial_root.is_dir():
        return rows
    for path in sorted(trial_root.glob("*/result.json")):
        value = base._read_json(path)
        if float(value.get("lr", -1)) != 0:
            continue
        key = (str(value.get("branch")), str(value.get("config")))
        if key in rows:
            raise base.CampaignError(f"duplicate primary zero control: {key}")
        rows[key] = {
            **value,
            "result_path": str(path),
            "result_sha256": base._file_sha256(path),
        }
    return rows


def _audit_zero(_: argparse.Namespace) -> int:
    plan = base._read_json(PLAN_PATH)
    base._verify_plan(plan)
    results = _zero_results()
    rows = []
    failures = []
    for config in base.CONFIG_IDS:
        pair = []
        for branch in base.BRANCH_VALUES:
            key = (branch, config)
            value = results.get(key)
            if value is None:
                failures.append(f"missing:{branch}/{config}")
                continue
            if value.get("status") != "succeeded":
                failures.append(
                    f"failed:{branch}/{config}:{value.get('failure_class')}"
                )
            pair.append(value)
        if len(pair) != 2:
            continue
        by_branch = {str(value["branch"]): value for value in pair}
        left = by_branch["full_block"]
        right = by_branch["single_linear"]
        exact_kl = float(left["kl"]) == float(right["kl"])
        exact_ppl = float(left["ppl"]) == float(right["ppl"])
        if not exact_kl or not exact_ppl:
            failures.append(
                f"mismatch:{config}:kl={left['kl']}/{right['kl']}:ppl={left['ppl']}/{right['ppl']}"
            )
        rows.append(
            {
                "config": config,
                "full_block": {
                    "kl": left["kl"],
                    "ppl": left["ppl"],
                    "result": {
                        "path": left["result_path"],
                        "sha256": left["result_sha256"],
                    },
                },
                "single_linear": {
                    "kl": right["kl"],
                    "ppl": right["ppl"],
                    "result": {
                        "path": right["result_path"],
                        "sha256": right["result_sha256"],
                    },
                },
                "exact_kl_equal": exact_kl,
                "exact_ppl_equal": exact_ppl,
            }
        )
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "status": "passed" if not failures and len(rows) == 20 else "failed",
        "audited_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "counts": {
            "expected_pairs": 20,
            "audited_pairs": len(rows),
            "exact_pairs": sum(
                row["exact_kl_equal"] and row["exact_ppl_equal"] for row in rows
            ),
        },
        "failures": failures,
        "rows": rows,
    }
    identity = dict(payload)
    identity.pop("audited_at")
    payload["audit_fingerprint"] = base._canonical_sha256(identity)
    base._atomic_json(ZERO_AUDIT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise base.CampaignError(
            f"paired zero-control gate failed with {len(failures)} findings"
        )
    return 0


def _load_zero_audit() -> dict[str, Any]:
    audit = base._read_json(ZERO_AUDIT_PATH)
    if audit.get("campaign_id") != CAMPAIGN_ID or audit.get("status") != "passed":
        raise base.CampaignError("paired zero-control audit has not passed")
    identity = dict(audit)
    fingerprint = identity.pop("audit_fingerprint", None)
    identity.pop("audited_at", None)
    if base._canonical_sha256(identity) != fingerprint:
        raise base.CampaignError("paired zero-control audit fingerprint mismatch")
    if audit.get("counts") != {
        "expected_pairs": 20,
        "audited_pairs": 20,
        "exact_pairs": 20,
    }:
        raise base.CampaignError("paired zero-control audit count mismatch")
    return audit


def _make_bracket_manifest(args: argparse.Namespace) -> int:
    audit = _load_zero_audit()
    path = _write_manifest(
        name=args.name,
        stage="F2-positive-global-bracket",
        rationale=(
            "positive four-decade full-model bracket released only after paired "
            f"zero-control audit {audit['audit_fingerprint']} passed"
        ),
        lrs=POSITIVE_LRS,
    )
    print(path)
    return 0


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    zero = subparsers.add_parser("make-zero-manifest")
    zero.add_argument("--name", default="paired_zero_control_v1")
    zero.set_defaults(handler=_make_zero_manifest)
    audit = subparsers.add_parser("audit-zero")
    audit.set_defaults(handler=_audit_zero)
    bracket = subparsers.add_parser("make-bracket-manifest")
    bracket.add_argument("--name", default="positive_global_bracket_v1")
    bracket.set_defaults(handler=_make_bracket_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {
        "make-zero-manifest",
        "audit-zero",
        "make-bracket-manifest",
    }:
        args = _custom_parser().parse_args(arguments)
        try:
            return int(args.handler(args))
        except (
            base.CampaignError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            print(f"fullmodel-retune-v2: {exc}", file=sys.stderr)
            return 2
    return base.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
