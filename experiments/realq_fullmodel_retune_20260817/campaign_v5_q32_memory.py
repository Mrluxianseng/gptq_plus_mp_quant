#!/usr/bin/env python3
"""Qwen3-32B V5 retries with deterministic math-SDPA memory tiling.

V4 remains immutable.  This derivative covers only the eight Qwen3-32B
branch/config curves whose first positive point cannot fit the strict math
SDPA path.  It adopts V4's successful zero endpoints as immutable evidence,
counts both V4 launches against the 20-launch cap, and changes only execution
memory: logical batches, samples, seeds, scheduler, loss, and Adam cadence are
unchanged.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base


CAMPAIGN_ID = "realq-fullmodel-two-branch-retune-20260818-v5-q32-memory"
OUTPUT_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_fullmodel_retune_20260818_v5_q32_memory"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
V4_ROOT = (
    base.REPO_ROOT.parent
    / "experiment_data"
    / "realq_fullmodel_retune_20260817_v4"
)
V4_PLAN_PATH = V4_ROOT / "plan.json"
ENTRY_MODULE = (
    "experiments.realq_fullmodel_retune_20260817.q32_memory_entry"
)
MODULE_PATH = (
    "experiments/realq_fullmodel_retune_20260817/"
    "campaign_v5_q32_memory.py"
)
MEMORY_PATH = (
    "experiments/realq_fullmodel_retune_20260817/"
    "deterministic_sdpa_memory.py"
)
ENTRY_PATH = (
    "experiments/realq_fullmodel_retune_20260817/q32_memory_entry.py"
)
Q32_CONFIGS = tuple(
    config for config in base.CONFIG_IDS if config.startswith("qwen3-32b_")
)


def _bootstrap() -> None:
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.PLAN_PATH = PLAN_PATH
    base.CONFIG_IDS = Q32_CONFIGS
    base.MODEL_SLUGS = ("qwen3-32b",)
    base.INITIAL_LRS = ()
    base._a_loss_ratio_for = lambda _config: 1.0
    base._a_loss_ratio_text = lambda _config: "1"
    inputs = tuple(
        path
        for path in base.CODE_INPUTS
        if path not in {MODULE_PATH, MEMORY_PATH, ENTRY_PATH}
    )
    base.CODE_INPUTS = (*inputs, MEMORY_PATH, ENTRY_PATH, MODULE_PATH)
    base._build_plan = _build_plan


def _replace_entry(command: list[str]) -> None:
    if len(command) < 3 or command[1:3] != ["-m", "realq.ptq"]:
        raise base.CampaignError(
            f"unexpected PTQ entry in source command: {command[:3]!r}"
        )
    command[2] = ENTRY_MODULE


def _snapshot_result(path: Path) -> dict[str, Any]:
    result = base._read_json(path)
    if result.get("status") != "succeeded" or float(result.get("lr", -1)) != 0.0:
        raise base.CampaignError(f"adopted V4 zero is not successful: {path}")
    return {
        "path": str(path),
        "sha256": base._file_sha256(path),
        "branch": result["branch"],
        "config": result["config"],
        "kl": result["kl"],
        "ppl": result["ppl"],
    }


def _build_plan() -> dict[str, Any]:
    v4_plan = base._read_json(V4_PLAN_PATH)
    stable_v4 = dict(v4_plan)
    v4_fingerprint = stable_v4.pop("protocol_fingerprint", None)
    stable_v4.pop("created_at", None)
    if base._canonical_sha256(stable_v4) != v4_fingerprint:
        raise base.CampaignError("source V4 plan fingerprint mismatch")

    configurations: dict[str, Any] = {}
    paired_profile_audit: dict[str, Any] = {}
    adopted_zero: list[dict[str, Any]] = []
    for config in Q32_CONFIGS:
        paired_profile_audit[config] = copy.deepcopy(
            v4_plan["paired_profile_audit"][config]
        )
        for branch in base.BRANCH_VALUES:
            key = f"{branch}/{config}"
            row = copy.deepcopy(v4_plan["configurations"][key])
            command = row["source_command"]
            _replace_entry(command)
            base._validate_full_profile(command, branch=branch, config=config)
            row["source_command"] = command
            configurations[key] = row

            identity = base._trial_id(
                {
                    "branch": branch,
                    "config": config,
                    "schedule": base._schedule_for(config),
                    "lr": 0.0,
                }
            )
            adopted_zero.append(
                _snapshot_result(V4_ROOT / "trials" / identity / "result.json")
            )

    cache_snapshots = {
        key: copy.deepcopy(value)
        for key, value in v4_plan["cache_snapshots"].items()
        if key.endswith("/qwen3-32b")
    }
    body = {
        "campaign_id": CAMPAIGN_ID,
        "supersedes_campaign_id": v4_plan["campaign_id"],
        "supersedes_protocol_fingerprint": v4_plan["protocol_fingerprint"],
        "supersedes_reason": (
            "strict deterministic math SDPA needs a 32-GiB workspace at "
            "logical backward_bsz=32; V5 adds only batch-axis SDPA tiling "
            "and non-reentrant block recomputation"
        ),
        "output_root": str(OUTPUT_ROOT),
        "branches": base.BRANCH_VALUES,
        "config_ids": list(Q32_CONFIGS),
        "initial_lrs": [],
        # Every curve already spent V4 zero + V4 first-positive attempt.
        "prior_launch_count_per_branch_config": 2,
        "max_launches_per_branch_config": 18,
        "determinism": copy.deepcopy(v4_plan["determinism"]),
        "selection_protocol": {
            **copy.deepcopy(v4_plan["selection_protocol"]),
            "candidate_release": (
                "agent-reviewed directed points; adopted V4 zero plus V5 "
                "deterministic-memory positive points"
            ),
        },
        "memory_execution": {
            "scope": "Qwen3-32B grad-enabled Transformer-block calls only",
            "logical_backward_samples": 32,
            "logical_backward_bsz": 32,
            "final_layer_backward_bsz": 32,
            "hessian_accum_bsz": 32,
            "attention_backend": "sdpa",
            "deterministic_sdpa": "math only",
            "math_sdpa_batch_tile": 4,
            "checkpoint": "non-reentrant, preserve_rng_state=true",
            "loss_and_adam_cadence_changed": False,
        },
        "code_snapshot": base._code_snapshot(),
        "calibration_token_hashes": {
            "qwen3-32b": copy.deepcopy(
                v4_plan["calibration_token_hashes"]["qwen3-32b"]
            )
        },
        "cache_snapshots": cache_snapshots,
        "paired_profile_audit": paired_profile_audit,
        "adopted_v4_zero_results": sorted(
            adopted_zero, key=lambda row: (row["config"], row["branch"])
        ),
        "configurations": configurations,
    }
    body["protocol_fingerprint"] = base._canonical_sha256(body)
    body["created_at"] = base._utc_now()
    return body


def _canary_environment(cuda_id: str) -> dict[str, str]:
    environment = os.environ.copy()
    for key in base.TORCH_DISTRIBUTED_ENV:
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES=cuda_id,
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


def _run_canary(args: argparse.Namespace) -> int:
    plan = base._read_json(PLAN_PATH)
    base._verify_plan(plan)
    trial = {
        "branch": args.branch,
        "config": args.config,
        "schedule": base._schedule_for(args.config),
        "lr": float(args.lr),
        "stage": "V5-Q32-layer0-memory-canary",
    }
    base._validate_trial(plan, trial)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    canary_dir = OUTPUT_ROOT / "canaries" / (
        f"{args.branch}__{args.config}__lr_{base._stable_float(float(args.lr))}__{stamp}"
    )
    canary_dir.mkdir(parents=True, exist_ok=False)
    command = base._build_trial_command(plan, trial, canary_dir)
    base._set_or_append_arg(command, "--quant_stop_layer", "0")
    base._set_arg(command, "--skip_eval", "true")
    base._set_arg(command, "--skip_kl_ppl_eval", "true")
    gpu = base._gpu_inventory(args.cuda_id)
    started_at = base._utc_now()
    spec = {
        "campaign_id": CAMPAIGN_ID,
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "stage": trial["stage"],
        "branch": args.branch,
        "config": args.config,
        "lr": float(args.lr),
        "cuda_visible_devices": args.cuda_id,
        "gpu": gpu,
        "command": command,
        "command_sha256": base._canonical_sha256(command),
        "created_at": started_at,
    }
    base._atomic_json(canary_dir / "spec.json", spec)
    log_path = canary_dir / "execution.log"
    started = time.monotonic()
    with log_path.open("wb") as handle:
        handle.write(
            (
                f"[{started_at}] command={json.dumps(command, ensure_ascii=False)}\n"
                f"[{started_at}] gpu={json.dumps(gpu, ensure_ascii=False)}\n"
                f"[{started_at}] protocol_fingerprint={plan['protocol_fingerprint']}\n"
            ).encode("utf-8")
        )
        process = subprocess.Popen(
            command,
            cwd=base.REPO_ROOT,
            env=_canary_environment(args.cuda_id),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        returncode = process.wait()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    reached = "[realq] quant_stop_layer=0 reached" in text
    wrapper_marker = "[realq.q32_memory] wrappers installed" in text
    result = {
        **spec,
        "started_at": started_at,
        "finished_at": base._utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "returncode": returncode,
        "layer0_reached": reached,
        "wrapper_marker": wrapper_marker,
        "status": (
            "succeeded"
            if returncode == 0 and reached and wrapper_marker
            else "failed"
        ),
        "log_path": str(log_path),
    }
    base._atomic_json(canary_dir / "result.json", result)
    print(canary_dir)
    return 0 if result["status"] == "succeeded" else 1


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    canary = subparsers.add_parser("run-canary")
    canary.add_argument("--branch", choices=base.BRANCH_VALUES, required=True)
    canary.add_argument("--config", choices=Q32_CONFIGS, required=True)
    canary.add_argument("--lr", type=float, default=1e-7)
    canary.add_argument("--cuda-id", required=True)
    canary.set_defaults(handler=_run_canary)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _bootstrap()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "run-canary":
        args = _custom_parser().parse_args(arguments)
        try:
            return int(args.handler(args))
        except (
            base.CampaignError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
        ) as exc:
            print(f"q32-memory-v5: {exc}", file=sys.stderr)
            return 2
    return base.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
