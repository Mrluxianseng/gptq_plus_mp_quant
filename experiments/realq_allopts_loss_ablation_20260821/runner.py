#!/usr/bin/env python3
"""Source-locked three-setting run15 all-on/all-off loss ablation."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from tools import realq_auto_tune as tuner


OUTPUT_ROOT = (
    REPO_ROOT.parent
    / "experiment_data"
    / "realq_allopts_loss_ablation_20260821_v1"
)
PLAN_PATH = OUTPUT_ROOT / "plan.json"
FORMAL_PLAN_PATH = (
    REPO_ROOT.parent
    / "experiment_data"
    / "realq_fullmodel_retune_20260818_v6_run15"
    / "formal_plan.json"
)
V6_PLAN_PATH = FORMAL_PLAN_PATH.parent / "plan.json"
LEGACY_CACHE_ROOT = (
    REPO_ROOT.parent
    / "experiment_data"
    / "realq_20group_20260808"
    / "shared_cache"
)
CANONICAL_CACHE_ROOT = OUTPUT_ROOT / "canonical_sdpa_cache"
LOCK_ROOT = (
    REPO_ROOT.parent
    / "experiment_data"
    / "_fair20_physical_gpu_locks_20260821"
)

CONFIGS = (
    "qwen3-0.6b_w4a16",
    "qwen3-4b_w3a16",
    "qwen3-8b_w4a4kv4",
)
MODEL_FOR_CONFIG = {
    "qwen3-0.6b_w4a16": "qwen3-0.6b",
    "qwen3-4b_w3a16": "qwen3-4b",
    "qwen3-8b_w4a4kv4": "qwen3-8b",
}
ALL_ON_FLAGS = {
    "--attention_backend": "flash_attention_4",
    "--hessian_tf32": "true",
    "--quantizer_inner_fastpath": "true",
    "--w_clip_search_impl": "symmetric_union_exact",
    "--fisher_fp32_cache": "true",
    "--act_order_stitch_impl": "prefix_q_trailing_w_exact",
    "--w_clip_update_impl": "where_out",
    "--w_group_param_layout": "compact",
    "--prepared_clamp_bound_cache": "true",
    "--triton_column_block": "true",
    "--fused_block_adam": "true",
}
ALL_OFF_FLAGS = {
    "--attention_backend": "sdpa",
    "--hessian_tf32": "false",
    "--quantizer_inner_fastpath": "false",
    "--w_clip_search_impl": "cartesian_legacy",
    "--fisher_fp32_cache": "false",
    "--act_order_stitch_impl": "full_weight_legacy",
    "--w_clip_update_impl": "guarded",
    "--w_group_param_layout": "expanded",
    "--prepared_clamp_bound_cache": "false",
    "--triton_column_block": "false",
    "--fused_block_adam": "false",
}
SOURCE_FILES = (
    "experiments/realq_allopts_loss_ablation_20260821/runner.py",
    "experiments/realq_allopts_loss_ablation_20260821/legacy_driver.py",
    "realq/config.py",
    "realq/pipeline.py",
    "realq/quant/realq_layer.py",
    "realq/refresh/block_gd.py",
    "realq/refresh/fisher_loss.py",
    "realq/runner/layer_loop.py",
    "utils/hadamard_utils.py",
    "utils/quant_utils.py",
    "utils/rotation_utils.py",
)


class AblationError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flags(command: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 3
    while index < len(command):
        flag = command[index]
        if not flag.startswith("--") or index + 1 >= len(command):
            raise AblationError(f"malformed command near {command[index:]}")
        if flag in result:
            raise AblationError(f"duplicate command flag: {flag}")
        result[flag] = command[index + 1]
        index += 2
    return result


def _set(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) == 1:
        command[positions[0] + 1] = value
    elif not positions:
        command.extend((flag, value))
    else:
        raise AblationError(f"duplicate flag {flag}")


def _drop(command: list[str], flag: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) > 1:
        raise AblationError(f"duplicate flag {flag}")
    if positions:
        del command[positions[0] : positions[0] + 2]


def _formal_rows() -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    formal = _read(FORMAL_PLAN_PATH)
    stable = dict(formal)
    fingerprint = stable.pop("formal_plan_fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise AblationError("formal plan fingerprint mismatch")
    rows = {
        row["config"]: row
        for row in formal["rows"]
        if row["branch"] == "full_block" and row["config"] in CONFIGS
    }
    if set(rows) != set(CONFIGS):
        raise AblationError("formal plan is missing a selected ablation config")
    for config, row in rows.items():
        command = list(map(str, row["command"]))
        if base._canonical_sha256(command) != row["command_sha256"]:
            raise AblationError(f"formal command hash mismatch: {config}")
        values = _flags(command)
        for flag, expected in ALL_ON_FLAGS.items():
            if values.get(flag) != expected:
                raise AblationError(
                    f"source {config} {flag}={values.get(flag)!r}, "
                    f"expected {expected!r}"
                )
        frozen = {
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
            "--act_order": "true",
            "--rotate": "true",
            "--full_block_refresh": "true",
            "--a_loss_ratio": "1",
        }
        for flag, expected in frozen.items():
            if values.get(flag) != expected:
                raise AblationError(f"source {config} drifted {flag}")
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise AblationError(f"all-on checkpoint missing: {checkpoint}")
    return formal, rows


def _source_snapshot() -> dict[str, Any]:
    files = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        files.append(
            {
                "path": relative,
                "sha256": base._file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"files": files, "sha256": base._canonical_sha256(files)}


def _build_plan() -> dict[str, Any]:
    formal, rows = _formal_rows()
    body = {
        "schema_version": 1,
        "experiment_id": "realq-run15-allopts-loss-ablation-20260821-v1",
        "formal_plan": {
            "path": str(FORMAL_PLAN_PATH),
            "sha256": base._file_sha256(FORMAL_PLAN_PATH),
            "fingerprint": formal["formal_plan_fingerprint"],
        },
        "v6_plan": {
            "path": str(V6_PLAN_PATH),
            "sha256": base._file_sha256(V6_PLAN_PATH),
        },
        "source_snapshot": _source_snapshot(),
        "configs": [
            {
                "config": config,
                "model": MODEL_FOR_CONFIG[config],
                "selected_lr": rows[config]["selected_lr"],
                "source_command_sha256": rows[config]["command_sha256"],
                "all_on_checkpoint": rows[config]["checkpoint"],
                "all_on_checkpoint_size": Path(rows[config]["checkpoint"]).stat().st_size,
            }
            for config in CONFIGS
        ],
        "all_on_flags": ALL_ON_FLAGS,
        "all_off_flags": ALL_OFF_FLAGS,
        "fixed_contract": {
            "same_model_config_calibration_tokens_seeds_lr_schedule": True,
            "all_off_driver": (
                "eager Qwen3 RMSNorm+RoPE/SwiGLU, eager FP32 Fisher, eager "
                "activation QDQ, separate Fast-Hadamard scaling"
            ),
            "canonical_metric": "WikiText2 exact full-vocabulary KL and PPL",
            "canonical_evaluator": "deterministic SDPA, FP32 matmul, all fixed fusions off",
            "all_on_reuses_source_locked_formal_checkpoint": True,
        },
    }
    body["plan_fingerprint"] = base._canonical_sha256(body)
    return body


def _verify_plan() -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    plan = _read(PLAN_PATH)
    stable = dict(plan)
    fingerprint = stable.pop("plan_fingerprint", None)
    if base._canonical_sha256(stable) != fingerprint:
        raise AblationError("ablation plan fingerprint mismatch")
    rebuilt = _build_plan()
    if rebuilt != plan:
        raise AblationError("ablation plan inputs or source files drifted")
    _, rows = _formal_rows()
    return plan, rows


def init_plan() -> dict[str, Any]:
    plan = _build_plan()
    if PLAN_PATH.is_file():
        if _read(PLAN_PATH) != plan:
            raise AblationError("existing plan differs from current inputs")
        return plan
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    base._atomic_json(PLAN_PATH, plan)
    return plan


def _legacy_environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    for key in base.TORCH_DISTRIBUTED_ENV:
        environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
            "REALQ_DETERMINISTIC_SDPA": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": os.pathsep.join(
                value
                for value in (str(REPO_ROOT), environment.get("PYTHONPATH", ""))
                if value
            ),
        }
    )
    return environment


def _all_off_checkpoint(config: str) -> Path:
    return OUTPUT_ROOT / "runs" / config / "all_off" / "checkpoint" / "quantized.pt"


def _stage_dir(config: str, stage: str) -> Path:
    return OUTPUT_ROOT / "runs" / config / stage


def _legacy_module(command: list[str]) -> None:
    if command[:2] != [command[0], "-m"] or command[2] != "realq.ptq":
        raise AblationError("unexpected source executable/module")
    command[2] = "experiments.realq_allopts_loss_ablation_20260821.legacy_driver"


def _all_off_quant_command(row: Mapping[str, Any], config: str) -> list[str]:
    command = list(map(str, row["command"]))
    _legacy_module(command)
    for flag, value in ALL_OFF_FLAGS.items():
        _set(command, flag, value)
    model = MODEL_FOR_CONFIG[config]
    cache = LEGACY_CACHE_ROOT / model
    _set(command, "--static_cache_path", str(cache / "static"))
    _set(command, "--cache_dir", str(cache / "runtime"))
    _set(command, "--require_static_cache_hit", "true")
    _set(command, "--require_reference_cache_hit", "false")
    _set(command, "--skip_eval", "true")
    _set(command, "--skip_kl_ppl_eval", "true")
    _set(command, "--lm_eval", "false")
    _set(command, "--reasoning_eval", "false")
    _set(command, "--save_qmodel_path", str(_all_off_checkpoint(config)))
    output = _stage_dir(config, "all_off") / "realq_output"
    _set(command, "--output_dir", str(output))
    _set(command, "--exp", f"allopts_off_{config}")
    return command


def _canonical_eval_command(
    row: Mapping[str, Any], config: str, arm: str
) -> list[str]:
    if arm not in {"all_on", "all_off"}:
        raise AblationError(f"unknown canonical arm: {arm}")
    command = list(map(str, row["command"]))
    _legacy_module(command)
    checkpoint = (
        Path(row["checkpoint"])
        if arm == "all_on"
        else _all_off_checkpoint(config)
    )
    _drop(command, "--save_qmodel_path")
    _set(command, "--load_qmodel_path", str(checkpoint))
    _set(command, "--attention_backend", "sdpa")
    _set(command, "--hessian_tf32", "false")
    _set(command, "--skip_eval", "false")
    _set(command, "--skip_kl_ppl_eval", "false")
    _set(command, "--lm_eval", "false")
    _set(command, "--reasoning_eval", "false")
    _set(command, "--require_static_cache_hit", "false")
    _set(
        command,
        "--require_reference_cache_hit",
        "false" if arm == "all_on" else "true",
    )
    model = MODEL_FOR_CONFIG[config]
    _set(command, "--cache_dir", str(CANONICAL_CACHE_ROOT / model))
    output = _stage_dir(config, f"canonical_{arm}") / "realq_output"
    _set(command, "--output_dir", str(output))
    _set(command, "--exp", f"canonical_sdpa_{arm}_{config}")
    return command


def _validate_pair_commands(
    source: Sequence[str], all_off: Sequence[str], evaluations: Sequence[Sequence[str]]
) -> None:
    source_flags = _flags(source)
    off_flags = _flags(all_off)
    mutable = set(ALL_OFF_FLAGS) | {
        "--static_cache_path",
        "--cache_dir",
        "--require_static_cache_hit",
        "--require_reference_cache_hit",
        "--save_qmodel_path",
        "--output_dir",
        "--exp",
    }
    for flag, value in source_flags.items():
        if flag not in mutable and off_flags.get(flag) != value:
            raise AblationError(f"all-off command drifted scientific flag {flag}")
    for flag, expected in ALL_OFF_FLAGS.items():
        if off_flags.get(flag) != expected:
            raise AblationError(f"all-off command did not set {flag}")
    for command in evaluations:
        values = _flags(command)
        if values.get("--attention_backend") != "sdpa":
            raise AblationError("canonical evaluation must use SDPA")
        if values.get("--skip_kl_ppl_eval") != "false":
            raise AblationError("canonical exact KL/PPL is disabled")
        if "--save_qmodel_path" in values or "--load_qmodel_path" not in values:
            raise AblationError("canonical evaluation is not checkpoint-only")


def _execute_stage(
    config: str,
    stage: str,
    command: list[str],
    gpu: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    directory = _stage_dir(config, stage)
    result_path = directory / "result.json"
    if result_path.is_file():
        result = _read(result_path)
        if result.get("status") == "succeeded":
            return result
        raise AblationError(f"existing stage failed: {result_path}")
    if directory.exists() or directory.is_symlink():
        raise AblationError(f"refusing incomplete stage directory: {directory}")
    directory.mkdir(parents=True)
    log_path = directory / "execution.log"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "config": config,
        "stage": stage,
        "hostname": socket.gethostname(),
        "physical_gpu": gpu,
        "plan_fingerprint": plan["plan_fingerprint"],
        "command": command,
        "command_sha256": base._canonical_sha256(command),
        "started_at": base._utc_now(),
    }
    base._atomic_json(directory / "manifest.json", manifest)
    started = time.monotonic()
    with log_path.open("xb") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=_legacy_environment(gpu),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {
        **manifest,
        "status": "failed",
        "returncode": completed.returncode,
        "finished_at": base._utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log_path),
        "log_sha256": base._file_sha256(log_path),
    }
    if completed.returncode == 0:
        if stage.startswith("canonical_"):
            kl, ppl = tuner.parse_exact_metric(
                log_path.read_text(encoding="utf-8", errors="replace"),
                "wikitext2",
            )
            result.update(exact_kl=kl, ppl=ppl)
        elif not _all_off_checkpoint(config).is_file():
            raise AblationError("all-off quantization returned zero without a checkpoint")
        result["status"] = "succeeded"
    base._atomic_json(result_path, result)
    if result["status"] != "succeeded":
        raise AblationError(f"stage failed: {config}/{stage}")
    return result


def execute_pair(config: str, gpu: int) -> dict[str, Any]:
    if config not in CONFIGS:
        raise AblationError(f"unsupported config: {config}")
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise AblationError("pair must run inside a Canoe debug pod")
    if gpu < 0 or gpu > 7:
        raise AblationError(f"invalid physical GPU: {gpu}")
    plan, rows = _verify_plan()
    row = rows[config]
    off_command = _all_off_quant_command(row, config)
    eval_on = _canonical_eval_command(row, config, "all_on")
    eval_off = _canonical_eval_command(row, config, "all_off")
    _validate_pair_commands(row["command"], off_command, (eval_on, eval_off))

    lock = LOCK_ROOT / hostname / f"gpu{gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        on_result = _execute_stage(config, "canonical_all_on", eval_on, gpu, plan)
        _execute_stage(config, "all_off", off_command, gpu, plan)
        off_result = _execute_stage(config, "canonical_all_off", eval_off, gpu, plan)

    on_kl = float(on_result["exact_kl"])
    off_kl = float(off_result["exact_kl"])
    summary = {
        "schema_version": 1,
        "status": "succeeded",
        "config": config,
        "selected_lr": row["selected_lr"],
        "canonical_all_on": {
            "exact_kl": on_kl,
            "ppl": on_result["ppl"],
        },
        "canonical_all_off": {
            "exact_kl": off_kl,
            "ppl": off_result["ppl"],
        },
        "off_minus_on_kl": off_kl - on_kl,
        "off_vs_on_relative_kl": (
            (off_kl - on_kl) / on_kl if on_kl != 0.0 else math.nan
        ),
        "finished_at": base._utc_now(),
        "plan_fingerprint": plan["plan_fingerprint"],
    }
    base._atomic_json(_stage_dir(config, "pair_result.json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-plan", action="store_true")
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    try:
        if args.init_plan:
            print(json.dumps(init_plan(), indent=2, sort_keys=True), flush=True)
            return 0
        if args.config is None or args.physical_gpu is None:
            parser.error("--config and --physical-gpu are required unless --init-plan")
        print(
            json.dumps(
                execute_pair(args.config, args.physical_gpu),
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
