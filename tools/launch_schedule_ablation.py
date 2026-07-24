#!/usr/bin/env python3
"""Expand one fixed schedule-ablation case into an exact torchrun command."""

from __future__ import annotations

import argparse
from dataclasses import fields
import os
from pathlib import Path
import shlex
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from realq.config import Config
from tools.run_schedule_ablation_variant import (
    _CASE_PROTOCOLS,
    _COMMON_CASE_CONFIG,
    _case_paths,
)


def _config_cli(config: dict) -> list[str]:
    known = {field.name for field in fields(Config)}
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(f"Manifest contains unknown Config fields: {unknown}")
    argv: list[str] = []
    for field in fields(Config):
        if field.name == "model_name" or field.name not in config:
            continue
        value = config[field.name]
        if value is None:
            continue
        flag = f"--{field.name}"
        if field.name == "allow_unsafe_legacy_checkpoint":
            if value:
                argv.append(flag)
        elif isinstance(value, bool):
            argv.extend([flag, str(value).lower()])
        elif isinstance(value, list):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        else:
            argv.extend([flag, str(value)])
    return argv


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    experiment_root = args.experiment_root.resolve()
    dataset_dir = args.wikitext_parquet_dir.resolve()
    model = args.model.resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"WikiText parquet directory does not exist: {dataset_dir}"
        )

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError(
            "--gpus must name exactly four distinct visible GPU IDs."
        )
    protocol = _CASE_PROTOCOLS[args.case]
    variant = "paper" if args.warm else args.variant
    if variant is None:
        raise ValueError("--variant is required unless --warm is set.")
    if variant not in protocol["variants"]:
        raise ValueError(
            f"Variant {variant!r} is not defined for case {args.case!r}."
        )

    paths = _case_paths(
        experiment_root,
        args.case,
        variant,
        warm=args.warm,
    )
    config = {
        **_COMMON_CASE_CONFIG,
        **protocol["config"],
        "model": str(model),
        "static_cache_path": str(paths["static_cache_path"]),
        "tokens_cache_path": str(paths["tokens_cache_path"]),
        "cache_dir": str(paths["cache_dir"]),
        "output_dir": str(paths["output_dir"]),
        "exp": (
            f"{args.case}__warm"
            if args.warm
            else f"{args.case}__{variant}"
        ),
        "exit_after_precompute": bool(args.warm),
    }

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "4",
        "--master_port",
        str(args.master_port),
        "--module",
        "tools.run_schedule_ablation_variant",
        "--experiment_case",
        args.case,
        "--experiment_root",
        str(experiment_root),
        "--schedule_ablation_variant",
        variant,
        "--wikitext_parquet_dir",
        str(dataset_dir),
        "--rotation_fingerprint_path",
        str(paths["rotation_fingerprint_path"]),
    ]
    if args.warm:
        command.append("--allow_create_rotation_fingerprint")
    else:
        command.extend(
            [
                "--metrics_json_path",
                str(paths["metrics_json_path"]),
                "--require_precomputed_caches",
            ]
        )
    command.extend(_config_cli(config))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    return command, env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--wikitext-parquet-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=tuple(_CASE_PROTOCOLS))
    parser.add_argument("--variant", choices=(
        "paper",
        "paper_sin_scheduled",
        "legacy_cos2_scheduled",
    ))
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    command, env = build_command(args)
    print(
        "CUDA_VISIBLE_DEVICES="
        f"{shlex.quote(env['CUDA_VISIBLE_DEVICES'])} "
        + shlex.join(command),
        flush=True,
    )
    if args.dry_run:
        return
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
