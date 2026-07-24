#!/usr/bin/env python3
"""Run a controlled REAL-Q learning-rate-schedule ablation.

This is an experiment-only entry point.  It executes the current refactored
pipeline and changes exactly one function: the transformer-layer learning-rate
schedule.  Checking out an old repository revision would mix the schedule
change with unrelated Fisher, clipping, rotation, grouped-quantization, and
evaluation fixes.

Three variants separate the two schedule corrections:

``paper``
    Current production behavior: ``sin(pi*x/2)`` for weight-only/unaware
    runs, and a constant reported LR for activation-aware runs.

``paper_sin_scheduled``
    Counterfactual intermediate arm: use the paper's sine curve even when
    activation-aware.  Comparing this with ``paper`` isolates disabling the
    schedule in aware mode.

``legacy_cos2_scheduled``
    Historical behavior: ``0.5*(1-cos(pi*x))`` for every run, including
    activation-aware.  Comparing this with ``paper_sin_scheduled`` isolates
    the curve-formula correction.

The wrapper also replaces the repository's historical local-dataset-script
lookup with an explicitly revision-pinned WikiText-2 parquet directory and
writes unrounded KL/PPL plus provenance to JSON.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import math
import os

# Match realq.ptq's process contract before any direct or indirect torch
# import.  This wrapper imports runner modules before dispatching ptq.main, so
# setting these only inside realq.ptq would be too late for CUDA/cuBLAS.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import platform
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from typing import Any


VARIANTS = (
    "paper",
    "paper_sin_scheduled",
    "legacy_cos2_scheduled",
)

_SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
_EXPECTED_SPLIT_SHA256 = {
    "train": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "validation": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
    "test": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
}

_COMMON_CASE_CONFIG = {
    "dataset": "wikitext2",
    "eval_datasets": ["wikitext2"],
    "seed": 1,
    "rotation_seed": 0,
    "refresh_seed": 0,
    "seq_len": 2048,
    "eval_seq_len": 2048,
    "w_asym": False,
    "w_clip": True,
    "num_groups": 4,
    "percdamp": 0.01,
    "blocksize": 128,
    "act_order": True,
    "group_parallel_quant": "rank",
    "saliency_clip_percentile": 0.99,
    "grad_hessian_topk": -1,
    "alignment_trace_path": None,
    "alignment_run_id": "default",
    "grad_clip": 1.0,
    "final_layer_grad_clip": None,
    "grad_lr_layer_schedule": "cosine",
    "grad_lr_layer_base_ratio": 0.01,
    "backward_samples": 32,
    "backward_bsz": 128,
    "final_layer_backward_bsz": 32,
    "bsz": 64,
    "fsdp": False,
    "fsdp_cpu_offload": False,
    "fsdp_max_shard_size": "5GB",
    "fsdp_prepared_dir": None,
    "cpu_master": False,
    "a_groupsize": -1,
    "a_asym": False,
    "v_groupsize": -1,
    "v_asym": False,
    "k_groupsize": -1,
    "k_asym": False,
    "loss_slide_window": True,
    "kl_topk": -1,
    "rotate": True,
    "optimized_rotation_path": None,
    "skip_eval": False,
    "lm_eval": False,
    "lm_eval_batch_size": 64,
    "quant_stop_layer": None,
    "nsys_profile": False,
    "load_qmodel_path": None,
    "allow_unsafe_legacy_checkpoint": False,
    "save_qmodel_path": None,
}

_CASE_PROTOCOLS = {
    "qwen3_4b_w4a16": {
        "model_name": "Qwen3-4B",
        "model_artifact_identity": (
            "64b5baa184e0fb3676b4d52c6b662a1a20ebf8e1e51ffb495e7d47b60063130d"
        ),
        "variants": {"paper", "legacy_cos2_scheduled"},
        "config": {
            "nsamples": 2048,
            "w_bits": 4,
            "w_groupsize": -1,
            "global_loss_bsz": 48,
            "grad_lr": 2e-4,
            "a_loss_ratio": 0.95,
            "hessian_accum_bsz": 256,
            "a_bits": 16,
            "a_clip_ratio": 1.0,
            "v_bits": 16,
            "v_clip_ratio": 1.0,
            "k_bits": 16,
            "k_clip_ratio": 1.0,
            "act_quant_aware_gptq": False,
            "k_cache_quant_aware_gptq": False,
            "final_layer_grad_lr": 1e-5,
        },
    },
    "qwen3_8b_w2a4kv4_aware": {
        "model_name": "Qwen3-8B",
        "model_artifact_identity": (
            "d5ce437fdf8f2437c50213abe869a8c4780273a9bb79076aba44477af875220b"
        ),
        "variants": set(VARIANTS),
        "config": {
            "nsamples": 256,
            "w_bits": 2,
            "w_groupsize": 128,
            "global_loss_bsz": 32,
            "grad_lr": 5e-6,
            "a_loss_ratio": 1.0,
            "hessian_accum_bsz": 64,
            "a_bits": 4,
            "a_clip_ratio": 0.9,
            "v_bits": 4,
            "v_clip_ratio": 0.9,
            "k_bits": 4,
            "k_clip_ratio": 0.9,
            "act_quant_aware_gptq": True,
            "k_cache_quant_aware_gptq": True,
            "final_layer_grad_lr": 1e-6,
        },
    },
}


def schedule_ablation_lr(
    variant: str,
    base_lr: float,
    layer_idx: int,
    num_layers: int,
    base_ratio: float,
    schedule: str,
    *,
    activation_aware: bool,
) -> float:
    """Independent scalar definition for the three controlled arms."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown schedule ablation variant {variant!r}.")
    if schedule == "none":
        return float(base_lr)
    if schedule != "cosine":
        raise ValueError(
            f"Unknown grad_lr_layer_schedule={schedule!r}; "
            "expected 'none' or 'cosine'."
        )
    if variant == "paper" and activation_aware:
        return float(base_lr)
    if num_layers <= 1:
        scale = 1.0
    else:
        x = float(layer_idx) / float(num_layers - 1)
        if variant == "legacy_cos2_scheduled":
            scale = 0.5 * (1.0 - math.cos(math.pi * x))
        else:
            scale = math.sin(math.pi * x / 2.0)
    return float(base_lr) * (
        float(base_ratio) + (1.0 - float(base_ratio)) * scale
    )


def _dispatch_schedule(
    variant: str,
    production_schedule,
    base_lr: float,
    layer_idx: int,
    num_layers: int,
    base_ratio: float,
    schedule: str,
    *,
    activation_aware: bool,
) -> float:
    """Delegate the paper arm to production; isolate only counterfactuals."""
    if variant == "paper":
        return float(
            production_schedule(
                base_lr,
                layer_idx,
                num_layers,
                base_ratio,
                schedule,
                activation_aware=activation_aware,
            )
        )
    return schedule_ablation_lr(
        variant,
        base_lr,
        layer_idx,
        num_layers,
        base_ratio,
        schedule,
        activation_aware=activation_aware,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_pin(
    parquet_dir: Path,
) -> tuple[dict[str, Path], dict[str, str], str]:
    """Validate the exact official WikiText files and return a cache tag."""
    split_paths = {
        split: parquet_dir / filename
        for split, filename in _SPLIT_FILES.items()
    }
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing revision-pinned WikiText-2 parquet files: "
            + ", ".join(missing)
        )
    observed = {split: _sha256(path) for split, path in split_paths.items()}
    mismatches = {
        split: {
            "expected": _EXPECTED_SPLIT_SHA256[split],
            "observed": digest,
        }
        for split, digest in observed.items()
        if digest != _EXPECTED_SPLIT_SHA256[split]
    }
    if mismatches:
        raise ValueError(
            "WikiText-2 parquet SHA256 mismatch for pinned revision "
            f"{_DATASET_REVISION}: {mismatches}"
        )
    identity_payload = "|".join(
        f"{split}={observed[split]}" for split in sorted(observed)
    )
    identity = hashlib.sha256(identity_payload.encode()).hexdigest()
    return split_paths, observed, identity


def _token_tensor_identity(tokens: list[Any]) -> str:
    """Hash the exact sampled token tensors, including order and shape."""
    digest = hashlib.sha256()
    digest.update(f"count={len(tokens)}|".encode())
    for index, token in enumerate(tokens):
        tensor = token.detach().cpu().contiguous()
        digest.update(
            (
                f"index={index}|dtype={tensor.dtype}|"
                f"shape={tuple(tensor.shape)}|"
            ).encode()
        )
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _revision_scoped_token_path(
    save_path: str | None,
    *,
    dataset_identity: str,
    tokenizer_identity: str,
) -> str | None:
    """Keep historical basename caches outside this pinned experiment."""
    if save_path is None:
        return None
    path = Path(save_path)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    scoped_name = (
        f"{stem}.dataset-{dataset_identity[:12]}."
        f"tokenizer-{tokenizer_identity[:12]}{suffix}"
    )
    return str(path.with_name(scoped_name))


def _git_output(workspace: Path, *args: str) -> str:
    workspace = workspace.resolve()
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={workspace}", *args],
        cwd=workspace,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {workspace}: "
            f"{completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _source_snapshot(workspace: Path) -> dict[str, str]:
    repo_root = Path(
        _git_output(workspace, "rev-parse", "--show-toplevel")
    ).resolve()
    # Compare against HEAD rather than only the work tree so staged changes
    # cannot disappear from the recorded source identity.
    diff = _git_output(repo_root, "diff", "HEAD", "--binary")
    wrapper = Path(__file__).resolve()
    return {
        "workspace": str(workspace),
        "repo_root": str(repo_root),
        "commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "status": _git_output(
            repo_root, "status", "--short", "--untracked-files=all"
        ),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "wrapper": str(wrapper),
        "wrapper_sha256": _sha256(wrapper),
    }


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _acquire_exclusive_json_lock(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Create a lock without a check-then-create race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o644,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _declared_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _declared_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _path_manifest(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    entry: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
    }
    if resolved.is_file():
        stat = resolved.stat()
        entry.update(
            {
                "size_bytes": stat.st_size,
                "mode": oct(stat.st_mode & 0o777),
                "sha256": _sha256(resolved),
            }
        )
    return entry


def _content_manifest(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the immutable file identity, deliberately ignoring chmod."""
    if entry is None:
        return None
    return {
        key: entry.get(key)
        for key in ("path", "exists", "size_bytes", "sha256")
    }


def _immutable_source(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: snapshot[key]
        for key in (
            "repo_root",
            "commit",
            "diff_sha256",
            "wrapper",
            "wrapper_sha256",
        )
    }


_METRIC_SPREAD_TOLERANCES = {
    "kl_raw": {"abs": 1e-7, "rel": 5e-6},
    "kl_x100": {"abs": 1e-5, "rel": 5e-6},
    "ppl": {"abs": 1e-4, "rel": 5e-6},
}


def _metric_spread_report(
    per_rank: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate cross-rank metric shape/finiteness/tolerance."""
    if not per_rank:
        raise RuntimeError("No per-rank evaluation metrics were provided.")
    expected_shape = {
        dataset: sorted(values)
        for dataset, values in per_rank[0]["metrics"].items()
    }
    for record in per_rank:
        observed_shape = {
            dataset: sorted(values)
            for dataset, values in record["metrics"].items()
        }
        if observed_shape != expected_shape:
            raise RuntimeError(
                f"Evaluation metric keys differ across ranks: {per_rank}"
            )

    spreads: dict[str, dict[str, Any]] = {}
    for dataset, fields in expected_shape.items():
        spreads[dataset] = {}
        for field in fields:
            if field not in _METRIC_SPREAD_TOLERANCES:
                raise RuntimeError(
                    f"No cross-rank tolerance is declared for metric {field!r}."
                )
            values = [
                float(record["metrics"][dataset][field])
                for record in per_rank
            ]
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError(
                    "Evaluation produced a non-finite metric across ranks: "
                    f"dataset={dataset}, field={field}, values={values}"
                )
            minimum = min(values)
            maximum = max(values)
            spread = maximum - minimum
            tolerance = _METRIC_SPREAD_TOLERANCES[field]
            allowed = tolerance["abs"] + tolerance["rel"] * max(
                abs(minimum),
                abs(maximum),
            )
            spreads[dataset][field] = {
                "values_by_rank": values,
                "min": minimum,
                "max": maximum,
                "spread": spread,
                "allowed_spread": allowed,
                "abs_tolerance": tolerance["abs"],
                "rel_tolerance": tolerance["rel"],
            }
            if spread > allowed:
                raise RuntimeError(
                    "Evaluation metrics differ materially across ranks: "
                    f"dataset={dataset}, field={field}, spread={spread}, "
                    f"allowed={allowed}, values={values}"
                )
    return spreads


def _model_state_sha256(model) -> str:
    """Hash every post-rotation parameter and buffer without dtype loss."""
    import torch

    digest = hashlib.sha256()
    entries = [
        ("parameter", name, tensor)
        for name, tensor in model.named_parameters()
    ]
    entries.extend(
        ("buffer", name, tensor)
        for name, tensor in model.named_buffers()
    )
    for kind, name, tensor in sorted(entries, key=lambda item: (item[0], item[1])):
        value = tensor.detach().cpu().contiguous()
        digest.update(
            json.dumps(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                },
                sort_keys=True,
            ).encode()
        )
        if value.numel():
            raw = value.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
        del value
    return digest.hexdigest()


def _case_paths(
    experiment_root: Path,
    experiment_case: str,
    variant: str,
    *,
    warm: bool,
) -> dict[str, Path]:
    case_root = experiment_root.resolve() / experiment_case
    run_name = "warm" if warm else variant
    return {
        "case_root": case_root,
        "static_cache_path": case_root / "cache" / "static",
        "tokens_cache_path": case_root / "cache" / "tokens",
        "cache_dir": case_root / "cache" / "runtime",
        "rotation_fingerprint_path": (
            case_root / "cache" / "rotation_fingerprint.json"
        ),
        "warm_manifest_path": case_root / "cache" / "warm_complete.json",
        "output_dir": case_root / "outputs" / run_name,
        "metrics_json_path": case_root / "metrics" / f"{variant}.json",
    }


def _parse_experiment_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    # Production Config has short prefixes such as --exp; abbreviation would
    # otherwise misclassify it as an ambiguous experiment-only option.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--experiment_case",
        required=True,
        choices=tuple(_CASE_PROTOCOLS),
    )
    parser.add_argument(
        "--experiment_root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--schedule_ablation_variant",
        required=True,
        choices=VARIANTS,
    )
    parser.add_argument(
        "--wikitext_parquet_dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--metrics_json_path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--rotation_fingerprint_path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--allow_create_rotation_fingerprint",
        action="store_true",
    )
    parser.add_argument(
        "--require_precomputed_caches",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite_metrics",
        action="store_true",
    )
    return parser.parse_known_args(argv)


def _validate_schedule_calls(
    calls: list[dict[str, Any]],
    cfg,
    variant: str,
) -> None:
    """Reject partial/miswired interventions before publishing metrics."""
    if not calls:
        raise RuntimeError("Schedule ablation recorded no LR schedule calls.")
    num_layers_values = {record["num_layers"] for record in calls}
    if len(num_layers_values) != 1:
        raise RuntimeError(
            f"Inconsistent num_layers in schedule calls: {num_layers_values}"
        )
    num_layers = int(next(iter(num_layers_values)))
    final_index = num_layers - 1
    expected_last = (
        min(int(cfg.quant_stop_layer), final_index)
        if cfg.quant_stop_layer is not None
        else final_index
    )
    expected_indices = list(range(expected_last + 1))
    observed_indices = [int(record["layer_idx"]) for record in calls]
    if observed_indices != expected_indices:
        raise RuntimeError(
            "Schedule intervention did not execute exactly once in transformer "
            f"order: expected={expected_indices}, observed={observed_indices}"
        )

    aware_enabled = bool(cfg.activation_aware_quantization_enabled)
    for record in calls:
        layer_idx = int(record["layer_idx"])
        expected_aware = aware_enabled and layer_idx != final_index
        if bool(record["activation_aware"]) != expected_aware:
            raise RuntimeError(
                f"Unexpected activation-aware schedule flag at layer {layer_idx}: "
                f"expected={expected_aware}, observed={record['activation_aware']}"
            )
        expected_base_lr = float(cfg.grad_lr)
        if layer_idx == final_index and cfg.final_layer_grad_lr is not None:
            expected_base_lr = float(cfg.final_layer_grad_lr)
        if not math.isclose(
            float(record["reported_or_final_base_lr"]),
            expected_base_lr,
            rel_tol=0.0,
            abs_tol=1e-18,
        ):
            raise RuntimeError(
                f"Unexpected base LR at layer {layer_idx}: "
                f"expected={expected_base_lr}, "
                f"observed={record['reported_or_final_base_lr']}"
            )
        expected_lr = schedule_ablation_lr(
            variant,
            expected_base_lr,
            layer_idx,
            num_layers,
            float(cfg.grad_lr_layer_base_ratio),
            str(cfg.grad_lr_layer_schedule),
            activation_aware=expected_aware,
        )
        if not math.isclose(
            float(record["effective_lr"]),
            expected_lr,
            rel_tol=1e-13,
            abs_tol=1e-20,
        ):
            raise RuntimeError(
                f"Unexpected effective LR at layer {layer_idx}: "
                f"expected={expected_lr}, observed={record['effective_lr']}"
            )


def _validate_measured_arm(cfg, metrics_json_path: Path | None) -> None:
    if cfg.dataset != "wikitext2":
        raise ValueError(
            "Schedule ablations require --dataset wikitext2 so calibration "
            "matches the verified Salesforce/wikitext artifact."
        )
    if list(cfg.eval_datasets) != ["wikitext2"]:
        raise ValueError(
            "Schedule ablations require exactly "
            "--eval_datasets wikitext2."
        )
    if cfg.grad_lr_layer_schedule != "cosine":
        raise ValueError(
            "Schedule ablations require "
            "--grad_lr_layer_schedule cosine."
        )
    if cfg.exit_after_precompute:
        return
    if metrics_json_path is None:
        raise ValueError(
            "--metrics_json_path is required for every quantization arm."
        )
    if cfg.skip_eval:
        raise ValueError(
            "--skip_eval must be false for a measured quantization arm."
        )


def _validate_experiment_protocol(
    cfg,
    experiment: argparse.Namespace,
    *,
    model_artifact_identity: str,
) -> None:
    """Fail closed unless this is exactly one manifest case or its warm-up."""
    _validate_measured_arm(cfg, experiment.metrics_json_path)
    protocol = _CASE_PROTOCOLS[experiment.experiment_case]
    warm = bool(cfg.exit_after_precompute)

    mismatches: dict[str, dict[str, Any]] = {}
    expected_config = {
        **_COMMON_CASE_CONFIG,
        **protocol["config"],
    }
    for field_name, expected in expected_config.items():
        observed = getattr(cfg, field_name)
        if observed != expected:
            mismatches[field_name] = {
                "expected": expected,
                "observed": observed,
            }

    if cfg.model_name != protocol["model_name"]:
        mismatches["model_name"] = {
            "expected": protocol["model_name"],
            "observed": cfg.model_name,
        }
    if model_artifact_identity != protocol["model_artifact_identity"]:
        mismatches["model_artifact_identity"] = {
            "expected": protocol["model_artifact_identity"],
            "observed": model_artifact_identity,
        }
    if not Path(cfg.model).is_absolute():
        mismatches["model_path_absolute"] = {
            "expected": True,
            "observed": cfg.model,
        }

    allowed_variants = protocol["variants"]
    if experiment.schedule_ablation_variant not in allowed_variants:
        mismatches["schedule_ablation_variant"] = {
            "expected": sorted(allowed_variants),
            "observed": experiment.schedule_ablation_variant,
        }
    if warm and experiment.schedule_ablation_variant != "paper":
        mismatches["warm_schedule_ablation_variant"] = {
            "expected": "paper",
            "observed": experiment.schedule_ablation_variant,
        }

    if _declared_world_size() != 4:
        mismatches["world_size"] = {
            "expected": 4,
            "observed": _declared_world_size(),
        }
    if experiment.overwrite_metrics:
        mismatches["overwrite_metrics"] = {
            "expected": False,
            "observed": True,
        }

    expected_paths = _case_paths(
        experiment.experiment_root,
        experiment.experiment_case,
        experiment.schedule_ablation_variant,
        warm=warm,
    )
    path_values = {
        "static_cache_path": cfg.static_cache_path,
        "tokens_cache_path": cfg.tokens_cache_path,
        "cache_dir": cfg.cache_dir,
        "output_dir": cfg.output_dir,
        "rotation_fingerprint_path": experiment.rotation_fingerprint_path,
    }
    for name, observed in path_values.items():
        expected = expected_paths[name].resolve()
        actual = Path(observed).resolve() if observed is not None else None
        if actual != expected:
            mismatches[name] = {
                "expected": str(expected),
                "observed": str(actual) if actual is not None else None,
            }

    expected_exp = (
        f"{experiment.experiment_case}__warm"
        if warm
        else (
            f"{experiment.experiment_case}__"
            f"{experiment.schedule_ablation_variant}"
        )
    )
    if cfg.exp != expected_exp:
        mismatches["exp"] = {
            "expected": expected_exp,
            "observed": cfg.exp,
        }

    metrics_path = (
        experiment.metrics_json_path.resolve()
        if experiment.metrics_json_path is not None
        else None
    )
    if warm:
        if metrics_path is not None:
            mismatches["metrics_json_path"] = {
                "expected": None,
                "observed": str(metrics_path),
            }
        if not experiment.allow_create_rotation_fingerprint:
            mismatches["allow_create_rotation_fingerprint"] = {
                "expected": True,
                "observed": False,
            }
        if experiment.require_precomputed_caches:
            mismatches["require_precomputed_caches"] = {
                "expected": False,
                "observed": True,
            }
    else:
        expected_metrics = expected_paths["metrics_json_path"].resolve()
        if metrics_path != expected_metrics:
            mismatches["metrics_json_path"] = {
                "expected": str(expected_metrics),
                "observed": (
                    str(metrics_path) if metrics_path is not None else None
                ),
            }
        if experiment.allow_create_rotation_fingerprint:
            mismatches["allow_create_rotation_fingerprint"] = {
                "expected": False,
                "observed": True,
            }
        if not experiment.require_precomputed_caches:
            mismatches["require_precomputed_caches"] = {
                "expected": True,
                "observed": False,
            }

    for name, path in (
        ("experiment_root", experiment.experiment_root),
        ("wikitext_parquet_dir", experiment.wikitext_parquet_dir),
        (
            "rotation_fingerprint_path_arg",
            experiment.rotation_fingerprint_path,
        ),
    ):
        if not path.is_absolute():
            mismatches[name] = {
                "expected": "absolute path",
                "observed": str(path),
            }

    if mismatches:
        raise ValueError(
            "Schedule-ablation manifest protocol mismatch for "
            f"{experiment.experiment_case}: "
            f"{json.dumps(mismatches, sort_keys=True, default=str)}"
        )


def main() -> None:
    original_argv = list(sys.argv)
    experiment, realq_argv = _parse_experiment_args(sys.argv[1:])
    parquet_dir = experiment.wikitext_parquet_dir.resolve()
    split_paths, dataset_sha256, dataset_identity = _dataset_pin(parquet_dir)
    workspace = Path.cwd().resolve()
    source_start = _source_snapshot(workspace)

    # Import after consuming the experiment-only flags.  realq.ptq will parse
    # the remaining argv with the production Config parser.
    import pyarrow.parquet as pq
    import torch
    import transformers
    from realq import config as realq_config
    import realq.pipeline as pipeline_mod
    from realq.precompute import cache as static_cache
    import realq.precompute.static_e2e as static_e2e
    from realq.parallel import env as parallel_env
    import realq.refresh.block_gd as block_gd
    import realq.runner.layer_loop as layer_loop
    from utils.cache_identity import artifact_identity
    from utils import data_utils, dist_utils, eval_utils

    production_cfg = realq_config.parse_cli(realq_argv)
    model_identity_start = artifact_identity(production_cfg.model)
    _validate_experiment_protocol(
        production_cfg,
        experiment,
        model_artifact_identity=model_identity_start,
    )
    metrics_path = (
        experiment.metrics_json_path.resolve()
        if experiment.metrics_json_path is not None
        else None
    )
    metrics_lock_path = (
        metrics_path.with_name(f"{metrics_path.name}.lock")
        if metrics_path is not None
        else None
    )
    case_paths = _case_paths(
        experiment.experiment_root,
        experiment.experiment_case,
        experiment.schedule_ablation_variant,
        warm=bool(production_cfg.exit_after_precompute),
    )
    warm_manifest_path = case_paths["warm_manifest_path"].resolve()

    original_pipeline_run = pipeline_mod.run
    original_schedule = block_gd.layer_lr_for_schedule
    original_get_tokens = data_utils.get_tokens
    original_get_ref_logits = eval_utils.get_ref_logits
    original_load_reference_cache = eval_utils._load_reference_cache
    original_static_cache_key = static_cache.build_cache_key
    original_static_try_load = static_cache.try_load
    original_all_ranks_have_cache = static_e2e._all_ranks_have_cache
    original_maybe_rotate = pipeline_mod._maybe_rotate
    schedule_calls: list[dict[str, Any]] = []
    calibration_tokens: dict[str, Any] = {}
    static_cache_runtime: dict[str, Any] = {}
    reference_cache_runtime: dict[str, dict[str, Any]] = {}
    rotation_runtime: dict[str, Any] = {}
    warm_manifest_runtime: dict[str, Any] = {}
    evaluation_rank_runtime: dict[str, Any] = {}

    def broadcast_rank0_outcome(
        outcome: dict[str, Any] | None,
    ) -> dict[str, Any]:
        values = [outcome]
        if parallel_env.is_dist_available_and_initialized():
            backend = str(torch.distributed.get_backend()).lower()
            broadcast_device = (
                torch.device(f"cuda:{torch.cuda.current_device()}")
                if "nccl" in backend
                else torch.device("cpu")
            )
            torch.distributed.broadcast_object_list(
                values,
                src=0,
                device=broadcast_device,
            )
        result = values[0]
        if result is None:
            raise RuntimeError("Rank-0 outcome broadcast returned no result.")
        return result

    def raise_on_any_rank_error(
        local_error: str | None,
        *,
        context: str,
    ) -> None:
        local = {
            "rank": parallel_env.get_rank(),
            "error": local_error,
        }
        if parallel_env.is_dist_available_and_initialized():
            gathered: list[dict[str, Any] | None] = [
                None
            ] * parallel_env.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        errors = [
            record
            for record in gathered
            if record is None or record["error"] is not None
        ]
        if errors:
            raise RuntimeError(
                f"{context} failed on at least one rank: {errors}"
            )

    def require_all_ranks_equal(value: Any, *, context: str) -> None:
        local = {
            "rank": parallel_env.get_rank(),
            "value": value,
        }
        if parallel_env.is_dist_available_and_initialized():
            gathered: list[dict[str, Any] | None] = [
                None
            ] * parallel_env.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        serialized = {
            json.dumps(record["value"], sort_keys=True, default=str)
            for record in gathered
            if record is not None
        }
        if any(record is None for record in gathered) or len(serialized) != 1:
            raise RuntimeError(f"{context} differs across ranks: {gathered}")

    def validate_metrics_across_ranks(
        metrics: dict[str, dict[str, float]],
    ) -> None:
        """Allow only numerical-kernel noise, and preserve its full spread."""
        local = {
            "rank": parallel_env.get_rank(),
            "metrics": metrics,
        }
        if parallel_env.is_dist_available_and_initialized():
            gathered: list[dict[str, Any] | None] = [
                None
            ] * parallel_env.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if any(record is None for record in gathered):
            raise RuntimeError(
                f"Evaluation metrics missing from at least one rank: {gathered}"
            )
        concrete = [record for record in gathered if record is not None]
        spreads = _metric_spread_report(concrete)
        evaluation_rank_runtime.clear()
        evaluation_rank_runtime.update(
            {
                "rank_zero_is_reported": True,
                "per_rank": concrete,
                "spread_validation": spreads,
            }
        )

    rotation_fingerprint_path = (
        experiment.rotation_fingerprint_path.resolve()
    )

    def pinned_maybe_rotate(cfg, analyzer) -> None:
        original_maybe_rotate(cfg, analyzer)
        try:
            local_hash_result = {
                "rank": parallel_env.get_rank(),
                "sha256": _model_state_sha256(analyzer.model),
                "error": None,
            }
        except Exception as exc:
            local_hash_result = {
                "rank": parallel_env.get_rank(),
                "sha256": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if parallel_env.is_dist_available_and_initialized():
            all_rank_hashes: list[dict[str, Any] | None] = [
                None
            ] * parallel_env.get_world_size()
            torch.distributed.all_gather_object(
                all_rank_hashes,
                local_hash_result,
            )
        else:
            all_rank_hashes = [local_hash_result]

        outcome: list[dict[str, Any] | None] = [None]
        if parallel_env.is_main():
            try:
                hash_errors = [
                    result
                    for result in all_rank_hashes
                    if result is None or result["error"] is not None
                ]
                if hash_errors:
                    raise RuntimeError(
                        "At least one rank could not hash its post-rotation "
                        f"model: {hash_errors}"
                    )
                state_hashes = [
                    str(result["sha256"])
                    for result in all_rank_hashes
                    if result is not None
                ]
                if len(set(state_hashes)) != 1:
                    raise RuntimeError(
                        "Post-rotation model state differs across ranks: "
                        f"{all_rank_hashes}"
                    )
                state_sha256 = state_hashes[0]
                expected_payload = {
                    "schema_version": 1,
                    "experiment_case": experiment.experiment_case,
                    "model_artifact_identity": model_identity_start,
                    "rotate": bool(cfg.rotate),
                    "rotation_seed": int(cfg.rotation_seed),
                    "optimized_rotation_path": cfg.optimized_rotation_path,
                    "post_rotation_model_state_sha256": state_sha256,
                    "all_rank_model_state_sha256": state_hashes,
                }
                created = False
                if rotation_fingerprint_path.is_file():
                    observed_payload = json.loads(
                        rotation_fingerprint_path.read_text(encoding="utf-8")
                    )
                    if observed_payload != expected_payload:
                        raise RuntimeError(
                            "Post-rotation model fingerprint mismatch: "
                            f"expected_file={observed_payload}, "
                            f"observed_runtime={expected_payload}"
                        )
                elif experiment.allow_create_rotation_fingerprint:
                    _atomic_json_dump(
                        expected_payload,
                        rotation_fingerprint_path,
                    )
                    created = True
                else:
                    raise FileNotFoundError(
                        "Measured arm requires the pre-created rotation "
                        f"fingerprint: {rotation_fingerprint_path}"
                    )
                outcome[0] = {
                    **expected_payload,
                    "path": str(rotation_fingerprint_path),
                    "file_sha256": _sha256(rotation_fingerprint_path),
                    "created_by_this_run": created,
                    "error": None,
                }
            except Exception as exc:
                outcome[0] = {
                    "path": str(rotation_fingerprint_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        result = broadcast_rank0_outcome(outcome[0])
        rotation_runtime.clear()
        rotation_runtime.update(result)
        if result["error"] is not None:
            raise RuntimeError(result["error"])

    pipeline_mod._maybe_rotate = pinned_maybe_rotate

    def patched_schedule(
        base_lr: float,
        layer_idx: int,
        num_layers: int,
        base_ratio: float,
        schedule: str,
        *,
        activation_aware: bool = False,
    ) -> float:
        value = _dispatch_schedule(
            experiment.schedule_ablation_variant,
            original_schedule,
            base_lr,
            layer_idx,
            num_layers,
            base_ratio,
            schedule,
            activation_aware=activation_aware,
        )
        record = {
            "layer_idx": int(layer_idx),
            "num_layers": int(num_layers),
            "reported_or_final_base_lr": float(base_lr),
            "base_ratio": float(base_ratio),
            "schedule": str(schedule),
            "activation_aware": bool(activation_aware),
            "effective_lr": float(value),
        }
        schedule_calls.append(record)
        logging.info(
            "[schedule-ablation] variant=%s layer=%d/%d aware=%s lr=%.12g",
            experiment.schedule_ablation_variant,
            layer_idx,
            num_layers,
            activation_aware,
            value,
        )
        return value

    # layer_loop imported the symbol directly; patch both the defining module
    # and that bound reference to make the intervention explicit.
    block_gd.layer_lr_for_schedule = patched_schedule
    layer_loop.layer_lr_for_schedule = patched_schedule

    def local_wikitext(split: str):
        if split not in split_paths:
            raise ValueError(f"Unknown WikiText-2 split {split!r}.")
        table = pq.read_table(split_paths[split], columns=["text"])
        return table.column("text").to_pylist()

    data_utils._get_wikitext2 = local_wikitext

    def pinned_get_tokens(
        dataset_name,
        split,
        tokenizer,
        seq_len,
        num_samples,
        save_path=None,
        seed=0,
    ):
        tokenizer_source = getattr(tokenizer, "name_or_path", None)
        tokenizer_identity = artifact_identity(tokenizer_source)
        scoped_save_path = _revision_scoped_token_path(
            save_path,
            dataset_identity=dataset_identity,
            tokenizer_identity=tokenizer_identity,
        )
        is_calibration = dataset_name == "wikitext2" and split == "train"
        cache_before = None
        precheck_error = None
        try:
            cache_before = (
                _path_manifest(Path(scoped_save_path))
                if scoped_save_path is not None
                else None
            )
            if (
                is_calibration
                and experiment.require_precomputed_caches
                and (cache_before is None or not cache_before["exists"])
            ):
                precheck_error = (
                    "Measured arm requires a precomputed calibration-token "
                    f"cache: {scoped_save_path}"
                )
        except Exception as exc:
            precheck_error = f"{type(exc).__name__}: {exc}"
        if is_calibration:
            raise_on_any_rank_error(
                precheck_error,
                context="Calibration-token cache precheck",
            )

        tokens = None
        load_error = None
        try:
            tokens = original_get_tokens(
                dataset_name,
                split,
                tokenizer,
                seq_len,
                num_samples,
                scoped_save_path,
                seed,
            )
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
        if is_calibration:
            raise_on_any_rank_error(
                load_error,
                context="Calibration-token cache load",
            )
        elif load_error is not None:
            raise RuntimeError(load_error)
        if tokens is None:
            raise RuntimeError("Token loader returned no tokens.")

        if is_calibration:
            record = None
            postcheck_error = None
            try:
                observed_identity = _token_tensor_identity(tokens)
                cache_after = (
                    _path_manifest(Path(scoped_save_path))
                    if scoped_save_path is not None
                    else None
                )
                if (
                    experiment.require_precomputed_caches
                    and cache_after != cache_before
                ):
                    raise RuntimeError(
                        "Calibration-token cache changed while loading it: "
                        f"before={cache_before}, after={cache_after}"
                    )
                record = {
                    "dataset": str(dataset_name),
                    "split": str(split),
                    "num_samples": int(num_samples),
                    "seq_len": int(seq_len),
                    "seed": int(seed),
                    "sha256": observed_identity,
                    "cache_path": scoped_save_path,
                    "tokenizer_source": str(tokenizer_source),
                    "tokenizer_artifact_identity": tokenizer_identity,
                    "cache_required": bool(
                        experiment.require_precomputed_caches
                    ),
                    "cache_before": cache_before,
                    "cache_after_load": cache_after,
                }
                if experiment.require_precomputed_caches:
                    warm_payload = warm_manifest_runtime.get("payload")
                    if warm_payload is None:
                        raise RuntimeError(
                            "Measured token check has no validated warm "
                            "completion manifest."
                        )
                    warm_token = warm_payload["calibration_tokens"]["train"]
                    comparisons = {
                        "dataset": str(dataset_name),
                        "split": str(split),
                        "num_samples": int(num_samples),
                        "seq_len": int(seq_len),
                        "seed": int(seed),
                        "sha256": observed_identity,
                        "cache_path": scoped_save_path,
                        "tokenizer_artifact_identity": tokenizer_identity,
                    }
                    mismatches = {
                        key: {
                            "warm": warm_token.get(key),
                            "measured": value,
                        }
                        for key, value in comparisons.items()
                        if warm_token.get(key) != value
                    }
                    if (
                        _content_manifest(cache_before)
                        != _content_manifest(warm_token.get("cache_end"))
                    ):
                        mismatches["cache_content"] = {
                            "warm": _content_manifest(
                                warm_token.get("cache_end")
                            ),
                            "measured": _content_manifest(cache_before),
                        }
                    if mismatches:
                        raise RuntimeError(
                            "Measured calibration tokens do not match the "
                            f"completed warm run: {mismatches}"
                        )
                previous = calibration_tokens.get("train")
                if previous is not None and previous != record:
                    raise RuntimeError(
                        "Calibration-token identity changed within one run: "
                        f"first={previous}, later={record}"
                    )
            except Exception as exc:
                postcheck_error = f"{type(exc).__name__}: {exc}"
            raise_on_any_rank_error(
                postcheck_error,
                context="Calibration-token cache postcheck",
            )
            if record is None:
                raise RuntimeError("Calibration-token record was not created.")
            require_all_ranks_equal(
                record["sha256"],
                context="Calibration-token tensor SHA256",
            )
            calibration_tokens["train"] = record
        return tokens

    data_utils.get_tokens = pinned_get_tokens

    def pinned_static_cache_key(cfg, world_size: int) -> str:
        # The production cache identity historically included only the
        # dataset name.  Append the verified file identity here so an older
        # WikiText cache cannot silently bypass this experiment's data pin.
        return (
            f"{original_static_cache_key(cfg, world_size)}"
            f"_dataset{dataset_identity[:12]}"
        )

    static_cache.build_cache_key = pinned_static_cache_key

    def static_cache_manifest(cfg, world_size: int) -> list[dict[str, Any]]:
        key = pinned_static_cache_key(cfg, world_size)
        if not cfg.static_cache_path:
            return []
        return [
            {
                "rank": rank,
                **_path_manifest(
                    Path(
                        static_cache.cache_path(
                            cfg.static_cache_path,
                            key,
                            world_size,
                            rank,
                        )
                    )
                ),
            }
            for rank in range(world_size)
        ]

    static_manifest_start: list[dict[str, Any]] = []

    def tracked_static_try_load(
        cache_dir: str,
        key: str,
        world_size: int,
        rank: int,
    ):
        path = Path(
            static_cache.cache_path(
                cache_dir,
                key,
                world_size,
                rank,
            )
        ).resolve()
        cached = original_static_try_load(
            cache_dir,
            key,
            world_size,
            rank,
        )
        static_cache_runtime.update(
            {
                "required": bool(experiment.require_precomputed_caches),
                "key": key,
                "rank": int(rank),
                "path": str(path),
                "local_hit": cached is not None,
            }
        )
        return cached

    def tracked_all_ranks_have_cache(
        local_hit: bool,
        world_size: int,
    ) -> bool:
        all_hit = original_all_ranks_have_cache(local_hit, world_size)
        static_cache_runtime["all_ranks_hit"] = bool(all_hit)
        if experiment.require_precomputed_caches and not all_hit:
            raise RuntimeError(
                "Measured arm requires a collective static-cache hit on "
                "every rank; recomputation is forbidden."
            )
        return all_hit

    static_cache.try_load = tracked_static_try_load
    static_e2e._all_ranks_have_cache = tracked_all_ranks_have_cache

    preloaded_reference_cache: dict[str, Any] = {}

    def tracked_load_reference_cache(path: str, metadata: dict):
        resolved = str(Path(path).resolve())
        if resolved in preloaded_reference_cache:
            cached = preloaded_reference_cache.pop(resolved)
        else:
            cached = original_load_reference_cache(path, metadata)
        record = reference_cache_runtime.setdefault(
            resolved,
            {
                "path": resolved,
                "required": bool(experiment.require_precomputed_caches),
            },
        )
        record["loader_hit"] = cached is not None
        return cached

    eval_utils._load_reference_cache = tracked_load_reference_cache

    def tracked_get_ref_logits(args, analyzer, dataset, dataloader):
        metadata = eval_utils._reference_cache_metadata(
            args,
            analyzer,
            dataset,
            dataloader,
        )
        cache_tag = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()[:20]
        path = (
            Path(args.cache_dir)
            / "ref_logits"
            / (
                f"{args.model_name}_{dataset}_test_"
                f"{args.eval_seq_len}_{cache_tag}.cache"
            )
        ).resolve()
        before = None
        preloaded = None
        precheck_error = None
        try:
            before = _path_manifest(path)
            if experiment.require_precomputed_caches:
                if not before["exists"]:
                    raise FileNotFoundError(
                        "Measured arm requires a precomputed reference-logit "
                        f"cache: {path}"
                    )
                preloaded = original_load_reference_cache(
                    str(path),
                    metadata,
                )
                if preloaded is None:
                    raise RuntimeError(
                        "Reference-logit cache exists but failed metadata or "
                        f"payload validation: {path}"
                    )
        except Exception as exc:
            precheck_error = f"{type(exc).__name__}: {exc}"
        raise_on_any_rank_error(
            precheck_error,
            context=f"Reference-logit cache precheck ({dataset})",
        )
        if preloaded is not None:
            preloaded_reference_cache[str(path)] = preloaded

        record = reference_cache_runtime.setdefault(
            str(path),
            {
                "path": str(path),
                "dataset": str(dataset),
                "required": bool(experiment.require_precomputed_caches),
            },
        )
        record["dataset"] = str(dataset)
        record["before"] = before
        record["metadata_sha256"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()
        result = None
        load_error = None
        try:
            result = original_get_ref_logits(
                args,
                analyzer,
                dataset,
                dataloader,
            )
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
        raise_on_any_rank_error(
            load_error,
            context=f"Reference-logit cache load ({dataset})",
        )
        if result is None:
            raise RuntimeError("Reference-logit loader returned no result.")
        # On a warm miss only rank 0 writes the shared cache.  Wait until that
        # atomic save is visible before any rank hashes the file.
        if parallel_env.is_dist_available_and_initialized():
            parallel_env.barrier()

        postcheck_error = None
        try:
            after = _path_manifest(path)
            record["after_load"] = after
            if experiment.require_precomputed_caches:
                if not record.get("loader_hit", False):
                    raise RuntimeError(
                        "Reference-logit cache existed but was not accepted "
                        f"by the loader: {path}"
                    )
                if after != before:
                    raise RuntimeError(
                        "Reference-logit cache changed while loading it: "
                        f"before={before}, after={after}"
                    )
                warm_payload = warm_manifest_runtime.get("payload")
                if warm_payload is None:
                    raise RuntimeError(
                        "Measured reference-cache check has no validated "
                        "warm completion manifest."
                    )
                warm_matches = [
                    warm_record
                    for warm_record in warm_payload["reference_cache"]
                    if warm_record.get("dataset") == str(dataset)
                    and warm_record.get("path") == str(path)
                ]
                if len(warm_matches) != 1:
                    raise RuntimeError(
                        "Completed warm run does not identify exactly one "
                        f"reference cache for {dataset}: {warm_matches}"
                    )
                warm_record = warm_matches[0]
                if (
                    warm_record.get("metadata_sha256")
                    != record["metadata_sha256"]
                    or _content_manifest(warm_record.get("end"))
                    != _content_manifest(after)
                ):
                    raise RuntimeError(
                        "Measured reference cache does not match the completed "
                        f"warm run: warm={warm_record}, measured={record}"
                    )
        except Exception as exc:
            postcheck_error = f"{type(exc).__name__}: {exc}"
        raise_on_any_rank_error(
            postcheck_error,
            context=f"Reference-logit cache postcheck ({dataset})",
        )
        require_all_ranks_equal(
            {
                "path": str(path),
                "metadata_sha256": record["metadata_sha256"],
                "cache_sha256": record["after_load"].get("sha256"),
            },
            context=f"Reference-logit cache identity ({dataset})",
        )
        return result

    eval_utils.get_ref_logits = tracked_get_ref_logits

    warm_lock_path = warm_manifest_path.with_name(
        f"{warm_manifest_path.name}.lock"
    )

    def expected_dataset_provenance() -> dict[str, Any]:
        return {
            "repo": "Salesforce/wikitext",
            "revision": _DATASET_REVISION,
            "config": "wikitext-2-raw-v1",
            "parquet_dir": str(parquet_dir),
            "sha256": dataset_sha256,
            "identity": dataset_identity,
        }

    def validate_warm_manifest(
        payload: dict[str, Any],
        cfg,
        static_now: list[dict[str, Any]],
    ) -> None:
        expected_protocol = {
            **_COMMON_CASE_CONFIG,
            **_CASE_PROTOCOLS[experiment.experiment_case]["config"],
        }
        exact_checks = {
            "schema_version": 1,
            "experiment_case": experiment.experiment_case,
            "protocol_config": expected_protocol,
            "dataset": expected_dataset_provenance(),
            "model_artifact_identity": model_identity_start,
            "source": _immutable_source(source_start),
        }
        mismatches = {
            key: {
                "warm": payload.get(key),
                "measured": expected,
            }
            for key, expected in exact_checks.items()
            if payload.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                "Warm completion manifest does not match this measured run: "
                f"{mismatches}"
            )

        warm_static = payload.get("static_cache", {})
        expected_key = pinned_static_cache_key(
            cfg,
            parallel_env.get_world_size(),
        )
        if warm_static.get("key") != expected_key:
            raise RuntimeError(
                "Warm static-cache key differs from the measured key: "
                f"warm={warm_static.get('key')}, measured={expected_key}"
            )
        if any(not entry.get("exists", False) for entry in static_now):
            raise FileNotFoundError(
                f"Measured arm is missing static-cache shards: {static_now}"
            )
        warm_static_content = [
            _content_manifest(entry)
            for entry in warm_static.get("files", [])
        ]
        static_now_content = [
            _content_manifest(entry)
            for entry in static_now
        ]
        if warm_static_content != static_now_content:
            raise RuntimeError(
                "Static-cache content differs from the completed warm run: "
                f"warm={warm_static_content}, measured={static_now_content}"
            )

        fingerprint_now = _path_manifest(rotation_fingerprint_path)
        if (
            not fingerprint_now["exists"]
            or _content_manifest(payload.get("rotation_fingerprint"))
            != _content_manifest(fingerprint_now)
        ):
            raise RuntimeError(
                "Rotation fingerprint differs from the completed warm run: "
                f"warm={payload.get('rotation_fingerprint')}, "
                f"measured={fingerprint_now}"
            )

    def pipeline_preflight(cfg) -> dict[str, Any]:
        """Rank-0 filesystem checks, broadcast only after dist.init()."""
        outcome: dict[str, Any] | None = None
        if parallel_env.is_main():
            created_lock: Path | None = None
            try:
                world_size = parallel_env.get_world_size()
                static_now = static_cache_manifest(cfg, world_size)
                if cfg.exit_after_precompute:
                    if warm_manifest_path.exists():
                        raise FileExistsError(
                            "Refusing to replace an existing completed warm "
                            f"manifest: {warm_manifest_path}"
                        )
                    if any(
                        entry.get("exists", False)
                        for entry in static_now
                    ):
                        raise FileExistsError(
                            "Incomplete warm-up already left static-cache "
                            "shards. Use a fresh case root (or explicitly "
                            "archive and clear the failed case) rather than "
                            f"silently reusing partial state: {static_now}"
                        )
                    _acquire_exclusive_json_lock(
                        warm_lock_path,
                        {
                            "kind": "schedule-ablation-warm",
                            "pid": os.getpid(),
                            "hostname": socket.gethostname(),
                            "experiment_case": experiment.experiment_case,
                        },
                    )
                    created_lock = warm_lock_path
                    warm_payload = None
                    warm_file = None
                else:
                    if metrics_path is None or metrics_lock_path is None:
                        raise RuntimeError(
                            "Measured arm has no fixed metrics destination."
                        )
                    if metrics_path.exists():
                        raise FileExistsError(
                            "Refusing to overwrite existing metrics: "
                            f"{metrics_path}"
                        )
                    _acquire_exclusive_json_lock(
                        metrics_lock_path,
                        {
                            "kind": "schedule-ablation-metrics",
                            "pid": os.getpid(),
                            "hostname": socket.gethostname(),
                            "experiment_case": experiment.experiment_case,
                            "variant": experiment.schedule_ablation_variant,
                        },
                    )
                    created_lock = metrics_lock_path
                    if warm_lock_path.exists():
                        raise RuntimeError(
                            "Warm run is incomplete or still active; refusing "
                            f"to measure while its lock exists: {warm_lock_path}"
                        )
                    if not warm_manifest_path.is_file():
                        raise FileNotFoundError(
                            "Measured arm requires a completed warm manifest: "
                            f"{warm_manifest_path}"
                        )
                    warm_payload = json.loads(
                        warm_manifest_path.read_text(encoding="utf-8")
                    )
                    validate_warm_manifest(warm_payload, cfg, static_now)
                    warm_file = _path_manifest(warm_manifest_path)
                outcome = {
                    "error": None,
                    "static_manifest_start": static_now,
                    "warm_payload": warm_payload,
                    "warm_manifest_file": warm_file,
                }
            except Exception as exc:
                # A lock created by this live preflight is not a crash marker.
                # Remove it so a corrected command can be submitted directly.
                if created_lock is not None:
                    try:
                        created_lock.unlink()
                    except OSError:
                        pass
                outcome = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "static_manifest_start": [],
                    "warm_payload": None,
                    "warm_manifest_file": None,
                }
        result = broadcast_rank0_outcome(outcome)
        if result["error"] is not None:
            raise RuntimeError(result["error"])
        static_manifest_start[:] = result["static_manifest_start"]
        warm_manifest_runtime.clear()
        warm_manifest_runtime.update(
            {
                "payload": result["warm_payload"],
                "file_start": result["warm_manifest_file"],
            }
        )
        return result

    def finalize_warm_manifest(cfg) -> None:
        """Publish warm_complete.json only after all four ranks finish."""
        if parallel_env.is_dist_available_and_initialized():
            parallel_env.barrier()
        outcome: dict[str, Any] | None = None
        if parallel_env.is_main():
            try:
                _, dataset_sha256_end, dataset_identity_end = _dataset_pin(
                    parquet_dir
                )
                if (
                    dataset_sha256_end != dataset_sha256
                    or dataset_identity_end != dataset_identity
                ):
                    raise RuntimeError(
                        "Pinned WikiText files changed during warm-up."
                    )
                source_end = _source_snapshot(workspace)
                if _immutable_source(source_end) != _immutable_source(
                    source_start
                ):
                    raise RuntimeError(
                        "Repository source changed during warm-up: "
                        f"start={source_start}, end={source_end}"
                    )
                model_identity_end = artifact_identity(cfg.model)
                if model_identity_end != model_identity_start:
                    raise RuntimeError(
                        "Model artifact changed during warm-up: "
                        f"start={model_identity_start}, "
                        f"end={model_identity_end}"
                    )

                world_size = parallel_env.get_world_size()
                static_end = static_cache_manifest(cfg, world_size)
                if (
                    len(static_end) != world_size
                    or any(
                        not entry.get("exists", False)
                        for entry in static_end
                    )
                ):
                    raise FileNotFoundError(
                        "Warm-up did not produce every static-cache shard: "
                        f"{static_end}"
                    )

                token_record = calibration_tokens.get("train")
                if token_record is None:
                    raise RuntimeError(
                        "Warm-up did not sample and hash calibration tokens."
                    )
                token_cache_path = token_record.get("cache_path")
                token_record["cache_end"] = (
                    _path_manifest(Path(token_cache_path))
                    if token_cache_path is not None
                    else None
                )
                if (
                    token_record["cache_end"] is None
                    or not token_record["cache_end"].get("exists", False)
                ):
                    raise FileNotFoundError(
                        "Warm-up did not publish its calibration-token cache: "
                        f"{token_record}"
                    )

                for record in reference_cache_runtime.values():
                    record["end"] = _path_manifest(Path(record["path"]))
                reference_records = sorted(
                    reference_cache_runtime.values(),
                    key=lambda record: record["path"],
                )
                if (
                    len(reference_records) != len(cfg.eval_datasets)
                    or any(
                        not record["end"].get("exists", False)
                        for record in reference_records
                    )
                ):
                    raise FileNotFoundError(
                        "Warm-up did not publish exactly one reference cache "
                        f"per evaluation dataset: {reference_records}"
                    )

                if not rotation_runtime:
                    raise RuntimeError(
                        "Warm-up did not record a post-rotation model hash."
                    )
                fingerprint_end = _path_manifest(
                    rotation_fingerprint_path
                )
                if (
                    not fingerprint_end["exists"]
                    or fingerprint_end["sha256"]
                    != rotation_runtime.get("file_sha256")
                ):
                    raise RuntimeError(
                        "Rotation fingerprint changed during warm-up: "
                        f"runtime={rotation_runtime}, end={fingerprint_end}"
                    )

                payload = {
                    "schema_version": 1,
                    "experiment_case": experiment.experiment_case,
                    "protocol_config": {
                        **_COMMON_CASE_CONFIG,
                        **_CASE_PROTOCOLS[
                            experiment.experiment_case
                        ]["config"],
                    },
                    "dataset": expected_dataset_provenance(),
                    "model_artifact_identity": model_identity_start,
                    "source": _immutable_source(source_start),
                    "calibration_tokens": calibration_tokens,
                    "rotation": rotation_runtime,
                    "rotation_fingerprint": fingerprint_end,
                    "static_cache": {
                        "key": pinned_static_cache_key(cfg, world_size),
                        "files": static_end,
                    },
                    "reference_cache": reference_records,
                    "runtime": {
                        "hostname": socket.gethostname(),
                        "world_size": world_size,
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "transformers": transformers.__version__,
                        "canoe_job_id": os.environ.get("CANOE_JOB_ID"),
                    },
                }
                _atomic_json_dump(payload, warm_manifest_path)
                warm_file = _path_manifest(warm_manifest_path)
                warm_lock_path.unlink()
                outcome = {
                    "error": None,
                    "payload": payload,
                    "file": warm_file,
                }
            except Exception as exc:
                # Keep the warm lock on a post-run validation failure.  It is
                # an intentional marker that partial caches must not be used.
                outcome = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "payload": None,
                    "file": None,
                }
        result = broadcast_rank0_outcome(outcome)
        if result["error"] is not None:
            raise RuntimeError(result["error"])
        warm_manifest_runtime.clear()
        warm_manifest_runtime.update(
            {
                "payload": result["payload"],
                "file_end": result["file"],
            }
        )

    def guarded_pipeline_run(cfg) -> None:
        pipeline_preflight(cfg)
        original_pipeline_run(cfg)
        if cfg.exit_after_precompute:
            finalize_warm_manifest(cfg)

    pipeline_mod.run = guarded_pipeline_run

    def publish_metrics_payload(
        args,
        raw_metrics: dict[str, dict[str, float]],
    ) -> None:
        if metrics_path is None or metrics_lock_path is None:
            raise RuntimeError("Measured arm has no metrics path or lock.")
        if not metrics_lock_path.is_file():
            raise RuntimeError(
                "Measured metrics lock disappeared before publication: "
                f"{metrics_lock_path}"
            )
        _validate_schedule_calls(
            schedule_calls,
            args,
            experiment.schedule_ablation_variant,
        )
        _, dataset_sha256_end, dataset_identity_end = _dataset_pin(parquet_dir)
        if (
            dataset_sha256_end != dataset_sha256
            or dataset_identity_end != dataset_identity
        ):
            raise RuntimeError(
                "Pinned WikiText files changed while the experiment ran."
            )
        source_end = _source_snapshot(workspace)
        if _immutable_source(source_end) != _immutable_source(source_start):
            raise RuntimeError(
                "Repository source changed while the experiment ran: "
                f"start={source_start}, end={source_end}"
            )
        model_identity_end = artifact_identity(args.model)
        if model_identity_end != model_identity_start:
            raise RuntimeError(
                "Model artifact changed while the experiment ran: "
                f"start={model_identity_start}, end={model_identity_end}"
            )

        world_size = parallel_env.get_world_size()
        static_key = pinned_static_cache_key(args, world_size)
        static_manifest_end = static_cache_manifest(args, world_size)
        if not static_cache_runtime.get("local_hit", False):
            raise RuntimeError(
                "Rank 0 did not actually load its static-cache shard."
            )
        if not static_cache_runtime.get("all_ranks_hit", False):
            raise RuntimeError(
                "Static-cache consensus did not report an all-rank hit."
            )
        if static_manifest_end != static_manifest_start:
            raise RuntimeError(
                "Static-cache files changed during the measured arm: "
                f"start={static_manifest_start}, end={static_manifest_end}"
            )

        for record in reference_cache_runtime.values():
            record["end"] = _path_manifest(Path(record["path"]))
            if record["end"] != record["before"]:
                raise RuntimeError(
                    "Reference-logit cache changed during the measured arm: "
                    f"{record}"
                )
        if len(reference_cache_runtime) != len(args.eval_datasets):
            raise RuntimeError(
                "Did not observe exactly one reference-logit cache per eval "
                f"dataset: {reference_cache_runtime}"
            )
        if any(
            not record.get("loader_hit", False)
            for record in reference_cache_runtime.values()
        ):
            raise RuntimeError(
                "At least one reference-logit cache was not a real loader "
                f"hit: {reference_cache_runtime}"
            )

        token_record = calibration_tokens.get("train")
        if token_record is None:
            raise RuntimeError(
                "Measured arm did not load and hash calibration tokens."
            )
        token_cache_path = token_record["cache_path"]
        token_record["cache_end"] = (
            _path_manifest(Path(token_cache_path))
            if token_cache_path is not None
            else None
        )
        if token_record["cache_end"] != token_record["cache_before"]:
            raise RuntimeError(
                "Calibration-token cache changed during the measured arm: "
                f"{token_record}"
            )

        if not rotation_runtime:
            raise RuntimeError(
                "Measured arm did not record a post-rotation model hash."
            )
        fingerprint_end = _path_manifest(rotation_fingerprint_path)
        if (
            not fingerprint_end["exists"]
            or fingerprint_end["sha256"]
            != rotation_runtime["file_sha256"]
        ):
            raise RuntimeError(
                "Rotation fingerprint file changed during the measured arm: "
                f"runtime={rotation_runtime}, end={fingerprint_end}"
            )

        warm_payload = warm_manifest_runtime.get("payload")
        warm_file_start = warm_manifest_runtime.get("file_start")
        warm_file_end = _path_manifest(warm_manifest_path)
        if warm_payload is None or warm_file_start != warm_file_end:
            raise RuntimeError(
                "Warm completion manifest changed during the measured arm: "
                f"start={warm_file_start}, end={warm_file_end}"
            )

        payload = {
            "schema_version": 4,
            "experiment_case": experiment.experiment_case,
            "experiment_root": str(experiment.experiment_root.resolve()),
            "schedule_ablation_variant": (
                experiment.schedule_ablation_variant
            ),
            "metric_units": {
                "kl_raw": "mean token KL divergence",
                "kl_x100": "paper table unit (raw KL multiplied by 100)",
                "ppl": "shifted-token perplexity",
            },
            "metrics": raw_metrics,
            "schedule_calls": schedule_calls,
            "config": asdict(args),
            "dataset": {
                **expected_dataset_provenance(),
                "calibration_tokens": calibration_tokens,
            },
            "model_artifact_identity": model_identity_start,
            "warm_completion": {
                "manifest": warm_file_end,
                "payload": warm_payload,
            },
            "post_rotation_model": {
                **rotation_runtime,
                "fingerprint_file_end": fingerprint_end,
            },
            "static_cache": {
                "key": static_key,
                "runtime": static_cache_runtime,
                "files_start": static_manifest_start,
                "files_end": static_manifest_end,
            },
            "reference_cache": sorted(
                reference_cache_runtime.values(),
                key=lambda record: record["path"],
            ),
            "evaluation_rank_consistency": evaluation_rank_runtime,
            "runtime": {
                "hostname": socket.gethostname(),
                "world_size": world_size,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "transformers": transformers.__version__,
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "canoe_job_id": os.environ.get("CANOE_JOB_ID"),
            },
            "argv": original_argv,
            "source": {
                "start": source_start,
                "end": source_end,
            },
        }
        _atomic_json_dump(payload, metrics_path)
        metrics_lock_path.unlink()

    def exact_kl_ppl_eval(
        args,
        analyzer,
        orig_lm_head,
        test_loader_dict,
        ref_logits_dict,
    ):
        raw_metrics: dict[str, dict[str, float]] = {}
        formatted: dict[str, str] = {}
        for eval_dataset in args.eval_datasets:
            logging.info("Evaluating KL&PPL on %s", eval_dataset)
            ppl, kl_loss = eval_utils._kl_ppl_eval(
                args,
                analyzer,
                orig_lm_head,
                test_loader_dict[eval_dataset],
                ref_logits_dict[eval_dataset],
            )
            raw_metrics[eval_dataset] = {
                "kl_raw": float(kl_loss),
                "kl_x100": float(kl_loss) * 100.0,
                "ppl": float(ppl),
            }
            formatted[f"KL-{eval_dataset}"] = f"{kl_loss:.9g}"
            formatted[f"PPL-{eval_dataset}"] = f"{ppl:.9g}"
            logging.info(
                "Exact KL&PPL on %s: %.12g, %.12g",
                eval_dataset,
                kl_loss,
                ppl,
            )
        eval_utils.pretty_print_results(formatted)

        require_all_ranks_equal(
            schedule_calls,
            context="Schedule-call trace",
        )
        validate_metrics_across_ranks(raw_metrics)
        outcome: dict[str, Any] | None = None
        if parallel_env.is_main():
            try:
                publish_metrics_payload(args, raw_metrics)
                outcome = {"error": None}
            except Exception as exc:
                # Leave the metrics lock in place as a visible crash/validation
                # marker.  The fixed protocol never overwrites such a run.
                outcome = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        result = broadcast_rank0_outcome(outcome)
        if result["error"] is not None:
            raise RuntimeError(result["error"])
        return raw_metrics

    eval_utils.kl_ppl_eval = exact_kl_ppl_eval

    # Hand only production flags to realq.ptq.
    sys.argv = [sys.argv[0], *realq_argv]
    from realq.ptq import main as realq_main

    realq_main()


if __name__ == "__main__":
    main()
