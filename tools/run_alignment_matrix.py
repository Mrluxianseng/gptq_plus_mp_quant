#!/usr/bin/env python3
"""Run deterministic legacy/refactored REAL-Q trace-alignment experiments.

The runner deliberately uses a real Hugging Face decoder checkpoint and the
same cached token tensors for both implementations.  Every case is repeated,
then checked in three directions:

* legacy repeatability;
* refactored repeatability;
* legacy versus refactored, for both repeats.

Any missing refresh, metadata mismatch, sample-order mismatch, or symmetric
relative loss difference >= 1% fails the matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

from realq.alignment import compare_refresh_traces


CASES: dict[str, dict[str, Any]] = {
    "w4_row_fp": {
        "w_groupsize": -1,
        "w_clip": False,
        "act_order": False,
        "rotate": False,
        "loss_slide_window": False,
        "a_loss_ratio": 1.0,
    },
    "w4_row_rotate_slide": {
        "w_groupsize": -1,
        "w_clip": True,
        "act_order": True,
        "rotate": True,
        "loss_slide_window": True,
        "a_loss_ratio": 0.95,
    },
    "w4_g128_actorder": {
        "w_groupsize": 128,
        "w_clip": True,
        "act_order": True,
        "rotate": False,
        "loss_slide_window": True,
        "a_loss_ratio": 1.0,
    },
    "w4a4kv4_unaware_clip09": {
        "w_groupsize": 128,
        "w_clip": True,
        "act_order": True,
        "rotate": True,
        "loss_slide_window": True,
        "a_loss_ratio": 0.95,
        "a_bits": 4,
        "k_bits": 4,
        "v_bits": 4,
        "a_clip_ratio": 0.9,
        "k_clip_ratio": 0.9,
        "v_clip_ratio": 0.9,
    },
    "w4a4kv4_unaware_clip1": {
        "w_groupsize": 128,
        "w_clip": True,
        "act_order": True,
        "rotate": True,
        "loss_slide_window": True,
        "a_loss_ratio": 1.0,
        "a_bits": 4,
        "k_bits": 4,
        "v_bits": 4,
        "a_clip_ratio": 1.0,
        "k_clip_ratio": 1.0,
        "v_clip_ratio": 1.0,
    },
    "w4a4v4_aware": {
        "w_groupsize": 128,
        "w_clip": True,
        "act_order": True,
        "rotate": True,
        "loss_slide_window": True,
        "a_loss_ratio": 0.95,
        "a_bits": 4,
        "v_bits": 4,
        "a_clip_ratio": 0.9,
        "v_clip_ratio": 0.9,
        "act_quant_aware_gptq": True,
    },
    "w4k4_aware": {
        "w_groupsize": 128,
        "w_clip": True,
        "act_order": True,
        "rotate": True,
        "loss_slide_window": True,
        "a_loss_ratio": 0.95,
        "k_bits": 4,
        "k_clip_ratio": 0.9,
        "k_cache_quant_aware_gptq": True,
    },
    "w4a4kv4_aware": {
        "w_groupsize": 128,
        "w_clip": True,
        "act_order": True,
        "rotate": True,
        "loss_slide_window": True,
        "a_loss_ratio": 0.95,
        "a_bits": 4,
        "k_bits": 4,
        "v_bits": 4,
        "a_clip_ratio": 0.9,
        "k_clip_ratio": 0.9,
        "v_clip_ratio": 0.9,
        "act_quant_aware_gptq": True,
        "k_cache_quant_aware_gptq": True,
    },
    # Focused interaction cases retained for diagnosing any divergence in the
    # rotate + slide + activation-loss clipping configuration.
    "w4_row_rotate_only": {
        "w_groupsize": -1,
        "w_clip": True,
        "act_order": True,
        "rotate": True,
        "loss_slide_window": False,
        "a_loss_ratio": 1.0,
    },
    "w4_row_aloss095_only": {
        "w_groupsize": -1,
        "w_clip": True,
        "act_order": True,
        "rotate": False,
        "loss_slide_window": False,
        "a_loss_ratio": 0.95,
    },
    "w4_row_slide_aloss1": {
        "w_groupsize": -1,
        "w_clip": True,
        "act_order": True,
        "rotate": False,
        "loss_slide_window": True,
        "a_loss_ratio": 1.0,
    },
    "w4_row_rotate_aloss095": {
        "w_groupsize": -1,
        "w_clip": True,
        "act_order": True,
        "rotate": True,
        "loss_slide_window": False,
        "a_loss_ratio": 0.95,
    },
    "w4_row_slide_aloss095": {
        "w_groupsize": -1,
        "w_clip": True,
        "act_order": True,
        "rotate": False,
        "loss_slide_window": True,
        "a_loss_ratio": 0.95,
    },
}

DEFAULT_CASES = (
    "w4_row_fp",
    "w4_row_rotate_slide",
    "w4_g128_actorder",
    "w4a4kv4_unaware_clip09",
    "w4a4kv4_unaware_clip1",
    "w4a4v4_aware",
    "w4k4_aware",
    "w4a4kv4_aware",
)


DEFAULTS: dict[str, Any] = {
    "w_groupsize": -1,
    "w_clip": False,
    "act_order": False,
    "rotate": False,
    "loss_slide_window": False,
    "a_loss_ratio": 1.0,
    "a_bits": 16,
    "k_bits": 16,
    "v_bits": 16,
    "a_groupsize": -1,
    "k_groupsize": -1,
    "v_groupsize": -1,
    "a_clip_ratio": 1.0,
    "k_clip_ratio": 1.0,
    "v_clip_ratio": 1.0,
    "act_quant_aware_gptq": False,
    "k_cache_quant_aware_gptq": False,
}


def _flag_value(value: bool) -> str:
    return "true" if value else "false"


def _common_old(
    args: argparse.Namespace,
    case: dict[str, Any],
    run_dir: Path,
    run_id: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.world_size}",
        "ptq.py",
        "--model", args.model,
        "--exp", run_id,
        "--output_dir", str(run_dir),
        "--cache_dir", args.cache_dir,
        "--dataset", "wikitext2",
        "--nsamples", str(args.nsamples),
        "--seq_len", str(args.seq_len),
        "--eval_seq_len", str(args.seq_len),
        "--w_method", "gptq_plus",
        "--w_bits", "4",
        "--w_groupsize", str(case["w_groupsize"]),
        "--no_pre_clip",
        "--num_groups", "4",
        "--percdamp", "0.01",
        "--blocksize", "128",
        "--group_parallel_quant", "rank",
        "--enable_gptq_plus", "0",
        "--alpha", "0",
        "--second_order_scale", "1",
        "--g_update_mode", "block_gd",
        "--grad_optimizer", "adam",
        "--grad_reg_strategy", "none",
        "--grad_refresh_loss", "fisher_diag_mse",
        "--global_loss",
        "--global_loss_bsz", str(args.global_loss_bsz),
        "--saliency_clip_percentile", "0.99",
        "--grad_hessian_topk", "-1",
        "--kl_topk", "-1",
        "--bsz", str(args.stats_bsz),
        "--hessian_accum_bsz", str(args.hessian_accum_bsz),
        "--final_layer_stats_bsz", str(args.stats_bsz),
        "--backward_samples", str(args.backward_samples),
        "--backward_bsz", str(args.backward_bsz),
        "--final_layer_backward_bsz", str(args.backward_bsz),
        "--grad_lr", "0.0003",
        "--final_layer_grad_lr", "0.00001",
        "--grad_clip", "1",
        "--final_layer_grad_clip", "1",
        "--grad_lr_layer_schedule", "cosine",
        "--grad_lr_layer_base_ratio", "0.01",
        "--dp_global_shuffle",
        "--skip_eval",
        "--seed", str(args.seed),
        "--rotation_seed", str(args.rotation_seed),
        "--refresh_seed", str(args.refresh_seed),
        "--static_cache_path", args.static_old,
        "--alignment_trace_path", str(run_dir / "trace.jsonl"),
        "--alignment_run_id", run_id,
        "--save_qmodel_path", str(run_dir / "model.pt"),
        "--a_loss_ratio", str(case["a_loss_ratio"]),
        "--a_bits", str(case["a_bits"]),
        "--k_bits", str(case["k_bits"]),
        "--v_bits", str(case["v_bits"]),
        "--a_groupsize", str(case["a_groupsize"]),
        "--k_groupsize", str(case["k_groupsize"]),
        "--v_groupsize", str(case["v_groupsize"]),
        "--a_clip_ratio", str(case["a_clip_ratio"]),
        "--k_clip_ratio", str(case["k_clip_ratio"]),
        "--v_clip_ratio", str(case["v_clip_ratio"]),
    ]
    if case["w_clip"]:
        cmd.append("--w_clip")
    if case["act_order"]:
        cmd.append("--act_order")
    if case["rotate"]:
        cmd.append("--rotate")
    if case["loss_slide_window"]:
        cmd.append("--loss_slide_window")
    if case["act_quant_aware_gptq"]:
        cmd.append("--act_quant_aware_gptq")
    if case["k_cache_quant_aware_gptq"]:
        cmd.append("--k_cache_quant_aware_gptq")
    return cmd


def _common_new(
    args: argparse.Namespace,
    case: dict[str, Any],
    run_dir: Path,
    run_id: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.world_size}",
        "-m",
        "realq.ptq",
        "--model", args.model,
        "--exp", run_id,
        "--output_dir", str(run_dir),
        "--cache_dir", args.cache_dir,
        "--tokens_cache_path", str(Path(args.cache_dir) / "tokens"),
        "--dataset", "wikitext2",
        "--nsamples", str(args.nsamples),
        "--seq_len", str(args.seq_len),
        "--eval_seq_len", str(args.seq_len),
        "--w_bits", "4",
        "--w_groupsize", str(case["w_groupsize"]),
        "--w_clip", _flag_value(case["w_clip"]),
        "--num_groups", "4",
        "--percdamp", "0.01",
        "--blocksize", "128",
        "--act_order", _flag_value(case["act_order"]),
        "--group_parallel_quant", "rank",
        "--global_loss_bsz", str(args.global_loss_bsz),
        "--saliency_clip_percentile", "0.99",
        "--grad_hessian_topk", "-1",
        "--kl_topk", "-1",
        "--bsz", str(args.stats_bsz),
        "--hessian_accum_bsz", str(args.hessian_accum_bsz),
        "--backward_samples", str(args.backward_samples),
        "--backward_bsz", str(args.backward_bsz),
        "--final_layer_backward_bsz", str(args.backward_bsz),
        "--grad_lr", "0.0003",
        "--final_layer_grad_lr", "0.00001",
        "--grad_clip", "1",
        "--final_layer_grad_clip", "1",
        "--grad_lr_layer_schedule", "cosine",
        "--grad_lr_layer_base_ratio", "0.01",
        "--a_loss_ratio", str(case["a_loss_ratio"]),
        "--loss_slide_window", _flag_value(case["loss_slide_window"]),
        "--rotate", _flag_value(case["rotate"]),
        "--skip_eval", "true",
        "--seed", str(args.seed),
        "--rotation_seed", str(args.rotation_seed),
        "--refresh_seed", str(args.refresh_seed),
        "--static_cache_path", args.static_new,
        "--alignment_trace_path", str(run_dir / "trace.jsonl"),
        "--alignment_run_id", run_id,
        "--save_qmodel_path", str(run_dir / "model.pt"),
        "--a_bits", str(case["a_bits"]),
        "--k_bits", str(case["k_bits"]),
        "--v_bits", str(case["v_bits"]),
        "--a_groupsize", str(case["a_groupsize"]),
        "--k_groupsize", str(case["k_groupsize"]),
        "--v_groupsize", str(case["v_groupsize"]),
        "--a_clip_ratio", str(case["a_clip_ratio"]),
        "--k_clip_ratio", str(case["k_clip_ratio"]),
        "--v_clip_ratio", str(case["v_clip_ratio"]),
        "--act_quant_aware_gptq",
        _flag_value(case["act_quant_aware_gptq"]),
        "--k_cache_quant_aware_gptq",
        _flag_value(case["k_cache_quant_aware_gptq"]),
    ]


def _run(command: list[str], run_dir: Path, env: dict[str, str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8",
    )
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=env["REALQ_WORKSPACE"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        tail = (run_dir / "run.log").read_text(
            encoding="utf-8", errors="replace",
        ).splitlines()[-60:]
        raise RuntimeError(
            f"run failed ({completed.returncode}): {' '.join(command)}\n"
            + "\n".join(tail)
        )


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint {path} is not a mapping")
    return payload


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = _checkpoint_payload(path)
    return payload["model"]


def _canonical_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, tuple[str, torch.Tensor]]:
    """Normalize ActQuantWrapper's ``.module`` without hiding its weights."""
    canonical: dict[str, tuple[str, torch.Tensor]] = {}
    for raw_key, value in state.items():
        key = raw_key.replace(".module.", ".")
        if key in canonical:
            other_key, other_value = canonical[key]
            # ActQuantWrapper deliberately registers the same Parameter both
            # as ``wrapper.weight`` and ``wrapper.module.weight``. PyTorch
            # serializes both aliases. Merge only after proving equality.
            if (
                isinstance(other_value, torch.Tensor)
                and isinstance(value, torch.Tensor)
                and other_value.shape == value.shape
                and other_value.dtype == value.dtype
                and torch.equal(other_value, value)
            ):
                preferred_key = (
                    raw_key if ".module." in raw_key else other_key
                )
                canonical[key] = (preferred_key, value)
                continue
            raise RuntimeError(
                "state_dict canonicalization collision with unequal values: "
                f"{other_key!r} and {raw_key!r} both map to {key!r}"
            )
        canonical[key] = (raw_key, value)
    return canonical


def _compare_model_weights(a_path: Path, b_path: Path) -> dict[str, Any]:
    # Compare the complete tensor state, including wrapped linear weights and
    # persistent rotation/quantization buffers.  The old filter explicitly
    # discarded ``*.module.weight`` — precisely the quantized weights once an
    # ActQuantWrapper was installed — and could therefore report a false pass.
    a = _canonical_state_dict(_state_dict(a_path))
    b = _canonical_state_dict(_state_dict(b_path))
    keys = sorted(set(a) & set(b))
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    mismatched = []
    max_abs = 0.0
    for key in keys:
        a_raw, a_value = a[key]
        b_raw, b_value = b[key]
        if not isinstance(a_value, torch.Tensor) or not isinstance(
            b_value, torch.Tensor
        ):
            mismatched.append(
                {
                    "key": key,
                    "reason": "non_tensor",
                    "left_key": a_raw,
                    "right_key": b_raw,
                }
            )
            continue
        if a_value.shape != b_value.shape:
            mismatched.append({"key": key, "reason": "shape"})
            continue
        if a_value.dtype != b_value.dtype:
            mismatched.append(
                {
                    "key": key,
                    "reason": "dtype",
                    "left_dtype": str(a_value.dtype),
                    "right_dtype": str(b_value.dtype),
                }
            )
            continue
        diff = (a_value.float() - b_value.float()).abs()
        current_max = float(diff.max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, current_max)
        if not torch.equal(a_value, b_value):
            mismatched.append(
                {
                    "key": key,
                    "left_key": a_raw,
                    "right_key": b_raw,
                    "max_abs": current_max,
                    "different_elements": int((diff != 0).sum().item()),
                }
            )
    return {
        "passed": not missing and not extra and not mismatched,
        "compared_tensor_keys": len(keys),
        "compared_weight_keys": sum(
            key.endswith(".weight") for key in keys
        ),
        "max_abs_difference": max_abs,
        "missing_keys": missing,
        "extra_keys": extra,
        "mismatched_weights": mismatched,
    }


def _normalized_checkpoint_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Return semantic provenance, tolerating the legacy-only method label."""
    weight = dict(payload.get("weight_quantization", {}))
    method = weight.pop("w_method", "gptq_plus")
    return {
        "format": payload.get("format"),
        "format_version": payload.get("format_version"),
        "base_model": payload.get("base_model"),
        "runtime_quantization": payload.get("runtime_quantization"),
        "weight_quantization": weight,
        # The refactor has a single GPTQ+ implementation and therefore no
        # w_method field.  A legacy artifact is semantically comparable only
        # when its explicit selector names that same implementation.
        "weight_method": method,
        "artifact_identity": payload.get("artifact_identity"),
    }


def _compare_checkpoint_manifests(
    a_path: Path,
    b_path: Path,
) -> dict[str, Any]:
    left = _normalized_checkpoint_manifest(_checkpoint_payload(a_path))
    right = _normalized_checkpoint_manifest(_checkpoint_payload(b_path))
    differences = {
        key: {"left": left.get(key), "right": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    return {
        "passed": not differences,
        "differences": differences,
        "left": left,
        "right": right,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _capture_provenance(
    workspace: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Capture enough source/runtime identity to make a report auditable."""

    def git_output(*command: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *command],
                cwd=workspace,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable: {exc}"

    diff = git_output("diff", "--no-ext-diff", "--binary", "HEAD")
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "not-installed"
    gpu_names = (
        [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else []
    )
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output("status", "--short"),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "visible_gpu_names": gpu_names,
        "argv": sys.argv,
        "resolved_args": vars(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=os.getcwd())
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--static-old", required=True)
    parser.add_argument("--static-new", required=True)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rotation-seed", type=int, default=11)
    parser.add_argument("--refresh-seed", type=int, default=13)
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--global-loss-bsz", type=int, default=4)
    parser.add_argument("--stats-bsz", type=int, default=4)
    parser.add_argument("--hessian-accum-bsz", type=int, default=4)
    parser.add_argument("--backward-samples", type=int, default=4)
    parser.add_argument("--backward-bsz", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    unknown = sorted(set(args.cases) - set(CASES))
    if unknown:
        parser.error(f"unknown cases: {unknown}; choices={sorted(CASES)}")
    if args.repeats < 2:
        parser.error("--repeats must be >=2 for determinism evidence")
    if args.world_size <= 0:
        parser.error("--world-size must be positive")
    for name in (
        "global_loss_bsz",
        "stats_bsz",
        "hessian_accum_bsz",
        "backward_bsz",
    ):
        if getattr(args, name) % args.world_size:
            parser.error(
                f"--{name.replace('_', '-')} must be divisible by --world-size"
            )

    workspace = str(Path(args.workspace).resolve())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["REALQ_WORKSPACE"] = workspace
    env["PYTHONPATH"] = (
        workspace
        if not env.get("PYTHONPATH")
        else workspace + os.pathsep + env["PYTHONPATH"]
    )
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    matrix_report: dict[str, Any] = {
        "threshold": args.threshold,
        "world_size": args.world_size,
        "provenance": _capture_provenance(workspace, args),
        "cases": {},
    }
    all_passed = True
    for case_name in args.cases:
        case = {**DEFAULTS, **CASES[case_name]}
        case_root = output / case_name
        paths: dict[str, list[Path]] = {"legacy": [], "realq": []}
        for implementation in ("legacy", "realq"):
            for repeat in range(args.repeats):
                run_id = f"{case_name}_{implementation}_r{repeat}"
                run_dir = case_root / implementation / f"repeat_{repeat}"
                command = (
                    _common_old(args, case, run_dir, run_id)
                    if implementation == "legacy"
                    else _common_new(args, case, run_dir, run_id)
                )
                print(f"[alignment] running {run_id}", flush=True)
                _run(command, run_dir, env)
                paths[implementation].append(run_dir)

        comparisons: dict[str, Any] = {}
        pairs = [
            ("legacy_repeat", paths["legacy"][0], paths["legacy"][1]),
            ("realq_repeat", paths["realq"][0], paths["realq"][1]),
        ]
        pairs.extend(
            (
                f"old_new_repeat_{repeat}",
                paths["legacy"][repeat],
                paths["realq"][repeat],
            )
            for repeat in range(args.repeats)
        )
        for label, left, right in pairs:
            trace_report = compare_refresh_traces(
                str(left / "trace.jsonl"),
                str(right / "trace.jsonl"),
                max_relative_difference=args.threshold,
            )
            weight_report = _compare_model_weights(
                left / "model.pt", right / "model.pt",
            )
            manifest_report = _compare_checkpoint_manifests(
                left / "model.pt", right / "model.pt",
            )
            report = {
                "passed": (
                    trace_report["passed"]
                    and weight_report["passed"]
                    and manifest_report["passed"]
                ),
                "trace": trace_report,
                "weights": weight_report,
                "manifest": manifest_report,
            }
            comparisons[label] = report
            _write_json(case_root / f"{label}.json", report)
            all_passed = all_passed and report["passed"]
        matrix_report["cases"][case_name] = {
            "config": case,
            "comparisons": {
                key: {
                    "passed": value["passed"],
                    "matched_steps": value["trace"]["matched_steps"],
                    "max_relative_difference": value["trace"][
                        "max_relative_difference"
                    ],
                    "weight_max_abs_difference": value["weights"][
                        "max_abs_difference"
                    ],
                    "manifest_passed": value["manifest"]["passed"],
                }
                for key, value in comparisons.items()
            },
            "passed": all(value["passed"] for value in comparisons.values()),
        }
        _write_json(output / "matrix_report.partial.json", matrix_report)

    matrix_report["passed"] = all_passed
    _write_json(output / "matrix_report.json", matrix_report)
    print(json.dumps(matrix_report, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
