#!/usr/bin/env python3
"""Fail-closed command renderer for the low-bit activation experiment matrix.

This tool does not submit Canoe jobs and does not launch experiments.  It
turns the reviewed machine-readable plan into shell commands only after every
configuration choice relevant to that command has been resolved.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = REPO_ROOT / "experiments" / "lowbit_activation" / "plan.json"

MODEL_ORDER = ("llama3.2-3b", "qwen3-4b", "qwen3-32b")
SETTING_ORDER = ("3W16A", "2W16A", "4W4A", "3W4A", "2W4A")
QUANT_METHODS = ("realq", "gptaq", "guided_gptq")
EXPECTED_TASKS = (
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "lambada_openai",
    "ceval-valid",
    "boolq",
    "openbookqa",
    "social_iqa",
)
EXPECTED_CANOE_JOB_ID = "j-abmvtvxw97"
EXPECTED_PREFLIGHT_MANIFEST = (
    "output/lowbit_activation/runtime_preflight_j-abmvtvxw97.json"
)
EXPECTED_MODELS = {
    "llama3.2-3b": "modelzoo/Llama/Llama-3.2-3B",
    "qwen3-4b": "modelzoo/Qwen3/Qwen3-4B",
    "qwen3-32b": "modelzoo/Qwen3/Qwen3-32B",
}
EXPECTED_MODEL_SIGNATURES = {
    "llama3.2-3b": {
        "model_type": "llama",
        "hidden_size": 3072,
        "num_hidden_layers": 28,
        "architectures": ["LlamaForCausalLM"],
    },
    "qwen3-4b": {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "architectures": ["Qwen3ForCausalLM"],
    },
    "qwen3-32b": {
        "model_type": "qwen3",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "architectures": ["Qwen3ForCausalLM"],
    },
}
EXPECTED_SETTINGS = {
    "3W16A": {"w_bits": 3, "a_bits": 16, "k_bits": 16, "v_bits": 16},
    "2W16A": {"w_bits": 2, "a_bits": 16, "k_bits": 16, "v_bits": 16},
    "4W4A": {"w_bits": 4, "a_bits": 4, "k_bits": 4, "v_bits": 4},
    "3W4A": {"w_bits": 3, "a_bits": 4, "k_bits": 4, "v_bits": 4},
    "2W4A": {"w_bits": 2, "a_bits": 4, "k_bits": 4, "v_bits": 4},
}
EXPECTED_SEARCH_POLICY = {
    "strategy": "adaptive_bracket_then_canonical_refine",
    "canonical_mantissas": [1, 2, 3, 5, 7],
    "boundary_expansion_batch_size": 2,
    "reserved_refinement_attempts": 3,
    "zero_boundary_probe_ratios": [0.1, 0.3, 0.7],
    "fixed_upper_lr_ceiling": None,
    "require_complete_coarse_round": True,
    "require_local_refinement": True,
}
EXPECTED_TUNING = {
    "world_size": 1,
    "nsamples": 256,
    "seq_len": 2048,
    "bsz": 32,
    "global_loss_bsz": 8,
    "hessian_accum_bsz": 32,
    "backward_samples": 16,
    "backward_bsz": 16,
    "final_layer_backward_bsz": 16,
    "blocksize": 256,
    "lm_eval": False,
    "max_attempts_per_model_setting": 20,
    "search_policy": EXPECTED_SEARCH_POLICY,
    "lr_candidates": [
        0.0,
        0.000001,
        0.000005,
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
    ],
}
EXPECTED_FINAL = {
    "world_size": 4,
    "nsamples": 256,
    "seq_len": 2048,
    "bsz": 128,
    "global_loss_bsz": 32,
    "hessian_accum_bsz": 128,
    "backward_samples": 32,
    "backward_bsz": 32,
    "final_layer_backward_bsz": 32,
    "blocksize": 128,
    "lm_eval": True,
    "lm_eval_batch_size": 32,
}
EXPECTED_FIXED_NUMERICS = {
    "dataset": "wikitext2",
    "eval_datasets": ["wikitext2"],
    "eval_seq_len": 2048,
    "num_groups": 4,
    "w_clip": True,
    "act_order": True,
    "symmetric": True,
    "percdamp": 0.01,
    "rotate": True,
    "seed": 1,
    "rotation_seed": 0,
    "refresh_seed": 0,
    "grad_clip": 1.0,
    "final_layer_grad_clip": None,
    "grad_hessian_topk": -1,
    "kl_topk": -1,
    "saliency_clip_percentile": 0.99,
    "a_loss_clip_scope": "local_backward_chunk",
    "loss_slide_window": True,
    "group_parallel_quant": "rank",
    "fsdp": False,
    "cpu_master": False,
    "quantizer_inner_fastpath": False,
    "w_clip_search_impl": "cartesian_legacy",
    "fisher_fp32_cache": False,
    "act_order_stitch_impl": "full_weight_legacy",
    "w_clip_update_impl": "guarded",
    "w_group_param_layout": "expanded",
    "activation_clip_ratio": 0.9,
    "qwen3_4b_a_loss_ratio": 0.95,
    "other_model_a_loss_ratio": 1.0,
}
EXPECTED_QWEN3_32B_MEMORY_POLICY = {
    "fsdp": False,
    "adjust_only_batch_knobs_on_oom": True,
    "bsz_note": "record_only_in_refactored_pipeline_do_not_use_as_an_oom_knob",
    "tuning_ladders": {
        "global_loss_bsz": [8, 4, 2, 1],
        "hessian_accum_bsz": [32, 16, 8, 4, 2, 1],
        "backward_bsz": [16, 8, 4, 2, 1],
        "final_layer_backward_bsz": [16, 8, 4, 2, 1],
    },
    "final_ladders": {
        "global_loss_bsz": [32, 16, 8, 4],
        "hessian_accum_bsz": [128, 64, 32, 16, 8, 4, 2, 1],
        "backward_bsz": [32, 16, 8, 4],
        "final_layer_backward_bsz": [32, 16, 8, 4],
        "lm_eval_batch_size": [32, 16, 8, 4, 2, 1],
    },
}
EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.9.1",
    "torch_runtime": "2.9.1+cu128",
    "transformers": "4.56.2",
    "lm-eval": "0.4.4",
}
EXPECTED_RUNTIME_ENVIRONMENT = {
    "HF_HOME": (
        "/minimax-avatar-new/zhangqian/realq/gptq_plus/"
        "datasets/lm_eval_hf_cache"
    ),
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_TRUST_REMOTE_CODE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


class PlanError(ValueError):
    """The experiment plan is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class RenderedCommand:
    env: dict[str, str]
    argv: list[str]
    run_id: str
    output_dir: Path

    def shell(self) -> str:
        prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in sorted(self.env.items())
        )
        command = shlex.join(self.argv)
        return f"{prefix} {command}" if prefix else command


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    validate_structure(plan)
    return plan


def validate_structure(plan: dict[str, Any]) -> None:
    errors: list[str] = []
    if plan.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if plan.get("models") != EXPECTED_MODELS:
        errors.append(
            "models must bind the reviewed names to the exact 3B/4B/32B "
            f"artifact paths: {EXPECTED_MODELS!r}"
        )
    if plan.get("model_signatures") != EXPECTED_MODEL_SIGNATURES:
        errors.append(
            "model_signatures must lock the reviewed architecture, hidden "
            "size, and layer count for every model"
        )
    if plan.get("settings") != EXPECTED_SETTINGS:
        errors.append(
            "settings must equal the exact reviewed W/A/K/V bit mapping "
            f"{EXPECTED_SETTINGS!r}"
        )
    if tuple(plan.get("paper_zero_shot_tasks", ())) != EXPECTED_TASKS:
        errors.append("paper_zero_shot_tasks does not match the paper's ten-task set")
    if tuple(plan.get("methods", ())) != (
        "realq",
        "gptaq",
        "guided_gptq",
        "bf16",
    ):
        errors.append("methods must contain only REAL-Q, GPTAQ, GuidedGPTQ, BF16")
    if "llama3.2-2b" in plan.get("models", {}):
        errors.append("obsolete llama3.2-2b target is forbidden; use llama3.2-3b")

    fixed = plan.get("fixed_numerics", {})
    if fixed != EXPECTED_FIXED_NUMERICS:
        errors.append(
            "fixed_numerics must equal the complete reviewed paper protocol"
        )
    if plan.get("tuning") != EXPECTED_TUNING:
        errors.append(
            "tuning must equal the reviewed proxy world/batch/block/search "
            "protocol"
        )
    if plan.get("final") != EXPECTED_FINAL:
        errors.append(
            "final must equal the reviewed formal world/batch/block/eval protocol"
        )
    if fixed.get("eval_datasets") != ["wikitext2"]:
        errors.append("KL/PPL eval_datasets must be exactly ['wikitext2']")
    if fixed.get("fsdp") is not False or fixed.get("cpu_master") is not False:
        errors.append("all runs, especially Qwen3-32B, must keep fsdp/cpu_master false")
    if fixed.get("symmetric") is not True:
        errors.append("weight and activation quantization must be symmetric")
    if plan.get("final", {}).get("nsamples") != 256:
        errors.append("user-confirmed final nsamples must be 256")
    if plan.get("tuning", {}).get("nsamples") != 256:
        errors.append("tuning nsamples must be 256")
    if plan.get("tuning", {}).get("world_size") != 1:
        errors.append("LR proxy tuning must use one GPU")
    if plan.get("final", {}).get("world_size") != 4:
        errors.append("formal REAL-Q runs must use four GPUs")
    if plan.get("tuning", {}).get("blocksize") != 256:
        errors.append("LR proxy tuning blocksize must be 256")
    if plan.get("final", {}).get("blocksize") != 128:
        errors.append("formal blocksize must be 128")
    if plan.get("tuning", {}).get("lm_eval") is not False:
        errors.append("LR proxy tuning must disable lm_eval")
    if plan.get("tuning", {}).get("max_attempts_per_model_setting") != 20:
        errors.append(
            "each model/setting proxy sweep must be capped at 20 attempts"
        )
    if plan.get("tuning", {}).get("search_policy") != EXPECTED_SEARCH_POLICY:
        errors.append(
            "tuning.search_policy must use adaptive bracketing and canonical "
            "local refinement without a fixed upper LR ceiling"
        )
    if plan.get("final", {}).get("lm_eval") is not True:
        errors.append("formal runs must enable lm_eval")

    resolutions = plan.get("resolutions", {})
    if resolutions.get("canoe_submit_confirmed") is not True:
        errors.append("resolutions.canoe_submit_confirmed must be true")
    if resolutions.get("canoe_job_id") != EXPECTED_CANOE_JOB_ID:
        errors.append(
            "resolutions.canoe_job_id must equal the dedicated experiment job "
            f"{EXPECTED_CANOE_JOB_ID!r}"
        )
    if (
        resolutions.get("runtime_preflight_manifest")
        != EXPECTED_PREFLIGHT_MANIFEST
    ):
        errors.append(
            "resolutions.runtime_preflight_manifest must equal "
            f"{EXPECTED_PREFLIGHT_MANIFEST!r}"
        )
    expected_resolutions = {
        "q4_protocol": "paper_a4k4v4_no_independent_q",
        "a16_aware_semantics": "weight_only_aware_not_applicable",
        "final_nsamples_policy": "all_quantized_runs_256",
        "final_layer_lr_policy": "model_size_le_4b_1e-5_gt_4b_1e-6",
        "lr_boundary_policy": (
            "expand_outward_until_an_interior_kl_minimum_is_bracketed"
        ),
        "tune_group_block_policy": "proxy_per_row_block256",
    }
    for key, expected in expected_resolutions.items():
        if resolutions.get(key) != expected:
            errors.append(f"{key} must equal {expected!r}")

    exact_impl_defaults = {
        "quantizer_inner_fastpath": False,
        "w_clip_search_impl": "cartesian_legacy",
        "fisher_fp32_cache": False,
        "act_order_stitch_impl": "full_weight_legacy",
        "w_clip_update_impl": "guarded",
        "w_group_param_layout": "expanded",
    }
    for key, expected in exact_impl_defaults.items():
        if fixed.get(key) != expected:
            errors.append(f"fixed_numerics.{key} must equal {expected!r}")

    if plan.get("qwen3_32b_memory_policy") != EXPECTED_QWEN3_32B_MEMORY_POLICY:
        errors.append(
            "Qwen3-32B memory policy must use the exact reviewed no-FSDP "
            "batch-only OOM ladders"
        )
    if plan.get("runtime_requirements") != {
        "minimum_gpu_count": 8,
        "minimum_gpu_memory_gib": 120,
    }:
        errors.append(
            "runtime_requirements must gate on 8 GPUs with at least 120 GiB each"
        )
    if plan.get("runtime_versions") != EXPECTED_RUNTIME_VERSIONS:
        errors.append(
            "runtime_versions must pin torch, transformers, and lm-eval to "
            f"{EXPECTED_RUNTIME_VERSIONS!r}"
        )
    if plan.get("runtime_environment") != EXPECTED_RUNTIME_ENVIRONMENT:
        errors.append(
            "runtime_environment must equal the reviewed environment exactly; "
            "data-source, proxy, and offline-mode changes require a new review"
        )
    if plan.get("python_environment") != {
        "venv": "/minimax-avatar-new/zhangqian/realq/gptq_plus/.venv",
        "activation_required": True,
    }:
        errors.append(
            "python_environment must require the experiment venv recorded "
            "in AGENT.md"
        )
    expected_baseline_numerics = {
        "offload_inps": False,
        "gptaq_alpha": 0.25,
        "guided_saliency_gradient_scale": 1000.0,
        "cholesky_damp_auto_increment": 0.0015,
        "cholesky_retry_semantics": (
            "historical_cumulative_diagonal_additions"
        ),
        "weight_mse_norm": 2.4,
        "weight_mse_grid": 50,
        "weight_mse_maxshrink": 0.5,
        "weight_mse_search_impl": "cartesian_legacy",
        "weight_mse_update_impl": "guarded",
        "guided_num_groups": 4,
        "guided_act_order_score": "shared_unweighted_input_energy",
    }
    if plan.get("baseline_numerics") != expected_baseline_numerics:
        errors.append(
            "baseline_numerics must match the reviewed REAL-Q comparison "
            "implementation"
        )
    if expected_baseline_numerics["guided_num_groups"] != fixed.get(
        "num_groups"
    ):
        errors.append("GuidedGPTQ num_groups must match fixed_numerics.num_groups")

    final_lrs = plan.get("final_layer_grad_lr_by_model", {})
    expected_lrs = {
        "llama3.2-3b": 1e-5,
        "qwen3-4b": 1e-5,
        "qwen3-32b": 1e-6,
    }
    if final_lrs != expected_lrs:
        errors.append(f"final-layer LR mapping must equal {expected_lrs!r}")

    selected_lrs = plan.get("selected_grad_lr_by_model_setting")
    if not isinstance(selected_lrs, dict) or tuple(selected_lrs) != MODEL_ORDER:
        errors.append(
            "selected_grad_lr_by_model_setting must contain every model"
        )
    else:
        for model in MODEL_ORDER:
            model_values = selected_lrs.get(model)
            if (
                not isinstance(model_values, dict)
                or tuple(model_values) != SETTING_ORDER
            ):
                errors.append(
                    f"selected LR ledger for {model} must contain every setting"
                )
                continue
            for setting_name, value in model_values.items():
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    errors.append(
                        f"selected LR for {model}/{setting_name} must be null "
                        f"or a non-negative finite number, got {value!r}"
                    )

    for setting_name, setting in plan.get("settings", {}).items():
        for key in ("w_bits", "a_bits", "k_bits", "v_bits"):
            if key not in setting:
                errors.append(f"{setting_name} is missing {key}")
        if setting_name.endswith("16A") and any(
            setting.get(key) != 16 for key in ("a_bits", "k_bits", "v_bits")
        ):
            errors.append(f"{setting_name} must be weight-only A/K/V16")
        if setting_name.endswith("4A") and any(
            setting.get(key) != 4 for key in ("a_bits", "k_bits", "v_bits")
        ):
            errors.append(f"{setting_name} must map to the paper A4/K4/V4 protocol")

    if errors:
        raise PlanError("; ".join(errors))


def unresolved_items(plan: dict[str, Any]) -> list[str]:
    resolutions = plan["resolutions"]
    unresolved = [
        key
        for key in (
            "q4_protocol",
            "a16_aware_semantics",
            "final_nsamples_policy",
            "final_layer_lr_policy",
            "lr_boundary_policy",
            "tune_group_block_policy",
        )
        if resolutions.get(key) is None
    ]
    if not resolutions.get("canoe_submit_confirmed", False):
        unresolved.append("canoe_submit_confirmed")
    if resolutions.get("canoe_submit_confirmed") and not resolutions.get(
        "canoe_job_id"
    ):
        unresolved.append("canoe_job_id")
    preflight_value = resolutions.get("runtime_preflight_manifest")
    preflight_path = (
        Path(preflight_value)
        if isinstance(preflight_value, str) and preflight_value
        else None
    )
    if preflight_path is not None and not preflight_path.is_absolute():
        preflight_path = REPO_ROOT / preflight_path
    if preflight_path is None or not preflight_path.is_file():
        unresolved.append("runtime_preflight_manifest")
    selected_lrs = plan.get("selected_grad_lr_by_model_setting", {})
    for model in MODEL_ORDER:
        model_values = selected_lrs.get(model, {})
        for setting in SETTING_ORDER:
            if model_values.get(setting) is None:
                unresolved.append(f"selected_grad_lr:{model}:{setting}")
    return unresolved


def matrix_rows(plan: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        for setting in SETTING_ORDER:
            aware = (
                "A/K/V-aware"
                if plan["settings"][setting]["a_bits"] < 16
                else "N/A (weight-only)"
            )
            for method in QUANT_METHODS:
                rows.append(
                    {
                        "model": model,
                        "setting": setting,
                        "method": method,
                        "aware": aware,
                    }
                )
        rows.append(
            {
                "model": model,
                "setting": "BF16",
                "method": "bf16",
                "aware": "N/A",
            }
        )
    return rows


def _require_resolution(plan: dict[str, Any], key: str) -> str:
    value = plan["resolutions"].get(key)
    if value is None:
        raise PlanError(f"unresolved protocol choice: {key}")
    return str(value)


def _setting(plan: dict[str, Any], name: str) -> dict[str, int]:
    try:
        return plan["settings"][name]
    except KeyError as exc:
        raise PlanError(f"unknown setting {name!r}") from exc


def _model_path(plan: dict[str, Any], name: str) -> str:
    try:
        return plan["models"][name]
    except KeyError as exc:
        raise PlanError(f"unknown model {name!r}") from exc


def _venv_python(plan: dict[str, Any]) -> str:
    configured = plan.get("python_environment")
    if not isinstance(configured, dict):
        raise PlanError("python_environment must be configured")
    venv = configured.get("venv")
    if not isinstance(venv, str) or not venv:
        raise PlanError("python_environment.venv must be a non-empty path")
    path = Path(venv)
    if not path.is_absolute():
        raise PlanError("python_environment.venv must be absolute")
    return str(path / "bin" / "python")


def _cuda_indices(cuda_devices: str) -> list[int]:
    if not isinstance(cuda_devices, str) or not cuda_devices:
        raise PlanError("CUDA device list must be a non-empty comma-separated index list")
    raw_items = cuda_devices.split(",")
    if any(not item or not item.isascii() or not item.isdigit() for item in raw_items):
        raise PlanError(
            "CUDA devices must be non-negative decimal indices without "
            f"whitespace, got {cuda_devices!r}"
        )
    indices = [int(item) for item in raw_items]
    if len(set(indices)) != len(indices):
        raise PlanError(f"CUDA devices must be distinct, got {cuda_devices!r}")
    return indices


def _selected_grad_lr(
    plan: dict[str, Any],
    model: str,
    setting: str,
) -> float:
    value = plan["selected_grad_lr_by_model_setting"][model][setting]
    if value is None:
        raise PlanError(
            f"best grad_lr is not recorded for {model}/{setting}; complete "
            "the coarse + local-refinement sweep before formal rendering"
        )
    return float(value)


def _is_activation_quantized(setting: dict[str, int]) -> bool:
    return any(setting[key] < 16 for key in ("a_bits", "k_bits", "v_bits"))


def _weight_group_and_block(
    plan: dict[str, Any], phase: str
) -> tuple[int, int]:
    if phase == "final":
        return 128, int(plan["final"]["blocksize"])
    policy = _require_resolution(plan, "tune_group_block_policy")
    requested_block = int(plan["tuning"]["blocksize"])
    if policy == "proxy_per_row_block256":
        return -1, requested_block
    raise PlanError(f"unsupported tune_group_block_policy={policy!r}")


def _slug_float(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.9g}".replace("+", "").replace("-", "m").replace(".", "p")


def _add_bool(argv: list[str], flag: str, value: bool) -> None:
    argv.extend([flag, "true" if value else "false"])


def _set_option(argv: list[str], flag: str, value: str) -> None:
    try:
        index = argv.index(flag)
    except ValueError as exc:
        raise PlanError(f"rendered command is missing required option {flag}") from exc
    if index + 1 >= len(argv):
        raise PlanError(f"rendered command option {flag} has no value")
    argv[index + 1] = value


def _run_id(
    phase: str,
    method: str,
    model: str,
    setting: str,
    grad_lr: float | None,
    overrides: dict[str, int] | None = None,
) -> str:
    parts = [phase, method, model, setting.lower()]
    if grad_lr is not None:
        parts.append(f"lr{_slug_float(grad_lr)}")
    if overrides:
        parts.extend(
            f"{key}{value}" for key, value in sorted(overrides.items())
        )
    return "_".join(parts).replace("/", "_")


def render_realq(
    plan: dict[str, Any],
    *,
    phase: str,
    model: str,
    setting_name: str,
    grad_lr: float,
    cuda_devices: str,
    overrides: dict[str, int] | None = None,
) -> RenderedCommand:
    if phase not in {"tune", "final"}:
        raise PlanError(f"unsupported REAL-Q phase {phase!r}")
    _require_resolution(plan, "q4_protocol")
    _require_resolution(plan, "a16_aware_semantics")
    if phase == "final":
        _require_resolution(plan, "final_nsamples_policy")
        _require_resolution(plan, "final_layer_lr_policy")
        selected_grad_lr = _selected_grad_lr(
            plan,
            model,
            setting_name,
        )
        if float(grad_lr) != selected_grad_lr:
            raise PlanError(
                f"formal grad_lr for {model}/{setting_name} must equal the "
                f"recorded best value {selected_grad_lr}, got {grad_lr}"
            )

    setting = _setting(plan, setting_name)
    model_path = _model_path(plan, model)
    fixed = plan["fixed_numerics"]
    phase_cfg = dict(plan["tuning" if phase == "tune" else "final"])
    effective_overrides: dict[str, int] = {}
    if overrides:
        allowed = {
            "global_loss_bsz",
            "hessian_accum_bsz",
            "backward_bsz",
            "final_layer_backward_bsz",
            "lm_eval_batch_size",
        }
        unexpected = sorted(set(overrides) - allowed)
        if unexpected:
            raise PlanError(
                "only effective batch/memory knobs may be overridden "
                "(the refactored pipeline does not consume bsz): "
                f"unexpected={unexpected!r}"
            )
        non_positive = {
            key: value for key, value in overrides.items() if value <= 0
        }
        if non_positive:
            raise PlanError(
                f"batch/memory overrides must be positive: {non_positive!r}"
            )
        effective_overrides = {
            key: value
            for key, value in overrides.items()
            if phase_cfg.get(key) != value
        }
        phase_cfg.update(effective_overrides)

    w_groupsize, blocksize = _weight_group_and_block(plan, phase)
    aware = _is_activation_quantized(setting)
    schedule = "none" if aware else "cosine"
    a_loss_ratio = (
        float(fixed["qwen3_4b_a_loss_ratio"])
        if model == "qwen3-4b"
        else float(fixed["other_model_a_loss_ratio"])
    )
    final_layer_lr = float(plan["final_layer_grad_lr_by_model"][model])
    world_size = int(phase_cfg["world_size"])

    visible = _cuda_indices(cuda_devices)
    if len(visible) != world_size:
        raise PlanError(
            f"{phase} requires {world_size} distinct visible GPU(s), "
            f"got {cuda_devices!r}"
        )
    for key in (
        "nsamples",
        "bsz",
        "global_loss_bsz",
        "backward_samples",
        "backward_bsz",
        "final_layer_backward_bsz",
    ):
        if int(phase_cfg[key]) % world_size != 0:
            raise PlanError(
                f"{key}={phase_cfg[key]} must be divisible by "
                f"world_size={world_size}"
            )
    if phase == "tune" and overrides and "lm_eval_batch_size" in overrides:
        raise PlanError("lm_eval_batch_size is irrelevant while tune lm_eval=false")
    if (
        isinstance(grad_lr, bool)
        or not isinstance(grad_lr, (int, float))
        or not math.isfinite(float(grad_lr))
        or grad_lr < 0
    ):
        raise PlanError(f"grad_lr must be a finite non-negative number, got {grad_lr}")

    run_id = _run_id(
        phase,
        "realq",
        model,
        setting_name,
        grad_lr,
        effective_overrides,
    )
    output_root = Path(plan["output_root"])
    output_dir = output_root / model / setting_name / "realq" / phase
    static_cache = (
        Path(plan["static_cache_root"])
        / model
        / f"{phase}_world{world_size}_glbsz{phase_cfg['global_loss_bsz']}"
    )

    python = _venv_python(plan)
    if world_size == 1:
        argv = [python, "-m", "realq.ptq"]
    else:
        argv = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={world_size}",
            "-m",
            "realq.ptq",
        ]
    argv.extend(
        [
            "--model",
            model_path,
            "--dataset",
            fixed["dataset"],
            "--eval_datasets",
            "wikitext2",
            "--eval_seq_len",
            str(fixed["eval_seq_len"]),
            "--w_bits",
            str(setting["w_bits"]),
            "--w_groupsize",
            str(w_groupsize),
            "--a_bits",
            str(setting["a_bits"]),
            "--k_bits",
            str(setting["k_bits"]),
            "--v_bits",
            str(setting["v_bits"]),
            "--a_groupsize",
            "-1",
            "--k_groupsize",
            "-1",
            "--v_groupsize",
            "-1",
            "--a_clip_ratio",
            str(fixed["activation_clip_ratio"] if aware else 1.0),
            "--k_clip_ratio",
            str(fixed["activation_clip_ratio"] if aware else 1.0),
            "--v_clip_ratio",
            str(fixed["activation_clip_ratio"] if aware else 1.0),
            "--num_groups",
            str(fixed["num_groups"]),
            "--percdamp",
            str(fixed["percdamp"]),
            "--blocksize",
            str(blocksize),
            "--group_parallel_quant",
            fixed["group_parallel_quant"],
            "--grad_lr",
            str(grad_lr),
            "--final_layer_grad_lr",
            str(final_layer_lr),
            "--grad_clip",
            str(fixed["grad_clip"]),
            "--grad_lr_layer_schedule",
            schedule,
            "--grad_lr_layer_base_ratio",
            "0.01",
            "--a_loss_ratio",
            str(a_loss_ratio),
            "--a_loss_clip_scope",
            fixed["a_loss_clip_scope"],
            "--saliency_clip_percentile",
            str(fixed["saliency_clip_percentile"]),
            "--grad_hessian_topk",
            str(fixed["grad_hessian_topk"]),
            "--kl_topk",
            str(fixed["kl_topk"]),
            "--nsamples",
            str(phase_cfg["nsamples"]),
            "--seq_len",
            str(phase_cfg["seq_len"]),
            "--bsz",
            str(phase_cfg["bsz"]),
            "--global_loss_bsz",
            str(phase_cfg["global_loss_bsz"]),
            "--hessian_accum_bsz",
            str(phase_cfg["hessian_accum_bsz"]),
            "--backward_samples",
            str(phase_cfg["backward_samples"]),
            "--backward_bsz",
            str(phase_cfg["backward_bsz"]),
            "--final_layer_backward_bsz",
            str(phase_cfg["final_layer_backward_bsz"]),
            "--seed",
            str(fixed["seed"]),
            "--rotation_seed",
            str(fixed["rotation_seed"]),
            "--refresh_seed",
            str(fixed["refresh_seed"]),
            "--static_cache_path",
            str(static_cache),
            "--output_dir",
            str(output_dir),
            "--exp",
            run_id,
        ]
    )
    for flag, value in (
        ("--w_asym", False),
        ("--a_asym", False),
        ("--k_asym", False),
        ("--v_asym", False),
        ("--w_clip", bool(fixed["w_clip"])),
        ("--act_order", bool(fixed["act_order"])),
        ("--rotate", bool(fixed["rotate"])),
        ("--act_quant_aware_gptq", aware),
        ("--k_cache_quant_aware_gptq", aware),
        ("--loss_slide_window", bool(fixed["loss_slide_window"])),
        ("--fsdp", False),
        ("--cpu_master", False),
        ("--require_static_cache_hit", True),
        ("--require_reference_cache_hit", True),
        ("--skip_eval", False),
        ("--lm_eval", bool(phase_cfg["lm_eval"])),
        ("--log_column_block_loss", True),
        ("--quantizer_inner_fastpath", bool(fixed["quantizer_inner_fastpath"])),
        ("--fisher_fp32_cache", bool(fixed["fisher_fp32_cache"])),
    ):
        _add_bool(argv, flag, value)
    argv.extend(
        [
            "--w_clip_search_impl",
            fixed["w_clip_search_impl"],
            "--act_order_stitch_impl",
            fixed["act_order_stitch_impl"],
            "--w_clip_update_impl",
            fixed["w_clip_update_impl"],
            "--w_group_param_layout",
            fixed["w_group_param_layout"],
        ]
    )
    if phase == "final":
        argv.extend(
            [
                "--lm_eval_batch_size",
                str(phase_cfg["lm_eval_batch_size"]),
            ]
        )

    return RenderedCommand(
        env={
            **plan.get("runtime_environment", {}),
            "CUDA_VISIBLE_DEVICES": cuda_devices,
        },
        argv=argv,
        run_id=run_id,
        output_dir=output_dir / run_id,
    )


def render_baseline(
    plan: dict[str, Any],
    *,
    method: str,
    model: str,
    setting_name: str | None,
    cuda_devices: str,
    overrides: dict[str, int] | None = None,
) -> RenderedCommand:
    if method not in {"gptaq", "guided_gptq", "bf16"}:
        raise PlanError(f"unsupported baseline method {method!r}")
    _require_resolution(plan, "final_nsamples_policy")
    fixed = plan["fixed_numerics"]
    final_cfg = dict(plan["final"])
    effective_overrides: dict[str, int] = {}
    if overrides:
        unexpected = sorted(set(overrides) - {"lm_eval_batch_size"})
        if unexpected:
            raise PlanError(
                "GPTAQ/GuidedGPTQ do not consume REAL-Q batch knobs; only "
                "lm_eval_batch_size may be overridden for baselines: "
                f"unexpected={unexpected!r}"
            )
        if int(overrides["lm_eval_batch_size"]) <= 0:
            raise PlanError("lm_eval_batch_size override must be positive")
        effective_overrides = {
            key: value
            for key, value in overrides.items()
            if final_cfg.get(key) != value
        }
        final_cfg.update(effective_overrides)

    model_path = _model_path(plan, model)
    visible = _cuda_indices(cuda_devices)
    setting_label = "BF16" if method == "bf16" else str(setting_name)
    if method != "bf16" and setting_name is None:
        raise PlanError(f"{method} requires --setting")
    setting = (
        {"w_bits": 16, "a_bits": 16, "k_bits": 16, "v_bits": 16}
        if method == "bf16"
        else _setting(plan, str(setting_name))
    )
    aware = method != "bf16" and _is_activation_quantized(setting)
    if aware:
        _require_resolution(plan, "q4_protocol")
    else:
        _require_resolution(plan, "a16_aware_semantics")

    mapped_method = (
        "gptq_guided"
        if method == "guided_gptq"
        else ("gptaq" if method == "gptaq" else None)
    )
    run_id = _run_id(
        "final",
        method,
        model,
        setting_label,
        None,
        effective_overrides,
    )
    output_root = Path(plan["output_root"])
    output_dir = output_root / model / setting_label / method / "final"

    argv = [
        _venv_python(plan),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        "ptq.py",
        "--model",
        model_path,
        "--output_dir",
        str(output_dir),
        "--cache_dir",
        str(plan["legacy_cache_root"]),
        "--exp",
        run_id,
        "--dataset",
        fixed["dataset"],
        "--eval_datasets",
        "wikitext2",
        "--eval_seq_len",
        str(fixed["eval_seq_len"]),
        "--nsamples",
        str(final_cfg["nsamples"]),
        "--seq_len",
        str(final_cfg["seq_len"]),
        "--seed",
        str(fixed["seed"]),
        "--rotation_seed",
        str(fixed["rotation_seed"]),
        "--refresh_seed",
        str(fixed["refresh_seed"]),
        "--w_bits",
        str(setting["w_bits"]),
        "--w_groupsize",
        "128" if method != "bf16" else "-1",
        "--a_bits",
        str(setting["a_bits"]),
        "--k_bits",
        str(setting["k_bits"]),
        "--v_bits",
        str(setting["v_bits"]),
        "--a_groupsize",
        "-1",
        "--k_groupsize",
        "-1",
        "--v_groupsize",
        "-1",
        "--a_clip_ratio",
        str(fixed["activation_clip_ratio"] if aware else 1.0),
        "--k_clip_ratio",
        str(fixed["activation_clip_ratio"] if aware else 1.0),
        "--v_clip_ratio",
        str(fixed["activation_clip_ratio"] if aware else 1.0),
        "--num_groups",
        str(fixed["num_groups"]),
        "--percdamp",
        str(fixed["percdamp"]),
        "--blocksize",
        str(final_cfg["blocksize"]),
        "--kl_topk",
        str(fixed["kl_topk"]),
        "--lm_eval_batch_size",
        str(final_cfg["lm_eval_batch_size"]),
        "--lm_eval",
    ]
    if mapped_method is not None:
        argv.extend(["--w_method", mapped_method])
    if method != "bf16":
        argv.extend(["--w_clip", "--act_order", "--rotate"])
        if plan["baseline_numerics"]["offload_inps"]:
            argv.append("--offload_inps")
    if method == "gptaq":
        argv.extend(
            ["--alpha", str(plan["baseline_numerics"]["gptaq_alpha"])]
        )
    if aware:
        argv.extend(["--act_quant_aware_gptq", "--k_cache_quant_aware_gptq"])

    return RenderedCommand(
        env={
            **plan.get("runtime_environment", {}),
            "CUDA_VISIBLE_DEVICES": cuda_devices,
        },
        argv=argv,
        run_id=run_id,
        output_dir=output_dir / Path(model_path).name / run_id,
    )


def render_realq_static_precompute(
    plan: dict[str, Any],
    *,
    target_phase: str,
    model: str,
    cuda_devices: str,
    overrides: dict[str, int] | None = None,
) -> RenderedCommand:
    """Render one cache-building run before concurrent REAL-Q LR arms.

    Static Stage 0 is invariant to W/A bits and LR, so a canonical 3W16A,
    grad_lr=0 command can build the cache shared by every setting/LR arm for
    the same model, target phase, world size, and global_loss_bsz. Evaluation
    remains enabled so the single-GPU tune precompute also materialises the
    model's validated WikiText-2 FP reference cache before LR arms fan out.
    """

    if target_phase not in {"tune", "final"}:
        raise PlanError(
            f"static precompute target must be tune or final, got {target_phase!r}"
        )
    if overrides and set(overrides) != {"global_loss_bsz"}:
        raise PlanError(
            "realq_static accepts only a global_loss_bsz OOM override"
        )
    precompute_grad_lr = (
        _selected_grad_lr(plan, model, "3W16A")
        if target_phase == "final"
        else 0.0
    )
    rendered = render_realq(
        plan,
        phase=target_phase,
        model=model,
        setting_name="3W16A",
        grad_lr=precompute_grad_lr,
        cuda_devices=cuda_devices,
        overrides=overrides,
    )
    argv = list(rendered.argv)
    _set_option(argv, "--lm_eval", "false")
    _set_option(argv, "--require_static_cache_hit", "false")
    _set_option(argv, "--require_reference_cache_hit", "false")
    _set_option(argv, "--log_column_block_loss", "false")
    argv.extend(["--exit_after_precompute", "true"])

    nominal_global_loss_bsz = int(
        plan["tuning" if target_phase == "tune" else "final"][
            "global_loss_bsz"
        ]
    )
    effective_global_loss_bsz = int(
        (overrides or {}).get(
            "global_loss_bsz",
            nominal_global_loss_bsz,
        )
    )
    override_suffix = (
        f"_global_loss_bsz{effective_global_loss_bsz}"
        if effective_global_loss_bsz != nominal_global_loss_bsz
        else ""
    )
    run_id = (
        f"precompute_realq_static_{target_phase}_{model}{override_suffix}"
    )
    output_dir = (
        Path(plan["output_root"])
        / model
        / "realq_static"
        / target_phase
    )
    _set_option(argv, "--output_dir", str(output_dir))
    _set_option(argv, "--exp", run_id)
    return RenderedCommand(
        env=rendered.env,
        argv=argv,
        run_id=run_id,
        output_dir=output_dir / run_id,
    )


def render_guided_saliency(
    plan: dict[str, Any],
    *,
    model: str,
    cuda_devices: str,
) -> RenderedCommand:
    """Render the one-per-model saliency precompute required by GuidedGPTQ."""
    _require_resolution(plan, "final_nsamples_policy")
    fixed = plan["fixed_numerics"]
    final_cfg = plan["final"]
    model_path = _model_path(plan, model)
    visible = _cuda_indices(cuda_devices)
    run_id = f"precompute_guided_saliency_{model}"
    output_dir = (
        Path(plan["output_root"])
        / model
        / "guided_saliency"
        / "precompute"
    )
    argv = [
        _venv_python(plan),
        "save_grads.py",
        "--model",
        model_path,
        "--output_dir",
        str(output_dir),
        "--cache_dir",
        str(plan["legacy_cache_root"]),
        "--exp",
        run_id,
        "--mode",
        "gradients",
        "--dataset",
        fixed["dataset"],
        "--nsamples",
        str(final_cfg["nsamples"]),
        "--seq_len",
        str(final_cfg["seq_len"]),
        "--eval_seq_len",
        str(fixed["eval_seq_len"]),
        "--seed",
        str(fixed["seed"]),
        "--rotation_seed",
        str(fixed["rotation_seed"]),
        "--refresh_seed",
        str(fixed["refresh_seed"]),
        "--num_groups",
        str(fixed["num_groups"]),
        "--rotate",
    ]
    return RenderedCommand(
        env={
            **plan.get("runtime_environment", {}),
            "CUDA_VISIBLE_DEVICES": cuda_devices,
        },
        argv=argv,
        run_id=run_id,
        output_dir=output_dir / Path(model_path).name / run_id,
    )


def _parse_overrides(items: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise PlanError(f"override must be KEY=INTEGER, got {item!r}")
        try:
            result[key] = int(value)
        except ValueError as exc:
            raise PlanError(f"override value must be an integer: {item!r}") from exc
    return result


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--strict", action="store_true")

    matrix = sub.add_parser("matrix")
    matrix.add_argument("--json", action="store_true")

    render = sub.add_parser("render")
    render.add_argument(
        "--phase", choices=("precompute", "tune", "final"), required=True
    )
    render.add_argument(
        "--method",
        choices=(
            "realq",
            "realq_static",
            "gptaq",
            "guided_gptq",
            "guided_saliency",
            "bf16",
        ),
        required=True,
    )
    render.add_argument("--target-phase", choices=("tune", "final"))
    render.add_argument("--model", choices=MODEL_ORDER, required=True)
    render.add_argument("--setting", choices=SETTING_ORDER)
    render.add_argument("--grad-lr", type=float)
    render.add_argument("--cuda-devices", required=True)
    render.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=INTEGER",
        help="batch/memory-only override; may be repeated",
    )
    render.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.command == "validate":
            unresolved = unresolved_items(plan)
            payload = {
                "valid_structure": True,
                "matrix_rows": len(matrix_rows(plan)),
                "unresolved": unresolved,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 1 if args.strict and unresolved else 0

        if args.command == "matrix":
            rows = matrix_rows(plan)
            if args.json:
                print(json.dumps(rows, indent=2, ensure_ascii=False))
            else:
                for row in rows:
                    print(
                        f"{row['model']:<14} {row['setting']:<7} "
                        f"{row['method']:<12} {row['aware']}"
                    )
                print(f"TOTAL={len(rows)}")
            return 0

        if args.command == "render":
            overrides = _parse_overrides(args.override)
            if args.method == "realq_static":
                if args.phase != "precompute":
                    raise PlanError(
                        "realq_static is rendered only in the precompute phase"
                    )
                if args.target_phase is None:
                    raise PlanError("realq_static requires --target-phase")
                if args.setting is not None or args.grad_lr is not None:
                    raise PlanError(
                        "realq_static does not accept --setting or --grad-lr"
                    )
                rendered = render_realq_static_precompute(
                    plan,
                    target_phase=args.target_phase,
                    model=args.model,
                    cuda_devices=args.cuda_devices,
                    overrides=overrides,
                )
            elif args.method == "guided_saliency":
                if args.phase != "precompute":
                    raise PlanError(
                        "guided_saliency is rendered only in the precompute phase"
                    )
                if args.setting is not None or args.grad_lr is not None:
                    raise PlanError(
                        "guided_saliency does not accept --setting or --grad-lr"
                    )
                if overrides:
                    raise PlanError("guided_saliency does not accept overrides")
                rendered = render_guided_saliency(
                    plan,
                    model=args.model,
                    cuda_devices=args.cuda_devices,
                )
            elif args.method == "realq":
                if args.target_phase is not None:
                    raise PlanError("--target-phase is only for realq_static")
                if args.setting is None:
                    raise PlanError("REAL-Q requires --setting")
                if args.grad_lr is None:
                    raise PlanError("REAL-Q requires --grad-lr")
                rendered = render_realq(
                    plan,
                    phase=args.phase,
                    model=args.model,
                    setting_name=args.setting,
                    grad_lr=args.grad_lr,
                    cuda_devices=args.cuda_devices,
                    overrides=overrides,
                )
            else:
                if args.target_phase is not None:
                    raise PlanError("--target-phase is only for realq_static")
                if args.phase != "final":
                    raise PlanError("baselines are rendered only in the final phase")
                rendered = render_baseline(
                    plan,
                    method=args.method,
                    model=args.model,
                    setting_name=args.setting,
                    cuda_devices=args.cuda_devices,
                    overrides=overrides,
                )
            if args.json:
                print(
                    json.dumps(
                        {
                            "run_id": rendered.run_id,
                            "output_dir": str(rendered.output_dir),
                            "env": rendered.env,
                            "argv": rendered.argv,
                            "shell": rendered.shell(),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print(rendered.shell())
            return 0
    except PlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
