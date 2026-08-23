"""Strict model loaders used by the unified additional-method evaluator."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from . import common


def _bootstrap_yaqa() -> None:
    import sys

    paths = (
        common.REPO_ROOT / "YAQA_wclip",
        common.REPO_ROOT / "YAQA_wclip/hessian_llama",
    )
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(1, value)


def _validate_base_model(spec: common.EvalSpec, model_info: Mapping[str, Any]) -> Path:
    path = Path(model_info["path"]).resolve()
    if not path.is_dir():
        raise common.EvaluationError(f"base model is missing: {path}")
    config = path / "config.json"
    if common.sha256_file(config) != model_info["config_sha256"]:
        raise common.EvaluationError(f"base model config changed: {config}")
    if spec.model not in common.MODEL_ORDER:
        raise common.EvaluationError(f"unsupported model: {spec.model}")
    return path


def _validate_terminal_binding(
    spec: common.EvalSpec,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    terminal_path = Path(artifact["terminal"])
    terminal = common.read_object(terminal_path)
    if common.sha256_file(terminal_path) != artifact["terminal_sha256"]:
        raise common.EvaluationError(f"terminal receipt changed: {terminal_path}")
    run = terminal.get("run") or terminal.get("stage")
    if not isinstance(run, dict):
        raise common.EvaluationError("terminal receipt has no run/stage contract")
    actual_id = run.get("run_id", run.get("stage_id"))
    expected = {
        "id": spec.run_id,
        "model": spec.model,
        "setting": spec.setting,
        "w_bits": spec.w_bits,
    }
    actual = {
        "id": actual_id,
        "model": run.get("model"),
        "setting": run.get("setting"),
        "w_bits": run.get("w_bits"),
    }
    if actual != expected:
        raise common.EvaluationError(
            f"terminal receipt/spec mismatch: {actual!r} != {expected!r}"
        )
    return terminal


def load_efficientqat(
    spec: common.EvalSpec,
    artifact: Mapping[str, Any],
    model_info: Mapping[str, Any],
):
    import torch
    from transformers import AutoModelForCausalLM

    from experiments.efficientqat_compare.materialize import (
        load_materialization_manifest,
        validate_materialized_model,
    )
    from utils import model_utils
    from realq import attention

    base_model = _validate_base_model(spec, model_info)
    _validate_terminal_binding(spec, artifact)
    hf_dir = Path(artifact["path"]).resolve()
    manifest_path = Path(artifact["validation"]).resolve()
    if manifest_path.parent != hf_dir:
        raise common.EvaluationError("EfficientQAT manifest is outside its HF artifact")
    if common.sha256_file(manifest_path) != artifact["validation_sha256"]:
        raise common.EvaluationError("EfficientQAT materialization manifest changed")
    manifest = load_materialization_manifest(manifest_path)
    provenance = manifest.get("provenance", {})
    expected_provenance = {
        "run_id": spec.run_id,
        "base_model": str(base_model),
        "weight_bits": spec.w_bits,
        "group_size": 128,
        "source_quantizer": "efficientqat_symmetric_signed_grid_fixed_zero",
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise common.EvaluationError(
            f"EfficientQAT provenance mismatch: {mismatches!r}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    validate_materialized_model(
        model,
        manifest,
        expected_count=int(model.config.num_hidden_layers) * 7,
    )
    attention.configure_attention_backend(model, "sdpa")
    return model_utils.ModelAnalyzer(
        model,
        2048,
        tokenizer_source=str(base_model),
        skip_state_dict=True,
    )


def load_yaqa_wclip(
    spec: common.EvalSpec,
    artifact: Mapping[str, Any],
    model_info: Mapping[str, Any],
):
    import torch

    base_model = _validate_base_model(spec, model_info)
    _validate_terminal_binding(spec, artifact)
    _bootstrap_yaqa()
    from experiments.turboboa_yaqa_qwen3_rerun.eval_yaqa_kl_ppl import (
        _validate_artifact,
    )
    from lib.utils.unsafe_import import model_from_hf_path
    from realq import attention
    from utils import model_utils

    hf_dir = Path(artifact["path"]).resolve()
    validation = Path(artifact["validation"]).resolve()
    validation_sha, report = _validate_artifact(
        model=base_model,
        setting=spec.setting,
        hf_dir=hf_dir,
        validation_path=validation,
        expected_validation_sha256=artifact["validation_sha256"],
    )
    if validation_sha != artifact["validation_sha256"]:
        raise common.EvaluationError("YAQA validation hash changed")
    expected = {
        "weight_bits": spec.w_bits,
        "weight_groupsize": 128,
        "runtime_weight_representation": "dense_fake_quant",
        "packed": False,
        "lowbit_kernel": False,
    }
    mismatches = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise common.EvaluationError(f"YAQA validation mismatch: {mismatches!r}")
    layer_count = int(report.get("layer_count", -1))
    if spec.setting == "W4A4KV4":
        expected_runtime = {
            "mode": "deployment",
            "decoder_layers": layer_count,
            "activation_input_sites": 7 * layer_count,
            "value_output_sites": layer_count,
            "post_rope_k_sites": layer_count,
            "a_bits": 4,
            "k_bits": 4,
            "v_bits": 4,
            "groupsize": -1,
            "symmetric": True,
            "clip_ratio": 0.9,
            "query_quantized": False,
            "extra_qk_hadamard": False,
        }
    else:
        expected_runtime = {"mode": "n/a"}
    if report.get("runtime_akv") != expected_runtime:
        raise common.EvaluationError("YAQA A/K/V deployment receipt mismatch")
    model, _ = model_from_hf_path(str(hf_dir), device_map="cpu")
    model.eval()
    attention.configure_attention_backend(model, "sdpa")
    analyzer = model_utils.ModelAnalyzer(
        model,
        2048,
        tokenizer_source=str(base_model),
        skip_state_dict=True,
    )
    del model
    gc.collect()
    return analyzer


def _checkpoint_config(spec: common.EvalSpec, base_model: Path) -> SimpleNamespace:
    aware = spec.setting == "W4A4KV4"
    clip = 0.9 if aware else 1.0
    return SimpleNamespace(
        model=str(base_model),
        model_name=base_model.name,
        seq_len=2048,
        eval_seq_len=2048,
        w_bits=spec.w_bits,
        w_groupsize=128,
        w_asym=False,
        w_clip=True,
        rotate=True,
        rotation_seed=0,
        optimized_rotation_path=None,
        a_bits=spec.a_bits,
        a_groupsize=-1,
        a_asym=False,
        a_clip_ratio=clip,
        k_bits=spec.k_bits,
        k_groupsize=-1,
        k_asym=False,
        k_clip_ratio=clip,
        v_bits=spec.v_bits,
        v_groupsize=-1,
        v_asym=False,
        v_clip_ratio=clip,
        act_quant_aware_gptq=aware,
        k_cache_quant_aware_gptq=aware,
    )


def load_turboboa(
    spec: common.EvalSpec,
    artifact: Mapping[str, Any],
    model_info: Mapping[str, Any],
    *,
    analyzer=None,
):
    from realq import akv, attention, pipeline
    from utils import checkpoint_utils, model_utils

    base_model = _validate_base_model(spec, model_info)
    _validate_terminal_binding(spec, artifact)
    checkpoint_path = Path(artifact["path"]).resolve()
    checkpoint = checkpoint_utils.load_quantized_checkpoint(checkpoint_path)
    cfg = _checkpoint_config(spec, base_model)
    checkpoint_utils.apply_runtime_manifest(cfg, checkpoint)
    checkpoint_utils.validate_artifact_identity(cfg, checkpoint)
    if analyzer is None:
        analyzer = model_utils.ModelAnalyzer(
            str(base_model),
            2048,
            tokenizer_source=str(base_model),
            skip_state_dict=True,
        )
    attention.configure_attention_backend(analyzer.model, "sdpa")
    checkpoint_utils.validate_artifact_identity(
        cfg,
        checkpoint,
        model=analyzer.model,
        tokenizer=analyzer.tokenizer,
    )
    pipeline._prepare_loaded_runtime_wrappers(cfg, analyzer)
    akv.install_actquant_wrappers(analyzer)
    checkpoint_utils.load_model_state(analyzer.model, checkpoint)
    akv.setup_aware_pre_quant(analyzer, cfg)
    akv.setup_unaware_post_quant(analyzer, cfg)
    analyzer.model.eval()

    runtime = checkpoint.get("runtime_quantization") or {}
    aware = spec.setting == "W4A4KV4"
    clip = 0.9 if aware else 1.0
    expected_runtime = {
        "rotate": True,
        "a_bits": spec.a_bits,
        "a_groupsize": -1,
        "k_bits": spec.k_bits,
        "k_groupsize": -1,
        "v_bits": spec.v_bits,
        "v_groupsize": -1,
        "a_asym": False,
        "k_asym": False,
        "v_asym": False,
        "a_clip_ratio": clip,
        "k_clip_ratio": clip,
        "v_clip_ratio": clip,
        "act_quant_aware_gptq": aware,
        "k_cache_quant_aware_gptq": aware,
    }
    mismatch = {
        key: (runtime.get(key), value)
        for key, value in expected_runtime.items()
        if runtime.get(key) != value
    }
    if mismatch:
        raise common.EvaluationError(
            f"TurboBOA runtime manifest mismatch: {mismatch!r}"
        )
    weight_runtime = checkpoint.get("weight_quantization") or {}
    expected_weight = {
        "w_bits": spec.w_bits,
        "w_groupsize": 128,
        "w_asym": False,
        "w_clip": True,
        "w_method": "turboboa_rmsnorm_mean_jacobian_kfac",
    }
    weight_mismatch = {
        key: (weight_runtime.get(key), value)
        for key, value in expected_weight.items()
        if weight_runtime.get(key) != value
    }
    if weight_mismatch:
        raise common.EvaluationError(
            f"TurboBOA weight manifest mismatch: {weight_mismatch!r}"
        )
    return analyzer


def load_quantized(
    spec: common.EvalSpec,
    artifact: Mapping[str, Any],
    model_info: Mapping[str, Any],
    *,
    reference_analyzer=None,
):
    if spec.method == "efficientqat":
        return load_efficientqat(spec, artifact, model_info)
    if spec.method == "yaqa_wclip":
        return load_yaqa_wclip(spec, artifact, model_info)
    if spec.method == "turboboa":
        return load_turboboa(
            spec,
            artifact,
            model_info,
            analyzer=reference_analyzer,
        )
    raise common.EvaluationError(f"unsupported method: {spec.method}")
