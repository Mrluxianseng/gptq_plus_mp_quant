#!/usr/bin/env python3
"""Run one frozen EfficientQAT pipeline from the controlled comparison plan."""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EFFICIENTQAT_ROOT = REPO_ROOT / "EfficientQAT"
EVALUATOR_SHA256 = (
    "e486fcd2d7fe31b78eeb8a407e8667b6f4774c7f2395e728b4654f7c41e1cbff"
)


class RunError(RuntimeError):
    pass


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _bootstrap_efficientqat() -> None:
    workspace_utils = importlib.import_module("utils")
    expected_utils = (REPO_ROOT / "utils").resolve()
    actual_utils_paths = {
        Path(value).resolve()
        for value in getattr(workspace_utils, "__path__", ())
    }
    if actual_utils_paths != {expected_utils}:
        raise RunError(
            "workspace utils package was shadowed before EfficientQAT "
            f"bootstrap: {sorted(map(str, actual_utils_paths))} "
            f"!= {[str(expected_utils)]}"
        )

    value = str(EFFICIENTQAT_ROOT)
    if value not in sys.path:
        # Keep the workspace root ahead of EfficientQAT so the comparison's
        # shared ``utils`` package cannot be shadowed, while placing the
        # upstream checkout ahead of site-packages for ``quantize`` and
        # ``datautils_block``.
        insertion_index = 1
        for index, entry in enumerate(sys.path):
            try:
                if Path(entry or os.getcwd()).resolve() == REPO_ROOT:
                    insertion_index = index + 1
                    break
            except (OSError, RuntimeError):
                continue
        sys.path.insert(insertion_index, value)


def _configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"efficientqat.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    log_path = run_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _load_contract(args: argparse.Namespace):
    from . import launcher

    plan_path = Path(args.plan_file).resolve()
    actual_plan_sha = launcher.plan_sha256(plan_path)
    if actual_plan_sha != args.expected_plan_sha256:
        raise RunError(
            "runner plan SHA256 differs from the launcher-pinned value: "
            f"{actual_plan_sha} != {args.expected_plan_sha256}"
        )
    plan = launcher.load_plan(plan_path)
    launcher.verify_artifacts(plan)
    actual_lm_eval = importlib.metadata.version("lm_eval")
    expected_lm_eval = plan["unified_eval_contract"]["lm_eval_version"]
    if actual_lm_eval != expected_lm_eval:
        raise RunError(
            f"lm-eval version mismatch: {actual_lm_eval} "
            f"!= {expected_lm_eval}"
        )
    matching = [run for run in plan["runs"] if run["run_id"] == args.run_id]
    if len(matching) != 1:
        raise RunError(f"run_id must resolve exactly once: {args.run_id}")
    run = matching[0]
    if socket.gethostname() != plan["runtime"]["canoe_pod"]:
        raise RunError(
            "runner is on the wrong pod: "
            f"{socket.gethostname()} != {plan['runtime']['canoe_pod']}"
        )
    if int(args.physical_train_gpu) != int(run["training_gpu"]):
        raise RunError("physical training GPU does not match the plan")
    if int(args.physical_eval_gpu) != int(run["evaluation_gpu"]):
        raise RunError("physical evaluation GPU does not match the plan")
    expected_visible = (
        f"{run['training_gpu']},{run['evaluation_gpu']}"
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RunError(
            "CUDA_VISIBLE_DEVICES mismatch: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r} != {expected_visible!r}"
        )
    run_dir = (
        Path(plan["runtime"]["output_root"]) / run["output_subdir"]
    )
    if not run_dir.is_dir():
        raise RunError(f"launcher-created run directory is missing: {run_dir}")
    return plan_path, plan, run, run_dir, actual_plan_sha


def _set_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _configure_determinism(plan: dict[str, Any], seed: int) -> dict[str, Any]:
    """Apply the campaign's numerical reproducibility contract.

    Older EfficientQAT campaigns intentionally preserved the released
    ``cudnn.benchmark=True`` behavior.  New controlled campaigns can opt into
    this stricter path without silently changing those archived plans.
    """

    import torch

    contract = plan.get("runtime_contract", {}).get("determinism", {})
    enabled = bool(contract.get("enabled", False))
    _set_seeds(seed)
    if enabled:
        required_workspace = contract.get(
            "cublas_workspace_config", ":4096:8"
        )
        actual_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if actual_workspace != required_workspace:
            raise RunError(
                "CUBLAS_WORKSPACE_CONFIG must be installed before importing "
                f"torch: {actual_workspace!r} != {required_workspace!r}"
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.benchmark = True
    return {
        "enabled": enabled,
        "seed": seed,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _record_stage(
    run_dir: Path,
    stage: str,
    *,
    status: str,
    started_at: str,
    wall_seconds: float,
    details: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "stage": stage,
        "status": status,
        "started_at": started_at,
        "ended_at": _now(),
        "wall_seconds": float(wall_seconds),
        **details,
    }
    _write_json_atomic(run_dir / stage / "stage.json", value)
    return value


def _assert_evaluator_identity() -> str:
    path = REPO_ROOT / "utils" / "eval_utils.py"
    actual = _sha256_file(path)
    if actual != EVALUATOR_SHA256:
        raise RunError(
            "shared evaluator SHA256 changed: "
            f"{actual} != {EVALUATOR_SHA256}"
        )
    return actual


def _assert_imported_evaluator_identity(module: Any) -> str:
    expected_path = (REPO_ROOT / "utils" / "eval_utils.py").resolve()
    actual_path = Path(module.__file__).resolve()
    if actual_path != expected_path:
        raise RunError(
            "imported evaluator path changed: "
            f"{actual_path} != {expected_path}"
        )
    return _assert_evaluator_identity()


def _load_exact_tokens(plan: dict[str, Any]):
    from .data import load_exact_token_cache

    contract = plan["calibration_contract"]["token_artifact"]
    path = Path(contract["path"])
    actual = _sha256_file(path)
    if actual != contract["sha256"]:
        raise RunError(
            f"calibration artifact SHA256 mismatch: {actual}"
        )
    return load_exact_token_cache(
        path,
        expected_samples=plan["calibration_contract"]["num_samples"],
        expected_seq_len=plan["calibration_contract"]["seq_len"],
    )


def _count_modules(model, module_type) -> int:
    return sum(
        1 for module in model.modules() if isinstance(module, module_type)
    )


def run_block_ap(
    plan: dict[str, Any],
    run: dict[str, Any],
    run_dir: Path,
    train_device: str,
    logger: logging.Logger,
) -> tuple[Path, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    _bootstrap_efficientqat()
    from quantize import block_ap as block_ap_module
    from quantize import int_linear_real as int_linear_real_module

    block_ap = block_ap_module.block_ap
    QuantLinear = int_linear_real_module.QuantLinear
    for module in (block_ap_module, int_linear_real_module):
        if EFFICIENTQAT_ROOT.resolve() not in Path(module.__file__).resolve().parents:
            raise RunError(
                f"EfficientQAT module was shadowed: {module.__file__}"
            )

    from .compat import (
        assert_model_vocab_unchanged,
        load_fast_eos_tokenizer,
    )
    from .data import build_block_ap_trainloader

    stage = "block_ap"
    stage_dir = run_dir / stage
    packed_dir = stage_dir / "packed_model"
    stage_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    started = time.monotonic()
    torch.cuda.set_device(torch.device(train_device))
    determinism = _configure_determinism(plan, 2)

    model_contract = plan["models"][run["model"]]
    model_path = Path(model_contract["path"])
    expected_vocab = int(
        json.loads((model_path / "config.json").read_text())["vocab_size"]
    )
    expected_tokenizer_vocab = int(
        model_contract.get("tokenizer_vocab_size", expected_vocab)
    )
    tokenizer = load_fast_eos_tokenizer(
        model_path,
        expected_vocab_size=expected_tokenizer_vocab,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.config.torch_dtype = torch.float16
    assert_model_vocab_unchanged(
        model,
        tokenizer,
        expected_vocab_size=expected_vocab,
        expected_tokenizer_vocab_size=expected_tokenizer_vocab,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    tokens = _load_exact_tokens(plan)
    trainloader = build_block_ap_trainloader(tokens)
    method = plan["method_contract"]
    block_cfg = method["block_ap"]
    minimum_lr_ratio = float(block_cfg["minimum_lr_ratio"])
    if not 0.0 < minimum_lr_ratio <= 1.0:
        raise RunError(
            "Block-AP minimum_lr_ratio must be in (0, 1], got "
            f"{minimum_lr_ratio}"
        )
    weight_quantizer = method.get("weight_quantizer", {})
    weight_symmetric = weight_quantizer.get("symmetric", False)
    if type(weight_symmetric) is not bool:
        raise RunError("method_contract.weight_quantizer.symmetric must be bool")
    block_args = SimpleNamespace(
        off_load_to_disk=False,
        cache_dir=str(stage_dir / "cache"),
        train_size=len(tokens),
        val_size=0,
        training_seqlen=plan["calibration_contract"]["seq_len"],
        batch_size=block_cfg["batch_size"],
        epochs=block_cfg["epochs"],
        quant_lr=block_cfg["quantizer_lr"],
        weight_lr=block_cfg["weight_lr_by_weight_bits"][
            str(run["w_bits"])
        ],
        min_lr_factor=1.0 / minimum_lr_ratio,
        wd=block_cfg["weight_decay"],
        early_stop=0,
        wbits=run["w_bits"],
        group_size=method["weight_group_size"],
        weight_symmetric=weight_symmetric,
        real_quant=True,
        max_blocks=int(block_cfg.get("max_blocks", -1)),
    )
    logger.info("Block-AP frozen args: %s", vars(block_args))
    block_ap(model, block_args, trainloader, [], logger=logger)
    expected_linears = len(model.model.layers) * 7
    packed_count = _count_modules(model, QuantLinear)
    if packed_count != expected_linears:
        raise RunError(
            f"packed linear count mismatch: {packed_count} != {expected_linears}"
        )
    assert_model_vocab_unchanged(
        model,
        tokenizer,
        expected_vocab_size=expected_vocab,
        expected_tokenizer_vocab_size=expected_tokenizer_vocab,
    )
    packed_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        packed_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(packed_dir)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    details = {
        "gpu_count": 1,
        "gpu_seconds": elapsed,
        "precision": "fp16_amp",
        "seed": 2,
        "validation_size": 0,
        "packed_linear_count": packed_count,
        "expected_optimizer_steps_per_block": (
            block_args.train_size
            // block_args.batch_size
            * block_args.epochs
        ),
        "output": str(packed_dir),
        "weight_bits": run["w_bits"],
        "group_size": method["weight_group_size"],
        "weight_symmetric": weight_symmetric,
        "minimum_lr_ratio": minimum_lr_ratio,
        "determinism": determinism,
    }
    return packed_dir, _record_stage(
        run_dir,
        stage,
        status="succeeded",
        started_at=started_at,
        wall_seconds=elapsed,
        details=details,
    )


def _prepare_e2e_model(
    packed_dir: Path,
    *,
    wbits: int,
    group_size: int,
    train_device: str,
    expected_vocab: int,
    expected_tokenizer_vocab: int,
):
    import torch

    _bootstrap_efficientqat()
    from quantize.int_linear_real import load_quantized_model, QuantLinear

    from .compat import assert_model_vocab_unchanged, ensure_fast_eos_tokenizer

    model, tokenizer = load_quantized_model(
        str(packed_dir),
        wbits,
        group_size,
        device_map={"": "cpu"},
    )
    tokenizer = ensure_fast_eos_tokenizer(
        tokenizer, expected_vocab_size=expected_tokenizer_vocab
    )
    assert_model_vocab_unchanged(
        model,
        tokenizer,
        expected_vocab_size=expected_vocab,
        expected_tokenizer_vocab_size=expected_tokenizer_vocab,
    )
    model.to(torch.device(train_device))
    model.train()
    model.config.use_cache = False
    model.config.torch_dtype = torch.bfloat16

    for parameter in model.parameters():
        parameter.requires_grad = False
        if parameter.dtype in (torch.float16, torch.bfloat16):
            parameter.data = parameter.data.float()
    for name, module in model.named_modules():
        if (
            "norm" in name
            or "lm_head" in name
            or "embed_tokens" in name
        ) and hasattr(module, "weight"):
            module.to(torch.bfloat16)
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.scales.requires_grad = True
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    invalid = [name for name, _ in trainable if not name.endswith(".scales")]
    if not trainable or invalid:
        raise RunError(
            f"E2E trainable-parameter contract failed; invalid={invalid}"
        )
    return model, tokenizer, trainable


def run_e2e_qp(
    plan: dict[str, Any],
    run: dict[str, Any],
    run_dir: Path,
    train_device: str,
    block_packed_dir: Path,
    logger: logging.Logger,
) -> tuple[Path, dict[str, Any]]:
    import torch
    from transformers import Trainer, TrainingArguments, default_data_collator

    from .compat import assert_model_vocab_unchanged
    from .data import build_e2e_fixed_dataset

    stage = "e2e_qp"
    stage_dir = run_dir / stage
    packed_dir = stage_dir / "packed_model"
    stage_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    started = time.monotonic()
    torch.cuda.set_device(torch.device(train_device))
    determinism = _configure_determinism(plan, 42)

    model_contract = plan["models"][run["model"]]
    model_path = Path(model_contract["path"])
    expected_vocab = int(
        json.loads((model_path / "config.json").read_text())["vocab_size"]
    )
    expected_tokenizer_vocab = int(
        model_contract.get("tokenizer_vocab_size", expected_vocab)
    )
    group_size = int(plan["method_contract"]["weight_group_size"])
    model, tokenizer, trainable = _prepare_e2e_model(
        block_packed_dir,
        wbits=run["w_bits"],
        group_size=group_size,
        train_device=train_device,
        expected_vocab=expected_vocab,
        expected_tokenizer_vocab=expected_tokenizer_vocab,
    )
    tokens = _load_exact_tokens(plan)
    dataset = build_e2e_fixed_dataset(tokens).shuffle(seed=0)
    cfg = plan["method_contract"]["e2e_qp"]
    learning_rate = cfg["scale_lr_by_weight_bits"][str(run["w_bits"])]
    training_args = TrainingArguments(
        output_dir=str(stage_dir / "trainer"),
        overwrite_output_dir=False,
        per_device_train_batch_size=cfg["micro_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["epochs"],
        learning_rate=learning_rate,
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"],
        bf16=True,
        fp16=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )
    # The process exposes its reserved evaluation GPU for the later stage.
    # Prevent Trainer from wrapping this single-run model in DataParallel.
    training_args._n_gpu = 1
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for _, parameter in trainable],
                "lr": learning_rate,
                "weight_decay": 0.0,
            }
        ]
    )
    positive_scale_contract = cfg.get("positive_scale_projection", {})
    positive_scale_enabled = bool(
        positive_scale_contract.get("enabled", False)
    )
    minimum_scale = float(positive_scale_contract.get("minimum", 1e-4))
    projection_stats = {
        "enabled": positive_scale_enabled,
        "minimum": minimum_scale,
        "optimizer_steps_with_projection": 0,
        "projected_element_count": 0,
    }
    projection_handle = None
    if positive_scale_enabled:
        if not minimum_scale > 0:
            raise RunError("positive scale projection minimum must be > 0")

        def _project_positive_scales(_optimizer, _args, _kwargs):
            projected = 0
            with torch.no_grad():
                for _, parameter in trainable:
                    projected += int(
                        torch.count_nonzero(parameter < minimum_scale).item()
                    )
                    parameter.clamp_(min=minimum_scale)
            projection_stats["optimizer_steps_with_projection"] += 1
            projection_stats["projected_element_count"] += projected

        projection_handle = optimizer.register_step_post_hook(
            _project_positive_scales
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )
    try:
        train_result = trainer.train()
    finally:
        if projection_handle is not None:
            projection_handle.remove()
    expected_steps = (
        len(dataset)
        // (
            cfg["micro_batch_size"]
            * cfg["gradient_accumulation_steps"]
        )
        * cfg["epochs"]
    )
    if trainer.state.global_step != expected_steps:
        raise RunError(
            "E2E optimizer-step mismatch: "
            f"{trainer.state.global_step} != {expected_steps}"
        )
    scale_minimum_observed = min(
        float(parameter.detach().amin().cpu()) for _, parameter in trainable
    )
    # ``minimum_scale`` is a Python float, while the learned scales are FP32.
    # ``clamp_(min=1e-4)`` therefore stores the nearest representable FP32
    # value (9.999999747e-05), which is microscopically below the Python float
    # when converted back.  Gate against the exact per-parameter representable
    # threshold so a successful projection is not rejected by dtype rounding.
    representable_minimum_scale = min(
        float(
            torch.tensor(
                minimum_scale,
                dtype=parameter.dtype,
                device="cpu",
            ).item()
        )
        for _, parameter in trainable
    )
    scale_finite = all(
        bool(torch.isfinite(parameter.detach()).all().cpu())
        for _, parameter in trainable
    )
    if not scale_finite:
        raise RunError("E2E-QP produced non-finite learned scales")
    if (
        positive_scale_enabled
        and scale_minimum_observed < representable_minimum_scale
    ):
        raise RunError(
            "positive scale projection contract failed: "
            f"{scale_minimum_observed} < {representable_minimum_scale} "
            f"(requested {minimum_scale})"
        )
    assert_model_vocab_unchanged(
        model,
        tokenizer,
        expected_vocab_size=expected_vocab,
        expected_tokenizer_vocab_size=expected_tokenizer_vocab,
    )
    model.config.use_cache = True
    model.cpu()
    packed_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        packed_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(packed_dir)
    torch.cuda.empty_cache()
    elapsed = time.monotonic() - started
    train_metrics = {
        key: float(value)
        for key, value in train_result.metrics.items()
        if isinstance(value, (int, float))
    }
    trainable_parameter_count = sum(
        parameter.numel() for _, parameter in trainable
    )
    global_step = trainer.state.global_step
    details = {
        "gpu_count": 1,
        "gpu_seconds": elapsed,
        "precision": "bf16",
        "optimizer": "torch.optim.AdamW",
        "optimizer_compatibility_substitution": True,
        "seed": 42,
        "dataset_pre_shuffle_seed": 0,
        "optimizer_steps": global_step,
        "trainable_parameter_count": trainable_parameter_count,
        "train_metrics": train_metrics,
        "output": str(packed_dir),
        "determinism": determinism,
        "positive_scale_projection": projection_stats,
        "representable_minimum_scale": representable_minimum_scale,
        "minimum_learned_scale": scale_minimum_observed,
    }
    del trainer, optimizer, model, trainable, train_result
    gc.collect()
    torch.cuda.empty_cache()
    return packed_dir, _record_stage(
        run_dir,
        stage,
        status="succeeded",
        started_at=started_at,
        wall_seconds=elapsed,
        details=details,
    )


def run_materialize(
    plan: dict[str, Any],
    run: dict[str, Any],
    run_dir: Path,
    packed_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    _bootstrap_efficientqat()
    from quantize.int_linear_real import load_quantized_model

    from .compat import (
        assert_model_vocab_unchanged,
        ensure_fast_eos_tokenizer,
    )
    from .materialize import (
        SOURCE_QUANTIZER,
        SYMMETRIC_SOURCE_QUANTIZER,
        materialize_efficientqat_model,
        validate_materialized_model,
        write_materialization_manifest,
    )

    stage = "materialize"
    stage_dir = run_dir / stage
    output_dir = stage_dir / "hf_model"
    stage_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    started = time.monotonic()
    group_size = int(plan["method_contract"]["weight_group_size"])
    model, tokenizer = load_quantized_model(
        str(packed_dir),
        run["w_bits"],
        group_size,
        device_map={"": "cpu"},
    )
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RunError("materialization input model is not entirely on CPU")
    model_contract = plan["models"][run["model"]]
    model_path = Path(model_contract["path"])
    expected_vocab = int(
        json.loads((model_path / "config.json").read_text())["vocab_size"]
    )
    expected_tokenizer_vocab = int(
        model_contract.get("tokenizer_vocab_size", expected_vocab)
    )
    tokenizer = ensure_fast_eos_tokenizer(
        tokenizer, expected_vocab_size=expected_tokenizer_vocab
    )
    assert_model_vocab_unchanged(
        model,
        tokenizer,
        expected_vocab_size=expected_vocab,
        expected_tokenizer_vocab_size=expected_tokenizer_vocab,
    )
    layer_count = len(model.model.layers)
    weight_symmetric = plan["method_contract"].get(
        "weight_quantizer", {}
    ).get("symmetric", False)
    if type(weight_symmetric) is not bool:
        raise RunError("method_contract.weight_quantizer.symmetric must be bool")
    source_quantizer = (
        SYMMETRIC_SOURCE_QUANTIZER
        if weight_symmetric
        else SOURCE_QUANTIZER
    )
    expected_names = []
    for index in range(layer_count):
        prefix = f"model.layers.{index}"
        expected_names.extend(
            [
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.up_proj",
                f"{prefix}.mlp.down_proj",
            ]
        )
    report = materialize_efficientqat_model(
        model,
        expected_count=layer_count * 7,
        expected_bits=run["w_bits"],
        expected_group_size=group_size,
        expected_module_names=expected_names,
        provenance={
            "run_id": run["run_id"],
            "base_model": str(model_path),
            "weight_bits": run["w_bits"],
            "group_size": group_size,
            "source_quantizer": source_quantizer,
            "e2e_checkpoint": str(packed_dir),
        },
        source_quantizer=source_quantizer,
    )
    model.config.torch_dtype = torch.bfloat16
    model.config.use_cache = True
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_dir)
    manifest_path = output_dir / "efficientqat_manifest.json"
    write_materialization_manifest(manifest_path, report)
    del model
    gc.collect()
    fresh = AutoModelForCausalLM.from_pretrained(
        output_dir,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    validate_materialized_model(
        fresh,
        report,
        expected_count=layer_count * 7,
    )
    assert_model_vocab_unchanged(
        fresh,
        tokenizer,
        expected_vocab_size=expected_vocab,
        expected_tokenizer_vocab_size=expected_tokenizer_vocab,
    )
    del fresh
    gc.collect()
    elapsed = time.monotonic() - started
    details = {
        "gpu_count": 0,
        "gpu_seconds": 0.0,
        "cpu_wall_seconds": elapsed,
        "output": str(output_dir),
        "manifest": str(manifest_path),
        "report": report,
    }
    return output_dir, _record_stage(
        run_dir,
        stage,
        status="succeeded",
        started_at=started_at,
        wall_seconds=elapsed,
        details=details,
    )


def _reference_args(
    plan: dict[str, Any],
    run: dict[str, Any],
) -> SimpleNamespace:
    model_path = plan["models"][run["model"]]["path"]
    shared_cache = (
        REPO_ROOT.parent
        / "experiment_data"
        / "turboboa_compare_20260725"
        / "j-8j1en3m0aq"
        / "artifacts"
        / "shared_cache"
    )
    return SimpleNamespace(
        model=model_path,
        model_name=Path(model_path).name,
        cache_dir=str(shared_cache),
        eval_seq_len=plan["unified_eval_contract"]["eval_seq_len"],
        kl_topk=plan["unified_eval_contract"]["kl_topk"],
        rotate=True,
        rotation_seed=0,
        optimized_rotation_path=None,
        require_reference_cache_hit=True,
    )


def run_evaluation(
    plan: dict[str, Any],
    run: dict[str, Any],
    run_dir: Path,
    materialized_dir: Path,
    eval_device: str,
    physical_eval_gpu: int,
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from utils import data_utils, eval_utils, model_utils

    from .akv_unaware import setup_unaware_post_quant_no_rotation

    _assert_imported_evaluator_identity(eval_utils)

    stage = "evaluation"
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    started = time.monotonic()
    lock_dir = Path(plan["runtime"]["output_root"]) / "_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"eval_gpu_{physical_eval_gpu}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        logger.info("Waiting for evaluation GPU lock: %s", lock_path)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        logger.info("Acquired evaluation GPU lock: %s", lock_path)
        torch.cuda.set_device(torch.device(eval_device))
        base_model = plan["models"][run["model"]]["path"]
        ref_args = _reference_args(plan, run)
        reference_analyzer = model_utils.ModelAnalyzer(
            base_model,
            plan["unified_eval_contract"]["eval_seq_len"],
            tokenizer_source=base_model,
            skip_state_dict=True,
        )
        test_loader = data_utils.get_loaders(
            "wikitext2",
            "test",
            reference_analyzer.tokenizer,
            ref_args.eval_seq_len,
            plan["calibration_contract"]["num_samples"],
            plan["calibration_contract"]["seed"],
        )
        reference_hidden, original_lm_head = eval_utils.get_ref_logits(
            ref_args,
            reference_analyzer,
            "wikitext2",
            test_loader,
        )
        reference_head_dtype = str(original_lm_head.weight.dtype)
        del reference_analyzer
        gc.collect()
        torch.cuda.empty_cache()

        analyzer = model_utils.ModelAnalyzer(
            str(materialized_dir),
            plan["unified_eval_contract"]["eval_seq_len"],
            tokenizer_source=base_model,
            skip_state_dict=True,
        )
        if run["setting"] == "W2A4KV4":
            akv_cfg = SimpleNamespace(
                rotate=False,
                act_quant_aware_gptq=False,
                k_cache_quant_aware_gptq=False,
                a_bits=4,
                a_groupsize=-1,
                a_asym=False,
                a_clip_ratio=0.9,
                k_bits=4,
                k_groupsize=-1,
                k_asym=False,
                k_clip_ratio=0.9,
                v_bits=4,
                v_groupsize=-1,
                v_asym=False,
                v_clip_ratio=0.9,
            )
            akv_summary = setup_unaware_post_quant_no_rotation(
                analyzer,
                akv_cfg,
            )
        else:
            akv_summary = None
        ppl, kl = eval_utils._kl_ppl_eval(
            ref_args,
            analyzer,
            original_lm_head,
            test_loader,
            reference_hidden,
        )
        analyzer.model.to(torch.device(eval_device))
        qa = eval_utils.qa_eval(
            analyzer.model,
            analyzer.tokenizer,
            lm_eval_batch_size=plan["unified_eval_contract"][
                "lm_eval_batch_size"
            ],
        )
        expected_tasks = list(eval_utils.PAPER_QA_TASKS)
        actual_tasks = [name for name in qa if name != "acc_avg"]
        if actual_tasks != expected_tasks or "acc_avg" not in qa:
            raise RunError(
                "shared qa_eval returned a result outside the fixed "
                f"ten-task protocol: expected={expected_tasks!r}, "
                f"actual={actual_tasks!r}"
            )
        metrics = {
            "wikitext2": {
                "kl_raw": float(kl),
                "kl_x100": float(kl) * 100.0,
                "ppl": float(ppl),
            },
            "lm_eval": {
                "tasks": {
                    name: float(qa[name]) for name in expected_tasks
                },
                "acc_avg": float(qa["acc_avg"]),
            },
            "akv_runtime": (
                None if akv_summary is None else akv_summary.as_dict()
            ),
            "reference": {
                "model": base_model,
                "lm_head_dtype": reference_head_dtype,
                "required_cache_hit": True,
            },
        }
        del analyzer, reference_hidden, original_lm_head, test_loader
        torch.cuda.empty_cache()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    elapsed = time.monotonic() - started
    details = {
        "gpu_count": 0,
        "gpu_seconds": 0.0,
        "evaluation_wall_seconds": elapsed,
        "physical_eval_gpu": physical_eval_gpu,
        "evaluator_sha256": _assert_evaluator_identity(),
        "metrics": metrics,
    }
    return metrics, _record_stage(
        run_dir,
        stage,
        status="succeeded",
        started_at=started_at,
        wall_seconds=elapsed,
        details=details,
    )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    plan_path, plan, run, run_dir, plan_sha = _load_contract(args)
    logger = _configure_logging(run_dir)
    evaluator_sha = _assert_evaluator_identity()
    if (run_dir / "result.json").exists() or (run_dir / "failure.json").exists():
        raise RunError("immutable terminal state already exists")
    provenance = {
        "schema_version": 1,
        "status": "running",
        "started_at": _now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "plan_file": str(plan_path),
        "plan_sha256": plan_sha,
        "run_id": run["run_id"],
        "run": run,
        "evaluator_sha256": evaluator_sha,
        "efficientqat_upstream_commit": plan["code_identity"][
            "efficientqat_upstream_commit"
        ],
        "token_artifact_sha256": plan["calibration_contract"][
            "token_artifact"
        ]["sha256"],
        "train_device": args.train_device,
        "eval_device": args.eval_device,
        "physical_train_gpu": args.physical_train_gpu,
        "physical_eval_gpu": args.physical_eval_gpu,
    }
    _write_json_atomic(run_dir / "run_manifest.json", provenance)
    stages = {}
    packed_block, stages["block_ap"] = run_block_ap(
        plan, run, run_dir, args.train_device, logger
    )
    packed_final, stages["e2e_qp"] = run_e2e_qp(
        plan,
        run,
        run_dir,
        args.train_device,
        packed_block,
        logger,
    )
    materialized, stages["materialize"] = run_materialize(
        plan, run, run_dir, packed_final
    )
    metrics, stages["evaluation"] = run_evaluation(
        plan,
        run,
        run_dir,
        materialized,
        args.eval_device,
        args.physical_eval_gpu,
        logger,
    )
    quantization_gpu_seconds = sum(
        stages[name]["gpu_seconds"]
        for name in ("block_ap", "e2e_qp")
    )
    result = {
        **provenance,
        "status": "succeeded",
        "ended_at": _now(),
        "stages": stages,
        "metrics": metrics,
        "quantization_gpu_seconds": quantization_gpu_seconds,
        "quantization_gpu_hours": quantization_gpu_seconds / 3600.0,
        "materialized_checkpoint": str(materialized),
    }
    _write_json_atomic(run_dir / "result.json", result)
    logger.info("Pipeline succeeded: %s", json.dumps(metrics, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-device", required=True)
    parser.add_argument("--eval-device", required=True)
    parser.add_argument("--physical-train-gpu", type=int, required=True)
    parser.add_argument("--physical-eval-gpu", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = None
    try:
        _, plan, run, run_dir, _ = _load_contract(args)
        del plan, run
        run_pipeline(args)
        return 0
    except Exception as exc:
        if run_dir is not None:
            failure = {
                "schema_version": 1,
                "status": "failed",
                "ended_at": _now(),
                "error_type": type(exc).__qualname__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            try:
                _write_json_atomic(run_dir / "failure.json", failure)
            except Exception:
                pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
