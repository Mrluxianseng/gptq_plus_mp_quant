#!/usr/bin/env python3
"""One representative 1B/W2 gate before the six formal EfficientQAT runs."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import torch

from experiments.efficientqat_compare.akv_unaware import (
    setup_unaware_post_quant_no_rotation,
)
from experiments.efficientqat_compare.compat import (
    assert_model_vocab_unchanged,
    load_fast_eos_tokenizer,
)
from experiments.efficientqat_compare.data import (
    build_block_ap_trainloader,
    build_e2e_fixed_dataset,
    load_exact_token_cache,
)
from experiments.efficientqat_compare.materialize import (
    materialize_efficientqat_model,
    validate_materialized_model,
    write_materialization_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EQAT_ROOT = REPO_ROOT / "EfficientQAT"
EVALUATOR_SHA = (
    "e486fcd2d7fe31b78eeb8a407e8667b6f4774c7f2395e728b4654f7c41e1cbff"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def logger_for(output: Path) -> logging.Logger:
    logger = logging.getLogger("efficientqat.smoke")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output / "smoke.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def block_args(output: Path, *, epochs: int, max_blocks: int):
    return SimpleNamespace(
        off_load_to_disk=False,
        cache_dir=str(output / "cache"),
        train_size=2,
        val_size=0,
        training_seqlen=2048,
        batch_size=2,
        epochs=epochs,
        quant_lr=1e-4,
        weight_lr=2e-5,
        min_lr_factor=20,
        wd=0.0,
        early_stop=0,
        wbits=2,
        group_size=128,
        real_quant=True,
        max_blocks=max_blocks,
    )


def expected_names(layer_count: int) -> list[str]:
    result = []
    for index in range(layer_count):
        prefix = f"model.layers.{index}"
        result.extend(
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
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logger = logger_for(output)
    started = time.monotonic()
    report = {"status": "running", "started_unix": time.time(), "stages": {}}
    atomic_json(output / "smoke_state.json", report)

    if str(EQAT_ROOT) not in sys.path:
        sys.path.append(str(EQAT_ROOT))
    from quantize.block_ap import block_ap
    from quantize.int_linear_real import QuantLinear, load_quantized_model
    from transformers import (
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
    from utils import eval_utils, model_utils

    plan = json.loads(Path(args.plan_file).read_text())
    artifact = plan["calibration_contract"]["token_artifact"]
    if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
        raise RuntimeError("token artifact hash mismatch")
    if (
        sha256_file(REPO_ROOT / "utils" / "eval_utils.py")
        != EVALUATOR_SHA
    ):
        raise RuntimeError("shared evaluator hash mismatch")
    model_path = Path(plan["models"]["llama3.2-1b"]["path"])
    tokens = load_exact_token_cache(artifact["path"])
    tiny_tokens = tokens[:2]
    tokenizer = load_fast_eos_tokenizer(
        model_path, expected_vocab_size=128256
    )
    torch.cuda.set_device(0)
    torch.manual_seed(2)

    # Gate the actual FP16 Block-AP backward path at the formal sequence length.
    tick = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cpu"
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    block_ap(
        model,
        block_args(output / "block_backward", epochs=1, max_blocks=1),
        build_block_ap_trainloader(tiny_tokens),
        [],
        logger=logger,
    )
    one_block_count = sum(
        isinstance(module, QuantLinear) for module in model.modules()
    )
    if one_block_count != 7:
        raise RuntimeError(f"one-block packed count={one_block_count}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    report["stages"]["block_ap_backward"] = {
        "wall_seconds": time.monotonic() - tick,
        "optimizer_steps": 1,
        "sequence_length": 2048,
        "packed_linears": one_block_count,
    }
    atomic_json(output / "smoke_state.json", report)

    # Create a structurally complete packed checkpoint without a second QAT
    # sweep; epochs=0 exercises all official min-max pack paths.
    tick = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cpu"
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    block_ap(
        model,
        block_args(output / "full_pack", epochs=0, max_blocks=-1),
        build_block_ap_trainloader(tiny_tokens),
        [],
        logger=logger,
    )
    packed_count = sum(
        isinstance(module, QuantLinear) for module in model.modules()
    )
    if packed_count != 112:
        raise RuntimeError(f"full packed count={packed_count}")
    block_dir = output / "block_packed_model"
    model.save_pretrained(block_dir, safe_serialization=True)
    tokenizer.save_pretrained(block_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    report["stages"]["full_pack"] = {
        "wall_seconds": time.monotonic() - tick,
        "packed_linears": packed_count,
    }
    atomic_json(output / "smoke_state.json", report)

    # One real BF16 E2E-QP optimizer step at micro-batch 4 x 2048 tokens.
    tick = time.monotonic()
    model, packed_tokenizer = load_quantized_model(
        str(block_dir), 2, 128, device_map={"": "cpu"}
    )
    packed_tokenizer.pad_token = packed_tokenizer.eos_token
    model.to("cuda:0")
    model.train()
    model.config.use_cache = False
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
    trainable = []
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.scales.requires_grad = True
            trainable.append(module.scales)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    training_args = TrainingArguments(
        output_dir=str(output / "e2e_trainer"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=42,
        data_seed=42,
    )
    training_args._n_gpu = 1
    optimizer = torch.optim.AdamW(
        [{"params": trainable, "lr": 2e-5, "weight_decay": 0.0}]
    )
    dataset = build_e2e_fixed_dataset(tokens[:4]).shuffle(seed=0)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )
    trainer.train()
    if trainer.state.global_step != 1:
        raise RuntimeError(
            f"E2E smoke global_step={trainer.state.global_step}"
        )
    model.config.use_cache = True
    model.cpu()
    e2e_dir = output / "e2e_packed_model"
    model.save_pretrained(e2e_dir, safe_serialization=True)
    packed_tokenizer.save_pretrained(e2e_dir)
    report["stages"]["e2e_qp"] = {
        "wall_seconds": time.monotonic() - tick,
        "optimizer_steps": 1,
        "micro_batch_size": 4,
        "sequence_length": 2048,
        "trainable_scale_tensors": len(trainable),
    }
    del trainer, optimizer, trainable, model
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(output / "smoke_state.json", report)

    # Decode to the exact representation used by the common evaluator.
    tick = time.monotonic()
    model, packed_tokenizer = load_quantized_model(
        str(e2e_dir), 2, 128, device_map={"": "cpu"}
    )
    manifest = materialize_efficientqat_model(
        model,
        expected_count=112,
        expected_bits=2,
        expected_group_size=128,
        expected_module_names=expected_names(16),
        provenance={"kind": "representative_smoke"},
    )
    model.config.torch_dtype = torch.bfloat16
    model.eval()
    materialized_dir = output / "materialized_model"
    model.save_pretrained(materialized_dir, safe_serialization=True)
    packed_tokenizer.save_pretrained(materialized_dir)
    write_materialization_manifest(
        materialized_dir / "efficientqat_manifest.json", manifest
    )
    del model
    fresh = AutoModelForCausalLM.from_pretrained(
        materialized_dir, dtype=torch.bfloat16, device_map="cpu"
    )
    validate_materialized_model(fresh, manifest, expected_count=112)
    assert_model_vocab_unchanged(
        fresh, packed_tokenizer, expected_vocab_size=128256
    )
    del fresh
    gc.collect()
    report["stages"]["materialize"] = {
        "wall_seconds": time.monotonic() - tick,
        "linears": manifest["module_count"],
        "modules_sha256": manifest["modules_sha256"],
    }
    atomic_json(output / "smoke_state.json", report)

    # Install the exact post-QAT, no-rotation W2A4KV4 runtime and execute it.
    tick = time.monotonic()
    student = model_utils.ModelAnalyzer(
        str(materialized_dir),
        16,
        tokenizer_source=str(model_path),
        skip_state_dict=True,
    )
    cfg = SimpleNamespace(
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
    wrapper_summary = setup_unaware_post_quant_no_rotation(student, cfg)
    probe = tokens[0][:16].unsqueeze(0).to("cuda:0")
    student.model.to("cuda:0")
    with torch.no_grad():
        logits = student.model(probe).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("W2A4KV4 probe produced non-finite logits")
    student.model.cpu()
    report["stages"]["w2a4kv4_runtime"] = {
        "wall_seconds": time.monotonic() - tick,
        **wrapper_summary.as_dict(),
        "probe_logits_shape": list(logits.shape),
    }
    del logits
    torch.cuda.empty_cache()
    atomic_json(output / "smoke_state.json", report)

    # Exercise the exact shared RealQ KL/PPL implementation on a tiny fixture.
    tick = time.monotonic()
    testenc = SimpleNamespace(input_ids=tokens[0][:16].unsqueeze(0))
    eval_args = SimpleNamespace(eval_seq_len=16, kl_topk=-1)
    reference = model_utils.ModelAnalyzer(
        str(model_path),
        16,
        tokenizer_source=str(model_path),
        skip_state_dict=True,
    )
    reference_hidden, reference_ids = eval_utils._get_logits(
        eval_args, reference, testenc, torch.device("cuda:0")
    )
    if not torch.equal(reference_ids.cpu(), testenc.input_ids.cpu()):
        raise RuntimeError("shared evaluator changed smoke token IDs")
    original_head = copy.deepcopy(reference.model.lm_head)
    del reference
    gc.collect()
    ppl, kl = eval_utils._kl_ppl_eval(
        eval_args,
        student,
        original_head,
        testenc,
        reference_hidden,
    )
    if not (math.isfinite(ppl) and math.isfinite(kl) and kl >= 0):
        raise RuntimeError(f"invalid KL/PPL smoke values: {kl}, {ppl}")
    import lm_eval
    from lm_eval import utils as lm_eval_utils

    task_manager = lm_eval.tasks.TaskManager(include_defaults=True)
    task_names = eval_utils._resolve_paper_qa_tasks(
        lm_eval_utils.pattern_match, task_manager.all_tasks
    )
    if len(task_names) != 10:
        raise RuntimeError("shared lm-eval task resolver did not return ten tasks")
    report["stages"]["shared_evaluator"] = {
        "wall_seconds": time.monotonic() - tick,
        "kl": float(kl),
        "ppl": float(ppl),
        "lm_eval_task_count": len(task_names),
        "evaluator_sha256": EVALUATOR_SHA,
    }
    report["status"] = "succeeded"
    report["wall_seconds"] = time.monotonic() - started
    report["ended_unix"] = time.time()
    atomic_json(output / "smoke_result.json", report)
    atomic_json(output / "smoke_state.json", report)
    logger.info("Representative smoke succeeded: %s", json.dumps(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Preserve a machine-readable terminal state even when a gate fails
        # before ``main`` reaches its normal stage bookkeeping.
        try:
            output_index = sys.argv.index("--output-dir") + 1
            failed_output = Path(sys.argv[output_index]).resolve()
            state_path = failed_output / "smoke_state.json"
            failed_state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.exists()
                else {"stages": {}}
            )
            failed_state.update(
                {
                    "status": "failed",
                    "failed_unix": time.time(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            atomic_json(state_path, failed_state)
        except Exception:
            pass
        raise
