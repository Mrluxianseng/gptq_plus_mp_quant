#!/usr/bin/env python3
"""Evaluate one hfized YAQA run with RealQ's canonical evaluators."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from lib.utils.unsafe_import import model_from_hf_path
from quantize_llama.preflight_wclip import SETTING_BITS
from utils import data_utils, eval_utils, model_utils


REFERENCE_CACHE = Path(
    "/minimax-avatar-new/zhangqian/realq/experiment_data/"
    "turboboa_compare_20260725/j-8j1en3m0aq/artifacts/shared_cache"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reference_args(model_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model=str(model_path),
        model_name=model_path.name,
        cache_dir=str(REFERENCE_CACHE),
        eval_seq_len=2048,
        kl_topk=-1,
        rotate=True,
        rotation_seed=0,
        optimized_rotation_path=None,
        require_reference_cache_hit=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=tuple(SETTING_BITS),
    )
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--hf-validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lm-eval-batch-size", type=int, default=32)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Formal evaluation requires exactly one visible GPU")
    if args.lm_eval_batch_size != 32:
        raise RuntimeError("Formal RealQ lm-eval batch size must be exactly 32")
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"formal evaluation output already exists: {output}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    model_path = Path(args.model).resolve()
    hf_dir = Path(args.hf_dir).resolve()
    validation_path = Path(args.hf_validation).resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("status") != "validated"
        or validation.get("setting") != args.setting
        or Path(validation["model"]).resolve() != model_path
        or Path(validation["hf_dir"]).resolve() != hf_dir
    ):
        raise RuntimeError("hfized validation report mismatch")

    ref_args = _reference_args(model_path)
    reference_analyzer = model_utils.ModelAnalyzer(
        str(model_path),
        2048,
        tokenizer_source=str(model_path),
        skip_state_dict=True,
    )
    test_loader = data_utils.get_loaders(
        "wikitext2",
        "test",
        reference_analyzer.tokenizer,
        2048,
        256,
        1,
    )
    reference_metadata = eval_utils._reference_cache_metadata(
        ref_args, reference_analyzer, "wikitext2", test_loader
    )
    cache_tag = hashlib.sha256(
        json.dumps(reference_metadata, sort_keys=True).encode()
    ).hexdigest()[:20]
    reference_cache_path = (
        REFERENCE_CACHE
        / "ref_logits"
        / f"{model_path.name}_wikitext2_test_2048_{cache_tag}.cache"
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

    quantized_model, _ = model_from_hf_path(str(hf_dir), device_map="cpu")
    analyzer = model_utils.ModelAnalyzer(
        quantized_model,
        2048,
        tokenizer_source=str(model_path),
        skip_state_dict=True,
    )
    del quantized_model
    ppl, kl = eval_utils._kl_ppl_eval(
        ref_args,
        analyzer,
        original_lm_head,
        test_loader,
        reference_hidden,
    )
    analyzer.model.to(torch.device("cuda"))
    qa = eval_utils.qa_eval(
        analyzer.model,
        analyzer.tokenizer,
        lm_eval_batch_size=32,
    )
    expected_tasks = list(eval_utils.PAPER_QA_TASKS)
    actual_tasks = [name for name in qa if name != "acc_avg"]
    if actual_tasks != expected_tasks or "acc_avg" not in qa:
        raise RuntimeError(
            f"RealQ qa_eval task mismatch: {actual_tasks!r} != "
            f"{expected_tasks!r}"
        )

    metrics = {
        "wikitext2": {
            "kl_raw": float(kl),
            "kl_x100": float(kl) * 100.0,
            "ppl": float(ppl),
            "split": "test",
            "sequence_length": 2048,
            "kl_direction": "KL(FP||quantized)",
            "kl_full_vocabulary_fp32": True,
        },
        "lm_eval": {
            "version": importlib.metadata.version("lm-eval"),
            "num_fewshot": 0,
            "apply_chat_template": False,
            "batch_size": 32,
            "tasks": {name: float(qa[name]) for name in expected_tasks},
            "acc_avg": float(qa["acc_avg"]),
        },
    }
    report = {
        "schema_version": 1,
        "status": "evaluated",
        "model": str(model_path),
        "setting": args.setting,
        "hf_dir": str(hf_dir),
        "hf_validation": str(validation_path),
        "hf_validation_sha256": _sha256_file(validation_path),
        "evaluator": {
            "path": str(Path(eval_utils.__file__).resolve()),
            "sha256": _sha256_file(Path(eval_utils.__file__).resolve()),
            "paper_tasks": expected_tasks,
        },
        "reference": {
            "cache_path": str(reference_cache_path),
            "cache_sha256": _sha256_file(reference_cache_path),
            "metadata": reference_metadata,
            "lm_head_dtype": reference_head_dtype,
            "required_cache_hit": True,
        },
        "metrics": metrics,
    }
    _write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
