#!/usr/bin/env python3
"""Validate both frozen RealQ reference caches before formal QAT starts."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

from . import launcher
from .run_one import (
    _assert_imported_evaluator_identity,
    _bootstrap_efficientqat,
    _reference_args,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        raise RuntimeError(
            "Reference-cache preflight must run with CUDA_VISIBLE_DEVICES=''."
        )

    plan_path = Path(args.plan_file).resolve()
    plan = launcher.load_plan(plan_path)
    launcher.verify_artifacts(plan)
    _bootstrap_efficientqat()
    from utils import data_utils, eval_utils, model_utils

    evaluator_sha = _assert_imported_evaluator_identity(eval_utils)
    records: dict[str, object] = {}
    seen_models: set[str] = set()
    for run in plan["runs"]:
        model_key = run["model"]
        if model_key in seen_models:
            continue
        seen_models.add(model_key)
        reference_args = _reference_args(plan, run)
        model_path = plan["models"][model_key]["path"]
        analyzer = model_utils.ModelAnalyzer(
            model_path,
            plan["unified_eval_contract"]["eval_seq_len"],
            tokenizer_source=model_path,
            skip_state_dict=True,
        )
        test_loader = data_utils.get_loaders(
            "wikitext2",
            "test",
            analyzer.tokenizer,
            reference_args.eval_seq_len,
            plan["calibration_contract"]["num_samples"],
            plan["calibration_contract"]["seed"],
        )
        metadata = eval_utils._reference_cache_metadata(
            reference_args,
            analyzer,
            "wikitext2",
            test_loader,
        )
        cache_tag = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        cache_path = (
            Path(reference_args.cache_dir)
            / "ref_logits"
            / (
                f"{reference_args.model_name}_wikitext2_test_"
                f"{reference_args.eval_seq_len}_{cache_tag}.cache"
            )
        )
        if not cache_path.is_file():
            raise RuntimeError(
                f"Required reference cache does not exist: {cache_path}"
            )

        # Use the evaluator's exact cache validator directly.  Its public
        # get_ref_logits wrapper ends with an unconditional CUDA cleanup,
        # while this preflight intentionally hides every GPU.
        reference_hidden = eval_utils._load_reference_cache(
            str(cache_path),
            metadata,
        )
        if reference_hidden is None:
            raise RuntimeError(
                f"Reference cache failed exact metadata/tensor validation: "
                f"{cache_path}"
            )
        original_lm_head = copy.deepcopy(analyzer.model.lm_head)
        if list(reference_hidden.shape) != metadata["hidden_states_shape"]:
            raise RuntimeError(
                f"Reference shape mismatch for {model_key}: "
                f"{list(reference_hidden.shape)} != "
                f"{metadata['hidden_states_shape']}"
            )
        if str(reference_hidden.dtype) != metadata["hidden_states_dtype"]:
            raise RuntimeError(
                f"Reference dtype mismatch for {model_key}: "
                f"{reference_hidden.dtype} != "
                f"{metadata['hidden_states_dtype']}"
            )
        records[model_key] = {
            "cache_path": str(cache_path),
            "cache_size_bytes": cache_path.stat().st_size,
            "cache_sha256": _sha256(cache_path),
            "cache_tag": cache_tag,
            "hidden_states_shape": list(reference_hidden.shape),
            "hidden_states_dtype": str(reference_hidden.dtype),
            "lm_head_dtype": str(original_lm_head.weight.dtype),
            "reference_model": model_path,
            "rotation_metadata": metadata["rotation_identity"],
            "checkpoint_is_rotated": metadata["checkpoint_is_rotated"],
        }
        del (
            analyzer,
            test_loader,
            reference_hidden,
            original_lm_head,
        )
        gc.collect()

    report = {
        "status": "succeeded",
        "plan_sha256": launcher.plan_sha256(plan_path),
        "evaluator_sha256": evaluator_sha,
        "cuda_available": torch.cuda.is_available(),
        "models": records,
    }
    _write_json_atomic(Path(args.output_json).resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
