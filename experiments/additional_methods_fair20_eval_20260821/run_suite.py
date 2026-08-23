#!/usr/bin/env python3
"""Evaluate one completed checkpoint on the shared quality/reasoning protocol."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

from . import common, loaders


def _find_spec(eval_id: str) -> common.EvalSpec:
    matches = [spec for spec in common.iter_specs() if spec.eval_id == eval_id]
    if len(matches) != 1:
        raise common.EvaluationError(
            f"eval_id must resolve exactly once: {eval_id!r}"
        )
    return matches[0]


def _configure_runtime(seed: int) -> dict[str, Any]:
    import numpy as np
    import torch

    if torch.cuda.device_count() != 1:
        raise common.EvaluationError("suite worker must see exactly one GPU")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < 150_000_000_000:
        raise common.EvaluationError(
            f"target GPU is busy: free={free_bytes}, total={total_bytes}"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise common.EvaluationError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise common.EvaluationError(
            f"PYTHONHASHSEED must equal suite seed {seed}"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(0)
    versions = {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "lm_eval": importlib.metadata.version("lm-eval"),
        "evalplus": importlib.metadata.version("evalplus"),
    }
    expected_versions = {
        "torch": "2.9.1+cu128",
        "transformers": "4.56.2",
        "datasets": "3.6.0",
        "lm_eval": "0.4.4",
        "evalplus": "0.3.1",
    }
    if versions != expected_versions:
        raise common.EvaluationError(
            f"formal evaluation version mismatch: {versions!r} "
            f"!= {expected_versions!r}"
        )
    return {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free_bytes_at_start": int(free_bytes),
        "gpu_total_bytes": int(total_bytes),
        "python": ".".join(map(str, sys.version_info[:3])),
        **versions,
        "seed": seed,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def _reference_cache_path(args, analyzer, loader, eval_utils):
    metadata = eval_utils._reference_cache_metadata(
        args, analyzer, "wikitext2", loader
    )
    tag = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    path = (
        Path(args.cache_dir)
        / "ref_logits"
        / f"{args.model_name}_wikitext2_test_2048_{tag}.cache"
    ).resolve()
    return path, metadata


def _load_reference(
    spec: common.EvalSpec,
    model_info: dict[str, Any],
    *,
    expected_record: dict[str, Any],
):
    import torch
    from realq import attention
    from utils import data_utils, eval_utils, model_utils

    record = common.reference_cache_record(spec.model)
    path = Path(record["source"]).resolve()
    if (
        Path(expected_record["path"]).resolve() != path
        or not common.reference_stat_matches(path, expected_record)
    ):
        raise common.EvaluationError(f"reference cache release changed: {path}")
    base_model = Path(model_info["path"]).resolve()
    analyzer = model_utils.ModelAnalyzer(
        str(base_model),
        2048,
        tokenizer_source=str(base_model),
        skip_state_dict=True,
    )
    attention.configure_attention_backend(analyzer.model, "sdpa")
    args = SimpleNamespace(
        model=str(base_model),
        model_name=base_model.name,
        cache_dir=str(path.parent.parent),
        eval_seq_len=2048,
        nsamples=256,
        kl_topk=-1,
        rotate=True,
        rotation_seed=0,
        optimized_rotation_path=None,
        require_reference_cache_hit=True,
    )
    test_loader = data_utils.get_loaders(
        "wikitext2",
        split="test",
        tokenizer=analyzer.tokenizer,
        seq_len=2048,
        num_samples=256,
        seed=0,
    )
    computed, metadata = _reference_cache_path(
        args, analyzer, test_loader, eval_utils
    )
    if computed != path:
        raise common.EvaluationError(
            f"reference identity resolves to {computed}, expected {path}"
        )
    hidden, lm_head = eval_utils.get_ref_logits(
        args, analyzer, "wikitext2", test_loader
    )
    if not common.reference_stat_matches(path, expected_record):
        raise common.EvaluationError("reference cache changed while loading")
    if hidden.device.type != "cpu":
        raise common.EvaluationError("reference hidden states are invalid")
    return analyzer, args, test_loader, hidden, lm_head, path, metadata


def _validate_quality_result(
    path: Path, spec: common.EvalSpec, reference_sha256: str
) -> dict[str, Any]:
    result = common.read_object(path)
    if (
        result.get("schema_version") != 1
        or result.get("status") != "succeeded"
        or result.get("eval_id") != spec.eval_id
        or result.get("reference", {}).get("sha256") != reference_sha256
    ):
        raise common.EvaluationError(f"quality result gate failed: {path}")
    metrics = result.get("metrics", {})
    wiki = metrics.get("wikitext2", {})
    qa = metrics.get("paper_qa", {})
    qa_values = qa.get("tasks", {})
    finite_qa = (
        isinstance(qa_values, dict)
        and all(
            math.isfinite(float(qa_values.get(task, float("nan"))))
            and 0.0 <= float(qa_values[task]) <= 100.0
            for task in common.PAPER_QA_TASKS
        )
    )
    recomputed_qa = (
        round(
            sum(float(qa_values[task]) for task in common.PAPER_QA_TASKS)
            / len(common.PAPER_QA_TASKS),
            2,
        )
        if finite_qa
        else float("nan")
    )
    if (
        not math.isfinite(float(wiki.get("kl_raw", float("nan"))))
        or float(wiki["kl_raw"]) < 0
        or not math.isfinite(float(wiki.get("ppl", float("nan"))))
        or float(wiki["ppl"]) <= 0
        or qa.get("task_order") != list(common.PAPER_QA_TASKS)
        or set(qa.get("tasks", {})) != set(common.PAPER_QA_TASKS)
        or not finite_qa
        or not math.isfinite(float(qa.get("acc_avg", float("nan"))))
        or not math.isclose(
            recomputed_qa,
            float(qa.get("acc_avg", float("nan"))),
            rel_tol=0,
            abs_tol=0.011,
        )
    ):
        raise common.EvaluationError(f"quality metrics are incomplete: {path}")
    return result


def _run_quality(
    spec: common.EvalSpec,
    artifact: dict[str, Any],
    model_info: dict[str, Any],
    output_dir: Path,
    *,
    reference_record: dict[str, Any],
):
    reference_sha256 = str(reference_record["sha256"])
    import torch
    from utils import eval_utils

    destination = output_dir / "quality_result.json"
    if destination.is_file():
        result = _validate_quality_result(
            destination, spec, reference_sha256
        )
        analyzer = loaders.load_quantized(spec, artifact, model_info)
        return analyzer, result
    if destination.exists() or destination.is_symlink():
        raise common.EvaluationError(f"invalid quality output path: {destination}")

    started_at = common.now()
    started = time.monotonic()
    (
        reference_analyzer,
        reference_args,
        test_loader,
        reference_hidden,
        original_lm_head,
        reference_path,
        reference_metadata,
    ) = _load_reference(
        spec,
        model_info,
        expected_record=reference_record,
    )
    if spec.method == "turboboa":
        analyzer = loaders.load_quantized(
            spec,
            artifact,
            model_info,
            reference_analyzer=reference_analyzer,
        )
    else:
        del reference_analyzer
        gc.collect()
        torch.cuda.empty_cache()
        analyzer = loaders.load_quantized(spec, artifact, model_info)

    ppl, kl_raw = eval_utils._kl_ppl_eval(
        reference_args,
        analyzer,
        original_lm_head,
        test_loader,
        reference_hidden,
    )
    if (
        not math.isfinite(ppl)
        or ppl <= 0
        or not math.isfinite(kl_raw)
        or kl_raw < 0
    ):
        raise common.EvaluationError(
            f"invalid KL/PPL: kl={kl_raw}, ppl={ppl}"
        )
    print(
        f"Exact KL&PPL on wikitext2: {kl_raw:.17g}, {ppl:.17g}",
        flush=True,
    )
    eval_utils.pretty_print_results(
        OrderedDict(
            (
                ("KL-wikitext2", f"{kl_raw:.2e}"),
                ("PPL-wikitext2", f"{ppl:.2f}"),
            )
        )
    )

    del reference_hidden, original_lm_head, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    analyzer.model.to(torch.device("cuda"))
    qa = eval_utils.qa_eval(
        analyzer.model,
        analyzer.tokenizer,
        lm_eval_batch_size=32,
    )
    task_values = {
        task: float(qa[task]) for task in common.PAPER_QA_TASKS
    }
    if tuple(name for name in qa if name != "acc_avg") != common.PAPER_QA_TASKS:
        raise common.EvaluationError("lm-eval task order/set changed")
    recomputed = round(sum(task_values.values()) / len(task_values), 2)
    if not math.isclose(
        recomputed, float(qa["acc_avg"]), rel_tol=0, abs_tol=0.011
    ):
        raise common.EvaluationError("QA average arithmetic mismatch")
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "eval_id": spec.eval_id,
        "method": spec.method,
        "run_id": spec.run_id,
        "model": spec.model,
        "setting": spec.setting,
        "started_at": started_at,
        "finished_at": common.now(),
        "wall_seconds": time.monotonic() - started,
        "reference": {
            "path": str(reference_path),
            "sha256": reference_sha256,
            "metadata": reference_metadata,
            "shared_with_gptaq_guided_rescomp": True,
        },
        "artifact": artifact,
        "metrics": {
            "wikitext2": {
                "kl_raw": float(kl_raw),
                "kl_x100": float(kl_raw) * 100.0,
                "ppl": float(ppl),
                "kl_direction": "KL(FP||quantized)",
                "full_vocabulary_fp32": True,
            },
            "paper_qa": {
                "task_order": list(common.PAPER_QA_TASKS),
                "tasks": task_values,
                "acc_avg": float(qa["acc_avg"]),
                "lm_eval_version": "0.4.4",
                "batch_size": 32,
                "num_fewshot": 0,
                "apply_chat_template": False,
            },
        },
    }
    common.atomic_json(destination, result)
    _validate_quality_result(destination, spec, reference_sha256)
    return analyzer, result


def _reasoning_config(
    spec: common.EvalSpec,
    artifact: dict[str, Any],
    task: str,
    output_dir: Path,
) -> SimpleNamespace:
    protocol = common.REASONING_TASKS[task]
    return SimpleNamespace(
        model=str(artifact["path"]),
        load_qmodel_path=(
            str(artifact["path"])
            if artifact["kind"] == "realq_checkpoint"
            else None
        ),
        w_bits=spec.w_bits,
        w_groupsize=128,
        a_bits=spec.a_bits,
        a_groupsize=-1,
        k_bits=spec.k_bits,
        k_groupsize=-1,
        v_bits=spec.v_bits,
        v_groupsize=-1,
        rotate=spec.method == "turboboa",
        reasoning_tasks=[task],
        reasoning_data_dir=str(common.REPO_ROOT / "datasets/reasoning_eval"),
        reasoning_output_dir=str(output_dir),
        reasoning_batch_size=int(protocol["batch_size"]),
        reasoning_limit=-1,
        reasoning_max_new_tokens=int(protocol["max_new_tokens"]),
        reasoning_num_samples=1,
        reasoning_apply_chat_template=True,
        reasoning_enable_thinking=True,
        reasoning_do_sample=False,
        reasoning_temperature=0.0,
        reasoning_top_p=1.0,
        reasoning_top_k=0,
        reasoning_seed=1234,
        reasoning_resume=True,
        reasoning_protocol="realq_zero_shot_v1",
        reasoning_system_prompt=(
            "You are a careful reasoning assistant. Follow the requested "
            "output format exactly."
        ),
        reasoning_lcb_release="release_v6",
        reasoning_lcb_source_dir=str(
            common.REPO_ROOT / "datasets/reasoning_eval/vendor/LiveCodeBench"
        ),
        output_dir=str(output_dir),
        exp=f"{spec.eval_id}_{task}",
    )


def _validate_reasoning_contract(
    path: Path,
    task: str,
    *,
    require_completed: bool,
) -> dict[str, Any]:
    manifest = common.read_object(path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("tasks") != [task]
        or manifest.get("generation", {}).get("protocol")
        != "realq_zero_shot_v1"
        or manifest.get("generation", {}).get("seed") != 1234
        or manifest.get("generation", {}).get("do_sample") is not False
    ):
        raise common.EvaluationError(
            f"reasoning manifest failed closed: {path}"
        )
    protocol = common.REASONING_TASKS[task]
    expected = {
        "batch_size": protocol["batch_size"],
        "max_new_tokens": protocol["max_new_tokens"],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "apply_chat_template": True,
        "enable_thinking": True,
    }
    mismatches = {
        key: (manifest["generation"].get(key), value)
        for key, value in expected.items()
        if manifest["generation"].get(key) != value
    }
    if mismatches:
        raise common.EvaluationError(
            f"reasoning protocol mismatch for {task}: {mismatches!r}"
        )
    if require_completed:
        results = manifest.get("results", [])
        if (
            manifest.get("status") != "completed"
            or len(results) != 1
            or results[0].get("task") != task
            or results[0].get("num_examples") != protocol["count"]
        ):
            raise common.EvaluationError(
                f"reasoning completion/coverage mismatch for {task}"
            )
    elif manifest.get("status") not in {"running", "failed", "completed"}:
        raise common.EvaluationError(
            f"reasoning partial status is invalid for {task}"
        )
    return manifest


def _validate_reasoning_manifest(path: Path, task: str) -> dict[str, Any]:
    return _validate_reasoning_contract(
        path,
        task,
        require_completed=True,
    )


def _run_reasoning(
    spec: common.EvalSpec,
    artifact: dict[str, Any],
    analyzer,
    output_dir: Path,
) -> dict[str, Any]:
    from realq_benchmark.benchmarks.runner import run_reasoning_eval

    results = {}
    for task in common.REASONING_TASKS:
        task_dir = output_dir / "reasoning" / task
        manifest_path = task_dir / "manifest.json"
        completed = False
        if manifest_path.is_file():
            partial = _validate_reasoning_contract(
                manifest_path,
                task,
                require_completed=False,
            )
            completed = partial.get("status") == "completed"
        elif manifest_path.exists() or manifest_path.is_symlink():
            raise common.EvaluationError(
                f"invalid reasoning manifest path: {manifest_path}"
            )
        if completed:
            manifest = _validate_reasoning_manifest(manifest_path, task)
        else:
            task_dir.mkdir(parents=True, exist_ok=True)
            cfg = _reasoning_config(spec, artifact, task, task_dir)
            manifest = run_reasoning_eval(
                analyzer.model,
                analyzer.tokenizer,
                cfg,
            )
            if not manifest_path.is_file():
                raise common.EvaluationError(
                    f"reasoning runner did not publish {manifest_path}"
                )
            manifest = _validate_reasoning_manifest(manifest_path, task)
        results[task] = {
            "manifest": str(manifest_path),
            "manifest_sha256": common.sha256_file(manifest_path),
            "result": manifest["results"][0],
        }
    return results


def execute(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    spec = _find_spec(args.eval_id)
    output_dir = Path(args.output_dir).resolve()
    success_path = output_dir / "suite_success.json"
    if success_path.is_file():
        success = common.read_object(success_path)
        if success.get("eval_id") != spec.eval_id:
            raise common.EvaluationError("suite success identity mismatch")
        return success
    if success_path.exists() or success_path.is_symlink():
        raise common.EvaluationError(f"invalid suite success path: {success_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = _configure_runtime(seed=1234)
    plans = common.load_quant_plans()
    reference_manifest = common.load_reference_manifest()
    if reference_manifest["fingerprint"] != args.expected_reference_fingerprint:
        raise common.EvaluationError("reference manifest fingerprint changed")
    reference_record = dict(reference_manifest["references"][spec.model])
    if reference_record["sha256"] != args.expected_reference_sha256:
        raise common.EvaluationError("reference cache digest binding changed")
    model_info = common.model_contract(spec.model, plans)
    artifact = common.resolve_completed_artifact(spec)
    quality_path = output_dir / "quality_result.json"
    if quality_path.is_file():
        quality = _validate_quality_result(
            quality_path, spec, args.expected_reference_sha256
        )
        analyzer = loaders.load_quantized(spec, artifact, model_info)
    else:
        analyzer, quality = _run_quality(
            spec,
            artifact,
            model_info,
            output_dir,
            reference_record=reference_record,
        )
    analyzer.model.to(torch.device("cuda"))
    reasoning = _run_reasoning(spec, artifact, analyzer, output_dir)
    result = {
        "schema_version": 1,
        "status": "generation_succeeded_official_humaneval_pending",
        "eval_id": spec.eval_id,
        "method": spec.method,
        "run_id": spec.run_id,
        "model": spec.model,
        "setting": spec.setting,
        "runtime": runtime,
        "reference_manifest_fingerprint": reference_manifest["fingerprint"],
        "artifact": artifact,
        "quality": {
            "path": str(quality_path),
            "sha256": common.sha256_file(quality_path),
            "metrics": quality["metrics"],
        },
        "reasoning": reasoning,
        "official_humaneval_plus": "pending",
        "finished_at": common.now(),
    }
    common.atomic_json(success_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--expected-reference-fingerprint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
