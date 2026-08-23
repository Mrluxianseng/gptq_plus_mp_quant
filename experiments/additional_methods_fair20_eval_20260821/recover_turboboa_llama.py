#!/usr/bin/env python3
"""Recover TurboBOA Llama suites blocked by a model-specific manifest alias.

The frozen evaluator correctly reconstructs the checkpoint, but its final
weight-provenance gate requires the Qwen3-only method name for every model.
TurboBOA truthfully records plain ``turboboa`` for Llama because Llama has no
Q/K RMSNorm pullback.  This recovery changes only that final expected string;
the numerical loader and the complete evaluation suite remain identical.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

from . import common, loaders, run_suite


ALLOWED_EVAL_IDS = (
    "turboboa__TB20-L8-W4A4",
    "turboboa__TB20-L8-W4",
    "turboboa__TB20-L8-W3",
    "turboboa__TB20-L8-W2",
)
LOADER_RELATIVE = (
    "experiments/additional_methods_fair20_eval_20260821/loaders.py"
)
KNOWN_FAILURE = (
    "TurboBOA weight manifest mismatch: "
    "{'w_method': ('turboboa', "
    "'turboboa_rmsnorm_mean_jacobian_kfac')}"
)


class RecoveryError(RuntimeError):
    """A recovery precondition or publication gate failed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _find_spec(eval_id: str) -> common.EvalSpec:
    if eval_id not in ALLOWED_EVAL_IDS:
        raise RecoveryError(f"recovery is not allowed for {eval_id!r}")
    matches = [spec for spec in common.iter_specs() if spec.eval_id == eval_id]
    if len(matches) != 1:
        raise RecoveryError(f"expected one frozen spec for {eval_id!r}")
    spec = matches[0]
    if spec.method != "turboboa" or spec.model != "llama31-8b-instruct":
        raise RecoveryError("recovery spec is not TurboBOA Llama-3.1-8B")
    return spec


def _validate_failed_attempts(suite_dir: Path, eval_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for attempt in sorted((suite_dir / "attempts").glob("attempt[0-9][0-9][0-9]")):
        result_path = attempt / "result.json"
        log_path = attempt / "execution.log"
        if not result_path.is_file() or not log_path.is_file():
            raise RecoveryError(f"incomplete prior attempt: {attempt}")
        result = common.read_object(result_path)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if (
            result.get("eval_id") != eval_id
            or result.get("status") != "failed"
            or int(result.get("returncode", 0)) == 0
            or KNOWN_FAILURE not in log
        ):
            raise RecoveryError(f"prior attempt is not the known alias failure: {attempt}")
        records.append(
            {
                "attempt": int(result["attempt"]),
                "result": str(result_path.resolve()),
                "result_sha256": common.sha256_file(result_path),
                "execution_log": str(log_path.resolve()),
                "execution_log_sha256": common.sha256_file(log_path),
            }
        )
    return records


def _validate_quantization_result(
    spec: common.EvalSpec,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = Path(artifact["validation"]).resolve()
    result = common.read_object(result_path)
    cfg = result.get("configuration", {})
    aware = spec.setting == "W4A4KV4"
    expected = {
        "llm_type": "Llama",
        "w_bits": spec.w_bits,
        "group_size": 128,
        "w_sym": True,
        "w_asym": False,
        "w_clip": True,
        "w_method": "turboboa",
        "weight_quantizer": "realq_mse",
        "qparam_comput": "RealQ-MSE",
        "act_order_col": True,
        "act_order_row": False,
        "rotate": True,
        "rotation_seed": 0,
        "seed": 1,
        "global_seed": 1,
        "deterministic": True,
        "nsamples": 256,
        "seqlen": 2048,
        "a_bits": spec.a_bits,
        "k_bits": spec.k_bits,
        "v_bits": spec.v_bits,
        "a_groupsize": -1,
        "k_groupsize": -1,
        "v_groupsize": -1,
        "a_asym": False,
        "k_asym": False,
        "v_asym": False,
        "a_clip_ratio": 0.9 if aware else 1.0,
        "k_clip_ratio": 0.9 if aware else 1.0,
        "v_clip_ratio": 0.9 if aware else 1.0,
        "act_quant_aware_gptq": aware,
        "k_cache_quant_aware_gptq": aware,
        "qk_rmsnorm_hessian_mode": "mean_jacobian_kfac",
    }
    mismatch = {
        key: (cfg.get(key), value)
        for key, value in expected.items()
        if cfg.get(key) != value
    }
    if (
        result.get("status") != "quantized_only"
        or result.get("method") != "TurboBoA"
        or mismatch
    ):
        raise RecoveryError(
            f"TurboBOA Llama quantization contract mismatch: {mismatch!r}"
        )
    return result


def load_turboboa_llama(
    spec: common.EvalSpec,
    artifact: Mapping[str, Any],
    model_info: Mapping[str, Any],
    *,
    analyzer=None,
):
    """Numerically identical loader with the truthful Llama method alias."""

    from realq import akv, attention, pipeline
    from utils import checkpoint_utils, model_utils

    if spec.model != "llama31-8b-instruct":
        raise RecoveryError("corrected loader is restricted to Llama-3.1-8B")
    base_model = loaders._validate_base_model(spec, model_info)
    loaders._validate_terminal_binding(spec, artifact)
    checkpoint_path = Path(artifact["path"]).resolve()
    checkpoint = checkpoint_utils.load_quantized_checkpoint(checkpoint_path)
    cfg = loaders._checkpoint_config(spec, base_model)
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
        # This is the sole correction relative to the frozen v1 loader.
        "w_method": "turboboa",
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


def _recovery_attempt_dir(suite_dir: Path) -> Path:
    root = suite_dir / "contract_recovery"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 4):
        attempt = root / f"attempt{index:03d}"
        try:
            attempt.mkdir()
        except FileExistsError:
            continue
        return attempt
    raise RecoveryError("contract-recovery retry budget exhausted")


def recover(eval_id: str) -> dict[str, Any]:
    hostname = socket.gethostname()
    if not hostname.startswith("j-") or not hostname.endswith("-master-0"):
        raise RecoveryError("recovery must run on a Canoe pod")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RecoveryError("CUDA_VISIBLE_DEVICES must name exactly one physical GPU")

    reference = common.load_reference_manifest()
    loader_path = common.REPO_ROOT / LOADER_RELATIVE
    loader_sha = common.sha256_file(loader_path)
    if reference["sources"].get(LOADER_RELATIVE) != loader_sha:
        raise RecoveryError("frozen loader source no longer matches the release")
    spec = _find_spec(eval_id)
    suite_dir = common.OUTPUT_ROOT / "evals" / eval_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    success_path = suite_dir / "suite_success.json"
    if success_path.is_file():
        existing = common.read_object(success_path)
        if existing.get("contract_recovery", {}).get("kind") != (
            "turboboa_llama_manifest_alias_only"
        ):
            raise RecoveryError("suite exists without the expected recovery record")
        return existing

    claim = suite_dir / ".claim"
    owner_path = claim / "owner.json"
    if not claim.is_dir() or not owner_path.is_file():
        raise RecoveryError("the recovery queue does not own the suite claim")
    owner = common.read_object(owner_path)
    if (
        owner.get("kind") != "turboboa_llama_contract_recovery_queue"
        or owner.get("hostname") != hostname
    ):
        raise RecoveryError("suite claim is not owned by this recovery queue")

    artifact = common.resolve_completed_artifact(spec)
    quant_result = _validate_quantization_result(spec, artifact)
    prior_attempts = _validate_failed_attempts(suite_dir, eval_id)
    checkpoint_path = Path(artifact["path"]).resolve()
    before = _file_identity(checkpoint_path)
    attempt = _recovery_attempt_dir(suite_dir)
    source_path = Path(__file__).resolve()
    recovery_source_sha = common.sha256_file(source_path)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "kind": "turboboa_llama_manifest_alias_only",
        "eval_id": eval_id,
        "hostname": hostname,
        "physical_gpu": int(visible),
        "started_at": _now(),
        "frozen_reference_fingerprint": reference["fingerprint"],
        "frozen_loader": {"path": str(loader_path), "sha256": loader_sha},
        "recovery_source": {
            "path": str(source_path),
            "sha256": recovery_source_sha,
        },
        "manifest_alias": {
            "checkpoint_actual": "turboboa",
            "frozen_loader_incorrect_expected": (
                "turboboa_rmsnorm_mean_jacobian_kfac"
            ),
            "recovery_expected_for_llama": "turboboa",
        },
        "prior_known_failures": prior_attempts,
        "checkpoint_identity_before": before,
        "quantization_result": {
            "path": artifact["validation"],
            "sha256": artifact["validation_sha256"],
            "method": quant_result["method"],
            "w_method": quant_result["configuration"]["w_method"],
        },
    }
    common.atomic_json(attempt / "manifest.json", manifest)

    original_loader = loaders.load_turboboa
    started = time.monotonic()
    try:
        loaders.load_turboboa = load_turboboa_llama
        result = run_suite.execute(
            SimpleNamespace(
                eval_id=eval_id,
                expected_reference_sha256=reference["references"][spec.model][
                    "sha256"
                ],
                expected_reference_fingerprint=reference["fingerprint"],
                output_dir=str(suite_dir),
            )
        )
    finally:
        loaders.load_turboboa = original_loader

    after = _file_identity(checkpoint_path)
    if after != before:
        raise RecoveryError("checkpoint identity changed during recovery")
    if (
        result.get("eval_id") != eval_id
        or result.get("status")
        != "generation_succeeded_official_humaneval_pending"
    ):
        raise RecoveryError("canonical suite did not complete")

    recovery_record = {
        "kind": "turboboa_llama_manifest_alias_only",
        "numerical_evaluation_code": "frozen_v1_reused_without_change",
        "checkpoint_modified": False,
        "checkpoint_identity_before": before,
        "checkpoint_identity_after": after,
        "frozen_loader": manifest["frozen_loader"],
        "recovery_source": manifest["recovery_source"],
        "manifest_alias": manifest["manifest_alias"],
        "prior_known_failures": prior_attempts,
        "attempt_manifest": str((attempt / "manifest.json").resolve()),
        "attempt_manifest_sha256": common.sha256_file(attempt / "manifest.json"),
    }
    published = dict(result)
    published["contract_recovery"] = recovery_record
    common.atomic_json(success_path, published)
    receipt = {
        "schema_version": 1,
        "status": "succeeded",
        "kind": "turboboa_llama_manifest_alias_only",
        "eval_id": eval_id,
        "wall_seconds": time.monotonic() - started,
        "finished_at": _now(),
        "quality_result": str((suite_dir / "quality_result.json").resolve()),
        "quality_result_sha256": common.sha256_file(
            suite_dir / "quality_result.json"
        ),
        "suite_success": str(success_path.resolve()),
        "suite_success_sha256": common.sha256_file(success_path),
        "recovery": recovery_record,
    }
    common.atomic_json(attempt / "receipt.json", receipt)
    common.atomic_json(
        attempt / "result.json",
        {
            **manifest,
            "status": "succeeded",
            "finished_at": receipt["finished_at"],
            "wall_seconds": receipt["wall_seconds"],
            "receipt": str((attempt / "receipt.json").resolve()),
            "receipt_sha256": common.sha256_file(attempt / "receipt.json"),
        },
    )
    return published


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-id", required=True, choices=ALLOWED_EVAL_IDS)
    args = parser.parse_args(argv)
    try:
        result = recover(args.eval_id)
    except BaseException:
        traceback.print_exc()
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
