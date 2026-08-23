#!/usr/bin/env python3
"""Fail-closed static preflight for the 2026-08-21 EfficientQAT campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODELS = {
    "qwen3-0.6b",
    "llama31-8b-instruct",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-32b",
}
EXPECTED_BITS = {2, 3, 4}


class PreflightError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def tensor_list_digest(values) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        tensor = value.detach().cpu().contiguous()
        digest.update(index.to_bytes(8, "little"))
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise PreflightError("unsupported plan schema")
    if set(plan.get("models", {})) != EXPECTED_MODELS:
        raise PreflightError("model set differs from the frozen five-model set")
    runs = plan.get("runs", [])
    expected_pairs = {
        (model, bits) for model in EXPECTED_MODELS for bits in EXPECTED_BITS
    }
    pairs = {(run.get("model"), run.get("w_bits")) for run in runs}
    if len(runs) != 15 or pairs != expected_pairs:
        raise PreflightError("run matrix must be exactly five models x W2/W3/W4")
    if len({run.get("run_id") for run in runs}) != 15:
        raise PreflightError("run IDs are not unique")
    if any(
        run.get("a_bits") != 16
        or run.get("k_bits") != 16
        or run.get("v_bits") != 16
        or run.get("rotation") is not False
        for run in runs
    ):
        raise PreflightError("all runs must be A16/KV16 with rotation disabled")
    method = plan.get("method_contract", {})
    quantizer = method.get("weight_quantizer", {})
    if (
        method.get("weight_group_size") != 128
        or quantizer.get("family") != "uniform_scalar"
        or quantizer.get("symmetric") is not True
        or quantizer.get("signed") is not True
        or quantizer.get("learnable_zero_point") is not False
        or method.get("rotation", {}).get("enabled") is not False
        or method.get("activation_kv_quantization", {}).get("enabled") is not False
    ):
        raise PreflightError("method quantization contract changed")
    block = method.get("block_ap", {})
    if (
        block.get("batch_size") != 2
        or block.get("epochs") != 2
        or block.get("quantizer_lr") != 1e-4
        or block.get("minimum_lr_ratio") != 0.05
        or block.get("weight_lr_by_weight_bits")
        != {"2": 2e-5, "3": 1e-5, "4": 1e-5}
    ):
        raise PreflightError("Block-AP paper recipe changed")
    e2e = method.get("e2e_qp", {})
    if (
        e2e.get("micro_batch_size") != 4
        or e2e.get("gradient_accumulation_steps") != 8
        or e2e.get("epochs") != 1
        or e2e.get("scale_lr_by_weight_bits")
        != {"2": 2e-5, "3": 1e-5, "4": 1e-5}
        or not e2e.get("positive_scale_projection", {}).get("enabled")
    ):
        raise PreflightError("E2E-QP recipe changed")


def verify_artifacts(plan: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from experiments.efficientqat_compare.data import load_exact_token_cache

    models = {}
    for key, contract in sorted(plan["models"].items()):
        root = Path(contract["path"])
        if not root.is_dir():
            raise PreflightError(f"model directory is missing: {root}")
        identities = {
            "config.json": contract["config_sha256"],
            "tokenizer.json": contract["tokenizer_json_sha256"],
            "tokenizer_config.json": contract["tokenizer_config_sha256"],
        }
        actual = {}
        for filename, expected in identities.items():
            path = root / filename
            value = sha256_file(path)
            if value != expected:
                raise PreflightError(f"{key}/{filename} identity changed")
            actual[filename] = value
        shards = sorted(root.glob("*.safetensors"))
        total = sum(path.stat().st_size for path in shards)
        if (
            len(shards) != int(contract["checkpoint_file_count"])
            or total != int(contract["checkpoint_bytes"])
        ):
            raise PreflightError(f"{key} checkpoint inventory changed")
        index_sha = contract.get("index_sha256")
        if index_sha is not None and sha256_file(
            root / "model.safetensors.index.json"
        ) != index_sha:
            raise PreflightError(f"{key} checkpoint index changed")
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer_vocab_size = len(tokenizer)
        tokenizer_max_id = max(tokenizer.get_vocab().values())
        expected_tokenizer_vocab_size = int(contract["tokenizer_vocab_size"])
        model_vocab_size = int(
            json.loads((root / "config.json").read_text(encoding="utf-8"))[
                "vocab_size"
            ]
        )
        if tokenizer_vocab_size != expected_tokenizer_vocab_size:
            raise PreflightError(
                f"{key} tokenizer vocabulary changed: "
                f"{tokenizer_vocab_size} != {expected_tokenizer_vocab_size}"
            )
        if tokenizer_max_id >= model_vocab_size:
            raise PreflightError(
                f"{key} tokenizer token id exceeds model vocabulary: "
                f"{tokenizer_max_id} >= {model_vocab_size}"
            )
        models[key] = {
            "identity_files": actual,
            "checkpoint_file_count": len(shards),
            "checkpoint_bytes": total,
            "model_vocab_size": model_vocab_size,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "tokenizer_max_id": tokenizer_max_id,
        }

    calibrations = {}
    for key, contract in sorted(plan["calibrations"].items()):
        artifact = contract["token_artifact"]
        path = Path(artifact["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["size_bytes"])
            or sha256_file(path) != artifact["sha256"]
        ):
            raise PreflightError(f"calibration artifact identity changed: {key}")
        values = load_exact_token_cache(
            path,
            expected_samples=256,
            expected_seq_len=2048,
        )
        calibrations[key] = {
            "path": str(path),
            "archive_sha256": artifact["sha256"],
            "size_bytes": path.stat().st_size,
            "length": len(values),
            "item_shape": [2048],
            "item_dtype": "torch.int64",
            "ordered_tensor_sha256": tensor_list_digest(values),
        }
    return {"models": models, "calibrations": calibrations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    plan_path = Path(args.plan_file).resolve()
    actual_sha = sha256_file(plan_path)
    if actual_sha != args.expected_plan_sha256:
        raise PreflightError(
            f"plan SHA256 mismatch: {actual_sha} != {args.expected_plan_sha256}"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "plan_file": str(plan_path),
        "plan_sha256": actual_sha,
        "artifacts": verify_artifacts(plan),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
