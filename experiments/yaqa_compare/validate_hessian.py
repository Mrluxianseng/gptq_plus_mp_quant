#!/usr/bin/env python3
"""Strictly validate every YAQA Sketch-B Hessian tensor and its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _triangular_size(dimension: int) -> int:
    return dimension * (dimension + 1) // 2


def _dimensions(config) -> dict[str, tuple[int, int]]:
    hidden = int(config.hidden_size)
    head_dim = int(
        getattr(config, "head_dim", hidden // config.num_attention_heads)
    )
    query_hidden = int(config.num_attention_heads) * head_dim
    kv_hidden = int(config.num_key_value_heads) * head_dim
    intermediate = int(config.intermediate_size)
    return {
        "q": (hidden, query_hidden),
        "k": (hidden, kv_hidden),
        "v": (hidden, kv_hidden),
        "o": (query_hidden, hidden),
        "up": (hidden, intermediate),
        "gate": (hidden, intermediate),
        "down": (intermediate, hidden),
    }


def _validate_tensor(path: Path, dimension: int) -> dict[str, Any]:
    tensor = torch.load(
        path, map_location="cpu", weights_only=True, mmap=True
    )
    expected_numel = _triangular_size(dimension)
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or tensor.ndim != 1
        or tensor.numel() != expected_numel
    ):
        raise RuntimeError(
            f"{path}: expected FP32[{expected_numel}], got "
            f"{type(tensor).__qualname__}, {getattr(tensor, 'dtype', None)}, "
            f"{getattr(tensor, 'shape', None)}"
        )
    for start in range(0, tensor.numel(), 1_048_576):
        if not torch.isfinite(tensor[start : start + 1_048_576]).all():
            raise RuntimeError(f"{path}: contains non-finite values")
    diagonal_positions = (
        (torch.arange(dimension, dtype=torch.int64) + 1)
        * (torch.arange(dimension, dtype=torch.int64) + 2)
        // 2
        - 1
    )
    diagonal = tensor[diagonal_positions]
    diagonal_mean = float(diagonal.double().mean())
    if not math.isfinite(diagonal_mean) or diagonal_mean <= 0:
        raise RuntimeError(
            f"{path}: diagonal mean must be finite and positive, "
            f"got {diagonal_mean}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "dimension": dimension,
        "diagonal_mean": diagonal_mean,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hessian-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, default=256)
    parser.add_argument("--expected-seq-len", type=int, default=2048)
    parser.add_argument("--expected-calib-path", required=True)
    parser.add_argument("--expected-calib-sha256", required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--expected-batch-size-per-rank", type=int, default=2)
    parser.add_argument("--expected-deterministic", action="store_true")
    parser.add_argument("--expected-global-seed", type=int, default=1)
    parser.add_argument(
        "--expected-akv-mode", choices=("n/a", "aware"), required=True
    )
    args = parser.parse_args()

    hessian_dir = Path(args.hessian_dir).resolve()
    manifest_path = hessian_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(args.model)
    layer_count = int(config.num_hidden_layers)
    dimensions = _dimensions(config)

    if Path(manifest["model"]).resolve() != Path(args.model).resolve():
        raise RuntimeError("Hessian manifest model path mismatch")
    if Path(manifest["save_path"]).resolve() != hessian_dir:
        raise RuntimeError("Hessian manifest save path mismatch")
    if (
        Path(manifest["calib_tokens_path"]).resolve()
        != Path(args.expected_calib_path).resolve()
        or manifest["calib_tokens_sha256"] != args.expected_calib_sha256
    ):
        raise RuntimeError("Hessian manifest calibration artifact mismatch")
    if manifest["n_seqs"] != args.expected_samples:
        raise RuntimeError("Hessian manifest sample count mismatch")
    if manifest["ctx_size"] != args.expected_seq_len:
        raise RuntimeError("Hessian manifest sequence length mismatch")
    if manifest["iterations"] * manifest["global_batch_size"] != args.expected_samples:
        raise RuntimeError("Hessian manifest does not consume every sample once")
    if manifest["power_iters"] != 1 or manifest["hessian_sketch"] != "B":
        raise RuntimeError("Hessian manifest is not one-iteration Sketch-B")
    if (
        manifest["world_size"] != args.expected_world_size
        or manifest["batch_size_per_rank"]
        != args.expected_batch_size_per_rank
        or manifest["global_batch_size"]
        != args.expected_world_size * args.expected_batch_size_per_rank
    ):
        raise RuntimeError("Hessian manifest distributed batch mismatch")
    if (
        manifest["start_layer"] != 0
        or manifest["end_layer"] != layer_count
        or manifest.get("fp64_accum_effective") is not False
    ):
        raise RuntimeError("Hessian manifest layer range/accumulation mismatch")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("torch_float32_matmul_precision") != "highest"
        or manifest.get("cuda_matmul_fp32_precision") != "ieee"
    ):
        raise RuntimeError(
            "Hessian manifest does not prove IEEE FP32 matmul collection"
        )
    if manifest["akv"]["mode"] != args.expected_akv_mode:
        raise RuntimeError("Hessian manifest A/K/V mode mismatch")
    expected_determinism = (
        {
            "enabled": True,
            "global_seed": args.expected_global_seed,
            "python_hash_seed": args.expected_global_seed,
            "python_random_seed": args.expected_global_seed,
            "numpy_random_seed": args.expected_global_seed,
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_fp32_precision": "ieee",
            "cudnn_allow_tf32": False,
            "fake_target_seed_rule": "calibration_iteration_index",
        }
        if args.expected_deterministic
        else None
    )
    if manifest.get("determinism") != expected_determinism:
        raise RuntimeError("Hessian manifest determinism contract mismatch")
    expected_akv = (
        {
            "a_bits": 16,
            "k_bits": 16,
            "v_bits": 16,
            "clip_ratio": 1.0,
            "activation_input_sites": 0,
            "value_output_sites": 0,
            "post_rope_k_sites": 0,
        }
        if args.expected_akv_mode == "n/a"
        else {
            "a_bits": 4,
            "k_bits": 4,
            "v_bits": 4,
            "clip_ratio": 0.9,
            "activation_input_sites": 7 * layer_count,
            "value_output_sites": layer_count,
            "post_rope_k_sites": layer_count,
        }
    )
    expected_akv.update(
        {
            "decoder_layers": layer_count,
            "groupsize": -1,
            "symmetric": True,
            "query_quantized": False,
            "extra_qk_hadamard": False,
        }
    )
    mismatched_akv = {
        key: (manifest["akv"].get(key), expected)
        for key, expected in expected_akv.items()
        if manifest["akv"].get(key) != expected
    }
    if mismatched_akv:
        raise RuntimeError(
            f"Hessian manifest A/K/V topology mismatch: {mismatched_akv!r}"
        )

    expected_paths: dict[Path, int] = {}
    for layer_index in range(layer_count):
        for site, (input_dimension, output_dimension) in dimensions.items():
            expected_paths[
                hessian_dir / f"{layer_index}_{site}_hin.pt"
            ] = input_dimension
            expected_paths[
                hessian_dir / f"{layer_index}_{site}_hout.pt"
            ] = output_dimension
    actual_paths = set(hessian_dir.glob("*_h*.pt"))
    if actual_paths != set(expected_paths):
        missing = sorted(str(path) for path in set(expected_paths) - actual_paths)
        extra = sorted(str(path) for path in actual_paths - set(expected_paths))
        raise RuntimeError(
            f"Hessian file set mismatch: missing={missing!r}, extra={extra!r}"
        )

    records = [
        _validate_tensor(path, expected_paths[path])
        for path in sorted(expected_paths)
    ]
    report = {
        "schema_version": 2,
        "status": "validated",
        "hessian_dir": str(hessian_dir),
        "model": str(Path(args.model).resolve()),
        "layer_count": layer_count,
        "tensor_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "manifest": manifest,
        "tensors": records,
    }
    _write_json_atomic(Path(args.output).resolve(), report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "hessian_dir",
                    "layer_count",
                    "tensor_count",
                    "total_bytes",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
