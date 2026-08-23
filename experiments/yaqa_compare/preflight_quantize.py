#!/usr/bin/env python3
"""Fail-fast contract checks before a formal YAQA QTIP quantization."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig


TOKEN_PATH = Path(
    "/minimax-avatar-new/zhangqian/realq/gptq_plus/cache/tokens/"
    "Llama-3.2-1B_wikitext2_train_n256_sl2048_seed1.pt"
).resolve()
TOKEN_SHA256 = (
    "3090c0cae6c16fd8cd04b23f08e82a7be7d2cb4f6a7ceb995a50bc4472a9ee41"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_akv(setting: str, layer_count: int) -> dict:
    aware = setting == "W2A4KV4"
    bits = 4 if aware else 16
    return {
        "mode": "aware" if aware else "n/a",
        "a_bits": bits,
        "k_bits": bits,
        "v_bits": bits,
        "groupsize": -1,
        "clip_ratio": 0.9 if aware else 1.0,
        "symmetric": True,
        "decoder_layers": layer_count,
        "activation_input_sites": 7 * layer_count if aware else 0,
        "value_output_sites": layer_count if aware else 0,
        "post_rope_k_sites": layer_count if aware else 0,
        "query_quantized": False,
        "extra_qk_hadamard": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=("W4A16KV16", "W3A16KV16", "W2A4KV4"),
    )
    parser.add_argument("--hessian-dir", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lut", required=True)
    parser.add_argument("--lut-manifest", required=True)
    args = parser.parse_args()

    model = Path(args.model).resolve()
    hessian_dir = Path(args.hessian_dir).resolve()
    validation_path = Path(args.validation_report).resolve()
    output_dir = Path(args.output_dir).resolve()
    lut_path = Path(args.lut).resolve()
    lut_manifest_path = Path(args.lut_manifest).resolve()

    if output_dir.exists():
        raise RuntimeError(
            f"formal QTIP output must be a fresh path: {output_dir}"
        )
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 2 or report.get("status") != "validated":
        raise RuntimeError("Hessian validation report is not schema-v2 validated")
    if Path(report["hessian_dir"]).resolve() != hessian_dir:
        raise RuntimeError("Hessian validation report path mismatch")
    if Path(report["model"]).resolve() != model:
        raise RuntimeError("Hessian validation report model mismatch")

    model_config = AutoConfig.from_pretrained(model)
    layer_count = int(model_config.num_hidden_layers)
    if (
        report.get("model") != str(model)
        or report.get("layer_count") != layer_count
        or report.get("tensor_count") != 14 * layer_count
    ):
        raise RuntimeError("Hessian validation report model/count mismatch")

    manifest = report["manifest"]
    expected_manifest = {
        "schema_version": 2,
        "model": str(model),
        "save_path": str(hessian_dir),
        "calib_tokens_path": str(TOKEN_PATH),
        "calib_tokens_sha256": TOKEN_SHA256,
        "n_seqs": 256,
        "ctx_size": 2048,
        "world_size": 4,
        "batch_size_per_rank": 2,
        "global_batch_size": 8,
        "iterations": 32,
        "start_layer": 0,
        "end_layer": layer_count,
        "hessian_sketch": "B",
        "power_iters": 1,
        "fp64_accum_effective": False,
        "torch_float32_matmul_precision": "highest",
        "cuda_matmul_fp32_precision": "ieee",
        "akv": _expected_akv(args.setting, layer_count),
    }
    if manifest != expected_manifest:
        mismatch = {
            key: (manifest.get(key), expected)
            for key, expected in expected_manifest.items()
            if manifest.get(key) != expected
        }
        extra = sorted(set(manifest) - set(expected_manifest))
        raise RuntimeError(
            "Validated Hessian does not match quantization setting: "
            f"mismatch={mismatch!r}, extra={extra!r}"
        )

    lut_manifest = json.loads(
        lut_manifest_path.read_text(encoding="utf-8")
    )
    lut_sha256 = _sha256_file(lut_path)
    if (
        lut_manifest.get("schema_version") != 1
        or Path(lut_manifest["path"]).resolve() != lut_path
        or lut_manifest["file_sha256"] != lut_sha256
        or lut_manifest["shape"] != [512, 2]
        or lut_manifest["dtype"] != "torch.float32"
    ):
        raise RuntimeError("Frozen QTIP LUT manifest mismatch")
    tmp_lut = Path("/tmp/kmeans_9_2.pt")
    if not tmp_lut.is_file() or _sha256_file(tmp_lut) != lut_sha256:
        raise RuntimeError("/tmp QTIP LUT is missing or differs from frozen LUT")
    value = torch.load(lut_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or tuple(value.shape) != (512, 2)
        or not torch.isfinite(value).all()
    ):
        raise RuntimeError("Frozen QTIP LUT tensor is invalid")

    # Import the worker's complete rounding stack before creating the formal
    # output directory. This catches optional/missing dependencies without
    # leaving a misleading partial run behind.
    importlib.import_module("lib.algo.finetune")

    print(
        json.dumps(
            {
                "status": "ready",
                "model": str(model),
                "setting": args.setting,
                "hessian_dir": str(hessian_dir),
                "validation_report": str(validation_path),
                "validation_report_sha256": _sha256_file(validation_path),
                "lut": str(lut_path),
                "lut_sha256": lut_sha256,
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
