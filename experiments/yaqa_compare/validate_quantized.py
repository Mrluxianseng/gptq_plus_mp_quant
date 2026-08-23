#!/usr/bin/env python3
"""Strictly validate a complete raw YAQA QTIP output directory."""

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


TOKEN_PATH = Path(
    "/minimax-avatar-new/zhangqian/realq/gptq_plus/cache/tokens/"
    "Llama-3.2-1B_wikitext2_train_n256_sl2048_seed1.pt"
)
TOKEN_SHA256 = (
    "3090c0cae6c16fd8cd04b23f08e82a7be7d2cb4f6a7ceb995a50bc4472a9ee41"
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


def _projection_shapes(config) -> dict[str, tuple[int, int]]:
    hidden = int(config.hidden_size)
    head_dim = int(
        getattr(config, "head_dim", hidden // config.num_attention_heads)
    )
    kv_hidden = int(config.num_key_value_heads) * head_dim
    intermediate = int(config.intermediate_size)
    return {
        "q": (hidden, hidden),
        "k": (kv_hidden, hidden),
        "v": (kv_hidden, hidden),
        "o": (hidden, hidden),
        "up": (intermediate, hidden),
        "gate": (intermediate, hidden),
        "down": (hidden, intermediate),
    }


def _expected_akv(setting: str, layer_count: int) -> dict[str, Any]:
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


def _validate_hessian_report(
    report: dict[str, Any],
    model: Path,
    hessian_dir: Path,
    setting: str,
    layer_count: int,
    projection_names: set[str],
) -> None:
    if (
        report.get("schema_version") != 2
        or report.get("status") != "validated"
        or report.get("model") != str(model)
        or Path(report.get("hessian_dir", "")).resolve() != hessian_dir
        or report.get("layer_count") != layer_count
        or report.get("tensor_count") != 14 * layer_count
    ):
        raise RuntimeError("Hessian validation report header mismatch")

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
        "akv": _expected_akv(setting, layer_count),
    }
    manifest = report.get("manifest")
    if manifest != expected_manifest:
        raise RuntimeError("Hessian validation manifest mismatch")

    expected_paths = {
        hessian_dir / f"{layer_index}_{projection}_{side}.pt"
        for layer_index in range(layer_count)
        for projection in projection_names
        for side in ("hin", "hout")
    }
    records = report.get("tensors")
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise RuntimeError("Hessian tensor validation record count mismatch")
    records_by_path = {
        Path(record.get("path", "")).resolve(): record for record in records
    }
    if set(records_by_path) != expected_paths:
        raise RuntimeError("Hessian tensor validation path set mismatch")
    for path, record in records_by_path.items():
        if (
            not path.is_file()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256_file(path)
        ):
            raise RuntimeError(f"Hessian tensor changed after validation: {path}")


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{name} must be scalar")
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=("W4A16KV16", "W3A16KV16", "W2A4KV4"),
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--hessian-validation", required=True)
    parser.add_argument("--lut", required=True)
    parser.add_argument(
        "--scale-override",
        type=float,
        default=None,
        help=(
            "Expected QTIP scale. Omit only for the historical canonical "
            "W4/W3=1.0, W2=0.9 protocol."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = Path(args.model).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    hessian_validation_path = Path(args.hessian_validation).resolve()
    lut_path = Path(args.lut).resolve()
    config = AutoConfig.from_pretrained(model)
    layer_count = int(config.num_hidden_layers)
    projection_shapes = _projection_shapes(config)
    setting_values = {
        "W4A16KV16": (4, 1.0, (16, 16, 16), 1.0, False),
        "W3A16KV16": (3, 1.0, (16, 16, 16), 1.0, False),
        "W2A4KV4": (2, 0.9, (4, 4, 4), 0.9, True),
    }
    weight_bits, canonical_scale, akv_bits, clip_ratio, aware = setting_values[
        args.setting
    ]
    scale_override = (
        canonical_scale
        if args.scale_override is None
        else float(args.scale_override)
    )
    if not math.isfinite(scale_override) or scale_override <= 0:
        raise RuntimeError(
            f"Expected QTIP scale must be finite and positive: {scale_override}"
        )

    expected_files = {raw_dir / "config.pt"}
    for layer_index in range(layer_count):
        expected_files.add(raw_dir / f"{layer_index}_layernorm.pt")
        expected_files.update(
            raw_dir / f"{layer_index}_{name}.pt"
            for name in projection_shapes
        )
    actual_files = {path for path in raw_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            "Raw QTIP file set mismatch: "
            f"missing={sorted(str(p) for p in expected_files - actual_files)!r}, "
            f"extra={sorted(str(p) for p in actual_files - expected_files)!r}"
        )

    saved_config = torch.load(
        raw_dir / "config.pt", map_location="cpu", weights_only=False
    )
    if set(saved_config) != {"quant_args", "model_config"}:
        raise RuntimeError("Raw QTIP config.pt keys mismatch")
    quant_args = saved_config["quant_args"]
    model_config = saved_config["model_config"]
    quip_params = model_config.quip_params
    hessian_validation = json.loads(
        hessian_validation_path.read_text(encoding="utf-8")
    )
    hessian_dir = Path(hessian_validation["hessian_dir"]).resolve()
    _validate_hessian_report(
        hessian_validation,
        model,
        hessian_dir,
        args.setting,
        layer_count,
        set(projection_shapes),
    )
    if _sha256_file(TOKEN_PATH) != TOKEN_SHA256:
        raise RuntimeError("Shared 256x2048 calibration token artifact changed")

    expected_args = {
        "base_model": str(model),
        "save_path": str(raw_dir),
        "hess_path": str(hessian_dir),
        "seed": 0,
        "num_cpu_threads": 8,
        "ctx_size": 2048,
        "sigma_reg": 1e-2,
        "scale_override": scale_override,
        "codebook": "bitshift",
        "ft_epochs": 0,
        "td_x": 16,
        "td_y": 16,
        "L": 16,
        "K": weight_bits,
        "V": 2,
        "tlut_bits": 9,
        "decode_mode": "quantlut_sym",
        "calib_tokens_path": str(TOKEN_PATH),
        "calib_tokens_sha256": TOKEN_SHA256,
        "calib_n_seqs": 256,
        "a_bits": akv_bits[0],
        "k_bits": akv_bits[1],
        "v_bits": akv_bits[2],
        "akv_groupsize": -1,
        "akv_clip_ratio": clip_ratio,
        "akv_aware_hessian": aware,
        "use_fp64": False,
        "no_use_buffered": False,
        "lowmem_ldlq": False,
        "ft_grad_ckpt": False,
        "ft_train_lut": False,
        "split_for_tp": False,
        "skip_list": None,
        "tp_rank": 8,
    }
    argument_mismatches = {
        key: (getattr(quant_args, key, None), expected)
        for key, expected in expected_args.items()
        if getattr(quant_args, key, None) != expected
    }
    if argument_mismatches:
        raise RuntimeError(
            f"Raw QTIP quant_args mismatch: {argument_mismatches!r}"
        )

    expected_quip = {
        "codebook": "bitshift",
        "codebook_version": 0,
        "L": 16,
        "K": weight_bits,
        "V": 2,
        "tlut_bits": 9,
        "decode_mode": "quantlut_sym",
        "td_x": 16,
        "td_y": 16,
        "split_for_tp": False,
        "skip_list": None,
        "a_bits": akv_bits[0],
        "k_bits": akv_bits[1],
        "v_bits": akv_bits[2],
        "akv_groupsize": -1,
        "akv_clip_ratio": clip_ratio,
        "akv_aware_hessian": aware,
        "calib_tokens_path": str(TOKEN_PATH),
        "calib_tokens_sha256": TOKEN_SHA256,
        "calib_n_seqs": 256,
        "calib_ctx_size": 2048,
        "torch_float32_matmul_precision": "highest",
        "cuda_matmul_fp32_precision": "ieee",
    }
    if set(quip_params) != set(expected_quip):
        raise RuntimeError(
            "Raw QTIP quip_params key set mismatch: "
            f"missing={sorted(set(expected_quip) - set(quip_params))!r}, "
            f"extra={sorted(set(quip_params) - set(expected_quip))!r}"
        )
    quip_mismatches = {
        key: (quip_params.get(key), expected)
        for key, expected in expected_quip.items()
        if quip_params.get(key) != expected
    }
    if quip_mismatches:
        raise RuntimeError(f"Raw QTIP quip_params mismatch: {quip_mismatches!r}")

    frozen_lut = torch.load(lut_path, map_location="cpu", weights_only=True)
    expected_lut = frozen_lut.to(torch.bfloat16)
    file_records = []
    for layer_index in range(layer_count):
        layernorm_path = raw_dir / f"{layer_index}_layernorm.pt"
        layernorm = torch.load(
            layernorm_path, map_location="cpu", weights_only=True
        )
        if set(layernorm) != {
            "input_layernorm",
            "post_attention_layernorm",
        }:
            raise RuntimeError(f"{layernorm_path}: keys mismatch")
        for name, value in layernorm.items():
            if (
                value.dtype != torch.bfloat16
                or tuple(value.shape) != (int(config.hidden_size),)
                or not torch.isfinite(value).all()
            ):
                raise RuntimeError(f"{layernorm_path}: invalid {name}")
        file_records.append(
            {
                "path": str(layernorm_path),
                "bytes": layernorm_path.stat().st_size,
                "sha256": _sha256_file(layernorm_path),
            }
        )

        for projection, (out_features, in_features) in projection_shapes.items():
            path = raw_dir / f"{layer_index}_{projection}.pt"
            value = torch.load(path, map_location="cpu", weights_only=True)
            required_keys = {
                "trellis",
                "SU",
                "SV",
                "Wscale",
                "proxy_err",
                "tlut",
                "rcp",
                "tp_rank",
            }
            if set(value) != required_keys:
                raise RuntimeError(f"{path}: keys mismatch")
            expected_trellis_shape = (
                (out_features // 16) * (in_features // 16),
                16 * weight_bits,
            )
            if (
                value["trellis"].dtype != torch.int16
                or tuple(value["trellis"].shape) != expected_trellis_shape
            ):
                raise RuntimeError(f"{path}: invalid trellis")
            if (
                value["SU"].dtype != torch.bfloat16
                or tuple(value["SU"].shape) != (in_features,)
                or not torch.equal(
                    value["SU"].abs(), torch.ones_like(value["SU"])
                )
                or value["SV"].dtype != torch.bfloat16
                or tuple(value["SV"].shape) != (out_features,)
                or not torch.equal(
                    value["SV"].abs(), torch.ones_like(value["SV"])
                )
            ):
                raise RuntimeError(f"{path}: invalid SU/SV")
            if _scalar(value["Wscale"], f"{path}: Wscale") <= 0:
                raise RuntimeError(f"{path}: Wscale must be positive")
            _scalar(value["proxy_err"], f"{path}: proxy_err")
            if (
                value["tlut"].dtype != torch.bfloat16
                or not torch.equal(value["tlut"], expected_lut)
                or int(value["rcp"]) != 0
                or int(value["tp_rank"]) != 8
            ):
                raise RuntimeError(f"{path}: invalid LUT/parallel metadata")
            file_records.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "trellis_shape": list(value["trellis"].shape),
                    "wscale": _scalar(value["Wscale"], f"{path}: Wscale"),
                    "proxy_err": _scalar(
                        value["proxy_err"], f"{path}: proxy_err"
                    ),
                }
            )

    report = {
        "schema_version": 1,
        "status": "validated",
        "model": str(model),
        "setting": args.setting,
        "scale_override": scale_override,
        "raw_dir": str(raw_dir),
        "hessian_validation": str(hessian_validation_path),
        "hessian_validation_sha256": _sha256_file(hessian_validation_path),
        "lut": str(lut_path),
        "lut_sha256": _sha256_file(lut_path),
        "layer_count": layer_count,
        "projection_count": 7 * layer_count,
        "file_count": len(actual_files),
        "total_bytes": sum(path.stat().st_size for path in actual_files),
        "files": file_records,
    }
    _write_json_atomic(Path(args.output).resolve(), report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "model",
                    "setting",
                    "file_count",
                    "projection_count",
                    "total_bytes",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
