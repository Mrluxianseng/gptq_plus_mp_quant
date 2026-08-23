#!/usr/bin/env python3
"""Validate an hfized YAQA model and exercise generic/specialized QTIP paths."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from lib.linear import QuantizedLinear
from lib.utils.unsafe_import import model_from_hf_path


TOKEN_PATH = Path(
    "/minimax-avatar-new/zhangqian/realq/gptq_plus/cache/tokens/"
    "Llama-3.2-1B_wikitext2_train_n256_sl2048_seed1.pt"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
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


def _runtime_summary(model, setting: str, layer_count: int) -> dict[str, Any]:
    summary = getattr(model, "_yaqa_akv_quantization_summary", None)
    if setting == "W2A4KV4":
        if not isinstance(summary, dict):
            raise RuntimeError("W2 model did not restore deployed A/K/V QDQ")
        expected = {
            "mode": "deployment",
            "decoder_layers": layer_count,
            "activation_input_sites": 7 * layer_count,
            "value_output_sites": layer_count,
            "post_rope_k_sites": layer_count,
            "a_bits": 4,
            "k_bits": 4,
            "v_bits": 4,
            "groupsize": -1,
            "symmetric": True,
            "clip_ratio": 0.9,
            "query_quantized": False,
            "extra_qk_hadamard": False,
        }
        mismatch = {
            key: (summary.get(key), value)
            for key, value in expected.items()
            if summary.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"Deployed A/K/V topology mismatch: {mismatch!r}")
        return summary
    if summary is not None:
        raise RuntimeError("A16/KV16 model unexpectedly installed A/K/V QDQ")
    return {"mode": "n/a"}


@torch.no_grad()
def _load_and_smoke(
    hf_dir: Path,
    setting: str,
    layer_count: int,
    one_token: torch.Tensor,
    many_tokens: torch.Tensor,
) -> tuple[dict[str, str], dict[str, Any]]:
    model, _ = model_from_hf_path(
        str(hf_dir), device_map={"": torch.cuda.current_device()}
    )
    model.eval()
    quantized_count = sum(
        isinstance(module, QuantizedLinear) for module in model.modules()
    )
    if quantized_count != 7 * layer_count:
        raise RuntimeError(
            f"QuantizedLinear count mismatch: {quantized_count} != "
            f"{7 * layer_count}"
        )
    runtime = _runtime_summary(model, setting, layer_count)
    fingerprints = {}
    for name, input_ids in (
        ("one_token", one_token),
        ("thirty_two_tokens", many_tokens),
    ):
        logits = model(
            input_ids.cuda(),
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        ).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{name} smoke produced non-finite logits")
        fingerprints[name] = _tensor_sha256(logits)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return fingerprints, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=("W4A16KV16", "W3A16KV16", "W2A4KV4"),
    )
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--raw-validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("HF validation requires exactly one visible GPU")
    model_path = Path(args.model).resolve()
    hf_dir = Path(args.hf_dir).resolve()
    raw_validation_path = Path(args.raw_validation).resolve()
    raw_validation = json.loads(
        raw_validation_path.read_text(encoding="utf-8")
    )
    if (
        raw_validation.get("status") != "validated"
        or raw_validation.get("setting") != args.setting
        or Path(raw_validation["model"]).resolve() != model_path
    ):
        raise RuntimeError("Raw QTIP validation report mismatch")

    config = AutoConfig.from_pretrained(hf_dir)
    layer_count = int(config.num_hidden_layers)
    if not hasattr(config, "quip_params"):
        raise RuntimeError("hfized config lacks quip_params")
    raw_config = torch.load(
        Path(raw_validation["raw_dir"]) / "config.pt",
        map_location="cpu",
        weights_only=False,
    )["model_config"]
    hf_quip_params = dict(config.quip_params)
    raw_quip_params = dict(raw_config.quip_params)
    # YAQA's deployment model constructor canonicalizes the no-skip sentinel
    # from None to an empty list. They are the same quantized topology; no
    # other metadata difference is permitted.
    if raw_quip_params.get("skip_list") is None:
        raw_quip_params["skip_list"] = []
    if hf_quip_params != raw_quip_params:
        raise RuntimeError("hfized quip_params differ from raw QTIP config")

    base_tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    probe = "YAQA tokenizer parity probe: 256 sequences, 2048 tokens each."
    if (
        type(hf_tokenizer) is not type(base_tokenizer)
        or hf_tokenizer.get_vocab() != base_tokenizer.get_vocab()
        or hf_tokenizer.special_tokens_map != base_tokenizer.special_tokens_map
        or hf_tokenizer(probe)["input_ids"]
        != base_tokenizer(probe)["input_ids"]
    ):
        raise RuntimeError("hfized tokenizer differs from base tokenizer")

    calibration = torch.load(TOKEN_PATH, map_location="cpu", weights_only=True)
    many_tokens = calibration[0][:32].view(1, 32)
    one_token = calibration[0][:1].view(1, 1)
    first_fingerprints, runtime = _load_and_smoke(
        hf_dir,
        args.setting,
        layer_count,
        one_token,
        many_tokens,
    )
    second_fingerprints, second_runtime = _load_and_smoke(
        hf_dir,
        args.setting,
        layer_count,
        one_token,
        many_tokens,
    )
    if (
        first_fingerprints != second_fingerprints
        or runtime != second_runtime
    ):
        raise RuntimeError("hfized reload smoke is not deterministic")

    files = sorted(path for path in hf_dir.rglob("*") if path.is_file())
    required_names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    names = {path.name for path in files}
    if not required_names.issubset(names):
        raise RuntimeError(
            f"hfized directory misses tokenizer/config files: "
            f"{sorted(required_names - names)!r}"
        )
    if not any(
        name.endswith(".safetensors") or name == "model.safetensors.index.json"
        for name in names
    ):
        raise RuntimeError("hfized directory contains no safetensors weights")

    records = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    report = {
        "schema_version": 1,
        "status": "validated",
        "model": str(model_path),
        "setting": args.setting,
        "hf_dir": str(hf_dir),
        "raw_validation": str(raw_validation_path),
        "raw_validation_sha256": _sha256_file(raw_validation_path),
        "layer_count": layer_count,
        "quantized_linear_count": 7 * layer_count,
        "runtime_akv": runtime,
        "smoke_logits_sha256": first_fingerprints,
        "reload_deterministic": True,
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
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
                    "quantized_linear_count",
                    "runtime_akv",
                    "smoke_logits_sha256",
                    "reload_deterministic",
                    "file_count",
                    "total_bytes",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
