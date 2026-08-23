#!/usr/bin/env python3
"""Isolate A/K/V contributions on one frozen hfized W2 checkpoint.

This is a short, fixed-token diagnostic.  It is deliberately separate from
the formal WikiText-2/lm-eval result and never mutates the checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from experiments.yaqa_compare.akv_aware import install_akv_quantization
from model.llama import LlamaForCausalLM


VARIANTS = (
    ("weight_only", 16, 16, 16),
    ("a4_only", 4, 16, 16),
    ("k4_only", 16, 4, 16),
    ("v4_only", 16, 16, 4),
    ("full_a4k4v4", 4, 4, 4),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


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


def _load_tokens(path: Path, num_tokens: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("diagnostic token artifact must be a non-empty list")
    first = payload[0]
    if (
        not isinstance(first, torch.Tensor)
        or first.dtype != torch.int64
        or first.ndim != 1
        or first.numel() < num_tokens
    ):
        raise RuntimeError("diagnostic token artifact has an invalid first row")
    return first[:num_tokens].unsqueeze(0).contiguous()


@torch.inference_mode()
def _evaluate_variant(
    hf_dir: Path,
    input_ids_cpu: torch.Tensor,
    *,
    a_bits: int,
    k_bits: int,
    v_bits: int,
) -> dict[str, Any]:
    model = LlamaForCausalLM.from_pretrained(
        hf_dir,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        device_map="cpu",
    )
    model.config.use_cache = False
    summary = None
    if min(a_bits, k_bits, v_bits) < 16:
        summary = install_akv_quantization(
            model,
            a_bits=a_bits,
            k_bits=k_bits,
            v_bits=v_bits,
            groupsize=-1,
            symmetric=True,
            clip_ratio=0.9,
            mode="diagnostic",
        ).as_dict()
    model.eval().to(torch.device("cuda"))
    input_ids = input_ids_cpu.to(torch.device("cuda"))
    logits = model(input_ids=input_ids, use_cache=False).logits.float()
    loss = F.cross_entropy(
        logits[:, :-1, :].transpose(1, 2),
        input_ids[:, 1:],
        reduction="mean",
    )
    if not torch.isfinite(loss):
        raise RuntimeError("diagnostic cross entropy is non-finite")
    loss_value = float(loss.item())
    last_logits = logits[0, -1]
    result = {
        "a_bits": a_bits,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "cross_entropy": loss_value,
        "pseudo_ppl": (
            math.exp(loss_value) if loss_value < math.log(float("1e300")) else None
        ),
        "last_token_logits_sha256_fp32": _sha256_tensor(last_logits),
        "runtime_akv": summary or {"mode": "n/a"},
    }
    del logits, loss, input_ids, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--tokens-sha256", required=True)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires exactly one visible GPU")
    if not 2 <= args.num_tokens <= 2048:
        raise ValueError("num-tokens must be in [2, 2048]")
    hf_dir = Path(args.hf_dir).resolve()
    token_path = Path(args.tokens).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite diagnostic: {output}")
    actual_token_sha = _sha256_file(token_path)
    if actual_token_sha != args.tokens_sha256:
        raise RuntimeError(
            f"token SHA mismatch: {actual_token_sha} != {args.tokens_sha256}"
        )
    input_ids = _load_tokens(token_path, args.num_tokens)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "diagnosed",
        "formal_result": False,
        "hf_dir": str(hf_dir),
        "token_path": str(token_path),
        "token_sha256": actual_token_sha,
        "token_row": 0,
        "num_tokens": args.num_tokens,
        "variants": {},
    }
    for name, a_bits, k_bits, v_bits in VARIANTS:
        report["variants"][name] = _evaluate_variant(
            hf_dir,
            input_ids,
            a_bits=a_bits,
            k_bits=k_bits,
            v_bits=v_bits,
        )
        print(name, json.dumps(report["variants"][name], sort_keys=True))
    _write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
