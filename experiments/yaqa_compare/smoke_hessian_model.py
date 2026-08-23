#!/usr/bin/env python3
"""Check YAQA's Hessian Llama against the base HF implementation."""

from __future__ import annotations

import argparse
import gc
import json

import torch
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from experiments.yaqa_compare.akv_aware import install_akv_quantization
from llama_hess import LlamaForCausalLM as HessianLlamaForCausalLM


def _load_tokens(path: str, length: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, list) or not payload:
        raise TypeError("token artifact must be a non-empty list")
    return payload[0][:length].unsqueeze(0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--length", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    input_ids = _load_tokens(args.tokens, args.length).to(device)

    reference = HFLlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        attn_implementation="eager",
    ).to(device).eval()
    reference_logits = reference(
        input_ids, use_cache=False
    ).logits.float().cpu()
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    candidate = HessianLlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        attn_implementation="eager",
    ).to(device).eval()
    candidate_logits = candidate(
        input_ids, use_cache=False
    ).logits.float().cpu()
    difference = (reference_logits - candidate_logits).abs()
    torch.testing.assert_close(
        candidate_logits, reference_logits, rtol=5e-3, atol=5e-3
    )

    summary = install_akv_quantization(
        candidate,
        a_bits=4,
        k_bits=4,
        v_bits=4,
        groupsize=-1,
        symmetric=True,
        clip_ratio=0.9,
        mode="aware",
    )
    quantized_logits = candidate(
        input_ids, use_cache=False
    ).logits.float().cpu()
    if not torch.isfinite(quantized_logits).all():
        raise RuntimeError("A/K/V-quantized Hessian model produced non-finite logits")
    if torch.equal(quantized_logits, candidate_logits):
        raise RuntimeError("A/K/V hooks did not change Hessian-model logits")

    report = {
        "model": args.model,
        "length": args.length,
        "max_abs_hf_vs_hessian": float(difference.max()),
        "mean_abs_hf_vs_hessian": float(difference.mean()),
        "max_abs_fp_vs_akv4": float(
            (candidate_logits - quantized_logits).abs().max()
        ),
        "akv": summary.as_dict(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("YAQA Hessian-model smoke: PASS")


if __name__ == "__main__":
    main()
