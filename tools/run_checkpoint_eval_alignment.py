#!/usr/bin/env python3
"""Run the production KL/PPL evaluator on two quantized checkpoints.

The smoke uses deterministic synthetic token IDs so it does not require a
network dataset download.  It still executes the real decoder, restored
rotation/A/K/V runtime wrappers, next-token perplexity shift, and
full-vocabulary KL implementation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig

from realq import akv, pipeline
from realq.alignment import symmetric_relative_difference
from realq.config import Config
from utils import checkpoint_utils, eval_utils, model_utils


def evaluate_checkpoint(
    *,
    model_path: str,
    checkpoint_path: str,
    input_ids: torch.Tensor,
    seq_len: int,
) -> dict[str, float]:
    cfg = Config(
        model=model_path,
        seq_len=seq_len,
        eval_seq_len=seq_len,
        nsamples=input_ids.numel() // seq_len,
        load_qmodel_path=checkpoint_path,
        skip_eval=True,
        kl_topk=-1,
    )
    checkpoint = checkpoint_utils.load_quantized_checkpoint(checkpoint_path)
    checkpoint_utils.apply_runtime_manifest(cfg, checkpoint)
    checkpoint_utils.validate_artifact_identity(cfg, checkpoint)
    analyzer = model_utils.ModelAnalyzer(model_path, seq_len)
    checkpoint_utils.validate_artifact_identity(
        cfg,
        checkpoint,
        model=analyzer.model,
        tokenizer=analyzer.tokenizer,
    )
    testenc = SimpleNamespace(input_ids=input_ids)

    # This is the same ordering as pipeline.run: FP reference before restoring
    # the rotated/quantized student weights.
    reference_hidden, reference_ids = eval_utils._get_logits(
        cfg, analyzer, testenc, torch.device("cuda")
    )
    if not torch.equal(
        reference_ids.reshape(1, -1).cpu(), input_ids.cpu()
    ):
        raise RuntimeError("reference evaluator changed the input token IDs")
    original_head = copy.deepcopy(analyzer.model.lm_head)

    pipeline._prepare_loaded_runtime_wrappers(cfg, analyzer)
    akv.install_actquant_wrappers(analyzer)
    checkpoint_utils.load_model_state(analyzer.model, checkpoint)
    akv.setup_aware_pre_quant(analyzer, cfg)
    akv.setup_unaware_post_quant(analyzer, cfg)

    ppl, kl = eval_utils._kl_ppl_eval(
        cfg,
        analyzer,
        original_head,
        testenc,
        reference_hidden,
    )
    return {"ppl": float(ppl), "kl": float(kl)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()
    if args.seq_len < 2 or args.nsamples <= 0:
        parser.error("--seq-len must be >=2 and --nsamples must be positive")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    vocab_size = int(
        AutoConfig.from_pretrained(args.model).vocab_size
    )
    if vocab_size <= 4:
        parser.error("model vocabulary must contain IDs above the special range")
    # The tiny alignment fixture reserves IDs 0..3 for special tokens.
    input_ids = torch.randint(
        4,
        vocab_size,
        (1, args.seq_len * args.nsamples),
        generator=generator,
        dtype=torch.long,
    )
    left = evaluate_checkpoint(
        model_path=args.model,
        checkpoint_path=args.left,
        input_ids=input_ids,
        seq_len=args.seq_len,
    )
    right = evaluate_checkpoint(
        model_path=args.model,
        checkpoint_path=args.right,
        input_ids=input_ids,
        seq_len=args.seq_len,
    )
    differences = {
        metric: symmetric_relative_difference(left[metric], right[metric])
        for metric in ("ppl", "kl")
    }
    report = {
        "model": str(Path(args.model).resolve()),
        "left_checkpoint": str(Path(args.left).resolve()),
        "right_checkpoint": str(Path(args.right).resolve()),
        "seq_len": args.seq_len,
        "nsamples": args.nsamples,
        "seed": args.seed,
        "threshold": args.threshold,
        "left": left,
        "right": right,
        "symmetric_relative_differences": differences,
        "passed": all(value < args.threshold for value in differences.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
