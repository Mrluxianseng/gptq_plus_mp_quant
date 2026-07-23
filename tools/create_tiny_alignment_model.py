#!/usr/bin/env python3
"""Create a deterministic tiny Llama checkpoint and calibration-token caches.

The checkpoint is small enough for repeated old/new REAL-Q integration runs
while retaining real Llama attention, RoPE, grouped-query attention, MLP, final
norm, and LM-head behavior. Its dimensions deliberately cross a 128-column
block boundary so group-128 and block-GD refreshes are both exercised.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from utils.reproducibility import configure_reproducibility


def build_tokenizer(vocab_size: int) -> PreTrainedTokenizerFast:
    if vocab_size < 16:
        raise ValueError("vocab_size must be at least 16")
    vocab = {
        "[PAD]": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
    }
    vocab.update({f"tok_{idx}": idx for idx in range(4, vocab_size)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="[PAD]",
    )


def create_checkpoint(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    configure_reproducibility(args.model_seed, deterministic=True)
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=max(args.seq_len * 2, 64),
        rms_norm_eps=1e-6,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    model.save_pretrained(args.output, safe_serialization=True)
    build_tokenizer(args.vocab_size).save_pretrained(args.output)


def create_tokens(args: argparse.Namespace) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    tokens = []
    for _ in range(args.nsamples):
        sample = torch.randint(
            low=4,
            high=args.vocab_size,
            size=(args.seq_len,),
            generator=generator,
            dtype=torch.long,
        )
        sample[0] = 1
        tokens.append(sample)
    return tokens


def save_token_caches(args: argparse.Namespace, tokens: list[torch.Tensor]) -> list[str]:
    token_dir = os.path.join(args.cache_dir, "tokens")
    os.makedirs(token_dir, exist_ok=True)
    model_name = os.path.basename(os.path.abspath(args.output))
    names = [
        # Legacy name before the seed-key correction.
        f"{model_name}-wikitext2_s{args.nsamples}_blk{args.seq_len}.pt",
        # Seed-aware legacy name.
        (
            f"{model_name}-wikitext2_s{args.nsamples}_blk{args.seq_len}"
            f"_seed{args.seed}.pt"
        ),
        # Refactored Config.tokens_cache_path convention.
        (
            f"{model_name}_wikitext2_train_n{args.nsamples}"
            f"_sl{args.seq_len}_seed{args.seed}.pt"
        ),
    ]
    paths = []
    for name in names:
        path = os.path.join(token_dir, name)
        torch.save(tokens, path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-seed", type=int, default=20260724)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    args = parser.parse_args()

    if args.hidden_size % args.num_attention_heads:
        parser.error("hidden-size must be divisible by num-attention-heads")
    if args.num_attention_heads % args.num_key_value_heads:
        parser.error("num-attention-heads must be divisible by num-key-value-heads")
    if args.hidden_size < 256 or args.intermediate_size < 256:
        parser.error(
            "hidden/intermediate dimensions must cross the 128-column "
            "integration-test block boundary"
        )

    create_checkpoint(args)
    token_paths = save_token_caches(args, create_tokens(args))
    manifest = {
        "model_path": os.path.abspath(args.output),
        "model_seed": args.model_seed,
        "calibration_seed": args.seed,
        "nsamples": args.nsamples,
        "seq_len": args.seq_len,
        "dimensions": {
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_layers": args.num_layers,
            "num_attention_heads": args.num_attention_heads,
            "num_key_value_heads": args.num_key_value_heads,
        },
        "token_cache_paths": token_paths,
    }
    manifest_path = os.path.join(args.output, "alignment_fixture.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
