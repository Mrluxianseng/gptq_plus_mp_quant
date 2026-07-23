#!/usr/bin/env python3
"""Create a deterministic tiny Llama checkpoint and calibration-token caches.

The checkpoint is small enough for repeated old/new REAL-Q integration runs
while retaining real Llama attention, RoPE, grouped-query attention, MLP, final
norm, and LM-head behavior. Its dimensions deliberately cross a 128-column
block boundary so group-128 and block-GD refreshes are both exercised.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from utils.reproducibility import configure_reproducibility


_DIMENSION_FIELDS = {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size",
    "num_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
}


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


def _token_cache_names(args: argparse.Namespace) -> list[str]:
    model_name = os.path.basename(os.path.abspath(args.output))
    return [
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


def _token_payload_matches(
    existing: object,
    expected: list[torch.Tensor],
) -> bool:
    return (
        isinstance(existing, list)
        and len(existing) == len(expected)
        and all(
            isinstance(left, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right)
            for left, right in zip(existing, expected)
        )
    )


def save_token_caches(
    args: argparse.Namespace,
    tokens: list[torch.Tensor],
    *,
    overwrite: bool = True,
) -> list[str]:
    token_dir = os.path.join(args.cache_dir, "tokens")
    os.makedirs(token_dir, exist_ok=True)
    paths = [
        os.path.join(token_dir, name)
        for name in _token_cache_names(args)
    ]
    # Validate every collision before writing any missing sibling. This keeps
    # the legacy/refactored cache set logically transactional: a bad second
    # name cannot leave a newly-created first name behind.
    if not overwrite:
        for path in paths:
            if not os.path.isfile(path):
                continue
            existing = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            if not _token_payload_matches(existing, tokens):
                raise RuntimeError(
                    "Refusing to overwrite a non-matching calibration-token "
                    f"cache in --tokens-only mode: {path}"
                )

    for path in paths:
        if os.path.isfile(path) and not overwrite:
            continue
        tmp_path = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save(tokens, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return paths


def validate_existing_fixture(args: argparse.Namespace) -> dict:
    """Validate the immutable model side of a ``--tokens-only`` fixture.

    Token caches are global calibration samples and deliberately do not depend
    on distributed world size.  Before adding another ``nsamples`` cache, prove
    that the requested architectural/vocabulary/sequence parameters describe
    the existing fixture exactly.  No model or tokenizer state is loaded onto
    a torch module, and this function does not write to ``args.output``.
    """

    output = os.path.abspath(args.output)
    if not os.path.isdir(output):
        raise FileNotFoundError(
            f"--tokens-only requires an existing fixture directory: {output}"
        )

    config_path = os.path.join(output, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"--tokens-only requires an existing model config: {config_path}"
        )
    model_artifacts = []
    for pattern in (
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
    ):
        model_artifacts.extend(glob.glob(os.path.join(output, pattern)))
    if not model_artifacts:
        raise FileNotFoundError(
            "--tokens-only requires existing model weights under "
            f"{output}"
        )

    config = LlamaConfig.from_pretrained(output, local_files_only=True)
    if config.model_type != "llama":
        raise ValueError(
            "--tokens-only only supports the deterministic tiny Llama "
            f"fixture, got model_type={config.model_type!r}"
        )

    mismatches = []
    for cli_name, config_name in _DIMENSION_FIELDS.items():
        requested = int(getattr(args, cli_name))
        actual = int(getattr(config, config_name))
        if requested != actual:
            mismatches.append(
                f"{cli_name}: requested {requested}, existing config {actual}"
            )
    if int(args.seq_len) > int(config.max_position_embeddings):
        mismatches.append(
            "seq_len: requested "
            f"{args.seq_len}, max_position_embeddings "
            f"{config.max_position_embeddings}"
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        output,
        local_files_only=True,
    )
    tokenizer_vocab = int(tokenizer.vocab_size)
    tokenizer_size = len(tokenizer)
    if tokenizer_vocab != int(args.vocab_size):
        mismatches.append(
            "tokenizer.vocab_size: requested "
            f"{args.vocab_size}, existing tokenizer {tokenizer_vocab}"
        )
    if tokenizer_size != int(args.vocab_size):
        mismatches.append(
            "len(tokenizer): requested "
            f"{args.vocab_size}, existing tokenizer {tokenizer_size}"
        )

    manifest_path = os.path.join(output, "alignment_fixture.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "--tokens-only requires the original fixture manifest: "
            f"{manifest_path}"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_dimensions = manifest.get("dimensions")
    if not isinstance(manifest_dimensions, dict):
        mismatches.append("alignment_fixture.json: missing dimensions mapping")
    else:
        for cli_name in _DIMENSION_FIELDS:
            requested = int(getattr(args, cli_name))
            actual = manifest_dimensions.get(cli_name)
            if actual != requested:
                mismatches.append(
                    "alignment_fixture.json "
                    f"{cli_name}: requested {requested}, manifest {actual!r}"
                )
    if manifest.get("seq_len") != int(args.seq_len):
        mismatches.append(
            "alignment_fixture.json seq_len: requested "
            f"{args.seq_len}, manifest {manifest.get('seq_len')!r}"
        )
    if manifest.get("model_seed") != int(args.model_seed):
        mismatches.append(
            "alignment_fixture.json model_seed: requested "
            f"{args.model_seed}, manifest {manifest.get('model_seed')!r}"
        )
    if manifest.get("calibration_seed") != int(args.seed):
        mismatches.append(
            "alignment_fixture.json calibration_seed: requested "
            f"{args.seed}, manifest {manifest.get('calibration_seed')!r}"
        )

    if mismatches:
        raise ValueError(
            "--tokens-only fixture validation failed:\n  - "
            + "\n  - ".join(mismatches)
        )
    return manifest


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
    parser.add_argument(
        "--tokens-only",
        action="store_true",
        help=(
            "Add deterministic calibration-token caches for an existing "
            "alignment fixture without rewriting its model, tokenizer, "
            "config, or manifest."
        ),
    )
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

    if args.tokens_only:
        try:
            validate_existing_fixture(args)
            token_paths = save_token_caches(
                args,
                create_tokens(args),
                overwrite=False,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        result = {
            "mode": "tokens_only",
            "model_path": os.path.abspath(args.output),
            "calibration_seed": args.seed,
            "nsamples": args.nsamples,
            "seq_len": args.seq_len,
            "token_cache_paths": token_paths,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

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
