from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig

from tools import create_tiny_alignment_model as fixture_tool
from utils import data_utils


def _write_existing_fixture(output: Path) -> None:
    output.mkdir()
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config.save_pretrained(output)
    fixture_tool.build_tokenizer(512).save_pretrained(output)
    # Token-only validation checks that the immutable checkpoint artifact is
    # present, but deliberately does not load a potentially large model.
    (output / "model.safetensors").write_bytes(b"immutable-model-sentinel")
    manifest = {
        "model_path": str(output.resolve()),
        "model_seed": 20260724,
        "calibration_seed": 7,
        "nsamples": 8,
        "seq_len": 16,
        "dimensions": {
            "vocab_size": 512,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_layers": 3,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
        },
        "token_cache_paths": [],
    }
    (output / "alignment_fixture.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_tokens_only(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    cache_dir: Path,
    *extra: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_tiny_alignment_model.py",
            "--output",
            str(output),
            "--cache-dir",
            str(cache_dir),
            "--tokens-only",
            "--nsamples",
            "16",
            *extra,
        ],
    )
    fixture_tool.main()


def test_tokens_only_adds_all_entrypoint_caches_without_mutating_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "tiny_llama"
    cache_dir = tmp_path / "cache"
    _write_existing_fixture(output)
    immutable_before = {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file()
    }
    immutable_mtimes_before = {
        path.name: path.stat().st_mtime_ns
        for path in output.iterdir()
        if path.is_file()
    }

    _run_tokens_only(monkeypatch, output, cache_dir)

    assert {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file()
    } == immutable_before
    assert {
        path.name: path.stat().st_mtime_ns
        for path in output.iterdir()
        if path.is_file()
    } == immutable_mtimes_before
    expected_names = {
        "tiny_llama-wikitext2_s16_blk16.pt",
        "tiny_llama-wikitext2_s16_blk16_seed7.pt",
        "tiny_llama_wikitext2_train_n16_sl16_seed7.pt",
    }
    cache_paths = sorted((cache_dir / "tokens").glob("*.pt"))
    assert {path.name for path in cache_paths} == expected_names
    payloads = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in cache_paths
    ]
    assert all(len(payload) == 16 for payload in payloads)
    assert all(
        torch.equal(left, right)
        for payload in payloads[1:]
        for left, right in zip(payloads[0], payload)
    )
    assert all(
        token.shape == (16,) and token.dtype == torch.long
        for token in payloads[0]
    )
    assert not list((cache_dir / "tokens").glob("*.tmp.*"))

    # An exact rerun validates and reuses the artifacts rather than touching
    # them, so a long-running matrix cannot race a gratuitous rewrite.
    mtimes_before = {path.name: path.stat().st_mtime_ns for path in cache_paths}
    _run_tokens_only(monkeypatch, output, cache_dir)
    assert {
        path.name: path.stat().st_mtime_ns for path in cache_paths
    } == mtimes_before
    assert {
        path.name: path.stat().st_mtime_ns
        for path in output.iterdir()
        if path.is_file()
    } == immutable_mtimes_before

    # Calibration caches describe the global sample sequence, not a rank-local
    # shard, so the same three paths must serve every rank in a world-size-8
    # run without touching the offline dataset.
    monkeypatch.setattr(
        data_utils,
        "_get_dataset",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("shared token-cache hit unexpectedly fetched data")
        ),
    )
    for _simulated_rank in range(8):
        for path in cache_paths:
            loaded = data_utils.get_tokens(
                "wikitext2",
                "train",
                object(),
                16,
                16,
                str(path),
                seed=7,
            )
            assert all(
                torch.equal(left, right)
                for left, right in zip(payloads[0], loaded)
            )


@pytest.mark.parametrize(
    "extra",
    [
        ("--seq-len", "32"),
        ("--vocab-size", "256"),
        ("--model-seed", "1"),
        ("--seed", "8"),
    ],
)
def test_tokens_only_rejects_fixture_identity_mismatch_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, str],
) -> None:
    output = tmp_path / "tiny_llama"
    cache_dir = tmp_path / "cache"
    _write_existing_fixture(output)
    immutable_before = {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file()
    }

    with pytest.raises(SystemExit, match="2"):
        _run_tokens_only(monkeypatch, output, cache_dir, *extra)

    assert not cache_dir.exists()
    assert {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file()
    } == immutable_before


def test_tokens_only_refuses_to_overwrite_mismatched_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "tiny_llama"
    cache_dir = tmp_path / "cache"
    _write_existing_fixture(output)
    _run_tokens_only(monkeypatch, output, cache_dir)

    missing = (
        cache_dir / "tokens" / "tiny_llama-wikitext2_s16_blk16.pt"
    )
    collision = (
        cache_dir
        / "tokens"
        / "tiny_llama-wikitext2_s16_blk16_seed7.pt"
    )
    missing.unlink()
    wrong_payload = [torch.tensor([999], dtype=torch.long)]
    torch.save(wrong_payload, collision)
    collision_bytes = collision.read_bytes()

    with pytest.raises(SystemExit, match="2"):
        _run_tokens_only(monkeypatch, output, cache_dir)

    # Set-level fail-fast: validation reaches the mismatched second key before
    # recreating the missing first key.
    assert not missing.exists()
    assert collision.read_bytes() == collision_bytes
    assert not list((cache_dir / "tokens").glob("*.tmp.*"))
