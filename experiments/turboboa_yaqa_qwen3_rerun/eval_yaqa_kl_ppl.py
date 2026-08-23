#!/usr/bin/env python3
"""Evaluate one validated YAQA_wclip artifact with the shared paper KL/PPL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tempfile
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate_artifact(
    *,
    model: Path,
    setting: str,
    hf_dir: Path,
    validation_path: Path,
    expected_validation_sha256: str,
) -> tuple[str, dict[str, Any]]:
    validation_sha = _sha256(validation_path)
    if validation_sha != expected_validation_sha256:
        raise RuntimeError("YAQA HF validation receipt hash changed")
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 2
        or report.get("status") != "validated"
        or report.get("model") != str(model)
        or report.get("setting") != setting
        or Path(report.get("hf_dir", "")).resolve() != hf_dir
        or report.get("runtime_weight_representation")
        != "dense_fake_quant"
        or report.get("packed") is not False
        or report.get("lowbit_kernel") is not False
    ):
        raise RuntimeError("YAQA HF validation receipt header mismatch")
    records = report.get("hf_files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("YAQA HF validation receipt has no files")
    declared: dict[Path, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("YAQA HF validation file record is malformed")
        path = Path(record.get("path", "")).resolve()
        if path in declared:
            raise RuntimeError(f"duplicate YAQA HF file record: {path}")
        declared[path] = record
    live = {path.resolve() for path in hf_dir.rglob("*") if path.is_file()}
    if live != set(declared):
        raise RuntimeError("YAQA HF file set changed after validation")
    for path, record in declared.items():
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(f"YAQA HF artifact changed: {path}")
    return validation_sha, report


def _load_original_lm_head(model: Path):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    index_path = model / "model.safetensors.index.json"
    weight_key = "lm_head.weight"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        shard_name = weight_map.get(weight_key)
        if shard_name is None and bool(config.tie_word_embeddings):
            # Hugging Face omits the redundant LM-head tensor for tied
            # checkpoints.  REAL-Q's ModelAnalyzer clones the input embedding
            # into an independent output head before rotation; reproduce that
            # exact source-head value without loading the full base model.
            weight_key = "model.embed_tokens.weight"
            shard_name = weight_map.get(weight_key)
        if not shard_name:
            raise RuntimeError(
                "base checkpoint index has neither lm_head.weight nor a "
                "declared tied model.embed_tokens.weight"
            )
        shard_path = model / shard_name
        weights = load_file(str(shard_path), device="cpu")
        weight = weights.get(weight_key)
    else:
        shard_path = model / "model.safetensors"
        weights = load_file(str(shard_path), device="cpu")
        weight = weights.get(weight_key)
        if weight is None and bool(config.tie_word_embeddings):
            weight_key = "model.embed_tokens.weight"
            weight = weights.get(weight_key)
    if weight is None:
        raise RuntimeError("base checkpoint has no recoverable LM-head weight")
    expected = (int(config.vocab_size), int(config.hidden_size))
    if tuple(weight.shape) != expected or not torch.isfinite(weight).all():
        raise RuntimeError(
            f"invalid base lm_head: shape={tuple(weight.shape)}, "
            f"expected={expected}"
        )
    head = torch.nn.Linear(
        expected[1],
        expected[0],
        bias=False,
        dtype=weight.dtype,
        device="cpu",
    )
    head.weight.data.copy_(weight)
    return head, shard_path, weight_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=("W3A16KV16", "W4A4KV4"),
    )
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--hf-validation", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hostname = socket.gethostname()
    if not re.fullmatch(r"j-[a-z0-9]+-master-0", hostname):
        raise RuntimeError("formal KL/PPL may run only on a Canoe pod")

    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("one KL/PPL worker must see exactly one GPU")
    model_path = Path(args.model).resolve()
    hf_dir = Path(args.hf_dir).resolve()
    validation_path = Path(args.hf_validation).resolve()
    output_path = Path(args.output).resolve()
    validation_sha, _ = _validate_artifact(
        model=model_path,
        setting=args.setting,
        hf_dir=hf_dir,
        validation_path=validation_path,
        expected_validation_sha256=args.expected_validation_sha256,
    )

    from lib.utils.unsafe_import import model_from_hf_path
    from utils import data_utils, eval_utils, model_utils

    started_at = _now()
    model, _ = model_from_hf_path(
        str(hf_dir),
        device_map={"": torch.cuda.current_device()},
    )
    model.eval()
    analyzer = model_utils.ModelAnalyzer(
        model,
        2048,
        tokenizer_source=str(model_path),
        skip_state_dict=True,
    )
    cfg = SimpleNamespace(
        model=str(model_path),
        model_name=model_path.name,
        cache_dir=str(REPO_ROOT / "cache"),
        eval_seq_len=2048,
        nsamples=256,
        rotate=True,
        rotation_seed=0,
        optimized_rotation_path=None,
        require_reference_cache_hit=True,
        kl_topk=-1,
    )
    loader = data_utils.get_loaders(
        "wikitext2",
        split="test",
        tokenizer=analyzer.tokenizer,
        seq_len=cfg.eval_seq_len,
        num_samples=cfg.nsamples,
    )
    reference_hidden, quant_head_copy = eval_utils.get_ref_logits(
        cfg,
        analyzer,
        "wikitext2",
        loader,
    )
    del quant_head_copy
    (
        original_lm_head,
        original_lm_head_shard,
        original_lm_head_weight_key,
    ) = _load_original_lm_head(model_path)
    ppl, kl_raw = eval_utils._kl_ppl_eval(
        cfg,
        analyzer,
        original_lm_head,
        loader,
        reference_hidden,
    )
    if not (
        torch.isfinite(torch.tensor(ppl))
        and torch.isfinite(torch.tensor(kl_raw))
        and ppl > 0
        and kl_raw >= 0
    ):
        raise RuntimeError(f"invalid KL/PPL: ppl={ppl}, kl={kl_raw}")

    _write_atomic(
        output_path,
        {
            "schema_version": 1,
            "status": "evaluated",
            "method": "yaqa_wclip",
            "model": str(model_path),
            "setting": args.setting,
            "hf_dir": str(hf_dir),
            "hf_validation": str(validation_path),
            "hf_validation_sha256": validation_sha,
            "hostname": hostname,
            "started_at": started_at,
            "finished_at": _now(),
            "evaluation": {
                "dataset": "wikitext2",
                "split": "test",
                "sequence_length": 2048,
                "kl_direction": "KL(FP||quantized)",
                "kl_full_vocabulary_fp32": True,
                "kl_topk": -1,
                "kl_raw": kl_raw,
                "kl_x100": kl_raw * 100.0,
                "ppl": ppl,
                "reference_cache_required": True,
                "original_lm_head_shard": str(original_lm_head_shard),
                "original_lm_head_weight_key": original_lm_head_weight_key,
                "original_lm_head_shard_sha256": _sha256(
                    original_lm_head_shard
                ),
            },
        },
    )
    print(
        json.dumps(
            {"kl_raw": kl_raw, "kl_x100": kl_raw * 100.0, "ppl": ppl},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
