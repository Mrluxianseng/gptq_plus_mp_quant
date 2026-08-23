from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "validate_guided_saliency.py"
SPEC = importlib.util.spec_from_file_location("validate_guided_saliency", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _model(tmp_path: Path, layers: int = 2) -> Path:
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"num_hidden_layers": layers}),
        encoding="utf-8",
    )
    return path


def _write_layer(path: Path, shape=(2, 3, 4), dtype=torch.bfloat16):
    payload = {
        name: torch.ones(shape, dtype=dtype)
        for name in validator.EXPECTED_MODULES
    }
    torch.save(payload, path)


def test_complete_cache_is_valid(tmp_path):
    model = _model(tmp_path)
    cache = tmp_path / "saliency"
    cache.mkdir()
    _write_layer(cache / "l0.pt")
    _write_layer(cache / "l1.pt")
    result = validator.validate(
        model=model,
        saliency_dir=cache,
        nsamples=2,
        seq_len=3,
        num_groups=4,
        include_sha256=True,
    )
    assert result["valid"]
    assert result["validated_files"] == 2
    assert all("sha256" in item for item in result["files"])


def test_partial_or_wrong_shaped_cache_is_rejected(tmp_path):
    model = _model(tmp_path)
    cache = tmp_path / "saliency"
    cache.mkdir()
    _write_layer(cache / "l0.pt", shape=(1, 3, 4))
    result = validator.validate(
        model=model,
        saliency_dir=cache,
        nsamples=2,
        seq_len=3,
        num_groups=4,
        include_sha256=False,
    )
    assert not result["valid"]
    assert any("missing layer files" in error for error in result["errors"])
    assert any("shape=" in error for error in result["errors"])
