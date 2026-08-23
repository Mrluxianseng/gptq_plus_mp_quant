from __future__ import annotations

import torch

from experiments.realq_cross_model_backend_diagnostic_20260822 import runner
from experiments.realq_cross_model_backend_diagnostic_20260822.trace_driver import (
    _feature_indices,
)


def test_calibration_token_semantic_hash_is_frozen() -> None:
    for model, spec in runner.MODEL_SPECS.items():
        command = runner._stage_command(model, "capture", "sdpa")
        tokens = (
            runner.Path(runner._flags(command)["--tokens_cache_path"])
            / spec["tokens_filename"]
        )
        assert runner._token_semantic_sha256(tokens) == spec["tokens_semantic_sha256"]


def test_feature_indices_cover_endpoints() -> None:
    values = _feature_indices(4096)
    assert len(values) == 64
    assert values[0] == 0
    assert values[-1] == 4095
    assert list(values) == sorted(set(values))


def test_tensor_metrics_exact_and_scaled() -> None:
    exact = runner._tensor_metrics(torch.tensor([1.0, -2.0]), torch.tensor([1.0, -2.0]))
    assert exact["relative_rms_to_left"] == 0
    assert exact["cosine"] == 1
    scaled = runner._tensor_metrics(torch.tensor([1.0, -2.0]), torch.tensor([2.0, -4.0]))
    assert scaled["right_to_left_rms_ratio"] == 2
    assert scaled["relative_rms_to_left"] == 1
    assert scaled["cosine"] == 1


def test_trace_arms_only_differ_in_backend_and_bookkeeping() -> None:
    for model in runner.MODEL_SPECS:
        left = runner._stage_command(model, "trace", "sdpa")
        right = runner._stage_command(model, "trace", "flash_attention_4")
        assert runner._scientific_projection(left) == runner._scientific_projection(right)
        assert runner._flags(left)["--attention_backend"] == "sdpa"
        assert runner._flags(right)["--attention_backend"] == "flash_attention_4"
