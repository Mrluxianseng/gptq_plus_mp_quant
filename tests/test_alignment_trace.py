from __future__ import annotations

import json

import pytest
import torch

from realq.alignment import (
    RefreshStep,
    RefreshTraceWriter,
    compare_refresh_traces,
    refresh_step_from_metrics,
    symmetric_relative_difference,
)
from tools.run_alignment_matrix import (
    _canonical_state_dict,
    _compare_checkpoint_manifests,
)


def _write_trace(
    path,
    losses,
    *,
    samples=(0, 2),
    alpha=0.5,
    config=None,
    current_losses=None,
):
    with RefreshTraceWriter(
        str(path),
        implementation="test",
        run_id=path.stem,
        config={"seed": 7} if config is None else config,
    ) as writer:
        for block, loss in enumerate(losses):
            writer.record(
                RefreshStep(
                    layer=0,
                    module="self_attn.q_proj",
                    block=block,
                    col_start=block * 4,
                    col_end=(block + 1) * 4,
                    adam_step=block + 1,
                    loss=loss,
                    loss_current=(
                        None if current_losses is None else current_losses[block]
                    ),
                    slide_alpha=alpha,
                    sample_indices=samples,
                )
            )


def test_symmetric_relative_difference():
    assert symmetric_relative_difference(1.0, 1.0) == 0.0
    assert symmetric_relative_difference(1.0, 1.005) < 0.01
    assert symmetric_relative_difference(1.0, 1.02) > 0.01


def test_compare_refresh_traces_accepts_sub_one_percent(tmp_path):
    ref = tmp_path / "ref.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_trace(ref, [1.0, 2.0])
    _write_trace(candidate, [1.005, 2.01])
    report = compare_refresh_traces(str(ref), str(candidate))
    assert report["passed"]
    assert report["matched_steps"] == 2


def test_compare_refresh_traces_rejects_missing_or_misaligned_steps(tmp_path):
    ref = tmp_path / "ref.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_trace(ref, [1.0, 2.0])
    _write_trace(candidate, [1.0], samples=(1, 3))
    report = compare_refresh_traces(str(ref), str(candidate))
    assert not report["passed"]
    assert report["missing_steps"]
    assert report["failed_steps"]


def test_loader_rejects_duplicate_step_identity(tmp_path):
    ref = tmp_path / "ref.jsonl"
    _write_trace(ref, [1.0])
    lines = ref.read_text().splitlines()
    ref.write_text("\n".join(lines + [lines[-1]]) + "\n")
    with pytest.raises(ValueError, match="duplicate step identity"):
        compare_refresh_traces(str(ref), str(ref))


def test_compare_refresh_traces_rejects_config_or_component_mismatch(tmp_path):
    ref = tmp_path / "ref.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_trace(ref, [1.0], config={"seed": 7}, current_losses=[1.0])
    _write_trace(
        candidate,
        [1.0],
        config={"seed": 8},
        current_losses=[1.1],
    )
    report = compare_refresh_traces(str(ref), str(candidate))
    assert not report["passed"]
    assert report["config_differences"]["seed"] == {
        "reference": 7,
        "candidate": 8,
    }
    assert report["failed_steps"][0]["current_loss_relative_difference"] > 0.01


def test_compare_refresh_traces_rejects_metadata_only_files(tmp_path):
    ref = tmp_path / "ref.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    for path in (ref, candidate):
        with RefreshTraceWriter(
            str(path),
            implementation="test",
            run_id=path.stem,
            config={"seed": 7},
        ):
            pass
    report = compare_refresh_traces(str(ref), str(candidate))
    assert not report["passed"]
    assert report["matched_steps"] == 0


def test_legacy_metrics_map_to_shared_refresh_identity():
    step = refresh_step_from_metrics(
        layer=2,
        module="mlp.down_proj",
        metrics={
            "block_idx": 3,
            "col_start": 384,
            "col_end": 512,
            "mean_refresh_loss": 0.25,
            "mean_refresh_loss_current": 0.2,
            "mean_refresh_loss_next": 0.3,
            "slide_alpha": 0.5,
            "sample_indices": [7, 1, 5],
        },
    )
    assert step.identity == (2, "mlp.down_proj", 3, 384, 512, 4)
    assert step.loss == 0.25
    assert step.loss_current == 0.2
    assert step.loss_next == 0.3
    assert step.slide_alpha == 0.5
    assert step.sample_indices == (7, 1, 5)


def test_state_canonicalization_merges_only_equal_wrapper_aliases():
    weight = torch.arange(6).reshape(2, 3)
    canonical = _canonical_state_dict(
        {
            "layer.weight": weight,
            "layer.module.weight": weight.clone(),
        }
    )
    assert list(canonical) == ["layer.weight"]
    assert torch.equal(canonical["layer.weight"][1], weight)

    with pytest.raises(RuntimeError, match="unequal values"):
        _canonical_state_dict(
            {
                "layer.weight": weight,
                "layer.module.weight": weight + 1,
            }
        )


def test_checkpoint_manifest_comparison_is_semantic_and_strict(tmp_path):
    common = {
        "format": "realq.quantized_model",
        "format_version": 1,
        "base_model": "tiny",
        "runtime_quantization": {"a_bits": 4, "rotate": True},
        "weight_quantization": {
            "w_bits": 4,
            "w_groupsize": 128,
            "w_clip": True,
        },
        "artifact_identity": {"source_model": "abc"},
        "model": {"weight": torch.ones(2, 2)},
    }
    legacy = {
        **common,
        "weight_quantization": {
            **common["weight_quantization"],
            "w_method": "gptq_plus",
        },
    }
    new = {**common}
    legacy_path = tmp_path / "legacy.pt"
    new_path = tmp_path / "new.pt"
    torch.save(legacy, legacy_path)
    torch.save(new, new_path)
    assert _compare_checkpoint_manifests(legacy_path, new_path)["passed"]

    divergent = {
        **new,
        "runtime_quantization": {"a_bits": 4, "rotate": False},
    }
    torch.save(divergent, new_path)
    report = _compare_checkpoint_manifests(legacy_path, new_path)
    assert not report["passed"]
    assert "runtime_quantization" in report["differences"]
