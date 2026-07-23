from __future__ import annotations

import json

import pytest

from realq.alignment import (
    RefreshStep,
    RefreshTraceWriter,
    compare_refresh_traces,
    symmetric_relative_difference,
)


def _write_trace(path, losses, *, samples=(0, 2), alpha=0.5):
    with RefreshTraceWriter(
        str(path),
        implementation="test",
        run_id=path.stem,
        config={"seed": 7},
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
