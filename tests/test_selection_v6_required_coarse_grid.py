from __future__ import annotations

from copy import deepcopy

from experiments.realq_fullmodel_retune_20260817 import selection_v6_run15 as select


def _row(lrs: list[float], *, ready: bool = True) -> dict[str, object]:
    return {
        "branch": "single_linear",
        "config": "qwen3-8b_w3a16",
        "aggregates": [{"lr": lr} for lr in lrs],
        "suggestions": [{"reason": "refine_log_bracket", "lr": 2e-6}],
        "selected_lr": 1e-6 if ready else None,
        "ready": ready,
    }


def test_incomplete_coarse_grid_defers_refinement_and_selection() -> None:
    payload = {
        "counts": {"ready": 1},
        "rows": [_row([1e-6, 1e-5])],
    }

    result = select._apply_required_coarse_grid(deepcopy(payload))
    row = result["rows"][0]

    assert result["counts"]["ready"] == 0
    assert result["coarse_grid_complete_groups"] == 0
    assert row["ready"] is False
    assert row["selected_lr"] is None
    assert row["coarse_grid_gate"] is False
    assert row["deferred_local_suggestions"] == payload["rows"][0]["suggestions"]
    assert [item["lr"] for item in row["suggestions"]] == [
        5e-7,
        3e-6,
        7e-6,
        3e-5,
        5e-5,
    ]
    assert all(
        item["reason"] == "complete_user_required_coarse_grid"
        for item in row["suggestions"]
    )


def test_complete_coarse_grid_preserves_core_selection() -> None:
    payload = {
        "counts": {"ready": 1},
        "rows": [_row(list(select.REQUIRED_COARSE_LRS))],
    }

    result = select._apply_required_coarse_grid(deepcopy(payload))
    row = result["rows"][0]

    assert result["counts"]["ready"] == 1
    assert result["coarse_grid_complete_groups"] == 1
    assert row["coarse_grid_gate"] is True
    assert row["missing_coarse_lrs"] == []
    assert row["ready"] is True
    assert row["selected_lr"] == 1e-6
    assert row["suggestions"] == payload["rows"][0]["suggestions"]


def test_coarse_grid_matching_tolerates_float_serialization_roundoff() -> None:
    perturbed = [
        value * (1.0 + 5e-14) for value in select.REQUIRED_COARSE_LRS
    ]
    payload = {"counts": {"ready": 1}, "rows": [_row(perturbed)]}

    result = select._apply_required_coarse_grid(payload)

    assert result["rows"][0]["coarse_grid_gate"] is True
