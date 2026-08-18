from __future__ import annotations

from experiments.realq_fullmodel_retune_20260817 import aloss_clip_ablation as a
from experiments.realq_fullmodel_retune_20260817 import selection_a1_q4 as a1


def _result(group: str, lr: float, kl: float, replicate: int) -> dict:
    return {
        "group": group,
        "identity": f"{group}-{lr}-{replicate}",
        "status": "succeeded",
        "lr": lr,
        "kl": kl,
        "ppl": kl + 10,
        "gpu": {"uuid": "GPU-test"},
        "result_path": f"/tmp/{group}-{lr}-{replicate}.json",
    }


def _build_plan_body_from_its_frozen_code_scope() -> dict:
    # Pytest imports every test module before executing tests.  Importing the
    # V4 selector mutates the shared campaign module's CODE_INPUTS, while the
    # real controllers invoke each campaign in a fresh process.  Recreate that
    # process-local V3 input list from the immutable plan for this unit test.
    saved = a.c.CODE_INPUTS
    frozen = a._read_json(a.v3.PLAN_PATH)
    a.c.CODE_INPUTS = tuple(
        item["path"] for item in frozen["code_snapshot"]["files"]
    )
    try:
        return a._build_plan_body()
    finally:
        a.c.CODE_INPUTS = saved


def test_plan_has_eight_independently_tuned_full_model_groups() -> None:
    first = _build_plan_body_from_its_frozen_code_scope()
    second = _build_plan_body_from_its_frozen_code_scope()
    assert first == second
    assert len(first["groups"]) == 8
    for group in first["groups"].values():
        flags = a._flag_map(group["source_command"])
        assert flags["--a_loss_ratio"] == a._ratio_text(group["a_loss_ratio"])
        assert flags["--loss_slide_window"] == "true"
        assert flags["--w_groupsize"] == "128"
        assert flags["--blocksize"] == "128"
        assert flags["--backward_samples"] == "32"
        assert flags["--require_static_cache_hit"] == "true"
        assert flags["--require_reference_cache_hit"] == "true"
        assert "--quant_stop_layer" not in flags
        assert "--save_qmodel_path" not in flags


def test_selector_requires_bracket_two_high_points_and_top_two_repeats(
    monkeypatch,
) -> None:
    group = a._groups()[0]
    group_id = group["group"]
    rows = []
    for lr, kl in ((1e-6, 1.3), (1.4e-6, 1.0), (1.9e-6, 1.2), (3e-6, 1.5)):
        rows.extend([_result(group_id, lr, kl, 1), _result(group_id, lr, kl, 2)])
    monkeypatch.setattr(a, "_result_rows", lambda _: rows)
    analysis = a._analyze_group(group)
    assert analysis["high_side_gate"] is True
    assert analysis["bracket_gate"] is True
    assert analysis["repeat_gate"] is True
    assert analysis["ready"] is True
    assert analysis["selected_lr"] == 1.4e-6


def test_selector_does_not_use_two_percent_plateau_as_stop(monkeypatch) -> None:
    group = a._groups()[0]
    group_id = group["group"]
    rows = [
        _result(group_id, 0.0, 1.0, 1),
        _result(group_id, 1e-7, 1.001, 1),
    ]
    monkeypatch.setattr(a, "_result_rows", lambda _: rows)
    analysis = a._analyze_group(group)
    assert analysis["two_percent_plateau"] == [0.0, 1e-7]
    assert analysis["ready"] is False
    assert analysis["suggestions"] == [
        {"reason": "refine_zero_boundary", "lr": 1e-8}
    ]


def test_a1_selection_excludes_stopped_point95_curves(monkeypatch) -> None:
    rows = [
        {
            **group,
            "ready": True,
            "launches": 1,
            "failures": [],
        }
        for group in a._groups()
    ]
    monkeypatch.setattr(
        a,
        "_analysis",
        lambda: {
            "plan_fingerprint": "test-plan",
            "rows": rows,
        },
    )
    value = a1._selected_analysis()
    assert value["counts"] == {
        "groups": 4,
        "ready": 4,
        "launches": 4,
        "failures": 0,
    }
    assert {row["a_loss_ratio"] for row in value["rows"]} == {1.0}
