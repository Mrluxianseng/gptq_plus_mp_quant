from __future__ import annotations

import pytest

from . import render_master_table as renderer


def _percent(hits: int, total: int) -> str:
    return str(100.0 * hits / total)


def _base_row(comparison_id: str, method: str) -> list[str]:
    gsm_hits = {
        "GPTAQ": 1,
        "GuidedQuant": 2,
        "ResComp-C": 3,
        "REALQ-F": 4,
        "REALQ-S": 5,
    }[method]
    math_hits = gsm_hits
    humaneval_hits = 8 if method == "REALQ-S" else gsm_hits
    return [
        comparison_id,
        "model",
        "setting",
        method,
        "0.1",
        "10.0",
        "50.0",
        _percent(gsm_hits, 1319),
        _percent(math_hits, 500),
        f"0.0 / {_percent(humaneval_hits, 164)}",
        "1.0",
    ]


def _release_row(method: str) -> dict:
    # YAQA wins GSM8K and ties REALQ-S on HumanEval+; TurboBOA wins MATH.
    gsm_hits = {"efficientqat": 6, "turboboa": 7, "yaqa_wclip": 8}[method]
    math_hits = {"efficientqat": 6, "turboboa": 8, "yaqa_wclip": 7}[method]
    humaneval_hits = {"efficientqat": 6, "turboboa": 7, "yaqa_wclip": 8}[method]
    return {
        "metrics": {
            "gsm8k": float(_percent(gsm_hits, 1319)),
            "math_500": float(_percent(math_hits, 500)),
            "humaneval_plus": float(_percent(humaneval_hits, 164)),
        }
    }


def test_winner_statistics_use_available_methods_and_integer_hits():
    grouped = {
        f"C{index:02d}": [
            _base_row(f"C{index:02d}", method) for method in renderer.BASE_METHODS
        ]
        for index in range(1, 21)
    }
    release_rows = {}
    for index in range(1, 21):
        comparison_id = f"C{index:02d}"
        for method, _label in renderer.NEW_METHODS:
            if method == "efficientqat" and comparison_id in renderer.W4A4_IDS:
                continue
            release_rows[(comparison_id, method)] = _release_row(method)

    stats = renderer._winner_statistics(grouped, release_rows)

    assert stats["efficientqat_excluded_cells"] == 15
    assert stats["unique_cells"] == 40
    assert stats["tied_cells"] == 20
    assert stats["all_zero_tied_cells"] == 0
    assert stats["including_ties"]["YAQA-wclip"] == 40
    assert stats["unique"]["YAQA-wclip"] == 20
    assert stats["including_ties"]["TurboBOA"] == 20
    assert stats["unique"]["TurboBOA"] == 20
    assert stats["including_ties"]["REALQ-S"] == 20
    assert stats["unique"]["REALQ-S"] == 0


def test_release_date_is_rendered_in_asia_shanghai():
    assert renderer._release_local_date(
        {"created_at": "2026-08-22T17:30:00+00:00"}
    ) == "2026-08-23"
    with pytest.raises(renderer.RenderError, match="timezone"):
        renderer._release_local_date({"created_at": "2026-08-22T17:30:00"})
