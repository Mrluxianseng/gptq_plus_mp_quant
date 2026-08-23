from __future__ import annotations

import pytest

from . import render_master_table_sdpa as renderer


def _release() -> dict:
    rows = []
    for index in range(1, 21):
        comparison_id = f"C{index:02d}"
        for method in renderer.REALQ_METHODS:
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "method": method,
                    "model_name": "model",
                    "setting": "setting",
                    "metrics": {
                        "kl_raw": 0.1,
                        "ppl": 10.0,
                        "qa_avg": 50.0,
                        "gsm8k": 20.0,
                        "math_500": 30.0,
                        "humaneval_base": 40.0,
                        "humaneval_plus": 35.0,
                    },
                    "quantization_gpu_hours": 1.0,
                }
            )
    evidence = {
        name: {"path": name, "sha256": "a", "fingerprint": "b"}
        for name in (
            "formal_plan",
            "formal_audit",
            "runtime_cache_audit",
            "quality_plan",
            "quality_audit",
            "quality_cache_audit",
            "reasoning_plan",
            "reasoning_audit",
        )
    }
    return {
        "release_id": renderer.EXPECTED_RELEASE_ID,
        "status": "complete",
        "row_count": 40,
        "created_at": "2026-08-22T17:30:00+00:00",
        "protocol": {
            "attention_backend": "deterministic math-SDPA",
            "learning_rates_retuned_for_sdpa": False,
            "exact_gptaq_guidedquant_token_physical_files": True,
            "exact_gptaq_guidedquant_sdpa_bf16_reference_files": True,
        },
        "inputs": evidence,
        "rows": rows,
        "gpu_hour_summary": {
            method: {"count": 20, "total": 20.0, "mean": 1.0}
            for method in renderer.REALQ_METHODS
        },
    }


def test_sdpa_release_gate_and_row_formatting():
    release = _release()
    rows = renderer._sdpa_release_rows(release)
    assert len(rows) == 40
    cells = [
        "C01",
        "model",
        "setting",
        "REALQ-F",
        "old",
        "old",
        "old",
        "old",
        "old",
        "old",
        "old",
    ]
    rendered = renderer._realq_cells(cells, rows[("C01", "REALQ-F")])
    assert rendered[4:] == [
        "0.1000000",
        "10.0000",
        "50.00",
        "20.00",
        "30.00",
        "40.00 / 35.00",
        "1.000000",
    ]


def test_sdpa_release_date_is_asia_shanghai():
    assert renderer._release_local_date(_release()) == "2026-08-23"


def test_quality_fairness_row_remains_a_two_cell_markdown_row():
    original = (
        "| 控制项 | 统一要求 |\n"
        "|---|---|\n"
        "| 质量评测 | old wording |\n"
    )
    rendered = renderer._replace_fairness_quality_row(original)
    row = next(line for line in rendered.splitlines() if line.startswith("| 质量评测 |"))
    assert row.count("| 质量评测 |") == 1
    assert len(row.split("|")[1:-1]) == 2
    assert "40/40 execution log" in row


def test_gpu_hour_renderer_records_subsecond_phase_fallback():
    document = (
        "- 每个 REALQ 分支：从正式日志的 `Fusing LN`、`Rotating`、"
        "`Quantising layers` 三个完整 phase timer 求和，再加该模型 "
        "deterministic-FA4 static/Fisher producer 的 `1/4`。\n"
        "- 八者统一排除模型加载、checkpoint I/O 和全部评测。REALQ 的 phase "
        "timer 为整秒精度，因此 `GPU·h*` 保留 6 位仅用于账本复算，不表示"
        "微秒级测量精度。\n"
        "| REALQ-F | 20 | 1.000000 | 0.050000 |\n"
        "| REALQ-S | 20 | 1.000000 | 0.050000 |\n"
    )

    rendered = renderer._update_gpu_hours(document, _release())

    assert "deterministic-SDPA static/Fisher" in rendered
    assert "显示 `00:00` 的亚秒 phase" in rendered
    assert "| REALQ-F | 20 | 20.000000 | 1.000000 |" in rendered
    assert "| REALQ-S | 20 | 20.000000 | 1.000000 |" in rendered


def _existing_eight_method_fixture(monkeypatch):
    release_rows = {}
    grouped = {}
    for index in range(1, 21):
        comparison_id = f"C{index:02d}"
        rows = []
        for method in renderer.additional.BASE_METHODS:
            rows.append(
                [
                    comparison_id,
                    "Qwen3-0.6B",
                    "W4A16",
                    method,
                    "0.1000000",
                    "10.0000",
                    "50.00",
                    "20.00",
                    "30.00",
                    "40.00 / 35.00",
                    "1.000000",
                ]
            )
        for method, label in renderer.additional.NEW_METHODS:
            if not (
                method == "efficientqat"
                and comparison_id in renderer.additional.W4A4_IDS
            ):
                release_rows[(comparison_id, method)] = {
                    "model": "qwen3-0.6b",
                    "setting": "W4A16KV16",
                    "metrics": {
                        "kl_raw": 0.2,
                        "ppl": 11.0,
                        "qa_avg": 51.0,
                        "gsm8k": 21.0,
                        "math_500": 31.0,
                        "humaneval_base": 41.0,
                        "humaneval_plus": 36.0,
                    },
                    "quantization_gpu_hours": 2.0,
                }
            rows.append(
                renderer.additional._new_cells(
                    comparison_id,
                    method,
                    label,
                    rows[0],
                    release_rows,
                )
            )
        grouped[comparison_id] = rows

    summary = {
        "efficientqat": {"count": 15, "total": 30.0, "mean": 2.0},
        "turboboa": {"count": 20, "total": 40.0, "mean": 2.0},
        "yaqa_wclip": {"count": 20, "total": 40.0, "mean": 2.0},
    }
    release = {
        "reference_fingerprint": renderer.additional.EXPECTED_REFERENCE_FINGERPRINT,
        "gpu_hour_summary": summary,
    }
    lines = [
        renderer.additional.FULL_TITLE,
        "",
        renderer.EIGHT_TABLE_START,
        "",
    ]
    for index in range(1, 21):
        lines.extend(
            "| " + " | ".join(row) + " |"
            for row in grouped[f"C{index:02d}"]
        )
    lines.extend(["", renderer.TABLE_END, ""])
    for method, label in renderer.additional.NEW_METHODS:
        value = summary[method]
        lines.append(
            f"| {label} | {value['count']} | {value['total']:.6f} | "
            f"{value['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "- 新增三方法 release：55/55 suite，reference fingerprint "
            f"`{release['reference_fingerprint']}`；evidence。",
        ]
    )
    monkeypatch.setattr(
        renderer.additional, "_release_rows", lambda _release: release_rows
    )
    return "\n".join(lines), release


def test_already_published_eight_method_document_is_validated(monkeypatch):
    document, release = _existing_eight_method_fixture(monkeypatch)
    assert renderer._prepare_additional_document(document, release) == document

    tampered = document.replace(
        "| TurboBOA | 20 | 40.000000 |",
        "| TurboBOA | 20 | 41.000000 |",
    )
    with pytest.raises(renderer.RenderError, match="GPU-hour row differs"):
        renderer._prepare_additional_document(tampered, release)
