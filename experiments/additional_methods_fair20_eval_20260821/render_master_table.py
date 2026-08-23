#!/usr/bin/env python3
"""Merge the completed 55-suite release into the existing 20-setting table."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENT = (
    REPO_ROOT / "docs/REALQ_GPTAQ_GuidedQuant_四算法20设定公平总表_20260821.md"
)
DEFAULT_RELEASE = (
    REPO_ROOT.parent
    / "experiment_data/additional_methods_fair20_eval_20260821_v1/final_release.json"
)
TABLE_START = "## 20 设定 × 5 算法总表"
TABLE_END = "## GPU-hour 汇总（20 设定）"
SUMMARY_END = "## 数据来源与审计身份"
WINNER_INSERT_BEFORE = "### 7 月旧 REAL-Q 与本表 REALQ-S 的直接复核"
FULL_WINNER_HEADING = "### 八算法的 60 项推理第一名计数"
BASE_TITLE = "# GPTAQ / GuidedQuant / ResComp-C / REALQ 双分支：20 设定公平横向总表"
FULL_TITLE = "# 八算法：20 设定公平横向总表"
BASE_METHODS = ("GPTAQ", "GuidedQuant", "ResComp-C", "REALQ-F", "REALQ-S")
NEW_METHODS = (
    ("efficientqat", "EfficientQAT"),
    ("turboboa", "TurboBOA"),
    ("yaqa_wclip", "YAQA-wclip"),
)
W4A4_IDS = {"C02", "C06", "C10", "C14", "C18"}
REASONING_COLUMNS = (
    ("gsm8k", 7, 1319),
    ("math_500", 8, 500),
    ("humaneval_plus", 9, 164),
)
EXPECTED_REFERENCE_FINGERPRINT = (
    "3ecffa5d4cb97efb24aac4a668b6570d8e8193f4f4fa21f98825c2d9d238be05"
)
MODEL_DISPLAY = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "llama31-8b-instruct": "Llama-3.1-8B-Instruct",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "qwen3-32b": "Qwen3-32B",
}
SETTING_DISPLAY = {
    "W4A16KV16": "W4A16",
    "W4A4KV4": "W4A4KV4",
    "W3A16KV16": "W3A16",
    "W2A16KV16": "W2A16",
}


class RenderError(RuntimeError):
    """The base table or completed release violates the merge contract."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RenderError(f"JSON root must be an object: {path}")
    return value


def _parse_row(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.split("|")[1:-1]]
    if len(cells) != 11:
        raise RenderError(f"unexpected table row width: {line}")
    return cells


def _base_rows(document: str) -> dict[str, list[list[str]]]:
    start = document.index(TABLE_START)
    end = document.index(TABLE_END, start)
    grouped: dict[str, list[list[str]]] = {}
    for line in document[start:end].splitlines():
        if not re.match(r"^\| C\d{2} \|", line):
            continue
        cells = _parse_row(line)
        grouped.setdefault(cells[0], []).append(cells)
    expected_ids = {f"C{index:02d}" for index in range(1, 21)}
    if set(grouped) != expected_ids:
        raise RenderError("base table does not contain exactly C01-C20")
    for comparison_id, rows in grouped.items():
        if tuple(row[3] for row in rows) != BASE_METHODS:
            raise RenderError(f"base method order changed for {comparison_id}")
        if len({(row[1], row[2]) for row in rows}) != 1:
            raise RenderError(f"base model/setting differs within {comparison_id}")
    return grouped


def _release_rows(release: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if (
        release.get("status") != "complete"
        or release.get("expected_suite_count") != 55
        or release.get("completed_suite_count") != 55
        or release.get("missing_eval_ids") != []
        or release.get("reference_fingerprint")
        != EXPECTED_REFERENCE_FINGERPRINT
    ):
        raise RenderError("release is not the complete 55-suite terminal")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in release.get("rows", []):
        key = (str(row.get("comparison_id")), str(row.get("method")))
        if key in indexed:
            raise RenderError(f"duplicate release row: {key}")
        indexed[key] = row
    expected = set()
    for index in range(1, 21):
        comparison_id = f"C{index:02d}"
        expected.update(
            (comparison_id, method)
            for method, _label in NEW_METHODS
            if method != "efficientqat" or comparison_id not in W4A4_IDS
        )
    if set(indexed) != expected:
        raise RenderError("release rows do not match the expected 15+20+20 matrix")
    summary = release.get("gpu_hour_summary", {})
    for method, _label in NEW_METHODS:
        method_rows = [
            row for (_comparison_id, name), row in indexed.items() if name == method
        ]
        expected_count = 15 if method == "efficientqat" else 20
        total = sum(float(row["quantization_gpu_hours"]) for row in method_rows)
        reported = summary.get(method, {})
        if (
            len(method_rows) != expected_count
            or reported.get("count") != expected_count
            or not math.isclose(float(reported.get("total", -1)), total, abs_tol=1e-12)
        ):
            raise RenderError(f"release GPU-hour summary differs for {method}")
    return indexed


def _new_cells(
    comparison_id: str,
    method: str,
    label: str,
    base: list[str],
    release_rows: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    if method == "efficientqat" and comparison_id in W4A4_IDS:
        return [
            comparison_id,
            base[1],
            base[2],
            label,
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        ]
    row = release_rows[(comparison_id, method)]
    if (
        MODEL_DISPLAY.get(str(row.get("model"))) != base[1]
        or SETTING_DISPLAY.get(str(row.get("setting"))) != base[2]
    ):
        raise RenderError(
            f"release model/setting mapping differs for {comparison_id}/{method}"
        )
    metrics = row["metrics"]
    return [
        comparison_id,
        base[1],
        base[2],
        label,
        f"{float(metrics['kl_raw']):.7f}",
        f"{float(metrics['ppl']):.4f}",
        f"{float(metrics['qa_avg']):.2f}",
        f"{float(metrics['gsm8k']):.2f}",
        f"{float(metrics['math_500']):.2f}",
        (
            f"{float(metrics['humaneval_base']):.2f} / "
            f"{float(metrics['humaneval_plus']):.2f}"
        ),
        f"{float(row['quantization_gpu_hours']):.6f}",
    ]


def _render_table(
    grouped: dict[str, list[list[str]]],
    release_rows: dict[tuple[str, str], dict[str, Any]],
) -> str:
    lines = [
        "## 20 设定 × 8 算法总表",
        "",
        "| ID | 模型 | 设定 | 算法 | KL† ↓ | PPL ↓ | QA Avg ↑ | GSM8K ↑ | MATH-500 ↑ | HumanEval+ base/plus ↑ | GPU·h* ↓ |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index in range(1, 21):
        comparison_id = f"C{index:02d}"
        rows = grouped[comparison_id]
        all_rows = list(rows)
        for method, label in NEW_METHODS:
            all_rows.append(
                _new_cells(comparison_id, method, label, rows[0], release_rows)
            )
        lines.extend("| " + " | ".join(row) + " |" for row in all_rows)
    return "\n".join(lines) + "\n\n"


def _render_summary(release: dict[str, Any]) -> str:
    existing = {
        "GPTAQ": (20, 30.749673),
        "GuidedQuant": (20, 86.096354),
        "ResComp-C": (20, 9.516878),
        "REALQ-F": (20, 22.979081),
        "REALQ-S": (20, 20.502692),
    }
    additions = release["gpu_hour_summary"]
    rows = [
        "## GPU-hour 汇总（20 设定）",
        "",
        "| 算法 | 有效设定数 | 统一 GPU·h* 合计 | 平均每有效设定 GPU·h* |",
        "|---|---:|---:|---:|",
    ]
    for label, (count, total) in existing.items():
        rows.append(f"| {label} | {count} | {total:.6f} | {total / count:.6f} |")
    for method, label in NEW_METHODS:
        value = additions[method]
        count = int(value["count"])
        total = float(value["total"])
        rows.append(f"| {label} | {count} | {total:.6f} | {total / count:.6f} |")
    return "\n".join(rows) + "\n\n"


def _percent_to_hits(value: float | str, total: int, identity: str) -> int:
    """Recover the exact integer score behind a displayed percentage.

    The historical table is printed to two decimal places, while one hit on
    every reasoning task is larger than 0.01 percentage point.  Therefore the
    displayed value still maps to exactly one integer hit count.  New release
    rows retain the unrounded percentage, and pass the same gate.
    """

    try:
        percentage = float(value)
    except (TypeError, ValueError) as exc:
        raise RenderError(f"invalid reasoning percentage for {identity}: {value}") from exc
    if not math.isfinite(percentage):
        raise RenderError(f"non-finite reasoning percentage for {identity}")
    hits = int(round(percentage * total / 100.0))
    reconstructed = 100.0 * hits / total
    if (
        hits < 0
        or hits > total
        or not math.isclose(percentage, reconstructed, abs_tol=0.0051)
    ):
        raise RenderError(
            f"reasoning percentage does not map to integer hits for {identity}: "
            f"value={percentage}, total={total}, nearest={reconstructed}"
        )
    return hits


def _base_reasoning_hits(row: list[str], metric: str, column: int, total: int) -> int:
    value = row[column]
    if metric == "humaneval_plus":
        parts = [part.strip() for part in value.split("/")]
        if len(parts) != 2:
            raise RenderError(f"invalid HumanEval base/plus cell: {value}")
        value = parts[1]
    return _percent_to_hits(value, total, f"{row[0]}/{row[3]}/{metric}")


def _winner_statistics(
    grouped: dict[str, list[list[str]]],
    release_rows: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    labels = list(BASE_METHODS) + [label for _method, label in NEW_METHODS]
    including_ties = {label: 0 for label in labels}
    unique = {label: 0 for label in labels}
    unique_cells = 0
    tied_cells = 0
    all_zero_tied_cells = 0
    efficientqat_excluded_cells = 0

    for index in range(1, 21):
        comparison_id = f"C{index:02d}"
        base_by_method = {row[3]: row for row in grouped[comparison_id]}
        for metric, column, total in REASONING_COLUMNS:
            hits = {
                method: _base_reasoning_hits(
                    base_by_method[method], metric, column, total
                )
                for method in BASE_METHODS
            }
            for method, label in NEW_METHODS:
                if method == "efficientqat" and comparison_id in W4A4_IDS:
                    efficientqat_excluded_cells += 1
                    continue
                row = release_rows[(comparison_id, method)]
                hits[label] = _percent_to_hits(
                    row["metrics"][metric],
                    total,
                    f"{comparison_id}/{label}/{metric}",
                )

            best = max(hits.values())
            winners = [label for label, value in hits.items() if value == best]
            for label in winners:
                including_ties[label] += 1
            if len(winners) == 1:
                unique[winners[0]] += 1
                unique_cells += 1
            else:
                tied_cells += 1
                if best == 0:
                    all_zero_tied_cells += 1

    if unique_cells + tied_cells != 60 or efficientqat_excluded_cells != 15:
        raise RenderError("reasoning winner accounting does not cover 60 cells")
    return {
        "method_order": labels,
        "including_ties": including_ties,
        "unique": unique,
        "unique_cells": unique_cells,
        "tied_cells": tied_cells,
        "all_zero_tied_cells": all_zero_tied_cells,
        "efficientqat_excluded_cells": efficientqat_excluded_cells,
    }


def _render_winner_summary(
    grouped: dict[str, list[list[str]]],
    release_rows: dict[tuple[str, str], dict[str, Any]],
) -> str:
    stats = _winner_statistics(grouped, release_rows)
    rows = [
        FULL_WINNER_HEADING,
        "",
        "加入 EfficientQAT、TurboBOA、YAQA-wclip 后，仍按每个设定的 GSM8K、",
        "MATH-500、HumanEval+ **plus** 组成 `20×3=60` 个比较单元。EfficientQAT",
        "按论文范围不做激活量化，因此在 5 个 W4A4KV4 设定（15 个推理单元）中",
        "记为 N/A 并从该单元的候选方法中排除；其余方法正常参与。第一名从整数命中数",
        "复算，避免显示百分比舍入制造假并列。",
        "",
        "| 方法 | 含并列第一 | 唯一第一 |",
        "|---|---:|---:|",
    ]
    for label in stats["method_order"]:
        rows.append(
            f"| {label} | {stats['including_ties'][label]} | "
            f"{stats['unique'][label]} |"
        )
    rows.extend(
        [
            "",
            f"60 个单元中有 {stats['unique_cells']} 个唯一第一、"
            f"{stats['tied_cells']} 个并列第一；其中 "
            f"{stats['all_zero_tied_cells']} 个是所有可用方法均为 0 的退化并列。",
            "因此跨方法判断优劣时优先看“唯一第一”；“含并列第一”用于完整记录并列，",
            "其列和可以超过 60。",
            "",
        ]
    )
    return "\n".join(rows) + "\n"


def _replace_once(text: str, old: str, new: str, identity: str) -> str:
    if text.count(old) != 1:
        raise RenderError(f"base document changed at {identity}")
    return text.replace(old, new, 1)


def _release_local_date(release: dict[str, Any]) -> str:
    created_at = release.get("created_at")
    if not isinstance(created_at, str):
        raise RenderError("release created_at is missing")
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise RenderError("release created_at is not ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise RenderError("release created_at must include a timezone")
    return timestamp.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()


def _upgrade_prefix(prefix: str, release: dict[str, Any]) -> str:
    prefix = _replace_once(prefix, BASE_TITLE, FULL_TITLE, "title")
    prefix = _replace_once(
        prefix,
        "> 生成日期：2026-08-21。数值由原始 marker / final audit 自动交叉核验后汇总；"
        "不是手工抄表。",
        f"> 原五方法表生成于 2026-08-21；八方法最终 release 生成于 "
        f"{_release_local_date(release)}（Asia/Shanghai）。数值由原始 marker / final "
        "audit 自动交叉核验后汇总；不是手工抄表。",
        "release date",
    )
    method_anchor = (
        "- `REALQ-S`：默认分支，不更新 transformer block 内其他线性层（single-linear）。"
    )
    method_additions = "\n".join(
        [
            method_anchor,
            "- `EfficientQAT`：论文范围的 signed-symmetric G128 weight-only 两阶段 QAT；"
            "只覆盖 W4/W3/W2，W4A4KV4 为 N/A，不做 QuaRot。",
            "- `TurboBOA`：`TurboBOA-C (RealQ-MSE, grouped-F3)`；保留 QuaRot、column "
            "act-order 与方法原生 grouped-F3。",
            "- `YAQA-wclip`：`YAQA-wclip-C`；使用 RealQ 等价 w-clip 标量量化器，保留"
            "方法原生双侧 RHT、anti-diagonal LDLQ，不额外叠加全局 QuaRot。",
        ]
    )
    prefix = _replace_once(prefix, method_anchor, method_additions, "method definitions")

    matrix_old = (
        "| 矩阵 | 5 个相同模型 × 4 个相同量化设定；每个设定均有 GPTAQ、GuidedQuant、"
        "ResComp-C、REALQ-F、REALQ-S。 |"
    )
    matrix_new = (
        "| 矩阵 | 5 个相同模型 × 4 个相同量化设定；TurboBOA、YAQA-wclip 与原五方法"
        "均覆盖 20 设定。EfficientQAT 按论文 weight-only 范围覆盖 W4/W3/W2 共 15 设定，"
        "5 个 W4A4KV4 格子为 N/A。 |"
    )
    prefix = _replace_once(prefix, matrix_old, matrix_new, "matrix contract")

    calibration_old = (
        "| 校准集 | 严格相同的 WikiText-2 train `256×2048=524,288` token tensor；"
        "三轮均直接引用 `realq_20group_20260808/shared_cache` 的同一物理文件，顺序不变。 |"
    )
    calibration_new = (
        "| 校准集 | 严格相同的 WikiText-2 train `256×2048=524,288` token tensor；"
        "八方法各轮均直接引用 `realq_20group_20260808/shared_cache` 的同一物理文件，"
        "解包 token 与顺序不变。 |"
    )
    prefix = _replace_once(
        prefix, calibration_old, calibration_new, "calibration contract"
    )

    prefix = _replace_once(
        prefix, "| 量化公共项 |", "| 原五方法量化公共项 |", "base quantization row"
    )
    base_quant_row_end = (
        "GPTAQ/GuidedQuant 的 legacy Cartesian search 与 ResComp-C/REALQ 的 "
        "optimized-exact search 在本次有限 FP32 对称域上产生相同 raw scale/zero。 |"
    )
    new_quant_row = (
        base_quant_row_end
        + "\n| 新增三方法表示与边界 | 三者权重均为 signed-symmetric scalar、natural-column "
        "G128。TurboBOA/YAQA-wclip 的 W4A4KV4 使用对称 per-token A/K/V、groupsize=-1、"
        "clip=0.9；EfficientQAT 的 A/K/V 恒为 BF16。各自保留论文/方法定义的训练、"
        "变换和误差补偿，因此属于 complete-method benchmark，不是 solver-only 消融。 |"
    )
    prefix = _replace_once(
        prefix, base_quant_row_end, new_quant_row, "new quantization row"
    )

    quality_old = (
        "| 质量评测 | WikiText-2 PPL、同一十项 QA 与同一平均口径；三轮各自 final gate "
        "均完整。GPTAQ/GuidedQuant/ResComp-C 走旧 HF/SDPA 路径，REALQ run15 质量评测走 "
        "deterministic FA4，因此属于完整方法结果，不是 attention-kernel-controlled 消融。 |"
    )
    quality_new = (
        "| 质量评测 | WikiText-2 PPL、同一十项 QA 与同一平均口径。新增三方法与 "
        "GPTAQ/GuidedQuant/ResComp-C 复用同一旧 SDPA BF16 teacher cache；REALQ run15 "
        "复用其 deterministic-FA4 teacher cache。因此属于完整方法结果，不是 "
        "attention-kernel-controlled 消融；KL 继续标记为 KL†。 |"
    )
    prefix = _replace_once(prefix, quality_old, quality_new, "quality contract")
    prefix = _replace_once(
        prefix,
        "| 硬件 | 五方法均为单张 `NVIDIA L20C`；评测时间不计入 GPU-hour。 |",
        "| 硬件 | 八方法均为单张 `NVIDIA L20C`；评测时间不计入 GPU-hour。 |",
        "hardware contract",
    )
    prefix = _replace_once(
        prefix,
        "GPTAQ/GuidedQuant/ResComp-C 使用同一旧 SDPA BF16 reference cache，REALQ run15",
        "GPTAQ/GuidedQuant/ResComp-C 及新增三方法使用同一旧 SDPA BF16 reference "
        "cache，REALQ run15",
        "KL reference scope",
    )
    prefix = _replace_once(
        prefix,
        "因此下表保留三轮实验各自通过审计的 Exact-KL",
        "因此下表保留各轮实验通过审计的 Exact-KL",
        "KL provenance wording",
    )
    prefix = _replace_once(
        prefix,
        "不满足“五方法共享同一 teacher logits、bitwise-controlled”的严格 KL 胜负条件",
        "不满足“八方法共享同一 teacher logits、bitwise-controlled”的严格 KL 胜负条件",
        "KL method count",
    )
    backend_limit = (
        "若要后者，需统一 attention backend 和 Hessian 精度后重新量化；仅重跑评测不足够。"
    )
    prefix = _replace_once(
        prefix,
        backend_limit,
        backend_limit
        + "新增三方法也保留各自核心数值路径：EfficientQAT 两阶段 QAT、TurboBOA "
        "eager-attention/grouped-F3、YAQA 双侧 RHT/anti-diagonal LDLQ；不把这些"
        "方法定义强行改成 REALQ 后端。",
        "new backend limits",
    )

    realq_gpu_hour = (
        "- 每个 REALQ 分支：从正式日志的 `Fusing LN`、`Rotating`、`Quantising layers` "
        "三个完整 phase timer 求和，再加该模型 deterministic-FA4 static/Fisher producer "
        "的 `1/4`。"
    )
    gpu_hour_additions = "\n".join(
        [
            realq_gpu_hour,
            "- EfficientQAT：Block-AP + E2E-QP；受 CPU 超卖污染的 Qwen3-0.6B 三组"
            "以 checkpoint byte-identical clean replay 计时替换。",
            "- TurboBOA：每组完整算法 timer；六个受并发污染的早期结果以 checkpoint "
            "content-identical clean replay 计时替换。",
            "- YAQA-wclip：raw quantization 加 Hessian；A16 Hessian 在同模型 W4/W3/W2 "
            "间按 `1/3` 摊销，W4A4-aware Hessian 完整计入；Qwen3-0.6B A16 使用 tensor "
            "byte-identical clean retime。",
        ]
    )
    prefix = _replace_once(
        prefix, realq_gpu_hour, gpu_hour_additions, "new GPU-hour definitions"
    )
    return _replace_once(
        prefix,
        "- 五者统一排除模型加载、checkpoint I/O 和全部评测。",
        "- 八者统一排除模型加载、checkpoint I/O 和全部评测。",
        "GPU-hour method count",
    )


def _upgrade_suffix(suffix: str, release: dict[str, Any]) -> str:
    anchor = "- 生成时再次核验：baseline 40 行均与"
    additions = "\n".join(
        [
            "- EfficientQAT：`docs/EfficientQAT_Qwen3_Llama31_15组权重量化实验记录_20260821.md`；"
            "15/15 quant、quality、三项 reasoning 与官方 EvalPlus 完整。",
            "- TurboBOA / YAQA-wclip："
            "`docs/TurboBOA_YAQA_wclip_Qwen3_Llama31_公平20组实验记录_20260821.md`；"
            "两方法各 20/20 quant、quality、三项 reasoning 与官方 EvalPlus 完整。",
            f"- 新增三方法 release：55/55 suite，reference fingerprint "
            f"`{release['reference_fingerprint']}`；每行绑定量化 terminal、checkpoint "
            "validation、quality result 与官方 HumanEval+ receipt 的 SHA-256。",
            anchor,
        ]
    )
    return _replace_once(suffix, anchor, additions, "new source provenance")


def render(document: str, release: dict[str, Any]) -> str:
    grouped = _base_rows(document)
    release_rows = _release_rows(release)
    table_start = document.index(TABLE_START)
    table_end = document.index(TABLE_END, table_start)
    summary_end = document.index(SUMMARY_END, table_end)
    summary = document[table_end:summary_end]
    for label, total, mean in (
        ("GPTAQ", "30.749673", "1.537484"),
        ("GuidedQuant", "86.096354", "4.304818"),
        ("ResComp-C", "9.516878", "0.475844"),
        ("REALQ-F", "22.979081", "1.148954"),
        ("REALQ-S", "20.502692", "1.025135"),
    ):
        if f"| {label} | {total} | {mean} |" not in summary:
            raise RenderError(f"base GPU-hour summary changed for {label}")
    prefix = _upgrade_prefix(document[:table_start], release)
    if FULL_WINNER_HEADING in prefix:
        raise RenderError("full eight-method winner section already exists")
    winner_insert = prefix.index(WINNER_INSERT_BEFORE)
    prefix = (
        prefix[:winner_insert]
        + _render_winner_summary(grouped, release_rows)
        + prefix[winner_insert:]
    )
    suffix = _upgrade_suffix(document[summary_end:], release)
    return (
        prefix
        + _render_table(grouped, release_rows)
        + _render_summary(release)
        + suffix
    )


def _atomic_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", default=str(DEFAULT_DOCUMENT))
    parser.add_argument("--release", default=str(DEFAULT_RELEASE))
    parser.add_argument("--output")
    args = parser.parse_args()
    document_path = Path(args.document).resolve()
    release_path = Path(args.release).resolve()
    rendered = render(
        document_path.read_text(encoding="utf-8"),
        _read_json(release_path),
    )
    if args.output:
        output = Path(args.output).resolve()
        _atomic_text(output, rendered)
        print(output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
