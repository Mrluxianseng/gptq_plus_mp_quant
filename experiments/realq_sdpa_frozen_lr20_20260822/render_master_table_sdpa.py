#!/usr/bin/env python3
"""Render the eight-method table with audited SDPA REALQ rows."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from experiments.additional_methods_fair20_eval_20260821 import (
    render_master_table as additional,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENT = additional.DEFAULT_DOCUMENT
DEFAULT_ADDITIONAL_RELEASE = additional.DEFAULT_RELEASE
DEFAULT_REALQ_RELEASE = (
    REPO_ROOT.parent
    / "experiment_data/realq_sdpa_frozen_lr20_20260822_v2_memory/"
    "final_release_v5_merged.json"
)
EIGHT_TABLE_START = "## 20 设定 × 8 算法总表"
TABLE_END = "## GPU-hour 汇总（20 设定）"
OLD_STATS_START = "### GPTAQ / GuidedQuant / REALQ 双分支的 60 项推理第一名计数"
GPU_HOUR_START = "### GPU-hour 的统一口径"
EXPECTED_RELEASE_ID = "realq-sdpa-frozen-run15-lr20-release-20260823-v5-merged"
REALQ_METHODS = ("REALQ-F", "REALQ-S")
EIGHT_METHODS = (
    *additional.BASE_METHODS,
    *(label for _method, label in additional.NEW_METHODS),
)


class RenderError(RuntimeError):
    """An input release or the document violates the overlay contract."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RenderError(f"JSON root must be an object: {path}")
    return value


def _release_local_date(release: Mapping[str, Any]) -> str:
    created_at = release.get("created_at")
    if not isinstance(created_at, str):
        raise RenderError("SDPA release created_at is missing")
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise RenderError("SDPA release created_at is not ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise RenderError("SDPA release created_at must include a timezone")
    return timestamp.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()


def _sdpa_release_rows(
    release: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    protocol = release.get("protocol", {})
    inputs = release.get("inputs", {})
    if (
        release.get("release_id") != EXPECTED_RELEASE_ID
        or release.get("status") != "complete"
        or release.get("row_count") != 40
        or protocol.get("attention_backend") != "deterministic math-SDPA"
        or protocol.get("learning_rates_retuned_for_sdpa") is not False
        or protocol.get("exact_gptaq_guidedquant_token_physical_files") is not True
        or protocol.get("exact_gptaq_guidedquant_sdpa_bf16_reference_files") is not True
        or set(inputs)
        != {
            "formal_plan",
            "formal_audit",
            "runtime_cache_audit",
            "quality_plan",
            "quality_audit",
            "quality_cache_audit",
            "reasoning_plan",
            "reasoning_audit",
        }
    ):
        raise RenderError("REALQ SDPA release identity/protocol is incomplete")
    for name, evidence in inputs.items():
        if not isinstance(evidence, Mapping) or not all(
            isinstance(evidence.get(field), str) and evidence.get(field)
            for field in ("path", "sha256", "fingerprint")
        ):
            raise RenderError(f"REALQ SDPA release evidence is incomplete: {name}")

    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in release.get("rows", []):
        key = (str(row.get("comparison_id")), str(row.get("method")))
        if key in indexed:
            raise RenderError(f"duplicate REALQ SDPA row: {key}")
        metrics = row.get("metrics", {})
        numbers = [
            metrics.get(name)
            for name in (
                "kl_raw",
                "ppl",
                "qa_avg",
                "gsm8k",
                "math_500",
                "humaneval_base",
                "humaneval_plus",
            )
        ] + [row.get("quantization_gpu_hours")]
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in numbers
        ):
            raise RenderError(f"invalid REALQ SDPA metrics: {key}")
        indexed[key] = row
    expected = {
        (f"C{index:02d}", method)
        for index in range(1, 21)
        for method in REALQ_METHODS
    }
    if set(indexed) != expected:
        raise RenderError("REALQ SDPA release is not the exact C01-C20 × F/S matrix")

    summary = release.get("gpu_hour_summary", {})
    for method in REALQ_METHODS:
        values = [
            float(indexed[(f"C{index:02d}", method)]["quantization_gpu_hours"])
            for index in range(1, 21)
        ]
        reported = summary.get(method, {})
        if (
            reported.get("count") != 20
            or not math.isclose(
                float(reported.get("total", -1)), sum(values), abs_tol=1e-12
            )
            or not math.isclose(
                float(reported.get("mean", -1)), sum(values) / 20, abs_tol=1e-12
            )
        ):
            raise RenderError(f"REALQ SDPA GPU-hour summary differs: {method}")
    return indexed


def _realq_cells(
    base_cells: list[str], row: Mapping[str, Any]
) -> list[str]:
    if (
        row.get("model_name") != base_cells[1]
        or row.get("setting") != base_cells[2]
        or row.get("comparison_id") != base_cells[0]
        or row.get("method") != base_cells[3]
    ):
        raise RenderError(
            f"REALQ SDPA model/setting identity differs: "
            f"{base_cells[0]}/{base_cells[3]}"
        )
    metrics = row["metrics"]
    return [
        base_cells[0],
        base_cells[1],
        base_cells[2],
        base_cells[3],
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


def _overlay_table(
    document: str,
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    start = document.index(EIGHT_TABLE_START)
    end = document.index(TABLE_END, start)
    lines = document[start:end].splitlines()
    replaced: set[tuple[str, str]] = set()
    output = []
    for line in lines:
        if not re.match(r"^\| C\d{2} \|", line):
            output.append(line)
            continue
        cells = additional._parse_row(line)
        key = (cells[0], cells[3])
        if cells[3] in REALQ_METHODS:
            cells = _realq_cells(cells, rows[key])
            line = "| " + " | ".join(cells) + " |"
            replaced.add(key)
        output.append(line)
    if replaced != set(rows):
        raise RenderError("not every REALQ SDPA row replaced a table row")
    return document[:start] + "\n".join(output) + "\n\n" + document[end:]


def _eight_grouped(document: str) -> dict[str, list[list[str]]]:
    start = document.index(EIGHT_TABLE_START)
    end = document.index(TABLE_END, start)
    grouped: dict[str, list[list[str]]] = {}
    for line in document[start:end].splitlines():
        if re.match(r"^\| C\d{2} \|", line):
            cells = additional._parse_row(line)
            grouped.setdefault(cells[0], []).append(cells)
    expected = {f"C{index:02d}" for index in range(1, 21)}
    if set(grouped) != expected:
        raise RenderError("rendered table does not contain C01-C20")
    for comparison_id, entries in grouped.items():
        if tuple(entry[3] for entry in entries) != EIGHT_METHODS:
            raise RenderError(f"eight-method order differs: {comparison_id}")
    return grouped


def _validate_existing_additional_rows(
    document: str,
    release: Mapping[str, Any],
    release_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    """Validate a previously published eight-method document in place.

    The additional-method renderer is intentionally one-way: it upgrades the
    original five-method table and rejects an already upgraded document.  The
    SDPA overlay normally runs after that publication, so it must validate the
    existing EfficientQAT/TurboBOA/YAQA cells instead of trying to insert them
    a second time.
    """

    if document.count(additional.FULL_TITLE) != 1:
        raise RenderError("existing eight-method title changed")
    grouped = _eight_grouped(document)
    release_rows_dict = dict(release_rows)
    for comparison_id, rows in grouped.items():
        indexed = {row[3]: row for row in rows}
        base = rows[0]
        for method, label in additional.NEW_METHODS:
            expected = additional._new_cells(
                comparison_id,
                method,
                label,
                base,
                release_rows_dict,
            )
            if indexed.get(label) != expected:
                raise RenderError(
                    "existing additional-method row differs from release: "
                    f"{comparison_id}/{label}"
                )

    summary = release.get("gpu_hour_summary", {})
    for method, label in additional.NEW_METHODS:
        value = summary.get(method, {})
        expected_count = 15 if method == "efficientqat" else 20
        try:
            line = (
                f"| {label} | {expected_count} | "
                f"{float(value['total']):.6f} | {float(value['mean']):.6f} |"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RenderError(
                f"additional-method GPU-hour summary is incomplete: {method}"
            ) from exc
        if value.get("count") != expected_count or document.count(line) != 1:
            raise RenderError(
                f"existing additional-method GPU-hour row differs: {method}"
            )

    fingerprint = release.get("reference_fingerprint")
    provenance = f"55/55 suite，reference fingerprint `{fingerprint}`"
    if not isinstance(fingerprint, str) or document.count(provenance) != 1:
        raise RenderError("existing additional-method provenance differs")


def _prepare_additional_document(document: str, release: dict[str, Any]) -> str:
    base_count = document.count(additional.TABLE_START)
    eight_count = document.count(EIGHT_TABLE_START)
    if (base_count, eight_count) == (1, 0):
        return additional.render(document, release)
    if (base_count, eight_count) == (0, 1):
        release_rows = additional._release_rows(release)
        _validate_existing_additional_rows(document, release, release_rows)
        return document
    raise RenderError(
        "document must contain exactly one five-method or eight-method table"
    )


def _replace_section(text: str, start: str, end: str, replacement: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RenderError(f"document section anchors changed: {start!r} / {end!r}")
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement.rstrip() + "\n\n" + text[finish:]


def _replace_fairness_quality_row(text: str) -> str:
    pattern = re.compile(r"^\| 质量评测 \|.*\|$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RenderError("quality fairness row changed")
    replacement = (
        "| 质量评测 | WikiText-2 Exact KL/PPL、同一十项 QA 与同一平均口径。"
        "八方法全部直接命中 GPTAQ/GuidedQuant 的同一旧 SDPA BF16 teacher "
        "物理文件；REALQ-SDPA 的 40/40 execution log 另有逐行 load-hit / "
        "no-regeneration 审计。 |"
    )
    return pattern.sub(replacement, text, count=1)


def _update_gpu_hours(
    text: str,
    release: Mapping[str, Any],
) -> str:
    realq_definition = re.compile(
        r"- 每个 REALQ 分支：从正式日志的 `Fusing LN`、`Rotating`、"
        r"`Quantising layers` 三个完整 phase timer 求和，再加该模型 "
        r"deterministic-FA4 static/Fisher producer 的 `1/4`。"
    )
    if len(realq_definition.findall(text)) != 1:
        raise RenderError("REALQ GPU-hour definition changed")
    text = realq_definition.sub(
        "- 每个 REALQ 分支：从 SDPA 正式日志的 `Fusing LN`、`Rotating`、"
        "`Quantising layers` 三个完整 phase timer 求和，再加该模型 "
        "deterministic-SDPA static/Fisher producer 的 `1/4`。",
        text,
        count=1,
    )
    old_precision = (
        "- 八者统一排除模型加载、checkpoint I/O 和全部评测。REALQ 的 phase "
        "timer 为整秒精度，因此 `GPU·h*` 保留 6 位仅用于账本复算，不表示"
        "微秒级测量精度。"
    )
    new_precision = (
        "- 八者统一排除模型加载、checkpoint I/O 和全部评测。REALQ 的非零 "
        "phase elapsed 使用 tqdm 整秒值；仅对已完成但显示 `00:00` 的亚秒 "
        "phase，以最终 completion rate 恢复正时长。因此 `GPU·h*` 保留 6 位仅"
        "用于账本复算，不表示微秒级测量精度。"
    )
    if text.count(old_precision) != 1:
        raise RenderError("REALQ GPU-hour precision note changed")
    text = text.replace(old_precision, new_precision, 1)
    for method in REALQ_METHODS:
        value = release["gpu_hour_summary"][method]
        total, mean = float(value["total"]), float(value["mean"])
        pattern = re.compile(
            rf"^\| {re.escape(method)} \| 20 \| [0-9.]+ \| [0-9.]+ \|$",
            re.MULTILINE,
        )
        if len(pattern.findall(text)) != 1:
            raise RenderError(f"REALQ GPU-hour summary row changed: {method}")
        text = pattern.sub(
            f"| {method} | 20 | {total:.6f} | {mean:.6f} |", text, count=1
        )
    return text


def _update_source_provenance(
    text: str,
    release: Mapping[str, Any],
    release_path: Path,
) -> str:
    pattern = re.compile(r"^- REALQ 双分支：.*?(?=^- EfficientQAT：)", re.MULTILINE | re.DOTALL)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RenderError("REALQ source provenance paragraph changed")
    inputs = release["inputs"]
    replacement = (
        "- REALQ 双分支（本表权威值）："
        "`docs/REALQ_SDPA双分支20设定复跑_20260822.md`；"
        f"release `{release_path}`，fingerprint "
        f"`{release['release_fingerprint']}`。formal/quality/reasoning audit "
        f"分别为 `{inputs['formal_audit']['fingerprint']}`、"
        f"`{inputs['quality_audit']['fingerprint']}`、"
        f"`{inputs['reasoning_audit']['fingerprint']}`；formal 与 quality 的 "
        "exact-cache audit 均为 40/40。历史 run15/FA4 数值不再进入本表主行，"
        "仍保留在原实验记录中。\n"
    )
    return pattern.sub(replacement, text, count=1)


def render(
    document: str,
    additional_release: dict[str, Any],
    realq_release: dict[str, Any],
    *,
    realq_release_path: Path = DEFAULT_REALQ_RELEASE,
) -> str:
    # Upgrade a pristine five-method source, or validate the already published
    # eight-method source without attempting a second one-way insertion.
    text = _prepare_additional_document(document, additional_release)
    sdpa_rows = _sdpa_release_rows(realq_release)
    text = _overlay_table(text, sdpa_rows)
    grouped = _eight_grouped(text)
    base_grouped = {
        comparison_id: [
            row for row in rows if row[3] in additional.BASE_METHODS
        ]
        for comparison_id, rows in grouped.items()
    }
    additional_rows = additional._release_rows(additional_release)
    winner = additional._render_winner_summary(base_grouped, additional_rows)
    backend = """### REALQ-SDPA 复跑与解释边界

本表的 REALQ-F/REALQ-S 已由 run15 deterministic-FA4 checkpoint 全部替换为
deterministic math-SDPA checkpoint；学习率逐行复用 run15 冻结选择，没有针对
SDPA 重新调参。校准 token 是 GPTAQ/GuidedQuant 的同一批物理文件，质量评测
teacher 也是同一旧 SDPA BF16 物理文件，static/Fisher cache 则按 backend 隔离。

为通过单卡容量门禁，Llama-8B/Qwen3-8B 的 global-loss batch 使用 4，Qwen3-32B
使用 1，Qwen3-4B W2A16 使用 4；Qwen3-32B 仅把数学上独立的 SDPA batch 轴切为
4 并使用 non-reentrant checkpoint，逻辑 backward batch、Hessian/Adam 节奏和
量化配置不变。因此本表满足 token、teacher、seed、quantizer 与评测协议的严格
控制，但不把它描述成相对 run15 只改变一个浮点 kernel 的单变量消融。FA4
诊断及旧 checkpoint 结果保留在 run15 专项文档中。"""
    text = _replace_section(
        text,
        "### 量化数值后端的严格限制",
        "### Exact-KL 的严格限制",
        """### 量化数值后端的严格限制

GPTAQ、GuidedQuant、ResComp-C 与本表 REALQ 双分支均使用 SDPA attention；
REALQ 的 Hessian/Fisher 仍使用 TF32 input / FP32 accumulate，其他方法也保留各自
核心数值路径。因此这是校准数据、量化表示、teacher 与评测协议受控的完整方法
benchmark，不宣称是只替换 solver 方程的逐 bit 数值消融。EfficientQAT、
TurboBOA、YAQA-wclip 同样保留各自论文/方法定义的训练、变换与误差补偿。""",
    )
    text = _replace_section(
        text,
        "### Exact-KL 的严格限制",
        OLD_STATS_START,
        """### Exact-KL teacher 控制

八方法的 WikiText-2 Exact KL 均以 GPTAQ/GuidedQuant 的同一旧 SDPA BF16
reference-logits 物理文件为 teacher。REALQ-SDPA 的 quality cache audit 对 40/40
成功日志逐项要求精确 `Loading reference logits`，并禁止任何 `Generating
reference logits`。因此本表 KL 的 teacher 已满足 bitwise-controlled 条件。""",
    )
    text = _replace_section(
        text,
        OLD_STATS_START,
        GPU_HOUR_START,
        winner + backend,
    )
    text = _replace_fairness_quality_row(text)
    text = _update_gpu_hours(text, realq_release)
    text = _update_source_provenance(text, realq_release, realq_release_path)
    text = text.replace("KL†", "KL")

    old_date = additional._release_local_date(additional_release)
    sdpa_date = _release_local_date(realq_release)
    date_pattern = re.compile(
        r"^> 原五方法表生成于 2026-08-21；八方法最终 release 生成于 .*?"
        r"不是手工抄表。$",
        re.MULTILINE | re.DOTALL,
    )
    matches = list(date_pattern.finditer(text[: text.index("## 方法与列定义")]))
    if len(matches) != 1:
        raise RenderError("release date paragraph changed")
    date_text = (
        "> 原五方法表生成于 2026-08-21；新增三方法 release 生成于 "
        f"{old_date}，REALQ-SDPA 最终 release 生成于 {sdpa_date}"
        "（Asia/Shanghai）。数值由原始 marker / final audit 自动交叉核验后汇总；"
        "不是手工抄表。"
    )
    begin, end = matches[0].span()
    text = text[:begin] + date_text + text[end:]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", default=str(DEFAULT_DOCUMENT))
    parser.add_argument("--additional-release", default=str(DEFAULT_ADDITIONAL_RELEASE))
    parser.add_argument("--realq-release", default=str(DEFAULT_REALQ_RELEASE))
    parser.add_argument("--output")
    args = parser.parse_args()
    document_path = Path(args.document).resolve()
    additional_path = Path(args.additional_release).resolve()
    realq_path = Path(args.realq_release).resolve()
    rendered = render(
        document_path.read_text(encoding="utf-8"),
        _read_json(additional_path),
        _read_json(realq_path),
        realq_release_path=realq_path,
    )
    if args.output:
        output = Path(args.output).resolve()
        additional._atomic_text(output, rendered)
        print(output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
