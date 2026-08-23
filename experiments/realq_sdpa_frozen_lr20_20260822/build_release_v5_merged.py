#!/usr/bin/env python3
"""Build the fail-closed 40-row REALQ deterministic-SDPA release."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Mapping, Sequence

from experiments.realq_fullmodel_retune_20260817 import campaign as base
from experiments.realq_sdpa_frozen_lr20_20260822 import (
    campaign_v2_memory as campaign,
)
from experiments.realq_sdpa_frozen_lr20_20260822 import formal_merged_v5 as formal
from experiments.realq_sdpa_frozen_lr20_20260822 import (
    quality_cache_audit_v5_merged as quality_cache,
)
from experiments.realq_sdpa_frozen_lr20_20260822 import quality_v5_merged as quality
from experiments.realq_sdpa_frozen_lr20_20260822 import (
    reasoning_v5_merged as reasoning,
)
from experiments.realq_sdpa_frozen_lr20_20260822 import (
    runtime_cache_audit_v5_merged as runtime_cache,
)


RELEASE_ID = "realq-sdpa-frozen-run15-lr20-release-20260823-v5-merged"
RELEASE_PATH = campaign.OUTPUT_ROOT / "final_release_v5_merged.json"
MODEL_LAYERS = {
    "qwen3-0.6b": 28,
    "llama31-8b-instruct": 32,
    "qwen3-4b": 36,
    "qwen3-8b": 36,
    "qwen3-32b": 64,
}
MODEL_NAMES = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "llama31-8b-instruct": "Llama-3.1-8B-Instruct",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "qwen3-32b": "Qwen3-32B",
}
SETTING_NAMES = {
    "w4a16": "W4A16",
    "w4a4kv4": "W4A4KV4",
    "w3a16": "W3A16",
    "w2a16": "W2A16",
}
METHOD_NAMES = {"full_block": "REALQ-F", "single_linear": "REALQ-S"}
PHASES = ("Fusing LN", "Rotating", "Quantising layers")
PRODUCER_AMORTIZATION_DIVISOR = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise base.CampaignError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON artifact: {path}")
    return base._read_json(path)


def _duration_seconds(value: str) -> float:
    fields = value.split(":")
    _require(
        len(fields) in {2, 3} and all(field.isdigit() for field in fields),
        f"invalid tqdm elapsed duration: {value!r}",
    )
    numbers = [int(field) for field in fields]
    if len(numbers) == 2:
        minutes, seconds = numbers
        _require(seconds < 60, f"invalid tqdm seconds field: {value!r}")
        return float(minutes * 60 + seconds)
    hours, minutes, seconds = numbers
    _require(
        minutes < 60 and seconds < 60,
        f"invalid tqdm minute/second field: {value!r}",
    )
    return float(hours * 3600 + minutes * 60 + seconds)


def _parse_phase_timing(text: str, *, expected_layers: int) -> dict[str, Any]:
    normalized = text.replace("\r", "\n")
    parsed: dict[str, Any] = {}
    for phase in PHASES:
        pattern = re.compile(
            rf"{re.escape(phase)}:\s+100%[^\n]*?"
            rf"(?P<done>\d+)/(?P<total>\d+)\s+"
            rf"\[(?P<elapsed>\d+(?::\d+){{1,2}})<"
            rf"(?P<remaining>\d+(?::\d+){{1,2}}),\s*"
            rf"(?P<rate>\d+(?:\.\d+)?)"
            rf"(?P<rate_unit>(?:it|layer)/s|s/(?:it|layer))\]"
        )
        matches = list(pattern.finditer(normalized))
        _require(matches, f"completed phase timer missing: {phase}")
        match = matches[-1]
        done, total = int(match.group("done")), int(match.group("total"))
        _require(
            done == total == expected_layers,
            f"phase coverage mismatch for {phase}: {done}/{total}, "
            f"expected {expected_layers}/{expected_layers}",
        )
        elapsed = _duration_seconds(match.group("elapsed"))
        timing_source = "tqdm elapsed field"
        if elapsed == 0:
            rate = float(match.group("rate"))
            _require(
                math.isfinite(rate) and rate > 0,
                f"invalid completed tqdm rate for {phase}: {rate}",
            )
            rate_unit = match.group("rate_unit")
            if rate_unit.endswith("/s"):
                elapsed = done / rate
            else:
                elapsed = done * rate
            timing_source = "tqdm completion rate fallback for subsecond phase"
        _require(
            math.isfinite(elapsed) and elapsed > 0,
            f"non-positive phase time: {phase}",
        )
        parsed[phase] = {
            "seconds": elapsed,
            "completed": done,
            "total": total,
            "completed_match_count": len(matches),
            "selected_match": "last completed tqdm line",
            "timing_source": timing_source,
            "completion_rate": float(match.group("rate")),
            "completion_rate_unit": match.group("rate_unit"),
        }
    parsed["phase_seconds_sum"] = sum(
        float(parsed[phase]["seconds"]) for phase in PHASES
    )
    return parsed


def _config_identity(config: str) -> tuple[str, str, str]:
    for index, expected in enumerate(base.CONFIG_IDS, start=1):
        if config != expected:
            continue
        model, setting = config.rsplit("_", 1)
        _require(model in MODEL_LAYERS, f"unknown model in config: {config}")
        _require(setting in SETTING_NAMES, f"unknown setting in config: {config}")
        return model, setting, f"C{index:02d}"
    raise base.CampaignError(f"unknown comparison config: {config}")


def _verify_final_audit(
    path: Path,
    *,
    identity_field: str,
    identity: str,
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    value = _read_json(path)
    fingerprint = value.get("audit_fingerprint")
    body = dict(value)
    body.pop("audit_fingerprint", None)
    body.pop("audited_at", None)
    _require(
        isinstance(fingerprint, str)
        and base._canonical_sha256(body) == fingerprint,
        f"audit fingerprint mismatch: {path}",
    )
    _require(
        value.get(identity_field) == identity and value.get("status") == "complete",
        f"audit identity/status mismatch: {path}",
    )
    _require(value.get("counts") == dict(expected_counts), f"audit counts mismatch: {path}")
    return value


def _verify_live_cache_audit(
    path: Path,
    *,
    build: Any,
    expected_id: str,
) -> dict[str, Any]:
    stored = _read_json(path)
    rebuilt = build(require_complete=True)
    _require(stored == rebuilt, f"cache audit is stale or drifted: {path}")
    _require(
        stored.get("audit_id") == expected_id
        and stored.get("status") == "complete"
        and stored.get("counts")
        == {"planned": 40, "runtime_succeeded_and_observed": 40},
        f"cache audit identity/count mismatch: {path}",
    )
    return stored


def _producer_timing(model: str) -> dict[str, Any]:
    marker_path = campaign.OUTPUT_ROOT / "shared_cache" / model / "producer_success.json"
    marker = _read_json(marker_path)
    _require(
        marker.get("status") == "succeeded"
        and marker.get("model") == model
        and marker.get("baseline_reference_unchanged") is True,
        f"invalid SDPA producer marker: {model}",
    )
    token = marker.get("token_contract", {})
    _require(
        token.get("same_physical_source_as_gptaq_guidedquant") is True,
        f"producer token source is not the exact baseline file: {model}",
    )
    if "source_result" in marker:
        result_ref = marker.get("source_result")
        _require(
            isinstance(result_ref, Mapping),
            f"producer source result reference invalid: {model}",
        )
        result_path = Path(str(result_ref.get("path", "")))
        _require(
            result_path.is_file()
            and base._file_sha256(result_path) == result_ref.get("sha256"),
            f"producer result hash mismatch: {model}",
        )
        result = _read_json(result_path)
    else:
        _require(
            marker.get("stage") == "deterministic-sdpa-static-cache-producer",
            f"producer result provenance missing: {model}",
        )
        result_path = marker_path
        result = marker
    seconds = result.get("elapsed_seconds")
    _require(
        result.get("status") == "succeeded"
        and result.get("returncode") == 0
        and isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and math.isfinite(float(seconds))
        and float(seconds) > 0,
        f"producer result/timing invalid: {model}",
    )
    flags = formal._flags(result.get("command", []))
    _require(
        flags.get("--attention_backend") == "sdpa"
        and flags.get("--tokens_cache_path") == str(Path(token["path"]).parent)
        and flags.get("--require_reference_cache_hit") == "true",
        f"producer command cache/backend drift: {model}",
    )
    return {
        "model": model,
        "producer_seconds": float(seconds),
        "per_branch_setting_allocation_seconds": (
            float(seconds) / PRODUCER_AMORTIZATION_DIVISOR
        ),
        "amortization_divisor": PRODUCER_AMORTIZATION_DIVISOR,
        "marker": {"path": str(marker_path), "sha256": base._file_sha256(marker_path)},
        "source_result": {
            "path": str(result_path),
            "sha256": base._file_sha256(result_path),
        },
        "token_contract": token,
    }


def _formal_timing(row: Mapping[str, Any], *, expected_layers: int) -> dict[str, Any]:
    result_ref = row.get("result", {})
    result_path = Path(str(result_ref.get("path", "")))
    _require(
        result_path.is_file()
        and base._file_sha256(result_path) == result_ref.get("sha256"),
        f"formal result hash mismatch: {row.get('branch')}/{row.get('config')}",
    )
    result = _read_json(result_path)
    log_ref = result.get("log", {})
    log_path = Path(str(log_ref.get("path", "")))
    _require(
        log_path.is_file() and base._file_sha256(log_path) == log_ref.get("sha256"),
        f"formal log hash mismatch: {row.get('branch')}/{row.get('config')}",
    )
    timing = _parse_phase_timing(
        log_path.read_text(encoding="utf-8", errors="strict"),
        expected_layers=expected_layers,
    )
    elapsed = result.get("elapsed_seconds")
    _require(
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and timing["phase_seconds_sum"] <= float(elapsed) + 3.0,
        f"formal phase time exceeds process elapsed: {row.get('config')}",
    )
    return {
        **timing,
        "formal_process_elapsed_seconds": float(elapsed),
        "result": {"path": str(result_path), "sha256": base._file_sha256(result_path)},
        "log": {"path": str(log_path), "sha256": base._file_sha256(log_path)},
    }


def build() -> dict[str, Any]:
    formal_plan = formal._load_plan()
    formal_audit = _verify_final_audit(
        formal.FORMAL_AUDIT_PATH,
        identity_field="formal_id",
        identity=formal.FORMAL_ID,
        expected_counts={"formal_checkpoints": 40},
    )
    runtime_audit = _verify_live_cache_audit(
        runtime_cache.AUDIT_PATH,
        build=runtime_cache.build,
        expected_id=runtime_cache.AUDIT_ID,
    )
    quality_plan = quality._load_plan()
    quality_audit = _verify_final_audit(
        quality.QUALITY_AUDIT_PATH,
        identity_field="quality_id",
        identity=quality.QUALITY_ID,
        expected_counts={
            "runs": 40,
            "wikitext2_ppl": 40,
            "paper_qa_task_runs": 400,
        },
    )
    quality_cache_audit = _verify_live_cache_audit(
        quality_cache.AUDIT_PATH,
        build=quality_cache.build,
        expected_id=quality_cache.AUDIT_ID,
    )
    reasoning_plan = reasoning._load_plan()
    reasoning_audit = _verify_final_audit(
        reasoning.REASONING_AUDIT_PATH,
        identity_field="reasoning_id",
        identity=reasoning.REASONING_ID,
        expected_counts={
            "generation_runs": 120,
            "gsm8k_runs": 40,
            "math_500_runs": 40,
            "humaneval_plus_generations": 40,
            "humaneval_plus_official_scores": 40,
        },
    )

    formal_rows = {
        f"{row['branch']}/{row['config']}": row for row in formal_audit["rows"]
    }
    quality_rows = {
        f"{row['branch']}/{row['config']}": row for row in quality_audit["rows"]
    }
    reasoning_rows = {
        f"{row['branch']}/{row['config']}": row
        for row in reasoning_audit["summaries"]
    }
    expected_keys = {
        f"{branch}/{config}"
        for branch, config in formal._balanced_pairs()
    }
    _require(
        set(formal_rows) == set(quality_rows) == set(reasoning_rows) == expected_keys,
        "formal/quality/reasoning release matrices differ",
    )

    producers = {model: _producer_timing(model) for model in MODEL_LAYERS}
    rows = []
    for branch, config in formal._balanced_pairs():
        key = f"{branch}/{config}"
        model, setting, comparison_id = _config_identity(config)
        timing = _formal_timing(
            formal_rows[key], expected_layers=MODEL_LAYERS[model]
        )
        allocated = float(
            producers[model]["per_branch_setting_allocation_seconds"]
        )
        core_seconds = float(timing["phase_seconds_sum"])
        gpu_hours = (core_seconds + allocated) / 3600.0
        qmetrics = quality_rows[key]["metrics"]
        rmetrics = reasoning_rows[key]
        metrics = {
            "kl_raw": float(qmetrics["wikitext2"]["kl"]),
            "ppl": float(qmetrics["wikitext2"]["ppl"]),
            "qa_avg": float(qmetrics["paper_qa"]["average"]),
            "gsm8k": 100.0 * float(rmetrics["gsm8k_pass_at_1"]),
            "math_500": 100.0 * float(rmetrics["math_500_pass_at_1"]),
            "humaneval_base": 100.0 * float(
                rmetrics["humaneval_base_pass_at_1"]
            ),
            "humaneval_plus": 100.0 * float(
                rmetrics["humaneval_plus_pass_at_1"]
            ),
        }
        numeric = [*metrics.values(), gpu_hours]
        _require(
            all(math.isfinite(value) and value >= 0 for value in numeric),
            f"invalid release metric: {key}",
        )
        rows.append(
            {
                "comparison_id": comparison_id,
                "branch": branch,
                "method": METHOD_NAMES[branch],
                "config": config,
                "model": model,
                "model_name": MODEL_NAMES[model],
                "setting": SETTING_NAMES[setting],
                "selected_lr": float(formal_rows[key]["selected_lr"]),
                "metrics": metrics,
                "quantization_gpu_hours": gpu_hours,
                "gpu_hour_accounting": {
                    "scope": (
                        "Fusing LN + Rotating + Quantising layers + "
                        "one-quarter of the model SDPA static/Fisher producer; "
                        "model load, checkpoint I/O and evaluation excluded"
                    ),
                    "phase_seconds": {
                        phase: timing[phase]["seconds"] for phase in PHASES
                    },
                    "formal_core_seconds": core_seconds,
                    "shared_producer_seconds": producers[model]["producer_seconds"],
                    "shared_producer_amortization_divisor": (
                        PRODUCER_AMORTIZATION_DIVISOR
                    ),
                    "allocated_shared_producer_seconds": allocated,
                    "total_seconds": core_seconds + allocated,
                    "timing_evidence": timing,
                    "producer_evidence": producers[model],
                },
                "formal": formal_rows[key],
                "quality": quality_rows[key],
                "reasoning": rmetrics,
            }
        )
    rows.sort(
        key=lambda row: (
            int(str(row["comparison_id"])[1:]),
            {"REALQ-F": 0, "REALQ-S": 1}[str(row["method"])],
        )
    )
    _require(len(rows) == 40, "release is not 40 REALQ rows")
    summaries = {}
    for method in METHOD_NAMES.values():
        values = [
            float(row["quantization_gpu_hours"])
            for row in rows
            if row["method"] == method
        ]
        _require(len(values) == 20, f"release method count mismatch: {method}")
        summaries[method] = {
            "count": len(values),
            "total": sum(values),
            "mean": sum(values) / len(values),
        }

    def evidence(path: Path, fingerprint: str) -> dict[str, Any]:
        return {
            "path": str(path),
            "sha256": base._file_sha256(path),
            "fingerprint": fingerprint,
        }

    value: dict[str, Any] = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "status": "complete",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "row_count": len(rows),
        "protocol": {
            "attention_backend": "deterministic math-SDPA",
            "learning_rates_retuned_for_sdpa": False,
            "exact_gptaq_guidedquant_token_physical_files": True,
            "exact_gptaq_guidedquant_sdpa_bf16_reference_files": True,
            "calibration_shape": "256 x [2048] torch.int64",
            "gpu_hour_formula": (
                "(Fusing LN + Rotating + Quantising layers + "
                "model SDPA static/Fisher producer / 4) / 3600"
            ),
        },
        "rows": rows,
        "gpu_hour_summary": summaries,
        "inputs": {
            "formal_plan": evidence(
                formal.FORMAL_PLAN_PATH, formal_plan["formal_plan_fingerprint"]
            ),
            "formal_audit": evidence(
                formal.FORMAL_AUDIT_PATH, formal_audit["audit_fingerprint"]
            ),
            "runtime_cache_audit": evidence(
                runtime_cache.AUDIT_PATH, runtime_audit["audit_fingerprint"]
            ),
            "quality_plan": evidence(
                quality.QUALITY_PLAN_PATH, quality_plan["quality_plan_fingerprint"]
            ),
            "quality_audit": evidence(
                quality.QUALITY_AUDIT_PATH, quality_audit["audit_fingerprint"]
            ),
            "quality_cache_audit": evidence(
                quality_cache.AUDIT_PATH,
                quality_cache_audit["audit_fingerprint"],
            ),
            "reasoning_plan": evidence(
                reasoning.REASONING_PLAN_PATH,
                reasoning_plan["reasoning_plan_fingerprint"],
            ),
            "reasoning_audit": evidence(
                reasoning.REASONING_AUDIT_PATH,
                reasoning_audit["audit_fingerprint"],
            ),
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": base._file_sha256(Path(__file__).resolve()),
        },
    }
    value["release_fingerprint"] = base._canonical_sha256(value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RELEASE_PATH)
    args = parser.parse_args(argv)
    try:
        value = build()
        output = args.output.expanduser().resolve()
        base._atomic_json(output, value)
        print(
            json.dumps(
                {
                    "status": value["status"],
                    "row_count": value["row_count"],
                    "gpu_hour_summary": value["gpu_hour_summary"],
                    "release_fingerprint": value["release_fingerprint"],
                    "output": str(output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
