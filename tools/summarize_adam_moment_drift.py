"""Summarize the read-only REAL-Q Adam moment drift JSONL observer."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            grouped[(int(item["layer"]), str(item["module"]))].append(item)

    results = []
    for (layer, module), records in sorted(grouped.items()):
        records.sort(key=lambda row: int(row["adam_step"]))
        later = records[1:]
        if not later:
            later = records
        final = records[-1]
        row = {
            "layer": layer,
            "module": module,
            "refreshes": len(records),
            "final_step": int(final["adam_step"]),
            "final_active_columns": int(final["active_columns"]),
            "final_mean_vhat_to_g2_ratio": float(final["mean_vhat_to_g2_ratio"]),
            "median_row_log_profile_pearson_after_first": statistics.median(
                float(x["row_log_profile_pearson"]) for x in later
            ),
            "median_tile_log_profile_pearson_after_first": statistics.median(
                float(x["tile_log_profile_pearson"]) for x in later
            ),
            "median_row_log_profile_rmse_after_first": statistics.median(
                float(x["row_log_profile_rmse"]) for x in later
            ),
            "median_tile_log_profile_rmse_after_first": statistics.median(
                float(x["tile_log_profile_rmse"]) for x in later
            ),
            "median_log10_scale_ratio_after_first": statistics.median(
                float(x["log10_mean_vhat_to_g2_ratio"]) for x in later
            ),
            "records": records,
        }
        results.append(row)

    output = args.output or args.input.with_name("moment_drift_summary.json")
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# REAL-Q Adam moment-drift observer summary",
        "",
        f"Input: `{args.input}`",
        f"Refresh records: {sum(int(x['refreshes']) for x in results)}; layer/module groups: {len(results)}.",
        "First refresh has \u005chat(v)=g² by Adam construction, so median profile metrics omit step 1.",
        "",
        "| Layer | Module | Refreshes | Final mean vhat/g² | Median row corr | Median tile corr | Median row RMSE | Median tile RMSE | Median log10 scale ratio |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['layer']} | `{r['module']}` | {r['refreshes']} | "
            f"{r['final_mean_vhat_to_g2_ratio']:.4g} | "
            f"{r['median_row_log_profile_pearson_after_first']:.3f} | "
            f"{r['median_tile_log_profile_pearson_after_first']:.3f} | "
            f"{r['median_row_log_profile_rmse_after_first']:.3f} | "
            f"{r['median_tile_log_profile_rmse_after_first']:.3f} | "
            f"{r['median_log10_scale_ratio_after_first']:.3f} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
