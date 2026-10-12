"""Export a full-pipeline pilot comparison with raw values and provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--configuration-label", required=True)
    args = parser.parse_args()
    path = args.audit / "comparison.json"
    report = json.loads(path.read_text())
    if report.get("scope") != "full_pipeline" or not report.get("strict_measured_memory_gate_passed"):
        raise ValueError("requires complete exactness and measured-memory gates")
    fields = {row["metric"]: row for row in report["global_metrics"]}
    rows = []
    for field, label, divisor in (
        ("max_nvml_memory_used_mib", "Device (NVML)", 1),
        ("full_process_peak_allocated_bytes", "CUDA allocated", 2**20),
        ("full_process_peak_reserved_bytes", "CUDA reserved", 2**20),
    ):
        metric = fields[field]
        rows.append(dict(metric=label, control_mib=metric["control"]/divisor,
                         candidate_mib=metric["candidate"]/divisor))
    rows.append(dict(metric="Host RSS", control_mib=report["control_host_rss_peak_bytes"]/2**20,
                     candidate_mib=report["candidate_host_rss_peak_bytes"]/2**20))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2.5, 1]})
    for ax, group in zip(axes, (rows[:3], rows[3:])):
        positions = list(range(len(group)))
        for shift, key, name, color in ((-.18, "control_mib", "Control", "#456D9A"),
                                       (.18, "candidate_mib", "CPU offload", "#D4813B")):
            bars = ax.bar([p+shift for p in positions], [r[key] for r in group], .36, label=name, color=color)
            ax.bar_label(bars, labels=[f"{r[key]:,.0f}" for r in group], padding=3, fontsize=9)
        ax.set_xticks(positions, [r["metric"] for r in group])
        ax.set_ylabel("Peak memory (MiB)")
        ax.set_ylim(0, max(max(r["control_mib"], r["candidate_mib"]) for r in group)*1.18)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Complete precompute-to-evaluation GPU peak")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_title("Host cost")
    fig.suptitle(f"{args.model_label}: saved-activation CPU offload (one paired pilot)", fontsize=12)
    fig.text(.5, .02, args.configuration_label + "; exact weights, layer outputs, KL/PPL. Host RSS may double-count shared pages.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0,.06,1,.95))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"full_pipeline_peak_tradeoff.{extension}", dpi=180)
    with (args.output_dir / "plot_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "audit_file": str(path.resolve()), "audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": report["scope"], "paired_repetitions": 1, "unit": "MiB", "model_label": args.model_label,
        "configuration_label": args.configuration_label, "figure_is_not_formal_scale_evidence": True,
    }, indent=2))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
