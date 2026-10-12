"""Publication figures from a completed, accepted memory campaign export.

Shows every repetition and the range of run peaks. These are not temporal
mean memory values or speed measurements. No failed/extreme run is removed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--configuration-label", required=True)
    args = parser.parse_args()
    summary_path = args.summary_directory / "summary.json"
    values_path = args.summary_directory / "paired_values.csv"
    summary = json.loads(summary_path.read_text())
    if not all(summary[key] for key in (
        "all_requested_pairs_finished", "campaign_valid",
        "all_audited_pairs_exact", "all_audited_pairs_pass_memory_gate",
    )) or summary["scope"] != "full_pipeline":
        raise ValueError("requires a completed, exact full-pipeline campaign accepted by all memory gates")
    with values_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    repeats = summary["audited_pairs"]
    if len(rows) != 2 * repeats:
        raise ValueError("CSV must retain both arms of every audited pair")
    metrics = (
        ("nvml_peak_mib", "Device (NVML)"),
        ("allocated_peak_mib", "CUDA allocated"),
        ("reserved_peak_mib", "CUDA reserved"),
        ("host_rss_peak_mib", "Host RSS"),
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={"width_ratios": [2.6, 1]})
    for ax, group in zip(axes, (metrics[:3], metrics[3:])):
        for shift, side, label, color in (
            (-.2, "control", "Control", "#456D9A"),
            (.2, "candidate", "Candidate", "#D4813B"),
        ):
            values = [[float(row[field]) for row in rows if row["side"] == side] for field, _ in group]
            if any(len(v) != repeats for v in values):
                raise ValueError("missing or duplicate arm")
            means = [statistics.mean(v) for v in values]
            bars = ax.bar([i + shift for i in range(len(group))], means, .36, color=color, label=label,
                          yerr=[[m-min(v) for m,v in zip(means,values)], [max(v)-m for m,v in zip(means,values)]],
                          capsize=4, error_kw={"linewidth": 1})
            ax.bar_label(bars, labels=[f"{v:,.0f}" for v in means], padding=7, fontsize=9)
            for i, v in enumerate(values):
                offsets = [(j-(repeats-1)/2)*.065 for j in range(repeats)]
                ax.scatter([i+shift+o for o in offsets], v, color="#222222", marker="o", s=17, zorder=4)
        ax.set_xticks(range(len(group)), [label for _,label in group])
        ax.set_ylabel("Complete-process peak memory (MiB)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, ax.get_ylim()[1]*1.15)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title("GPU resource requirement")
    axes[1].set_title("Host cost")
    fig.suptitle(f"{args.model_label}: {repeats} complete paired pilots", fontsize=12)
    fig.text(.5, .035, args.configuration_label, ha="center", fontsize=8.5)
    fig.text(.5, .01, "Bars: means of run peaks; dots: every run; whiskers: min/max. Exact state, all layers and KL/PPL. RSS may count shared pages twice.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.95))
    args.output_directory.mkdir(parents=True, exist_ok=False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output_directory/f"complete_pipeline_repeated_peaks.{extension}", dpi=180)
    (args.output_directory/"plot_values.csv").write_bytes(values_path.read_bytes())
    (args.output_directory/"provenance.json").write_text(json.dumps({
        "campaign": summary["campaign"], "paired_repetitions": repeats,
        "summary": str(summary_path.resolve()), "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "values": str(values_path.resolve()), "values_sha256": hashlib.sha256(values_path.read_bytes()).hexdigest(),
        "model_label": args.model_label, "configuration_label": args.configuration_label,
        "not_formal_paper_scale": True, "means_are_run_peaks_not_time_averages": True,
    }, indent=2))
    print(args.output_directory.resolve())


if __name__ == "__main__":
    main()
