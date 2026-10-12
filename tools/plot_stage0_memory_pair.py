"""Export a scope-labelled pilot figure from a completed Stage-0 audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.audit / "comparison.json"
    report = json.loads(audit_path.read_text())
    if report.get("scope") != "stage0_only" or not report.get("exactness_passed"):
        raise ValueError("requires a successful Stage-0-only exactness audit")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = {r["metric"]: r for r in report["global_metrics"]}
    rows = []
    for key, label, divisor in (
        ("max_nvml_memory_used_mib", "Device (NVML)", 1),
        ("stage0_process_peak_allocated_bytes", "CUDA allocated", 2**20),
        ("stage0_process_peak_reserved_bytes", "CUDA reserved", 2**20),
    ):
        row = fields[key]
        rows.append({"metric": label, "control_mib": row["control"]/divisor,
                     "candidate_mib": row["candidate"]/divisor})
    rows.append({"metric": "Host RSS", "control_mib": report["control_host_rss_peak_bytes"]/2**20,
                 "candidate_mib": report["candidate_host_rss_peak_bytes"]/2**20})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2.5, 1]})
    colors = ("#456D9A", "#D4813B")
    for ax, group in zip(axes, (rows[:3], rows[3:])):
        positions = list(range(len(group)))
        for offset, key, name, color in zip((-.18, .18), ("control_mib", "candidate_mib"), ("Control", "CPU offload"), colors):
            bars = ax.bar([p+offset for p in positions], [r[key] for r in group], width=.36, label=name, color=color)
            ax.bar_label(bars, labels=[f"{r[key]:,.0f}" for r in group], padding=3, fontsize=9)
        ax.set_xticks(positions, [r["metric"] for r in group])
        ax.set_ylabel("Peak memory (MiB)")
        ax.set_ylim(0, max(max(r["control_mib"], r["candidate_mib"]) for r in group)*1.18)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Stage-0 GPU peak")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_title("Host cost")
    fig.suptitle("Qwen3-0.6B: saved-activation CPU offload (one paired pilot)", fontsize=12)
    fig.text(.5, .02, "Stage-0 only; 2 x 1024 calibration tokens. No quantization/KL/PPL claim. Host RSS may count shared pages twice.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"stage0_peak_tradeoff.{extension}", dpi=180)
    with (args.output_dir / "plot_values.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output_dir / "provenance.json").write_text(json.dumps({"audit_file": str(audit_path.resolve()), "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(), "scope": "stage0_only", "paired_repetitions": 1, "unit": "MiB", "figure_is_not_formal_scale_evidence": True},indent=2))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
