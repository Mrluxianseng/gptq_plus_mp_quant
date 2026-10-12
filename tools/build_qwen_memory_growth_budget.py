"""CPU-only logical memory budgets from real model configs, not measured peaks."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--groups", type=int, default=4)
    args = p.parse_args()
    if min(args.samples, args.seq_len, args.world_size, args.groups) <= 0 or args.samples % args.world_size:
        p.error("positive dimensions and samples divisible by world-size required")
    rows = []
    local = args.samples // args.world_size
    for model in args.models:
        raw = (model / "config.json").read_bytes()
        cfg = json.loads(raw)
        hidden, intermediate, layers = (int(cfg[name]) for name in ("hidden_size", "intermediate_size", "num_hidden_layers"))
        rows.append({"model": model.name, "config_sha256": hashlib.sha256(raw).hexdigest(),
                     "hidden": hidden, "intermediate": intermediate, "layers": layers,
                     "two_bf16_input_buffers_bytes_per_data_rank": 2 * local * args.seq_len * hidden * 2,
                     "all_block_fp32_fisher_accumulators_bytes": layers * hidden * hidden * 4,
                     "all_block_bf16_fisher_cache_bytes": layers * hidden * hidden * 2,
                     "largest_ffn_fp32_group_hessian_bytes": args.groups * intermediate * intermediate * 4,
                     "largest_ffn_fp32_group_inverse_bytes_if_all_resident": args.groups * intermediate * intermediate * 4,
                     "down_proj_two_fp32_adam_moments_bytes_full_suffix": 2 * hidden * intermediate * 4})
    result = {"scope": "logical payload formulas; not measured or simultaneously live peaks",
              "conditions": {"samples": args.samples, "sequence_length": args.seq_len,
                             "world_size_data_sharding": args.world_size, "saliency_groups": args.groups,
                             "matrix_group_partitioning": "none; group matrices are not divided by world-size in these formulas"},
              "limitations": "Excludes allocator, workspaces, logits, activations, parameter copies, aliases and lifetime overlap. No hardware feasibility or multi-GPU validation claim.",
              "models": rows}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "budget.json").write_text(json.dumps(result, indent=2))
    lines = ["# Model-dependent logical memory budgets", "", result["scope"], "", result["limitations"], "",
             f"Calibration {args.samples}×{args.seq_len}; data ranks={args.world_size}; groups={args.groups}; BF16 inputs/FP32 accumulators.", "",
             "| Component | " + " | ".join(r["model"] + " (MiB)" for r in rows) + " |",
             "| --- | " + " | ".join("---:" for _ in rows) + " |"]
    for key in rows[0]:
        if "bytes" in key:
            lines.append("| " + key + " | " + " | ".join(f"{r[key] / 2**20:.2f}" for r in rows) + " |")
    (args.output_dir / "budget.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
