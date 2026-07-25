"""
Analyze per-module saliency scores from a cached static precompute file.

Usage:
    python scripts/analyze_mp_saliency.py \
        --cache_path ./cache/saliency/Qwen3-0.6B-wikitext2_s512_blk2048_rot0_seed42_g4 \
        --mp_target_avg_bits 3.5 \
        --mp_high_bits 4 --mp_low_bits 3

Outputs:
    1. Per-module sensitivity table (all four metrics side-by-side)
    2. Bit allocation under each metric (for 3.5-bit target)
    3. Metric correlation matrix
    4. Per-layer sensitivity summary (which layers are most sensitive)
    5. Per-type sensitivity summary (which module types are most sensitive)
"""
import argparse
import os
import sys

import torch

# allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


METRICS = ["fisher_mean", "fisher_max", "fisher_topk", "random"]


def module_score(sal, metric, topk_ratio=0.1, rng_val=None):
    flat = sal.float().reshape(-1)
    if metric == "fisher_mean":
        return flat.mean().item()
    elif metric == "fisher_max":
        return flat.max().item()
    elif metric == "fisher_topk":
        k = max(1, int(topk_ratio * flat.numel()))
        return flat.topk(k).values.mean().item()
    elif metric == "random":
        return rng_val if rng_val is not None else float(torch.rand(1).item())
    raise ValueError(metric)


def build_scores(saliency_by_layer, topk_ratio=0.1):
    """Returns {(layer_idx, module_name): {metric: score}} and a stable random mapping."""
    rng = {}
    scores = {}
    for layer_idx, layer_sal in enumerate(saliency_by_layer):
        if layer_sal is None:
            continue
        for module_name, sal in layer_sal.items():
            key = (layer_idx, module_name)
            r = float(torch.rand(1).item())
            rng[key] = r
            scores[key] = {
                m: module_score(sal, m, topk_ratio, r if m == "random" else None)
                for m in METRICS
            }
    return scores


def assign_bits(scores_dict, metric, target_avg, high_bits, low_bits):
    ratio = (target_avg - low_bits) / (high_bits - low_bits)
    items = sorted(scores_dict.items(), key=lambda x: x[1][metric], reverse=True)
    n_high = int(len(items) * ratio)
    return {key: (high_bits if i < n_high else low_bits) for i, (key, _) in enumerate(items)}


def corr(a, b):
    import math
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da * db > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", required=True,
                        help="Path to saliency cache directory (contains rank_*.pt) or a single .pt file")
    parser.add_argument("--mp_target_avg_bits", type=float, default=3.5)
    parser.add_argument("--mp_high_bits", type=int, default=4)
    parser.add_argument("--mp_low_bits", type=int, default=3)
    parser.add_argument("--mp_topk_ratio", type=float, default=0.1)
    parser.add_argument("--top_n", type=int, default=20, help="Show top-N modules by fisher_mean")
    args = parser.parse_args()

    # Load cache — supports both a single .pt and the rank_0.pt directory layout
    cache_path = args.cache_path
    if os.path.isdir(cache_path):
        rank_file = os.path.join(cache_path, "rank_0.pt")
        if not os.path.exists(rank_file):
            # try finding any .pt
            candidates = [f for f in os.listdir(cache_path) if f.endswith(".pt")]
            if not candidates:
                sys.exit(f"No .pt files found in {cache_path}")
            rank_file = os.path.join(cache_path, sorted(candidates)[0])
        cache_file = rank_file
    else:
        cache_file = cache_path

    print(f"Loading cache: {cache_file}")
    data = torch.load(cache_file, map_location="cpu", weights_only=True)
    saliency_by_layer = data["saliency"]

    scores = build_scores(saliency_by_layer, args.mp_topk_ratio)
    keys = sorted(scores.keys())
    n = len(keys)
    print(f"\nFound {n} modules across {len(saliency_by_layer)} layers.\n")

    # --- Table: per-module scores (top N by fisher_mean) ---
    sorted_by_mean = sorted(keys, key=lambda k: scores[k]["fisher_mean"], reverse=True)
    top = sorted_by_mean[:args.top_n]
    header = f"{'Layer':>5}  {'Module':<30}  {'fisher_mean':>12}  {'fisher_max':>11}  {'fisher_topk':>12}  {'random':>8}"
    print("=== Top-{} most sensitive modules (by fisher_mean) ===".format(args.top_n))
    print(header)
    print("-" * len(header))
    for key in top:
        li, mn = key
        s = scores[key]
        print(f"{li:>5}  {mn:<30}  {s['fisher_mean']:>12.6e}  {s['fisher_max']:>11.6e}"
              f"  {s['fisher_topk']:>12.6e}  {s['random']:>8.4f}")

    # --- Metric correlation matrix ---
    vals = {m: [scores[k][m] for k in keys] for m in METRICS}
    print("\n=== Pearson correlation between saliency metrics ===")
    print(f"{'':>14}", end="")
    for m in METRICS:
        print(f"  {m:>13}", end="")
    print()
    for m1 in METRICS:
        print(f"{m1:>14}", end="")
        for m2 in METRICS:
            c = corr(vals[m1], vals[m2])
            print(f"  {c:>13.4f}", end="")
        print()

    # --- Bit allocation under each metric ---
    bit_maps = {m: assign_bits(scores, m, args.mp_target_avg_bits, args.mp_high_bits, args.mp_low_bits)
                for m in METRICS}
    print(f"\n=== Bit allocations (target={args.mp_target_avg_bits} bits, "
          f"high={args.mp_high_bits}, low={args.mp_low_bits}) ===")
    for m in METRICS:
        bm = bit_maps[m]
        n_h = sum(1 for b in bm.values() if b == args.mp_high_bits)
        n_l = sum(1 for b in bm.values() if b == args.mp_low_bits)
        avg = sum(bm.values()) / len(bm)
        print(f"  {m:<15}: {n_h} modules @{args.mp_high_bits}bit + {n_l} modules @{args.mp_low_bits}bit  "
              f"(avg={avg:.4f} bits)")

    # Agreement between metrics: fraction of modules assigned same bits as fisher_mean
    print()
    ref_bm = bit_maps["fisher_mean"]
    for m in METRICS[1:]:
        agree = sum(1 for k in keys if bit_maps[m][k] == ref_bm[k])
        print(f"  Agreement {m} vs fisher_mean: {agree}/{n} = {100*agree/n:.1f}%")

    # --- Per-layer summary ---
    layer_indices = sorted(set(li for li, _ in keys))
    print("\n=== Per-layer sensitivity (fisher_mean, averaged over modules) ===")
    print(f"{'Layer':>5}  {'#Modules':>8}  {'mean_score':>12}  {'max_score':>11}")
    for li in layer_indices:
        layer_keys = [k for k in keys if k[0] == li]
        layer_scores = [scores[k]["fisher_mean"] for k in layer_keys]
        avg_s = sum(layer_scores) / len(layer_scores)
        max_s = max(layer_scores)
        print(f"{li:>5}  {len(layer_keys):>8}  {avg_s:>12.6e}  {max_s:>11.6e}")

    # --- Per-type summary ---
    type_map = {}
    for li, mn in keys:
        mtype = mn.split(".")[-1]
        type_map.setdefault(mtype, []).append((li, mn))
    print("\n=== Per-type sensitivity (fisher_mean, averaged over all layers) ===")
    print(f"{'Type':<25}  {'#Modules':>8}  {'mean_score':>12}  {'bit (3.5target)':>16}")
    type_avgs = {}
    for mtype, type_keys in type_map.items():
        s_vals = [scores[k]["fisher_mean"] for k in type_keys]
        type_avgs[mtype] = sum(s_vals) / len(s_vals)
    for mtype in sorted(type_avgs, key=lambda t: type_avgs[t], reverse=True):
        type_keys = type_map[mtype]
        avg_bits = sum(bit_maps["fisher_mean"][(li, mn)] for li, mn in type_keys) / len(type_keys)
        print(f"{mtype:<25}  {len(type_keys):>8}  {type_avgs[mtype]:>12.6e}  {avg_bits:>16.2f}")


if __name__ == "__main__":
    main()
