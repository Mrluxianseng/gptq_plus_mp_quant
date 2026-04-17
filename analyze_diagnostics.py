"""
Inspector for diagnostics dumped by `gptq_plus_utils.diagnostics.DiagnosticRecorder`.

Usage examples:

    # List everything captured for one module
    python analyze_diagnostics.py ./outputs/.../diagnostics/layer_02_mlp_down_proj

    # Show per-block stats side by side with the loss trajectory
    python analyze_diagnostics.py ./outputs/.../diagnostics/layer_02_mlp_down_proj --summary

    # Compare specific blocks between two dumps (e.g. layer 2 vs layer 1 baseline)
    python analyze_diagnostics.py \
        --compare ./outputs/.../layer_02_mlp_down_proj \
        --against  ./outputs/.../layer_01_mlp_down_proj \
        --subgroup 0

    # Pull a specific tensor
    python analyze_diagnostics.py ./outputs/.../layer_02_mlp_down_proj \
        --load subgroup_00/block_01/G_Update
"""
import argparse
import json
import os
from typing import Dict, Optional

import torch


def _load_meta(root: str) -> dict:
    path = os.path.join(root, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.float()
    if t.numel() == 0:
        return {"numel": 0}
    absv = t.abs()
    return {
        "shape": tuple(t.shape),
        "numel": t.numel(),
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std(unbiased=False).item() if t.numel() > 1 else 0.0,
        "abs_max": absv.max().item(),
        "abs_mean": absv.mean().item(),
        "row_l2_mean": (
            torch.linalg.norm(t, dim=1).mean().item()
            if t.ndim == 2 else None
        ),
        "has_nan": bool(torch.isnan(t).any().item()),
    }


def list_tags(root: str) -> Dict[str, str]:
    """Walk the dump directory, return {relative_tag_path: absolute_pt_path}."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".pt"):
                full = os.path.join(d, f)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                rel = rel[:-3]  # strip .pt
                out[rel] = full
    return out


def cmd_summary(root: str, subgroup: Optional[int] = None):
    meta = _load_meta(root)
    print(f"[{root}]")
    if not meta:
        print("  (no meta.json — ran without --diagnose_targets?)")
    else:
        print(f"  module:  layer {meta.get('layer_idx')}  {meta.get('module_name')}")
        per_sub = meta.get("per_subgroup_blocks", {})
        subs = sorted(per_sub.keys(), key=int)
        if subgroup is not None:
            subs = [str(subgroup)] if str(subgroup) in per_sub else []
        for sub_key in subs:
            blocks = per_sub[sub_key]
            print(f"\n  subgroup {sub_key}:")
            header = f"  {'block':>6}  {'loss':>12}  {'prev':>12}  {'ratio':>8}  spike"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for entry in blocks:
                loss = entry.get("loss")
                prev = entry.get("prev_loss")
                ratio = (loss / prev) if (loss is not None and prev is not None and prev > 0) else None
                ratio_str = f"{ratio:8.3f}" if ratio is not None else "    n/a "
                spike_str = "  ★" if entry.get("is_spike") else ""
                loss_str = f"{loss:12.6e}" if loss is not None else "        n/a "
                prev_str = f"{prev:12.6e}" if prev is not None else "        n/a "
                print(f"  {entry['block']:>6d}  {loss_str}  {prev_str}  {ratio_str}{spike_str}")


def cmd_list(root: str):
    tags = list_tags(root)
    for rel in sorted(tags.keys()):
        t = torch.load(tags[rel], map_location="cpu")
        if isinstance(t, torch.Tensor):
            stats = _tensor_stats(t)
            print(f"{rel:<70} shape={stats['shape']} dtype={t.dtype} abs_max={stats['abs_max']:.3e} abs_mean={stats['abs_mean']:.3e}")
        else:
            print(f"{rel:<70} {type(t).__name__}")


def cmd_load(root: str, tag: str):
    path = os.path.join(root, tag + ".pt")
    if not os.path.exists(path):
        candidates = [k for k in list_tags(root) if tag in k]
        print(f"Not found: {tag}")
        print(f"Candidates: {candidates[:10]}")
        return
    t = torch.load(path, map_location="cpu")
    print(f"Tag: {tag}")
    print(f"Tensor shape: {tuple(t.shape)}   dtype: {t.dtype}")
    stats = _tensor_stats(t) if isinstance(t, torch.Tensor) else {}
    for k, v in stats.items():
        print(f"  {k}: {v}")
    # Print a small sample.
    if isinstance(t, torch.Tensor) and t.ndim <= 2:
        snippet_rows = min(4, t.shape[0]) if t.ndim >= 1 else 0
        snippet_cols = min(8, t.shape[1]) if t.ndim == 2 else 0
        if t.ndim == 2 and snippet_rows > 0 and snippet_cols > 0:
            print("\n  top-left snippet:")
            print(t[:snippet_rows, :snippet_cols])
        elif t.ndim == 1:
            print(t[: min(8, t.numel())])
        else:
            print(t)


def cmd_compare(left_root: str, right_root: str, subgroup: Optional[int] = None):
    """Side-by-side diff of per-block stats between two dumps."""
    left_meta = _load_meta(left_root)
    right_meta = _load_meta(right_root)
    print(f"LEFT:  {left_root}   (layer {left_meta.get('layer_idx')} {left_meta.get('module_name')})")
    print(f"RIGHT: {right_root}  (layer {right_meta.get('layer_idx')} {right_meta.get('module_name')})")

    left_tags = list_tags(left_root)
    right_tags = list_tags(right_root)
    shared = sorted(set(left_tags) & set(right_tags))
    if subgroup is not None:
        shared = [t for t in shared if f"subgroup_{subgroup:02d}" in t]

    header = f"  {'tag':<60}  {'L abs_max':>12}  {'R abs_max':>12}  {'L abs_mean':>12}  {'R abs_mean':>12}"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for tag in shared:
        try:
            lt = torch.load(left_tags[tag], map_location="cpu")
            rt = torch.load(right_tags[tag], map_location="cpu")
        except Exception as e:
            print(f"  {tag:<60}  load-error: {e}")
            continue
        if not (isinstance(lt, torch.Tensor) and isinstance(rt, torch.Tensor)):
            continue
        ls = _tensor_stats(lt)
        rs = _tensor_stats(rt)
        print(
            f"  {tag:<60}  {ls.get('abs_max', 0):12.3e}  {rs.get('abs_max', 0):12.3e}  "
            f"{ls.get('abs_mean', 0):12.3e}  {rs.get('abs_mean', 0):12.3e}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="path to a single dump directory (e.g. .../layer_02_mlp_down_proj)")
    ap.add_argument("--summary", action="store_true", help="show per-block loss trajectory + spike flags")
    ap.add_argument("--list", action="store_true", help="list every .pt file with abs_max/abs_mean stats")
    ap.add_argument("--load", type=str, default=None, help="tag to load (e.g. subgroup_00/block_01/G_Update)")
    ap.add_argument("--compare", type=str, default=None, help="left dump path for diff mode")
    ap.add_argument("--against", type=str, default=None, help="right dump path for diff mode")
    ap.add_argument("--subgroup", type=int, default=None, help="restrict summary/compare to this subgroup")
    args = ap.parse_args()

    if args.compare or args.against:
        if not (args.compare and args.against):
            ap.error("--compare and --against must both be provided")
        cmd_compare(args.compare, args.against, subgroup=args.subgroup)
        return

    if not args.root:
        ap.error("root dump path is required unless --compare/--against is used")

    if args.load:
        cmd_load(args.root, args.load)
    elif args.list:
        cmd_list(args.root)
    else:
        cmd_summary(args.root, subgroup=args.subgroup)


if __name__ == "__main__":
    main()
