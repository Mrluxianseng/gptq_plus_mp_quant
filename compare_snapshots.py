"""
Tolerant tensor diff between two .pt snapshots saved by verify_gptq_plus.py.

Usage:
    python compare_snapshots.py <golden.pt> <current.pt> [--atol 1e-3] [--rtol 1e-4]

Prints per-tensor max/mean/abs diff (sorted by worst) and a PASS/FAIL summary.
Exits with code 0 on pass, 1 on fail — suitable for CI.
"""
import argparse
import sys

import torch


def compare(golden_path: str, current_path: str, atol: float, rtol: float,
            top_k: int = 15) -> bool:
    golden = torch.load(golden_path, map_location="cpu")
    current = torch.load(current_path, map_location="cpu")

    ok = True
    missing = sorted(set(golden) - set(current))
    extra = sorted(set(current) - set(golden))
    if missing:
        ok = False
        print(f"[cmp] keys only in golden ({len(missing)}): {missing[:5]}...")
    if extra:
        ok = False
        print(f"[cmp] keys only in current ({len(extra)}): {extra[:5]}...")

    common = sorted(set(golden) & set(current))
    print(f"[cmp] golden={golden_path}")
    print(f"[cmp] current={current_path}")
    print(f"[cmp] comparing {len(common)} tensors atol={atol} rtol={rtol}")
    rows = []
    mismatches = 0
    for key in common:
        a = golden[key].float()
        b = current[key].float()
        if a.shape != b.shape:
            print(f"[cmp] SHAPE  {key}: golden={a.shape} current={b.shape}")
            ok = False
            mismatches += 1
            continue
        diff = (a - b).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = a.abs().clamp(min=1e-12)
        max_rel = (diff / denom).max().item()
        ref = a.abs().max().item()
        matches = torch.allclose(a, b, atol=atol, rtol=rtol)
        if not matches:
            ok = False
            mismatches += 1
        rows.append((max_abs, max_rel, mean_abs, ref, matches, key))

    rows.sort(reverse=True)
    print(f"[cmp] mismatches: {mismatches}/{len(common)}; top {top_k} by max_abs:")
    for max_abs, max_rel, mean_abs, ref, matches, key in rows[:top_k]:
        tag = "OK  " if matches else "FAIL"
        print(
            f"  [{tag}] {key}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
            f"max_rel={max_rel:.3e} ref={ref:.3e}"
        )

    # Also print global stats for a quick numeric summary.
    if rows:
        max_abs_all = max(r[0] for r in rows)
        mean_abs_all = sum(r[2] for r in rows) / len(rows)
        print(f"[cmp] global: max_abs={max_abs_all:.3e} mean_abs(avg)={mean_abs_all:.3e}")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden", type=str, help="path to golden snapshot .pt")
    parser.add_argument("current", type=str, help="path to current snapshot .pt")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    ok = compare(args.golden, args.current, args.atol, args.rtol, args.top_k)
    if ok:
        print("[cmp] PASS — all tensors within tolerance")
    else:
        print("[cmp] FAIL — at least one tensor exceeds tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
