#!/usr/bin/env python
"""Turn a --horizon_trace into the per-module step-schedule exponent.

The schedule applied by --horizon_p weights a tail column's block-GD step by
its remaining update count h:

    w(h) = k * (h+1)^-p / sum_{i=1..k} i^-p

and the question this script answers is what p should be, from measurement
rather than from a sweep.

The argument for the schedule is SNR-maximisation: the weight on refresh n
should go as w_n ~ b_n / sigma_n^2, where b_n is the consistent part of the
refresh gradient and sigma_n its per-step noise.  Write both as powers of the
refresh index,

    b_n = B * n^alpha        sigma_n = S * n^s

then w_n ~ n^(alpha - 2s), and matching that dynamic range over a module of k
blocks (k^gamma) against the schedule's (k^p) gives

    p = gamma = alpha - 2s.

The earlier hand derivation assumed s = 0 and reported p = alpha ~ 0.43.  That
assumption was never checked; here both exponents are measured, gamma is the
answer, and alpha is kept alongside as the s = 0 special case.


Why this is a forward model and not a slope
-------------------------------------------
The trace records the two Adam moments over the live tail: M_n = E[mhat_n^2]
and V_n = E[vhat_n].  Neither is b_n^2 or sigma_n^2, and the gap between them
is not a constant:

  * mhat_n is an average of only n gradients, so it still carries
    Var = sigma^2 * (1-B1)(1+B1^n) / ((1+B1)(1-B1^n)) -- a factor that runs
    from 1.0 at n=1 to 0.063 at n=23.  Treating it as its asymptotic 0.0526
    overstates b^2 at small n and manufactures a downward slope, i.e. a
    spurious negative alpha.  At n=1 mhat IS the raw gradient and the two
    moments coincide exactly, which is visible in any trace.

  * mhat_n also lags: it is an exponentially weighted average of past
    gradients with an effective lag of B1/(1-B1) = 9 steps, so while b is
    growing mhat_n sits below b_n by an amount that itself varies with n --
    another slope that is not in the signal.

Both effects are exactly computable, so rather than correcting a slope this
fits the generating parameters through the EMA recursion.  With weights
w1[n,i] = (1-B1)B1^(n-i)/(1-B1^n) and w2[n,i] = (1-B2)B2^(n-i)/(1-B2^n),

    M_n = B^2 * p_n(alpha) + S^2 * q_n(s)
    V_n = B^2 * r_n(alpha) + S^2 * t_n(s)

    p_n(a) = (sum_i w1[n,i] i^a)^2      q_n(s) = sum_i w1[n,i]^2 i^(2s)
    r_n(a) =  sum_i w2[n,i] i^(2a)      t_n(s) = sum_i w2[n,i] i^(2s)

which is linear in (B^2, S^2) once (alpha, s) are fixed.  So: grid over
(alpha, s), solve the 2x2 weighted least squares in closed form, keep the
minimum.  `--estimator ratio` recovers the naive constant-c slope fit for
comparison -- it is reported next to the model fit so the size of the
correction is visible rather than assumed.

Usage
-----
    # 1. calibrate (production numerics, p = 0)
    HORIZON_TRACE=runs/trace/qwen06  ... bash scripts/formal_realq_compare.sh

    # 2. fit
    python scripts/fit_horizon_alpha.py runs/trace/qwen06 -o runs/alpha.json

    # 3. apply
    HORIZON_ALPHA=runs/alpha.json ... bash scripts/formal_realq_compare.sh
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

BETA1 = 0.9
BETA2 = 0.999
NL_S = chr(10)


# ---------------------------------------------------------------- loading
def load(prefix):
    """Read every rank shard under `prefix` and pool by (layer, module).

    Ranks hold disjoint row shards of the same module, so their series are
    measurements of the same quantity: the per-element means are averaged, not
    concatenated.  (The optimiser state is allocated at shard shape, so each
    rank's mean is already over live entries only -- there are no dead zeros
    diluting it.)
    """
    paths = sorted(glob.glob(prefix + ".rank*.jsonl"))
    if not paths:
        paths = sorted(glob.glob(prefix)) or sorted(glob.glob(prefix + "*.jsonl"))
    if not paths:
        raise SystemExit("no trace files matching %r" % (prefix + ".rank*.jsonl"))
    pooled = defaultdict(lambda: defaultdict(list))
    warn = set()
    meta = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("step") and rec["step"] != rec["n"]:
                    warn.add("step_ne_block")
                if rec.get("opt") not in (None, "adam"):
                    warn.add(rec.get("opt"))
                key = (rec["layer"], rec["module"])
                meta[key] = rec
                for n, b2, v in zip(rec["n"], rec["b2"], rec["v"]):
                    pooled[key][n].append((b2, v))
    out = {}
    for key, per_n in pooled.items():
        ns, ms, vs = [], [], []
        for n in sorted(per_n):
            rows = per_n[n]
            ns.append(n)
            ms.append(sum(r[0] for r in rows) / len(rows))
            vs.append(sum(r[1] for r in rows) / len(rows))
        out[key] = (np.array(ns, float), np.array(ms, float),
                    np.array(vs, float), meta[key])
    if "step_ne_block" in warn:
        print("WARNING: optimiser step count and block index disagree; the"
              + NL_S + "  refresh loop is not one step per block and the fit"
              + NL_S + "  is indexed on the wrong axis.")
    for w in sorted(warn - {"step_ne_block"}):
        print("WARNING: trace taken with grad_optimizer=%s; warm_adam seeds"
              % w + NL_S + "  the second moment with a prior, so E[vhat] is"
              + NL_S + "  not a pure gradient measurement.")
    return out, paths


# ------------------------------------------------------- forward EMA model
def _design(nmax, grid, beta1=BETA1, beta2=BETA2):
    """p_n(a), q_n(s), r_n(a), t_n(s) for every n<=nmax and every grid value.

    Returned shaped (len(grid), nmax) and indexed [g, n-1].
    """
    ns = np.arange(1, nmax + 1)
    P = np.zeros((len(grid), nmax))
    Q = np.zeros((len(grid), nmax))
    R = np.zeros((len(grid), nmax))
    T = np.zeros((len(grid), nmax))
    for n in ns:
        i = np.arange(1, n + 1, dtype=float)
        w1 = (1 - beta1) * beta1 ** (n - i) / (1 - beta1 ** n)
        w2 = (1 - beta2) * beta2 ** (n - i) / (1 - beta2 ** n)
        li = np.log(i)
        for gi, g in enumerate(grid):
            ia = np.exp(g * li)
            i2a = ia * ia
            P[gi, n - 1] = (w1 * ia).sum() ** 2
            Q[gi, n - 1] = (w1 * w1 * i2a).sum()
            R[gi, n - 1] = (w2 * i2a).sum()
            T[gi, n - 1] = (w2 * i2a).sum()
    return P, Q, R, T


def fit_model(ns, Ms, Vs, grid, design, n_min=1):
    """Grid over (alpha, s); closed-form (B^2, S^2) at each node.

    Residuals are relative (each row divided by its own observation), so the
    fit is not dominated by the largest n.
    """
    P, Q, R, T = design
    keep = ns >= n_min
    if keep.sum() < 3:
        return None
    idx = (ns[keep] - 1).astype(int)
    M = Ms[keep]
    V = Vs[keep]
    if np.any(M <= 0) or np.any(V <= 0):
        return None

    # design columns, scaled to relative residuals
    a = P[:, idx] / M            # (G, N)  coefficient of B^2 in the M rows
    b = Q[:, idx] / M            # (G, N)  coefficient of S^2 in the M rows
    c = R[:, idx] / V            # (G, N)  coefficient of B^2 in the V rows
    d = T[:, idx] / V            # (G, N)  coefficient of S^2 in the V rows

    # normal equations, vectorised over the (alpha, s) grid
    Saa = (a * a).sum(1)[:, None] + (c * c).sum(1)[:, None]   # alpha only
    Sbb = (b * b).sum(1)[None, :] + (d * d).sum(1)[None, :]   # s only
    Sab = a @ b.T + c @ d.T                                   # both
    Sa = a.sum(1)[:, None] + c.sum(1)[:, None]
    Sb = b.sum(1)[None, :] + d.sum(1)[None, :]

    det = Saa * Sbb - Sab * Sab
    det = np.where(np.abs(det) < 1e-300, np.nan, det)
    x = (Sa * Sbb - Sb * Sab) / det          # B^2
    y = (Sb * Saa - Sa * Sab) / det          # S^2
    x = np.clip(x, 0.0, None)
    y = np.clip(y, 0.0, None)

    # residual sum of squares at each node: ||A z - 1||^2, 2N rows of ones
    n_rows = 2 * keep.sum()
    rss = (n_rows
           - 2 * (x * Sa + y * Sb)
           + x * x * Saa + y * y * Sbb + 2 * x * y * Sab)
    rss = np.where(np.isfinite(rss), rss, np.inf)
    ia, isx = np.unravel_index(np.argmin(rss), rss.shape)
    alpha, s = float(grid[ia]), float(grid[isx])
    best = float(rss[ia, isx])
    # 1 - RSS/TSS with TSS = n_rows (targets are all 1, mean 1 -> TSS is the
    # spread of the *fitted* values; use the plain relative-error scale here)
    return {
        "alpha": alpha,
        "s": s,
        "gamma": alpha - 2.0 * s,
        "B2": float(x[ia, isx]),
        "S2": float(y[ia, isx]),
        "rel_rms": math.sqrt(max(best, 0.0) / n_rows),
        "npts": int(keep.sum()),
    }


# --------------------------------------------------- naive slope estimator
def fit_ratio(ns, Ms, Vs, n_min=2):
    """The constant-c slope fit, kept only to show the size of the bias."""
    c_inf = (1 - BETA1) / (1 + BETA1)
    bp, sp = [], []
    for n, m, v in zip(ns, Ms, Vs):
        if n < n_min:
            continue
        b2 = (m - c_inf * v) / (1 - c_inf)
        s2 = (v - m) / (1 - c_inf)
        if b2 > 0:
            bp.append((n, math.sqrt(b2)))
        if s2 > 0:
            sp.append((n, math.sqrt(s2)))

    def slope(pts):
        if len(pts) < 3:
            return None
        lx = np.log([p[0] for p in pts])
        ly = np.log([p[1] for p in pts])
        return float(np.polyfit(lx, ly, 1)[0])

    a, s = slope(bp), slope(sp)
    if a is None or s is None:
        return None
    return {"alpha": a, "s": s, "gamma": a - 2 * s}


# ----------------------------------------------------------------- helpers
def median(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def quartiles(xs):
    if not len(xs):
        return (float("nan"),) * 2
    return (float(np.percentile(xs, 25)), float(np.percentile(xs, 75)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="the --horizon_trace path (no .rankN.jsonl)")
    ap.add_argument("-o", "--out", default="", help="write the exponent table here")
    ap.add_argument(
        "--granularity", default="name",
        choices=("name", "layer_name", "global"),
        help=(
            "name (default): one exponent per module name, pooled over layers. "
            "layer_name: one per instance, each from <=23 points. global: one."
        ),
    )
    ap.add_argument(
        "--use", default="gamma", choices=("gamma", "alpha"),
        help=(
            "gamma (default) = alpha - 2s, the full b/sigma^2 rule. alpha = "
            "the sigma-constant special case, i.e. the original derivation."
        ),
    )
    ap.add_argument(
        "--estimator", default="model", choices=("model", "ratio"),
        help=(
            "model (default): fit (alpha, s) through the EMA recursion. "
            "ratio: the naive constant-c log-log slope, biased low in alpha; "
            "reported alongside either way."
        ),
    )
    ap.add_argument("--n_min", type=int, default=1,
                    help="drop refreshes below this index (model fit handles "
                         "n=1 correctly, so the default keeps everything)")
    ap.add_argument("--grid_step", type=float, default=0.02)
    ap.add_argument("--grid_lo", type=float, default=-1.0)
    ap.add_argument("--grid_hi", type=float, default=1.5)
    args = ap.parse_args()

    data, paths = load(args.prefix)
    print("read %d shard(s), %d module instance(s)" % (len(paths), len(data)))

    nmax = max(int(ns.max()) for ns, _, _, _ in data.values())
    grid = np.arange(args.grid_lo, args.grid_hi + 1e-9, args.grid_step)
    print("building design over %d grid nodes x n<=%d ..." % (len(grid), nmax))
    design = _design(nmax, grid)

    fits, ratios = {}, {}
    for key, (ns, Ms, Vs, meta) in sorted(data.items()):
        f = fit_model(ns, Ms, Vs, grid, design, args.n_min)
        if f is None:
            continue
        f["nblk"] = meta.get("nblk")
        fits[key] = f
        r = fit_ratio(ns, Ms, Vs)
        if r:
            ratios[key] = r
    if not fits:
        raise SystemExit("no module produced a usable fit")

    by_name = defaultdict(list)
    for (layer, module), f in fits.items():
        by_name[module].append(f)

    print()
    hdr = ("%-26s %4s %4s  %-17s %-17s %-17s %6s  %s"
           % ("module", "inst", "nblk", "alpha (signal)", "s (noise)",
              "gamma=alpha-2s", "relerr", "naive gamma"))
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for module in sorted(by_name):
        fs = by_name[module]
        A = [f["alpha"] for f in fs]
        S = [f["s"] for f in fs]
        G = [f["gamma"] for f in fs]
        nv = [ratios[k]["gamma"] for k in fits if k[1] == module and k in ratios]
        qa, qs, qg = quartiles(A), quartiles(S), quartiles(G)
        rows.append((module, fs))
        print("%-26s %4d %4s  %6.3f[%5.2f,%5.2f] %6.3f[%5.2f,%5.2f] "
              "%6.3f[%5.2f,%5.2f] %6.1f%%  %6.3f"
              % (module, len(fs), fs[0].get("nblk"),
                 median(A), qa[0], qa[1], median(S), qs[0], qs[1],
                 median(G), qg[0], qg[1],
                 100 * median([f["rel_rms"] for f in fs]),
                 median(nv) if nv else float("nan")))
    allf = list(fits.values())
    print("-" * len(hdr))
    print("%-26s %4d %4s  %6.3f%18s %6.3f%18s %6.3f"
          % ("ALL", len(allf), "", median([f["alpha"] for f in allf]), "",
             median([f["s"] for f in allf]), "",
             median([f["gamma"] for f in allf])))

    med_s = median([f["s"] for f in allf])
    print()
    print("sigma exponent s: median %.3f  (the p = alpha derivation assumed 0)"
          % med_s)
    edge = [k for k, f in fits.items()
            if min(abs(f["alpha"] - args.grid_lo), abs(f["alpha"] - args.grid_hi),
                   abs(f["s"] - args.grid_lo), abs(f["s"] - args.grid_hi))
            < args.grid_step]
    if edge:
        print("WARNING: %d/%d fits landed on a grid edge -- the power-law form"
              % (len(edge), len(fits)) + NL_S +
              "  does not describe those traces and their exponent is a bound,"
              + NL_S + "  not an estimate: " +
              " ".join("%s.%s" % k for k in sorted(edge)[:6]))
    relerr = median([f["rel_rms"] for f in allf])
    print("model relative RMS residual: %.1f%%%s" % (
        100 * relerr,
        "  -- high; the power-law form may not describe this trace"
        if relerr > 0.25 else ""))

    field = args.use
    src = fits if args.estimator == "model" else ratios
    table = {}
    if args.granularity == "global":
        pass
    elif args.granularity == "name":
        for module, fs in rows:
            vals = [src[k][field] for k in src if k[1] == module]
            if vals:
                table[module] = median(vals)
    else:
        for k, f in src.items():
            table["%s.%s" % (k[0], k[1])] = f[field]
    table["__default__"] = median([f[field] for f in src.values()])

    print("using %s from the %s estimator at granularity=%s: %d key(s)"
          % (field, args.estimator, args.granularity, len(table)))

    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({
                "alpha": table,
                "meta": {
                    "granularity": args.granularity,
                    "field": field,
                    "estimator": args.estimator,
                    "n_min": args.n_min,
                    "beta1": BETA1, "beta2": BETA2,
                    "sources": paths,
                    "instances": len(fits),
                },
                "detail": {"%s.%s" % (k[0], k[1]): v for k, v in fits.items()},
                "detail_ratio": {"%s.%s" % (k[0], k[1]): v
                                 for k, v in ratios.items()},
            }, fh, indent=2, sort_keys=True)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
