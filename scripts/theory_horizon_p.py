#!/usr/bin/env python
"""Per-module horizon exponent from the EMA transfer matrix. No measurement.

The schedule exists to undo the momentum EMA's truncated tail: a column
freezes before the EMA has delivered its last gradients' full weight. Writing
the displacement in terms of the raw gradients,

    W = A^T w,   A[n,i] = (1-b1) b1^(n-i) / (1-b1^n)  for i <= n

the exponent that best flattens W is a function of beta1 and of how many
refreshes the module's columns live through -- and nothing else. In
particular it is NOT global: a module whose input is 1024 wide has g = 8
blocks and wants a much steeper schedule than one 3072 wide with g = 24.

    beta1 = 0.9:   g = 8  -> 1.89     g = 16 -> 1.22     g = 24 -> 0.94

Testing a single global p therefore tests a deliberately weakened form of the
prediction. This writes the table --horizon_alpha consumes so the per-module
form can be run directly.

`--exact` instead reports what the unapproximated solve would do; use the
--horizon_exact flag for that rather than a table, since its shape (a spike on
a column's last refresh) is not in the power-law family at all.

Usage
-----
    python scripts/theory_horizon_p.py --model qwen3-0.6b -o runs/theory_p.json
    HORIZON_ALPHA=runs/theory_p.json ... bash scripts/formal_realq_compare.sh
"""
import argparse
import json
import os

import numpy as np

# in_features per module, which fixes g = in_features / blocksize
MODELS = {
    "qwen3-0.6b": {
        "self_attn.q_proj": 1024, "self_attn.k_proj": 1024,
        "self_attn.v_proj": 1024, "self_attn.o_proj": 2048,
        "mlp.gate_proj": 1024, "mlp.up_proj": 1024, "mlp.down_proj": 3072,
    },
    "qwen3-1.7b": {
        "self_attn.q_proj": 2048, "self_attn.k_proj": 2048,
        "self_attn.v_proj": 2048, "self_attn.o_proj": 4096,
        "mlp.gate_proj": 2048, "mlp.up_proj": 2048, "mlp.down_proj": 6144,
    },
}


def transfer(g, b1):
    """W = A^T w, indexed [raw gradient i, refresh n]. Upper triangular."""
    M = np.zeros((g, g))
    for i in range(1, g + 1):
        for n in range(i, g + 1):
            M[i - 1, n - 1] = (1 - b1) * b1 ** (n - i) / (1 - b1 ** n)
    return M


def exact_w(g, b1):
    """The schedule that flattens W outright, by back-substitution."""
    M = transfer(g, b1)
    w = np.zeros(g)
    for i in range(g, 0, -1):
        acc = sum(M[i - 1, n - 1] * w[n - 1] for n in range(i + 1, g + 1))
        w[i - 1] = (1.0 - acc) / M[i - 1, i - 1]
    return w * g / w.sum()


def p_star(g, b1, metric="rms_log"):
    """The exponent in the k*(h+1)^-p family that leaves W flattest.

    Reported across several flatness metrics elsewhere; they agree to about
    +-20%, so the number is a band and not a point.
    """
    if b1 <= 0.0:
        return 0.0                       # A = I: nothing to undo, exactly
    M = transfer(g, b1)
    h = np.arange(g - 1, -1, -1)
    best = None
    for p in np.arange(-0.5, 3.0001, 0.005):
        w = (h + 1.0) ** (-p)
        w = w / w.mean()
        W = M @ w
        W = W / W.mean()
        r = (np.sqrt(np.mean(np.log(W) ** 2)) if metric == "rms_log"
             else W.max() / W.min())
        if best is None or r < best[1]:
            best = (float(p), float(r))
    return best[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3-0.6b", choices=sorted(MODELS))
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    mods = MODELS[args.model]
    table, rows = {}, []
    for name, fin in sorted(mods.items()):
        g = fin // args.blocksize
        p = p_star(g, args.beta1)
        table[name] = round(p, 4)
        w = exact_w(g, args.beta1) if args.beta1 > 0 else np.ones(g)
        rows.append((name, fin, g, p, w[-1] / w[0]))
    table["__default__"] = round(float(np.median(list(table.values()))), 4)

    print("beta1 = %.2f, blocksize = %d, model = %s"
          % (args.beta1, args.blocksize, args.model))
    print()
    print("%-22s %-8s %-5s %-9s %s"
          % ("module", "in_feat", "g", "p*", "exact w(last)/w(first)"))
    print("-" * 66)
    for name, fin, g, p, ratio in rows:
        print("%-22s %-8d %-5d %-9.3f %.1f" % (name, fin, g, p, ratio))
    print("-" * 66)
    print("spread of p* across modules: %.3f - %.3f"
          % (min(r[3] for r in rows), max(r[3] for r in rows)))
    print()
    print("A single global p cannot sit at all of these at once; that is why")
    print("a global sweep tests a weaker claim than the theory makes.")

    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({
                "alpha": table,
                "meta": {
                    "source": "EMA transfer matrix, no measurement",
                    "beta1": args.beta1, "blocksize": args.blocksize,
                    "model": args.model,
                },
            }, fh, indent=2, sort_keys=True)
        print()
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
