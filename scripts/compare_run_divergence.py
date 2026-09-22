"""Where do two nominally identical runs start to differ?

Quantisation is a discontinuous map, so an infinitesimal difference anywhere
upstream can push one weight across a rounding boundary and change it by a
full grid step. That is not drift that averages out -- it propagates through
the closed-form compensation and grows.

Two runs of the same config, same seed, deterministic algorithms on, were
bit-identical through layer 0 and then jumped 1.2% at layer 2 mlp.down_proj.
The final KL differed by 0.44%. So a single toy-scale run cannot resolve an
effect below about 1.5%, and tightening the determinism settings will not fix
it -- only more seeds will.

Usage: python scripts/compare_run_divergence.py run_a.log run_b.log
"""
import re, sys
def traj(path):
    out = {}
    pat = re.compile(r"block-metrics layer=(\d+) module=(\S+) .*? block=(\d+) .*? loss=([0-9.eE+-]+)")
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            out[(int(m.group(1)), m.group(2), int(m.group(3)))] = float(m.group(4))
    return out
a, b = traj(sys.argv[1]), traj(sys.argv[2])
keys = sorted(set(a) & set(b))
print("shared block-metric points: %d" % len(keys))
first = None
worst = (0.0, None)
for k in keys:
    d = abs(a[k] - b[k]) / max(abs(a[k]), 1e-30)
    if d > 1e-12 and first is None:
        first = (k, a[k], b[k], d)
    if d > worst[0]:
        worst = (d, k)
if first is None:
    print("identical everywhere they overlap")
else:
    print("first difference at layer=%d %s block=%d : %.8g vs %.8g  (rel %.2e)"
          % (first[0][0], first[0][1], first[0][2], first[1], first[2], first[3]))
    print("largest relative difference: %.2e at layer=%d %s block=%d"
          % (worst[0], worst[1][0], worst[1][1], worst[1][2]))
    # how the divergence grows with layer
    import collections
    by_layer = collections.defaultdict(list)
    for k in keys:
        by_layer[k[0]].append(abs(a[k] - b[k]) / max(abs(a[k]), 1e-30))
    import statistics
    print()
    print("median relative difference by layer:")
    for L in sorted(by_layer):
        if L % 4 == 0 or L == max(by_layer):
            print("  layer %-3d  %.2e" % (L, statistics.median(by_layer[L])))
