"""Aggregate the reach and flip probes across a whole run.

Two questions:
  path/disp  -- at 1.0 the optimiser never doubles back and every unit of step
                becomes displacement, so the limit is magnitude, not
                oscillation. Above ~1.2 it is spending budget going nowhere.
  flip_rate  -- how often the accumulated push actually moved a grid point.
                The theory says this equals |disp|/grid exactly; if it does,
                the optimiser's effect on a column's OWN value is fully
                described by that one ratio.
"""
import re, sys
from collections import defaultdict
import numpy as np

path = sys.argv[1]
txt = open(path, encoding="utf-8", errors="ignore").read()

FLIP = re.compile(
    r"flip_probe layer=(\S+) module=(\S+) nblk=(\d+)\s*\n"
    r"\s*flip_rate\s+(.*?)\n"
    r"\s*\|disp\|/grid med\s+(.*?)\n", re.S)
REACH = re.compile(
    r"reach_probe layer=(\S+) module=(\S+) lr=(\S+) nblk=(\d+) ngrp=(\S+)\s*\n"
    r"\s*\|W_fp-Q\|\s+(.*?)\n\s*\|disp\|\s+(.*?)\n\s*path\s+(.*?)\n"
    r"\s*grid\s+(.*?)\n\s*R=disp/dev\s+(.*?)\n\s*path/disp\s+(.*?)\n", re.S)


def vals(s):
    return [float(x.split(":")[1]) for x in s.split() if ":" in x]


flips, pushes, byname = defaultdict(list), defaultdict(list), defaultdict(list)
for m in FLIP.finditer(txt):
    layer, mod, nblk = m.group(1), m.group(2), int(m.group(3))
    f, p = vals(m.group(4)), vals(m.group(5))
    for b, (fv, pv) in enumerate(zip(f, p)):
        flips[(mod, b)].append(fv)
        pushes[(mod, b)].append(pv)
    byname[mod].append((f, p))

pd, R, lrs = defaultdict(list), defaultdict(list), {}
for m in REACH.finditer(txt):
    mod = m.group(2)
    lrs.setdefault(mod, []).append(float(m.group(3)))
    for b, v in enumerate(vals(m.group(11))):
        if v > 0:
            pd[(mod, b)].append(v)
    for b, v in enumerate(vals(m.group(10))):
        R[(mod, b)].append(v)

mods = sorted(byname)
print("%-22s %6s %9s %9s %9s %9s %9s" % (
    "module", "nblk", "flip@last", "push@last", "flip/push", "path/disp", "R@last"))
print("-" * 82)
for mod in mods:
    nb = max(b for (m_, b) in flips if m_ == mod) + 1
    fl = np.median(flips[(mod, nb - 1)])
    pu = np.median(pushes[(mod, nb - 1)])
    pdv = np.median([v for (m_, b), vs in pd.items() if m_ == mod for v in vs])
    rv = np.median(R[(mod, nb - 1)])
    print("%-22s %6d %9.4f %9.4f %9.3f %9.4f %9.4f"
          % (mod, nb, fl, pu, fl / pu if pu > 0 else float("nan"), pdv, rv))

allf = [v for k, vs in flips.items() for v in vs]
allp = [v for k, vs in pushes.items() for v in vs]
allpd = [v for k, vs in pd.items() for v in vs]
print("-" * 82)
print("%-22s %6s %9.4f %9.4f %9.3f %9.4f" % (
    "ALL (median)", "", np.median(allf), np.median(allp),
    np.median(allf) / max(np.median(allp), 1e-12), np.median(allpd)))
print()
print("path/disp quartiles: %.3f / %.3f / %.3f"
      % tuple(np.percentile(allpd, [25, 50, 75])))
print("fraction of blocks with path/disp > 1.2 : %.3f"
      % float(np.mean(np.array(allpd) > 1.2)))
print()
print("flip rate, weighted by how many weights each block holds (all blocks")
print("carry the same count), i.e. the share of ALL rounding decisions the")
print("optimiser changed: %.4f" % np.mean(allf))
