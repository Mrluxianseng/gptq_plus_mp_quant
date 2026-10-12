"""Compare matched start AND end boundaries, never their combined maximum.

Read-only supplemental audit: preserve phase occurrence and point identity.
Partial logs can diagnose a regression but cannot certify a completed run.
"""
import argparse
import json
from pathlib import Path
import re


def boundaries(paths):
    values, counts = {}, {}
    for filename in paths:
        path = Path(filename)
        match = re.search(r"rank(\d+)", path.name)
        if match is None:
            raise ValueError(f"missing rank identity: {path}")
        rank = int(match.group(1))
        stage = "stage0" if "_stage0/" in path.as_posix() else "stage2"
        kind = "evaluation" if "evaluation" in path.name else "quantization"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "phase" not in row:
                continue
            phase = row["phase"]
            identity = (stage, kind, rank, phase)
            occurrence = counts.get(identity, 0)
            counts[identity] = occurrence + 1
            for point in ("start", "end"):
                for metric in ("allocated", "reserved"):
                    name = f"{point}_{metric}_bytes"
                    if name in row:
                        values[(*identity, occurrence, point, metric)] = row[name]
    return values


def compare(control, candidate, *, partial=False):
    common = sorted(control.keys() & candidate.keys())
    regressions = [dict(key=list(k), control_bytes=control[k], candidate_bytes=candidate[k],
                        delta_bytes=candidate[k] - control[k])
                   for k in common if candidate[k] > control[k]]
    missing = [list(k) for k in sorted(control.keys() ^ candidate.keys())]
    return dict(valid=bool(common) and not partial and not missing and not regressions,
                partial=partial, matched_values=len(common), regressions=regressions,
                missing=missing, criterion="Each matched start/end point separately; not max(start,end)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, nargs="+", required=True)
    p.add_argument("--candidate", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError("preserve earlier audit; select a new output file")
    result = compare(boundaries(args.control), boundaries(args.candidate), partial=args.partial)
    result.update(control=[str(f.resolve()) for f in args.control], candidate=[str(f.resolve()) for f in args.candidate])
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("regressions", "missing")}))
    print("regressions", len(result["regressions"]), "missing", len(result["missing"]))


if __name__ == "__main__":
    main()
