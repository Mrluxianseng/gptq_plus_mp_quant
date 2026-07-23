#!/usr/bin/env python3
"""Compare two REAL-Q per-refresh JSONL traces with a strict relative bound."""

from __future__ import annotations

import argparse
import json
import os
import sys

from realq.alignment import compare_refresh_traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--max-relative-difference", type=float, default=0.01)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    if not (0.0 < args.max_relative_difference < 1.0):
        parser.error("--max-relative-difference must be in (0, 1)")

    report = compare_refresh_traces(
        args.reference,
        args.candidate,
        max_relative_difference=args.max_relative_difference,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
