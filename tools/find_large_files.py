#!/usr/bin/env python3
"""Read-only disk inventory: report the largest regular files under given roots.

Does not follow directory symlinks and never deletes, moves, or modifies files.
Intended for server-side use when diagnosing low disk space.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from collections import Counter


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if amount < 1024 or unit == "PiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{value} B"


def scan(root: Path, errors: list[dict]) -> tuple[list[dict], Counter, int, int]:
    rows: list[dict] = []
    extensions: Counter = Counter()
    total_bytes = 0
    total_files = 0

    def onerror(exc: OSError) -> None:
        errors.append({"path": getattr(exc, "filename", None), "error": str(exc)})

    for current, dirs, files in os.walk(root, topdown=True, followlinks=False, onerror=onerror):
        # Explicitly avoid traversing directory symlinks and pseudo-filesystems mounted
        # below a requested root. A symlink to an ordinary file is also not counted.
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        for name in files:
            path = Path(current) / name
            try:
                info = path.stat(follow_symlinks=False)
            except OSError as exc:
                onerror(exc)
                continue
            if not path.is_file() or path.is_symlink():
                continue
            size = info.st_size
            total_bytes += size
            total_files += 1
            ext = path.suffix.lower() or "[no extension]"
            extensions[ext] += size
            rows.append({"path": str(path), "bytes": size, "modified": info.st_mtime})
    rows.sort(key=lambda row: (-row["bytes"], row["path"]))
    return rows, extensions, total_bytes, total_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True,
                        help="directory to scan; repeat for multiple roots")
    parser.add_argument("--top", type=int, default=100,
                        help="number of largest files to print per combined scan (default: 100)")
    parser.add_argument("--json", dest="json_path",
                        help="optional path for a full JSON inventory")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")

    all_rows: list[dict] = []
    all_errors: list[dict] = []
    by_extension: Counter = Counter()
    scanned_roots = []
    total_bytes = total_files = 0
    for raw in args.root:
        root = Path(raw).expanduser()
        if not root.exists() or not root.is_dir():
            all_errors.append({"path": str(root), "error": "root missing or not a directory"})
            continue
        root = root.resolve()
        rows, extensions, nbytes, nfiles = scan(root, all_errors)
        all_rows.extend(rows)
        by_extension.update(extensions)
        total_bytes += nbytes
        total_files += nfiles
        scanned_roots.append(str(root))

    all_rows.sort(key=lambda row: (-row["bytes"], row["path"]))
    print("Read-only large-file inventory")
    print("Roots:", ", ".join(scanned_roots) if scanned_roots else "(none scanned)")
    print(f"Files: {total_files:,} | apparent size: {human_bytes(total_bytes)}")
    print("\nLargest files:")
    for index, row in enumerate(all_rows[:args.top], 1):
        print(f"{index:4d}. {human_bytes(row['bytes']):>12}  {row['path']}")
    print("\nLargest extensions by summed file size:")
    for ext, size in by_extension.most_common(20):
        print(f"{human_bytes(size):>12}  {ext}")
    if all_errors:
        print(f"\nUnreadable/missing paths: {len(all_errors)} (see JSON if requested)", file=sys.stderr)

    if args.json_path:
        report = {
            "read_only": True,
            "roots": scanned_roots,
            "files_scanned": total_files,
            "apparent_bytes": total_bytes,
            "largest_files": all_rows[:args.top],
            "all_files": all_rows,
            "extension_bytes": dict(by_extension),
            "errors": all_errors,
        }
        destination = Path(args.json_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull inventory JSON: {destination}")
    return 0 if scanned_roots else 2


if __name__ == "__main__":
    raise SystemExit(main())
