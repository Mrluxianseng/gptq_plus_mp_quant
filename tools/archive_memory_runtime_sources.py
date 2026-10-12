"""Small local source snapshot; excludes data, model weights and Git metadata.

Hashes are useful for comparison but cannot reconstruct later edited files.
This zip preserves code contents before the next production-source change.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", default="current source snapshot; historical runs retain their own provenance")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise FileExistsError(args.output)
    # Explicit code folders also include local, untracked research tools.
    candidates = [path for folder in ("tools", "utils", "gptq_utils", "realq", "realq_benchmark", "tests", "docs")
                  for path in (root / folder).rglob("*")
                  if path.suffix in (".py", ".md", ".bib") and path.is_file()]
    candidates += [path for path in root.iterdir() if path.suffix in (".py", ".md", ".bib") and path.is_file()]
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(set(candidates)):
            if any(part in ("__pycache__", ".git", ".venv", "outputs", "cache", "datasets") for part in path.relative_to(root).parts):
                continue
            data = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            archive.writestr(relative, data)
        manifest = {"scope": "local source contents; no datasets, weights or environment packages",
                    "stage": args.stage,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "files": records}
        archive.writestr("SOURCE_CONTENTS_MANIFEST.json", json.dumps(manifest, indent=2))
    with ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("source archive CRC failed")
        for row in records:
            if hashlib.sha256(archive.read(row["path"])).hexdigest() != row["sha256"]:
                raise RuntimeError("source archive content hash failed")
    print(json.dumps({"archive": str(args.output.resolve()), "files": len(records),
                      "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "contents_verified": True}, indent=2))


if __name__ == "__main__":
    main()
