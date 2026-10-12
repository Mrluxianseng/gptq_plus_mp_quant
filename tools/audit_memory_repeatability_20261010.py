"""Read-only audit of all completed repeats, including fresh cache identity.

No CUDA initialization. This complements, rather than replaces, each paired
output/memory audit. Per-run peaks are not time-averaged occupancy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_full_pipeline_profiles import algorithm_command


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit(campaign):
    status = read(campaign / "status.json")
    rows = read(campaign / "paired_repetitions.json")
    arms = []
    inode_digests = {}
    errors = []
    if len(rows) != status["requested_pairs"]:
        errors.append("not all requested pairs were audited")
    if not status.get("valid"):
        errors.append("campaign validity gate failed")
    for record in rows:
        comparison = read(campaign / f"pair{record['pair']}_audit/comparison.json")
        if not comparison.get("exactness_passed") or not comparison.get("strict_measured_memory_gate_passed"):
            errors.append(f"pair {record['pair']} did not pass both gates")
        for side in ("control", "candidate"):
            folder = Path(comparison[f"{side}_directory"])
            run = read(folder / "status.json")
            manifest = read(folder / "manifest.json")
            cache = {}
            for path in sorted((folder / "static_cache").rglob("*")):
                if not path.is_file():
                    continue
                before = path.stat()
                identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                if identity not in inode_digests:
                    value = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(8*1024**2), b""):
                            value.update(block)
                    after = path.stat()
                    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
                        raise RuntimeError(f"completed cache changed during read: {path}")
                    inode_digests[identity] = value.hexdigest()
                cache[path.relative_to(folder / "static_cache").as_posix()] = inode_digests[identity]
            if not cache:
                errors.append(f"pair {record['pair']} {side} has no cache files")
            trace = read(folder / "layer_output_fingerprints.json")
            canonical = dict(state_sha256=run["metrics"]["state_sha256"], state_tensors=run["metrics"]["state_tensors"],
                             kl=run["kl"], ppl=run["ppl"], layers=trace, cache=cache,
                             stage0_algorithm_command=algorithm_command(manifest["stage0_command"]),
                             stage2_algorithm_command=algorithm_command(manifest["stage2_command"]))
            if arms:
                for field, value in canonical.items():
                    if value != arms[0]["canonical"][field]:
                        errors.append(f"pair {record['pair']} {side}: {field} differs from first control")
            arms.append(dict(pair=record["pair"], side=side, directory=str(folder), canonical=canonical))
    return dict(valid=bool(arms) and not errors, errors=errors, requested_pairs=status["requested_pairs"],
                audited_pairs=len(rows), arms=arms, unique_files_hashed=len(inode_digests),
                scope="Completed cross-repeat identity; no speed/average-occupancy/continuous-memory proof")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.campaign)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "arms"}), flush=True)
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
