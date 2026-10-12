"""CPU-only cross-repeat checks after the current timing queue completes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(exist_ok=False)
    print("[cross-repeat] waiting for GPU/timing queue completion", flush=True)
    started = time.monotonic()
    while not args.wait_for.exists():
        if time.monotonic()-started > 24*3600:
            raise RuntimeError("dependency did not complete within a day")
        time.sleep(5)
    records = []
    for tag in args.tags:
        campaign = ROOT / "outputs" / f"memory_campaign_{tag}"
        if not (campaign / "status.json").exists() or not (campaign / "paired_repetitions.json").exists():
            records.append(dict(tag=tag, audited=False, note="not completed/audited; not a negative result"))
            continue
        if not json.loads((campaign / "paired_repetitions.json").read_text()):
            records.append(dict(tag=tag, audited=False, note="zero completed pairs; inspect worker errors"))
            continue
        result = out / f"{tag}.json"
        with (out / f"{tag}.log").open("x") as log:
            proc = subprocess.run([sys.executable, str(ROOT / "tools/audit_memory_repeatability_20261010.py"),
                                  "--campaign", str(campaign), "--output", str(result)], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=600)
        records.append(dict(tag=tag, audited=result.exists(), returncode=proc.returncode,
                            result=str(result), repeatability_valid=(json.loads(result.read_text())["valid"] if result.exists() else None)))
    (out / "status.json").write_text(json.dumps(dict(completed=True, records=records), indent=2))
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
