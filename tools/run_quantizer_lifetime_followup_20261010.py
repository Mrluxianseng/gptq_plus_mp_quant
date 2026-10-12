"""Serial follow-up for a specific observed reserved-memory regression.

Waits for the current complete-model queue; preserves independent negatives.
Metadata must match in addition to the standard strict whole-prefix audit.
"""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_pipeline_memory_campaign import stop_process_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/quantizer-lifetime-followup-20261010")
    parser.add_argument("--wait-for", type=Path, default=ROOT / "outputs/cpu-input-checkpoint-followup-20261010/status.json")
    parser.add_argument("--tag-suffix", default="")
    parser.add_argument("--manifests-dir", type=Path, default=ROOT / "outputs/quantizer-residency-prefix-v3-manifests-20261010")
    options = parser.parse_args()
    out = options.output_dir
    out.mkdir(exist_ok=False)
    records = []
    proc = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)

    def execute(label, command):
        nonlocal proc
        (out / "active.json").write_text(json.dumps({"label": label, "command": command}, indent=2))
        print(f"[lifetime] starting {label}", flush=True)
        with (out / f"{label}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(3)
        row = {"label": label, "returncode": proc.returncode, "finished": time.strftime("%F %T")}
        records.append(row)
        (out / "jobs.json").write_text(json.dumps(records, indent=2))
        print(f"[lifetime] {json.dumps(row)}", flush=True)
        return proc.returncode == 0

    def campaign(label, folder):
        label += options.tag_suffix
        if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
            raise RuntimeError("requires 10 GiB free before a fresh 1.7B pair; preserve previous artifacts")
        command = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
                   "--tag", label, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "7200"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                command += [f"--{side}-{stage}-manifest", str(folder / f"{side}-{stage}.json")]
        accepted = execute(label, command)
        campaign_dir = ROOT / "outputs" / f"memory_campaign_{label}"
        rows = []
        for side in ("control", "candidate"):
            path = ROOT / "outputs" / f"full_pipeline_profile_{label}-r1-{side}" / "quantizer_metadata_lifetime_rank0.json"
            if not path.exists():
                raise RuntimeError(f"missing metadata observations: {path}")
            records_meta = json.loads(path.read_text())
            rows.append([{key: row[key] for key in ("layer", "module", "buffers")} for row in records_meta])
        metadata_exact = len(rows[0]) == 7 and rows[0] == rows[1]
        (campaign_dir / "metadata_byte_audit.json").write_text(json.dumps({"metadata_exact": metadata_exact,
                       "module_counts": [len(row) for row in rows], "scope": "completed layer0 quantizer buffers; devices intentionally differ"}, indent=2))
        if not metadata_exact:
            raise RuntimeError("metadata changed or module observations incomplete")
        return accepted

    try:
        print("[lifetime] waiting for current serial GPU queue", flush=True)
        began = time.monotonic()
        while not options.wait_for.exists():
            if time.monotonic() - began > 14400:
                raise RuntimeError("previous GPU queue did not finish")
            time.sleep(5)
        folder = options.manifests_dir
        accepted = campaign("qwen17b-quantizer-metadata-cpu-layer0-2x512-20261010", folder)
        if accepted:
            # The independent ablation alone does not repair the comparison
            # against the original algorithm/system control. Test it afresh.
            repaired = out / "original-control-composition-manifests"
            repaired.mkdir()
            original = ROOT / "outputs/memory_mechanism_followup_20261010/hessian-cpu-layer0-manifests"
            for side in ("control", "candidate"):
                for stage in ("stage0", "stage2"):
                    data = json.loads((original / f"{side}-{stage}.json").read_text())
                    data["command"][data["command"].index("baseline-profile") - 1] = str(folder / "entry.py")
                    if side == "candidate" and stage == "stage2":
                        data["command"].append("--research_stage2_quantizer_metadata_cpu")
                    data["scope_note"] = "Fresh original-control vs Hessian CPU+module-end metadata CPU prefix; not complete 28-layer quantization."
                    (repaired / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
            campaign("qwen17b-hessian-metadata-cpu-original-control-layer0-20261010", repaired)
        result = {"finished": time.strftime("%F %T"), "jobs": records,
                  "metadata_only_accepted": accepted, "scope": "fresh prefix mechanisms; no full-model/speed claim"}
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        (out / "status.json").write_text(json.dumps({"error": repr(exc), "jobs": records, "finished": time.strftime("%F %T")}, indent=2))
        raise
    (out / "status.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
