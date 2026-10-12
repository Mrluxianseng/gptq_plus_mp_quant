"""Extend accepted two-model memory results without changing the algorithm.

First expand calibration/evaluation coverage with a second seed, then measure
the recomputation cost of the already accepted system. No Triton tuning or
non-bottleneck component experiments are scheduled here.
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
from tools.run_full_pipeline_memory_profile import set_option
from tools.run_low_overhead_speed_campaign import core_sources, check_sources

REFERENCES = (
    ("qwen17b", "qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125"),
    ("qwen06b", "qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457"),
)


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag-suffix", required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(exist_ok=False)
    jobs, proc = [], None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    def execute(label, command):
        nonlocal proc
        (out / "active.json").write_text(json.dumps(dict(label=label, command=command), indent=2))
        print(f"[two-model] starting {label}", flush=True)
        with (out / f"{label}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            code = proc.wait()
        jobs.append(dict(label=label, returncode=code, finished=time.strftime("%F %T")))
        (out / "jobs.json").write_text(json.dumps(jobs, indent=2))
        print(f"[two-model] {label} exited {code}", flush=True)
        return code
    def link_completed(directory, anchor, label):
        if not (directory / "status.json").exists() or not read(directory / "status.json").get("valid"):
            return
        python = "/mnt/c/Users/65377/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
        def windows(path):
            path = str(path)
            if not path.startswith("/mnt/d/"):
                raise ValueError("housekeeping paths must remain on task D drive")
            return "D:/" + path[len("/mnt/d/"):]
        cmd = [python, windows(ROOT / "tools/deduplicate_completed_prepared_checkpoint.py"),
               "--control-run", windows(anchor), "--candidate-run", windows(directory),
               "--output", windows(out / ("disk-" + label + ".json")), "--apply"]
        execute("disk-" + label, cmd)
    try:
        print("[two-model] waiting for full allocator-capacity tests; no CUDA work", flush=True)
        while not args.wait_for.exists():
            time.sleep(5)
        dependency = read(args.wait_for)
        if not dependency.get("valid"):
            raise RuntimeError("capacity work needs review; no automatic expansion after a failed protocol")
        for row in dependency["models"]:
            model, accepted = row["model"], row["reference_campaign"]
            anchor = ROOT / "outputs" / f"full_pipeline_profile_{accepted}-r1-control"
            link_completed(Path(row["arms"]["candidate"]["directory"]), anchor, model + "-capacity-candidate")
        expanded_tags = []
        for model, accepted in REFERENCES:
            campaign = ROOT / "outputs" / f"memory_campaign_{accepted}"
            status = read(campaign / "status.json")
            if not status.get("valid") or status.get("completed_pairs") != 3:
                raise RuntimeError("requires complete accepted three-pair memory reference")
            reference = ROOT / "outputs" / f"full_pipeline_profile_{accepted}-r1-control"
            expected = core_sources(read(reference / "manifest.json"))
            check_sources(expected)
            folder = out / (model + "-expanded-manifests")
            folder.mkdir()
            for side in ("control", "candidate"):
                manifest = read(ROOT / "outputs" / f"full_pipeline_profile_{accepted}-r1-{side}/manifest.json")
                for stage in ("stage0", "stage2"):
                    data = read(manifest[f"{stage}_source_manifest"])
                    cmd = data["command"]
                    set_option(cmd, "--nsamples", "8")
                    set_option(cmd, "--seed", "45")
                    if stage == "stage2":
                        set_option(cmd, "--eval_samples", "8")
                    data.update(eval_samples=8, explicit_shared_conditions="Calibration 8x1024/seed45; refresh microbatch2; full28; eval8x1024. Both arms identical algorithm/data, candidate only checkpoint+CE.",
                                scope_note="Expanded local calibration and held-out coverage; independent paired numerical/resource audit, not paper2048 samples or multi-GPU proof")
                    (folder / f"{side}-{stage}.json").write_text(json.dumps(data, indent=2))
            if shutil.disk_usage(ROOT).free < 12 * 1024**3:
                raise RuntimeError("requires twelve GiB for fresh expanded profile arms")
            tag = model + "-checkpoint-ce-cal8-eval8-seed45-full-" + args.tag_suffix
            expanded_tags.append(tag)
            cmd = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
                   "--tag", tag, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "7200"]
            for side in ("control", "candidate"):
                for stage in ("stage0", "stage2"):
                    cmd += [f"--{side}-{stage}-manifest", str(folder / f"{side}-{stage}.json")]
            code = execute(tag, cmd)
            check_sources(expected)
            actual_campaign = ROOT / "outputs" / f"memory_campaign_{tag}"
            for side in ("control", "candidate"):
                link_completed(ROOT / "outputs" / f"full_pipeline_profile_{tag}-r1-{side}", reference, tag + "-" + side)
            if code or not read(actual_campaign / "status.json").get("valid"):
                raise RuntimeError(f"expanded {model} needs failure analysis before further experiments")
        # Report the new configurations before timing, never concurrently with
        # measured GPU work. Missing/negative audits remain in their directories.
        execute("record-expanded-pairs", [sys.executable, str(ROOT / "tools/record_completed_memory_followup_20261010.py"),
                "--wait-for", str(ROOT / "outputs" / f"memory_campaign_{expanded_tags[-1]}/status.json"),
                "--campaign-tags", *expanded_tags, "--output-dir", str(out / "expanded-records")])
        for model, accepted in REFERENCES:
            required_gib = 27 if model == "qwen17b" else 11
            if shutil.disk_usage(ROOT).free < required_gib * 1024**3:
                raise RuntimeError(f"requires {required_gib} GiB for three fresh cost pairs")
            tag = model + "-checkpoint-ce-recompute-cost-threepairs-" + args.tag_suffix
            reference = ROOT / "outputs" / f"full_pipeline_profile_{accepted}-r1"
            cmd = [sys.executable, "-u", str(ROOT / "tools/run_low_overhead_speed_campaign.py"),
                   "--tag", tag, "--pairs", "3", "--gpu-indices", "0", "--timeout-seconds", "7200",
                   "--control-reference", str(reference) + "-control", "--candidate-reference", str(reference) + "-candidate",
                   "--pair-audit", str(ROOT / "outputs" / f"memory_campaign_{accepted}/pair1_audit/comparison.json")]
            if execute(tag, cmd):
                raise RuntimeError(f"{model} recomputation cost protocol failed; inspect original logs")
        (out / "status.json").write_text(json.dumps(dict(valid=True, jobs=jobs, expanded_campaigns=expanded_tags,
            finished=time.strftime("%F %T"), scope="Two-model expanded local memory coverage and cost of accepted memory system; no new speed kernel"), indent=2))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        (out / "status.json").write_text(json.dumps(dict(valid=False, error=repr(exc), jobs=jobs,
                                                        finished=time.strftime("%F %T")), indent=2))
        raise


if __name__ == "__main__":
    main()
