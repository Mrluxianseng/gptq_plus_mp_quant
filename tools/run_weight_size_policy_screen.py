"""One fresh 0.6B pair after the independently confirmed layer25 failure.

Screen the new placement policy across all28 layers, not only a short prefix.
All numerical/source/cache/resource gates remain; no automatic repetitions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_replay_stats_full_followup_v2 as base
from tools.hessian_weight_size_policy_prototype import build_entry, MIN_STABLE_BYTES
from tools.run_full_pipeline_memory_profile import set_option


def prepare_entry(queue):
    return build_entry(base.prepare_entry(queue), helper_sha=base.digest(
        ROOT / "tools/hessian_weight_size_policy_prototype.py"))


def policy_coverage(inverse, dead, *, enabled):
    calls, writes = inverse["calls"], inverse["weight_write_calls"]
    layers, groups = Counter(range(28)), Counter({i: 4 for i in range(28)})
    if (len(calls) != 112 or Counter(r["layer"] for r in calls) != groups
            or len(writes) != 28 or Counter(r["layer"] for r in writes) != layers
            or inverse["solver_workspace"] != enabled or inverse["reuse"] != enabled
            or not inverse["helper_unchanged"] or not inverse["stable_helper_unchanged"]
            or dead["enabled"] != enabled or not dead["helpers_unchanged"]):
        return False
    if any(r["module"] != "mlp.down_proj" or r["shape"] != [3072, 3072]
           or r["dtype"] != "torch.float32" or r["enabled"] != enabled
           or r["identity_fallback"] for r in calls):
        return False
    for r in writes:
        selected = enabled and r["before"]["bytes"] >= MIN_STABLE_BYTES
        if (r.get("requested_enabled") != enabled or r["enabled"] != selected
                or r.get("min_stable_bytes") != MIN_STABLE_BYTES
                or r.get("policy") != "native-medium-size-placement"
                or r["before"]["bytes"] != 6 * 1024 * 1024
                or r["after"]["bytes"] != r["before"]["bytes"]
                or r["after"]["dtype"] != r["before"]["dtype"]
                or (selected and not r["pointer_preserved"])):
            return False
    if not enabled:
        return not inverse["reuse_calls"] and not dead["records"]
    reuse = inverse["reuse_calls"]
    return (len(reuse) == 112 and Counter(r["layer"] for r in reuse) == groups
        and all(r["factor_shares_consumed_h_storage"] and not r["identity_fallback"] for r in reuse)
        and len(dead["records"]) == 28 and Counter(r["layer"] for r in dead["records"]) == layers
        and all(r["released_private_input_bytes"] == 37748736 and r["retained_factors"] == 4
                for r in dead["records"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed-independent-audit", type=Path, required=True)
    p.add_argument("--workspace-queue", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    out = args.output_directory.resolve()
    out.mkdir(exist_ok=False)
    proc = reader = None
    report = dict(valid=False, scope="One fresh0.6B full28-layer size-policy screen; no repetitions or speed claim")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        failed = base.read(args.failed_independent_audit)
        mandatory = ("all_stages_exit_zero", "canonical_507_state_kl_ppl_exact",
            "canonical_all28_nonempty_outputs_exact", "sources_unchanged",
            "both_stage_commands_equal", "method_112_inverse_28_write_dead_stats_covered",
            "all13_cache_bytes_equal", "global_peaks_no_regression")
        if failed["valid"] or not all(failed["checks"].get(k) for k in mandatory):
            raise RuntimeError("requires independently confirmed exact numerical/source resource failure")
        if not failed["interval_regressions"] or failed["points"]["missing"]:
            raise RuntimeError("missing confirmed interval failure or incomplete points")
        old = base.read(ROOT / "outputs/replay-stats-full-followup-20261012-v2/source_bindings.json")
        base.check_bound_sources(old)
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                             text=True, capture_output=True, check=True)
        if any(x.strip().isdigit() for x in gpu.stdout.splitlines()):
            raise RuntimeError("another CUDA process active")
        entry = out / "entry.py"
        entry.write_text(prepare_entry(args.workspace_queue))
        bound = dict(old)
        for path in (entry, Path(__file__).resolve(), ROOT / "tools/hessian_weight_size_policy_prototype.py"):
            bound[str(path)] = base.digest(path)
        base.save_new(out / "source_bindings.json", bound)
        base.check_bound_sources(bound)
        reference = ROOT / "outputs/full_pipeline_profile_qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457-r1-candidate"
        ref, canonical = base.read(reference / "manifest.json"), base.read(reference / "status.json")
        folder = out / "manifests"
        folder.mkdir()
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                data = base.read(ref[stage + "_source_manifest"])
                command = data["command"]
                command[command.index("baseline-profile")-1] = str(entry)
                if stage == "stage2":
                    set_option(command, "--quant_stop_layer", "27")
                    if side == "candidate":
                        command.extend(["--research_hinv_solver_workspace", "--research_hinv_reuse_consumed_h",
                                        "--research_hinv_stable_weight", "--research_replay_early_stats"])
                data["scope_note"] = "Size-aware down weight placement;6MiB uses original conversion/allocation. Same math and fresh cache."
                base.save_new(folder / (side + "-" + stage + ".json"), data)
        for stage in ("stage0", "stage2"):
            commands = [base.read(folder / (side + "-" + stage + ".json"))["command"] for side in ("control", "candidate")]
            if base.normalized_command(commands[0]) != base.normalized_command(commands[1]):
                raise RuntimeError("algorithm/data commands changed")
        reader = subprocess.Popen([sys.executable, "-u", str(ROOT / "tools/stream_memory_research_progress.py"),
            "--queues", str(out), "--output", str(ROOT / "logs/realq-system-restart-20261010-1408.log"), "--append"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cmd = [sys.executable, "-u", str(ROOT / "tools/run_replay_stats_memory_campaign.py"),
               "--tag", args.tag, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "7200"]
        for side in ("control", "candidate"):
            for stage in ("stage0", "stage2"):
                cmd.extend([f"--{side}-{stage}-manifest", str(folder / (side + "-" + stage + ".json"))])
        base.save_new(out / "active.json", dict(label=args.tag, command=cmd))
        with (out / (args.tag + ".log")).open("x") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            code = proc.wait()
        runs = [ROOT / "outputs" / ("full_pipeline_profile_" + args.tag + "-r1-" + side)
                for side in ("control", "candidate")]
        statuses = [base.read(run / "status.json") for run in runs]
        points = base.audit_points(*statuses)
        base.save_new(out / "separate_start_end.json", points)
        coverage = [policy_coverage(base.read(run / "inverse_lifetime_stage2_rank0.json"),
                    base.read(run / "dead_subgroup_input_stage2_rank0.json"), enabled=i == 1)
                    for i, run in enumerate(runs)]
        replay = base.replay_coverage(*(base.read(run / "replay_stats_lifetime_stage2_rank0.json") for run in runs))
        quality = all(s["metrics"]["state_tensors"] == 507
            and s["metrics"]["state_sha256"] == canonical["metrics"]["state_sha256"]
            and (s["kl"], s["ppl"]) == (canonical["kl"], canonical["ppl"])
            and base.read(run / "layer_output_fingerprints.json") == base.read(reference / "layer_output_fingerprints.json")
            for s, run in zip(statuses, runs))
        base.check_bound_sources(bound)
        campaign = ROOT / "outputs" / ("memory_campaign_" + args.tag)
        audit = base.read(campaign / "pair1_audit/comparison.json")
        report.update(campaign_returncode=code, canonical_quality_passed=quality,
            size_policy_coverage=coverage, replay_coverage=replay,
            separate_points_passed=points["valid"], strict_memory_gate=audit["strict_measured_memory_gate_passed"],
            cache_files_match=audit["cache_files_match"], sources_unchanged=True,
            trigger_sha256=base.digest(args.failed_independent_audit))
        report["valid"] = (code == 0 and quality and all(coverage) and replay and points["valid"]
            and audit["strict_measured_memory_gate_passed"] and audit["cache_files_match"])
    except BaseException as exc:
        report["error"] = repr(exc)
        if proc and proc.poll() is None:
            base.stop_process_tree(proc.pid)
            proc.wait()
        raise
    finally:
        report["finished"] = time.strftime("%F %T")
        base.save_new(out / "status.json", report)
        if reader:
            reader.wait(timeout=15)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
