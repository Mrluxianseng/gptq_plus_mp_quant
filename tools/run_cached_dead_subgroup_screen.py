"""Dead-input lifetime candidate: cached real prefix0..10 after finite resource failure.

Keeps allocator history from layer0, catches layers3/10 replay regressions,
but is not arbitrary checkpoint resume or a full-process memory proof.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_full_pipeline_profiles import cache_hashes, digest, phase_peaks
from tools.audit_memory_boundary_points import boundaries, compare as compare_points
from tools.hessian_dead_subgroup_install import build_entry
from tools.run_full_pipeline_memory_profile import set_option
from tools.run_full_pipeline_memory_campaign import stop_process_tree


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-for", type=Path, required=True)
    p.add_argument("--workspace-queue", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--workspace-prefix-run", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    args = p.parse_args()
    out = args.output_directory.resolve()
    out.mkdir(exist_ok=False)
    proc = reader = None
    report = dict(valid=False, scope="Cached Stage-2 prefix0..10; no complete-process or multi-GPU claim")
    def progress(text):
        print("[dead-input-screen] " + text, flush=True)
    def interrupt(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        progress("waiting for workspace/capture/replay completion; no GPU process")
        while not args.wait_for.exists():
            time.sleep(5)
        prior = read(args.wait_for)
        parent = args.wait_for.parent
        if not prior.get("exactness_passed") or not prior.get("cache_unchanged") or not prior.get("bounded_inverse_coverage") or not prior.get("all_weight_write_coverage") or not prior.get("exact_interval_peaks") or prior.get("missing_phases"):
            raise RuntimeError("finite prefix quality/cache/coverage/infrastructure failure: inspect bug first")
        prior_manifest = read(parent / "manifest.json")
        for file, expected in prior_manifest["source_sha256"].items():
            if digest(file) != expected:
                raise RuntimeError("preceding source changed; no method switching")
        prior_points = compare_points(*(boundaries(read(parent / side / "status.json")["memory_profile_files"]) for side in ("control", "candidate")))
        (out / "preceding_separate_points.json").write_text(json.dumps(prior_points, indent=2))
        if prior_points["missing"] or not prior_points["matched_values"]:
            raise RuntimeError("preceding memory points incomplete; inspect tool before switching")
        if prior.get("valid") and prior_points["valid"]:
            report.update(valid=True, skipped=True, reason="Finite prefix accepted; let its full queue finish first")
            return
        if not prior.get("regressions") and not prior_points["regressions"] and prior.get("global_memory_gate"):
            raise RuntimeError("No confirmed numerical-exact resource failure")
        report["trigger_status_sha256"] = digest(args.wait_for)
        report["preceding_quality_and_source_passed"] = True
        time.sleep(5)
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                             text=True, capture_output=True, check=True)
        if any(x.strip().isdigit() for x in gpu.stdout.splitlines()):
            raise RuntimeError("another compute process is active; no concurrent CUDA screen")
        reader = subprocess.Popen([sys.executable, "-u", str(ROOT / "tools/stream_memory_research_progress.py"),
            "--queues", str(out), "--output", str(ROOT / "logs/realq-system-restart-20261010-1408.log"), "--append"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        reference = args.reference_run.resolve()
        ref_status, ref_manifest = read(reference / "status.json"), read(reference / "manifest.json")
        source_run = args.workspace_prefix_run.resolve()
        source_status, source_manifest = read(source_run / "status.json"), read(source_run / "manifest.json")
        if not ref_status.get("valid") or not source_status.get("valid") or source_manifest["expected_ranks"]["stage2"] != 1:
            raise RuntimeError("requires passed single-rank references")
        bound = source_manifest["source_sha256_at_launch"]
        dependencies = [Path(__file__).resolve(), ROOT / "tools/hessian_dead_subgroup_install.py",
            ROOT / "tools/hessian_dead_subgroup_input_prototype.py", ROOT / "tools/audit_memory_boundary_points.py",
            ROOT / "tools/hessian_solver_workspace_prototype.py", ROOT / "tools/hessian_stable_weight_prototype.py",
            ROOT / "tools/hessian_inverse_lifetime_prototype.py"]
        bound = dict(bound, **{str(file): digest(file) for file in dependencies})
        for file, expected in bound.items():
            if digest(file) != expected:
                raise RuntimeError(f"bound source changed: {file}")
        base = list(source_manifest["stage2_command"])
        ref_cmd = ref_manifest["stage2_command"]
        for flag in ("--model", "--dataset", "--nsamples", "--seq_len", "--seed", "--num_groups",
                     "--static_fisher_label_chunk_tokens"):
            if base[base.index(flag)+1] != ref_cmd[ref_cmd.index(flag)+1]:
                raise RuntimeError("cache-generating conditions mismatch: " + flag)
        cache = cache_hashes(reference, progress)
        entry_index = base.index("baseline-profile") - 1
        original_entry = build_entry(Path(base[entry_index]).read_text(),
            installer_sha=bound[str(ROOT / "tools/hessian_dead_subgroup_install.py")],
            prototype_sha=bound[str(ROOT / "tools/hessian_dead_subgroup_input_prototype.py")])
        anchor = "try:\n    runpy.run_path('ptq.py', run_name='__main__')"
        if original_entry.count(anchor) != 1 or "_hinv_reuse_source=" not in original_entry:
            raise RuntimeError("bound workspace entry boundary changed")
        hook = "def _forbid_precompute(*args, **kwargs):\n    raise RuntimeError('cached lifetime screen must not rerun precompute')\nimpl.collect_static_end_to_end_saliency_and_fisher=_forbid_precompute\n\n"
        entry = out / "entry.py"
        entry.write_text(original_entry.replace(anchor, hook + anchor))
        compile(entry.read_text(), str(entry), "exec")
        base[entry_index] = str(entry)
        set_option(base, "--static_cache_path", ref_manifest["fresh_static_cache"])
        set_option(base, "--quant_stop_layer", "10")
        set_option(base, "--quant_profile_target_layers", ",".join(map(str, range(11))))
        commands = {}
        for side in ("control", "candidate"):
            cmd = list(base)
            set_option(cmd, "--exp", out.name + "-" + side)
            if side == "candidate":
                cmd.extend(["--research_hinv_solver_workspace", "--research_hinv_reuse_consumed_h",
                            "--research_hinv_stable_weight"])
            commands[side] = cmd
        bound[str(entry)] = digest(entry)
        (out / "manifest.json").write_text(json.dumps(dict(commands=commands, source_sha256=bound,
            reference_manifest_sha256=digest(reference / "manifest.json"), static_cache_sha256=cache,
            stage0_run=False, quantized_through_layer=10, hypotheses="Workspace/consumed-H/down-only stable-write plus release of final dead private subgroup H; no finite/all-linear extension"), indent=2))
        statuses = {}
        for side, command in commands.items():
            run = out / side
            run.mkdir()
            env = os.environ.copy()
            env.update(RESEARCH_RUN_DIR=str(run), REALQ_PHASE_PEAK_MODE="exact", OMP_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            (out / "active.json").write_text(json.dumps(dict(label=side, command=command)))
            with (out / (side + ".log")).open("x") as log, (run / "nvml.csv").open("x") as gpu_log:
                monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--id=0",
                                            "--format=csv,noheader,nounits", "--loop-ms=100"], stdout=gpu_log, stderr=subprocess.STDOUT)
                try:
                    proc = subprocess.Popen(["timeout", "--signal=TERM", "--kill-after=30s", "7200", *command],
                                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = proc.wait()
                finally:
                    monitor.terminate()
                    monitor.wait(timeout=5)
            if code:
                raise RuntimeError(side + " trajectory failed; no further candidate arm")
            stats = read(run / "baseline-profile_metrics.json")
            quality = re.search(r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)", (out / (side + ".log")).read_text(errors="replace"))
            if not quality:
                raise RuntimeError("missing held-out KL/PPL")
            peak = read(run / "process_allocator_peak_stage2_rank0.json")
            memory_files = sorted({str(path) for path in (ROOT / "outputs").glob("*/" + out.name + "-" + side + "/memory_profile/*.jsonl")})
            values = [int(line.strip()) for line in (run / "nvml.csv").read_text().splitlines() if line.strip().isdigit()]
            if not values:
                raise RuntimeError("NVML telemetry missing")
            statuses[side] = dict(metrics=stats, kl=float(quality.group(1)), ppl=float(quality.group(2)),
                layers=read(run / "layer_output_fingerprints.json"), memory_profile_files=memory_files,
                process_peak=peak, nvml_mib=max(values), inverse=read(run / "inverse_lifetime_stage2_rank0.json"), dead=read(run / "dead_subgroup_input_stage2_rank0.json"))
            (run / "status.json").write_text(json.dumps(statuses[side], indent=2))
            if not all(digest(file) == expected for file, expected in bound.items()):
                raise RuntimeError("source changed during cached prefix")
        c, n = statuses["control"], statuses["candidate"]
        exact = all(c["metrics"][key] == n["metrics"][key] for key in ("state_sha256", "state_tensors")) and all(c[key] == n[key] for key in ("kl", "ppl", "layers"))
        phases = [phase_peaks(s, ROOT) for s in (c, n)]
        missing = sorted(set(phases[0]) ^ set(phases[1]))
        regressions = []
        for key in sorted(set(phases[0]) & set(phases[1])):
            delta = {metric: phases[1][key][metric] - phases[0][key][metric]
                     for metric in ("allocated", "reserved", "boundary_allocated", "boundary_reserved")}
            if any(v > 0 for v in delta.values()):
                regressions.append(dict(key=key, delta=delta))
        exact_peaks = bool(phases[0]) and all(p["measurement_mode"] == "exact" for arm in phases for p in arm.values())
        global_ok = n["nvml_mib"] <= c["nvml_mib"] and all(n["process_peak"][k] <= c["process_peak"][k]
                           for k in ("peak_allocated_bytes", "peak_reserved_bytes"))
        coverage = True
        for side, expected in (("control", False), ("candidate", True)):
            arm = statuses[side]
            inverse, dead = arm["inverse"], arm["dead"]
            writes, calls = inverse["weight_write_calls"], inverse["calls"]
            coverage = coverage and inverse["helper_unchanged"] and inverse["stable_helper_unchanged"] and dead["helpers_unchanged"]
            coverage = coverage and len(writes) == 11 and len(calls) == 44
            coverage = coverage and all(row["enabled"] == expected and (not expected or row["pointer_preserved"]) for row in writes)
            coverage = coverage and dead["enabled"] == expected
            coverage = coverage and (len(inverse["reuse_calls"]) == 44 if expected else not inverse["reuse_calls"])
            coverage = coverage and len(dead["records"]) == (11 if expected else 0)
            if expected:
                coverage = coverage and [row["layer"] for row in dead["records"]] == list(range(11))
                coverage = coverage and all(row["released_private_input_bytes"] == 144 * 2**20 and row["retained_factors"] == 4 for row in dead["records"])
        separate = compare_points(*(boundaries(arm["memory_profile_files"]) for arm in (c,n)))
        (out / "separate_start_end.json").write_text(json.dumps(separate, indent=2))
        report["separate_points_passed"] = separate["valid"]
        report.update(exactness_passed=exact, cache_unchanged=cache == cache_hashes(reference, progress),
            exact_interval_peaks=exact_peaks, missing_phases=missing, regressions=regressions,
            global_memory_gate=global_ok, dead_input_inverse_write_coverage=coverage,
            control_nvml_mib=c["nvml_mib"], candidate_nvml_mib=n["nvml_mib"])
        report["valid"] = exact and report["cache_unchanged"] and exact_peaks and not missing and not regressions and global_ok and coverage and separate["valid"]
        if not report["valid"]:
            raise RuntimeError("cached prefix numerical or memory gate failed; recorded diagnosis requires review")
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        report["error"] = repr(exc)
        raise
    finally:
        report["finished"] = time.strftime("%F %T")
        (out / "status.json").write_text(json.dumps(report, indent=2))
        with (ROOT / "docs/memory_speed_optimization_journal.md").open("a", encoding="utf-8") as journal:
            journal.write("\n\n### 末组已死私有H输入：真实缓存前缀条件验证\n\n"
                "**理论/参考。** OLLA最后消费者与位置（https://arxiv.org/html/2210.12924v2），Checkmate存储依赖与调度"
                "（https://arxiv.org/abs/1910.02653v3）。\n\n"
                "**具体方法。** 仅在两缓冲区全流程出现数值正确的资源失败、真实捕获/重放通过后，"
                "回到workspace/消费H复用/down-only稳定写回组合，在块循环前释放最后已死私有H_sub，保留四份因子；不启用finite/all-linear扩展。保持转换/补偿/Adam数学不变。"
                "两臂复用同一核验缓存，跑真实0–10层前缀保留Stage-2分配历史，覆盖先前层3/10问题。\n\n"
                f"**实际效果。** `{out / 'status.json'}`；valid={report['valid']}，skipped={report.get('skipped',False)}。"
                "层27及Stage-0未验证，不能写成新全程收益。\n\n"
                "**评价与反思。** 此处只证明释放点之后的存储生命周期；不声称可修复先前finite首子组早期回升。若失败先审查逐区间和逐点差值，"
                "不用校准/学习率调整掩盖资源失败，重复完整实验留给稳定候选。\n")
        if reader:
            reader.wait(timeout=15)
        progress(json.dumps(report))


if __name__ == "__main__":
    main()
