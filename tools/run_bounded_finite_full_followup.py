"""Fresh bounded-finite full acceptance after interval AND separate-point gates."""
import argparse
import ast
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_full_pipeline_profiles import digest
from tools.audit_memory_boundary_points import boundaries, compare as compare_points
from tools.hessian_bounded_solver_install import build_entry
from tools.run_full_pipeline_memory_profile import set_option
from tools.run_full_pipeline_memory_campaign import stop_process_tree

REFERENCES = (
    ("qwen17b", "qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125"),
    ("qwen06b", "qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457"),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def check_bound_sources(bound):
    for filename, expected in bound.items():
        if digest(filename) != expected:
            raise RuntimeError("bound source changed: " + filename)


def audit_points(control, candidate):
    paths = [arm.get("memory_profile_files", []) for arm in (control, candidate)]
    result = compare_points(*(boundaries(files) for files in paths))
    result.update(control=paths[0], candidate=paths[1])
    return result


def prepare_entry(workspace_queue):
    template = ROOT / "tools/run_cached_bounded_finite_screen.py"
    nodes = [node for node in ast.walk(ast.parse(template.read_text()))
             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "hook" for t in node.targets)]
    if len(nodes) != 1:
        raise RuntimeError("cached shared entry hook boundary changed")
    hook = ast.literal_eval(nodes[0].value)
    hook = hook.replace('_all_enabled="--research_all_weight_stable" in sys.argv', '_all_enabled=_hinv_stable')
    hook = hook.replace('if _all_enabled:\n    sys.argv.remove("--research_all_weight_stable")\n', '')
    hook = hook.replace('impl.collect_static_end_to_end_saliency_and_fisher=_forbid_precompute',
                        'if "--exit_after_precompute" not in sys.argv:\n    impl.collect_static_end_to_end_saliency_and_fisher=_forbid_precompute')
    helper = ROOT / "tools/hessian_all_weight_storage_prototype.py"
    hook = f'if hashlib.sha256(Path({str(helper)!r}).read_bytes()).hexdigest()!={digest(helper)!r}:\n    raise RuntimeError("all-weight helper changed")\n' + hook
    raw = build_entry((workspace_queue / "entry.py").read_text(),
        installer_sha=digest(ROOT / "tools/hessian_bounded_solver_install.py"),
        finite_sha=digest(ROOT / "tools/hessian_chunked_finite_prototype.py"),
        workspace_sha=digest(ROOT / "tools/hessian_solver_workspace_prototype.py"))
    anchor = "runpy.run_path('ptq.py', run_name='__main__')"
    if raw.count(anchor) != 1 or "# Report allocator peaks" in raw or "# Profile final evaluation separately" in raw:
        raise RuntimeError("full runner requires raw unprofiled portable entry")
    source = raw.replace(anchor, hook + anchor)
    compile(source, "bounded_full_entry", "exec")
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-for", type=Path, required=True)
    p.add_argument("--workspace-queue", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    args = p.parse_args()
    out = args.output_directory.resolve()
    out.mkdir(exist_ok=False)
    proc = reader = None
    report = dict(valid=False, scope="Conditional fresh full validation; accepted checkpoint+CE control")
    signal.signal(signal.SIGTERM, lambda sig, frame: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    def execute(label, command, *, allow_invalid=False):
        nonlocal proc
        (out / "active.json").write_text(json.dumps(dict(label=label, command=command)))
        print("[bounded-finite-full] starting " + label, flush=True)
        with (out / (label + ".log")).open("x") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            code = proc.wait()
        if code and not allow_invalid:
            raise RuntimeError(f"{label} failed; no unchanged later repeats")
        return code
    try:
        print("[bounded-finite-full] waiting for cached screen; no GPU process", flush=True)
        while not args.wait_for.exists():
            time.sleep(5)
        prior = read(args.wait_for)
        if prior.get("skipped"):
            report.update(valid=prior.get("valid", False), skipped=True, reason="Existing workspace full acceptance passed")
            return
        prefix = args.wait_for.parent
        if not prior.get("exactness_passed") or not prior.get("cache_unchanged"):
            raise RuntimeError("bounded prefix numerical/cache failure; inspect before full validation")
        supplemental = audit_points(read(prefix / "control/status.json"), read(prefix / "candidate/status.json"))
        save_new(out / "prefix_separate_start_end.json", supplemental)
        if not prior.get("valid"):
            raise RuntimeError("bounded prefix interval/resource screen failed; separate-point audit preserved, no full launch")
        if not supplemental["valid"]:
            raise RuntimeError("prefix separate start/end regression or missing telemetry; no full GPU launch")
        prefix_manifest = read(prefix / "manifest.json")
        check_bound_sources(prefix_manifest["source_sha256"])
        report["prefix_separate_points_passed"] = True
        report["prefix_status_sha256"] = digest(args.wait_for)
        time.sleep(5)
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
        if any(x.strip().isdigit() for x in gpu.stdout.splitlines()):
            raise RuntimeError("another CUDA compute task active; no concurrent full acceptance")
        reader = subprocess.Popen([sys.executable, "-u", str(ROOT / "tools/stream_memory_research_progress.py"),
            "--queues", str(out), "--output", str(ROOT / "logs/realq-system-restart-20261010-1408.log"), "--append"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        entry = out / "entry.py"
        entry.write_text(prepare_entry(args.workspace_queue))
        compile(entry.read_text(), str(entry), "exec")
        bound = dict(prefix_manifest["source_sha256"])
        bound[str(entry)] = digest(entry)
        for path in (Path(__file__).resolve(), ROOT / "tools/audit_memory_boundary_points.py", ROOT / "tools/run_full_pipeline_memory_campaign.py", ROOT / "tools/run_full_pipeline_memory_profile.py", ROOT / "tools/compare_full_pipeline_profiles.py", ROOT / "tools/stream_memory_research_progress.py", ROOT / "tools/deduplicate_completed_prepared_checkpoint.py"):
            bound[str(path)] = digest(path)
        save_new(out / "source_bindings.json", bound)
        check_bound_sources(bound)
        summaries = []
        for model, accepted in REFERENCES:
            reference = ROOT / "outputs" / ("full_pipeline_profile_" + accepted + "-r1-candidate")
            manifest, status = read(reference / "manifest.json"), read(reference / "status.json")
            if not status.get("valid") or manifest["expected_ranks"] != {"stage0": 1, "stage2": 1}:
                raise RuntimeError("single-rank accepted canonical reference required")
            for repeat in (1, 2, 3):
                check_bound_sources(bound)
                tag = f"{model}-bounded-finite-full-20261011-v1-p{repeat}"
                folder = out / (tag + "-manifests")
                folder.mkdir()
                for side in ("control", "candidate"):
                    for stage in ("stage0", "stage2"):
                        data = read(manifest[stage + "_source_manifest"])
                        command = data["command"]
                        command[command.index("baseline-profile")-1] = str(entry)
                        if stage == "stage2":
                            set_option(command, "--quant_stop_layer", "27")
                            if side == "candidate":
                                command.extend(["--research_hinv_solver_workspace", "--research_hinv_reuse_consumed_h", "--research_hinv_stable_weight"])
                        data["scope_note"] = "All sequential linear weights use stable destination when hinv_stable_weight is enabled; source-bound opt-in. Same REAL-Q math and fresh Stage-0 each arm."
                        (folder / (side + "-" + stage + ".json")).write_text(json.dumps(data, indent=2))
                command = [sys.executable, "-u", str(ROOT / "tools/run_full_pipeline_memory_campaign.py"),
                           "--tag", tag, "--pairs", "1", "--gpu-indices", "0", "--timeout-seconds", "7200"]
                for side in ("control", "candidate"):
                    for stage in ("stage0", "stage2"):
                        command.extend([f"--{side}-{stage}-manifest", str(folder / (side + "-" + stage + ".json"))])
                campaign_code = execute(tag, command, allow_invalid=True)
                campaign = ROOT / "outputs" / ("memory_campaign_" + tag)
                audit = read(campaign / "pair1_audit/comparison.json")
                full_points = audit_points(
                    read(ROOT / "outputs" / ("full_pipeline_profile_" + tag + "-r1-control") / "status.json"),
                    read(ROOT / "outputs" / ("full_pipeline_profile_" + tag + "-r1-candidate") / "status.json"))
                save_new(campaign / "pair1_audit/separate_start_end.json", full_points)
                check_bound_sources(bound)
                if campaign_code or not read(campaign / "status.json").get("valid") or not audit.get("strict_measured_memory_gate_passed"):
                    raise RuntimeError("full memory/numerical gate failed; no further repeated runs")
                if not full_points["valid"]:
                    raise RuntimeError("separate full start/end point regression or missing telemetry; no later repeats")
                for side in ("control", "candidate"):
                    run = ROOT / "outputs" / ("full_pipeline_profile_" + tag + "-r1-" + side)
                    actual, writes = read(run / "status.json"), read(run / "all_weight_writes.json")
                    if actual["metrics"]["state_sha256"] != status["metrics"]["state_sha256"] or (actual["kl"], actual["ppl"]) != (status["kl"], status["ppl"]) or read(run / "layer_output_fingerprints.json") != read(reference / "layer_output_fingerprints.json"):
                        raise RuntimeError("full state/28layers/KL/PPL differ from canonical accepted reference")
                    if len(writes["records"]) != 196 or any(row["enabled"] != (side == "candidate") or (side == "candidate" and not row["pointer_preserved"]) for row in writes["records"]):
                        raise RuntimeError("all-module stable-write coverage failed")
                    inverse = read(run / "inverse_lifetime_stage2_rank0.json")
                    if len(inverse["calls"]) != 112 or not inverse["helper_unchanged"] or not inverse["stable_helper_unchanged"] or (side == "candidate" and len(inverse["reuse_calls"]) != 112) or any(row.get("bounded_finite_enabled") != (side == "candidate") for row in inverse["calls"]):
                        raise RuntimeError("inverse/storage reuse coverage failed")
                    # Completed, verified prepared FP weights only; retain all
                    # Fisher caches and evidence paths. Free disk between pairs.
                    def win(path):
                        value = str(path)
                        if not value.startswith("/mnt/d/"):
                            raise RuntimeError("housekeeping path outside D workspace")
                        return "D:/" + value[len("/mnt/d/"):]
                    execute("disk-" + tag + "-" + side, ["/mnt/c/Users/65377/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
                        win(ROOT / "tools/deduplicate_completed_prepared_checkpoint.py"), "--control-run", win(reference),
                        "--candidate-run", win(run), "--output", win(out / ("disk-" + tag + "-" + side + ".json")), "--apply"])
                summaries.append(dict(model=model, repeat=repeat, tag=tag, audit=str(campaign / "pair1_audit/comparison.json"), valid=True))
                report["completed_pairs"] = summaries
                (out / "partial.json").write_text(json.dumps(report, indent=2))
        report["valid"] = True
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
            journal.write("\n\n### 分块有限值组合：逐点门槛下的条件完整验收\n\n"
                "**理论/参考与方法。** 延续OLLA位置/寿命原则及GMLake碎片分析，结合版本绑定PyTorch有限值临时量；前缀阶段峰及独立start/end严格通过后，"
                "逐对fresh Stage-0+28层量化+评估，两模型各三对；任何一对失败不继续相同配置重复。\n\n"
                f"**实际效果。** `{out / 'status.json'}`；valid={report['valid']}，skipped={report.get('skipped',False)}，"
                f"完成配对数={len(report.get('completed_pairs',[]))}。\n\n"
                "**评价与反思。** 完整结果必须保留所有阶段峰值/区间、全部28层及canonical state/KL/PPL核对，"
                "局部通过不作完整结论。诊断计时不是速度证据；此处仍是本机pilot，不是正式2048样本或多卡验收。\n")
        if reader:
            reader.wait(timeout=15)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
