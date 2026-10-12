"""Fresh, serial REAL-Q validation of down_proj inverse storage scheduling.

Control is the accepted checkpoint+CE system. Candidate adds only the tested
inverse lifetime/reuse method. Prefix screens are not full-model evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_full_pipeline_memory_campaign import stop_process_tree
from tools.run_full_pipeline_memory_profile import set_option
from tools.run_low_overhead_speed_campaign import core_sources,check_sources

REFERENCES=(
    ("qwen17b","qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125"),
    ("qwen06b","qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457"),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefix(helper_sha, pool_sha=None, stable_sha=None):
    return '''# Bound research helper; shared observers in both arms.
import atexit as _hinv_atexit
import hashlib as _hinv_hashlib
import json as _hinv_json
import os as _hinv_os
import sys as _hinv_sys
from pathlib import Path as _HinvPath
_hinv_sys.path.insert(0,str(_HinvPath.cwd()))
_hinv_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from gptq_utils import gptq_plus_utils as _hinv_impl
from tools.hessian_inverse_lifetime_prototype import install_down_proj as _hinv_install, build_reusing_fasterquant as _hinv_build_reuse
_hinv_helper=_HinvPath.cwd()/"tools/hessian_inverse_lifetime_prototype.py"
_hinv_expected="''' + helper_sha + '''"
if _hinv_hashlib.sha256(_hinv_helper.read_bytes()).hexdigest()!=_hinv_expected:
    raise RuntimeError("inverse helper changed since CUDA preflight")
_hinv_release="--research_hinv_early_release" in _hinv_sys.argv
_hinv_reuse="--research_hinv_reuse_consumed_h" in _hinv_sys.argv
_hinv_pool="--research_hinv_isolated_pool" in _hinv_sys.argv
_hinv_stable="--research_hinv_stable_weight" in _hinv_sys.argv
_hinv_stable_expected="''' + (stable_sha or "") + '''"
_hinv_stable_helper=_HinvPath.cwd()/"tools/hessian_stable_weight_prototype.py"
if _hinv_stable_expected and _hinv_hashlib.sha256(_hinv_stable_helper.read_bytes()).hexdigest()!=_hinv_stable_expected:
    raise RuntimeError("stable weight helper changed since CUDA preflight")
_hinv_pool_expected="''' + (pool_sha or "") + '''"
_hinv_pool_helper=_HinvPath.cwd()/"tools/hessian_inverse_pool_prototype.py"
if _hinv_pool_expected and _hinv_hashlib.sha256(_hinv_pool_helper.read_bytes()).hexdigest()!=_hinv_pool_expected:
    raise RuntimeError("private pool helper changed since CUDA preflight")
if _hinv_pool and not _hinv_reuse:
    raise RuntimeError("isolated pool requires consumed H destination reuse")
if _hinv_reuse and not _hinv_release:
    raise RuntimeError("reuse first requires the validated early-release combination")
for _hinv_marker in ("--research_hinv_early_release","--research_hinv_reuse_consumed_h","--research_hinv_isolated_pool","--research_hinv_stable_weight"):
    if _hinv_marker in _hinv_sys.argv:
        _hinv_sys.argv.remove(_hinv_marker)
_hinv_calls,_hinv_reuse_calls,_hinv_write_calls=[],[],[]
if _hinv_pool_expected:
    from tools.hessian_inverse_pool_prototype import install_isolated as _hinv_install_pool
    _hinv_source=_hinv_install_pool(_hinv_impl,enabled=_hinv_pool,records=_hinv_calls)
else:
    _hinv_source=_hinv_install(_hinv_impl,enabled=_hinv_release,records=_hinv_calls)
_hinv_reused,_hinv_reuse_source=_hinv_build_reuse(_hinv_impl)
if _hinv_stable_expected:
    from tools.hessian_stable_weight_prototype import install_fasterquant as _hinv_install_stable
    _hinv_reuse_source=_hinv_install_stable(_hinv_impl,enabled=_hinv_stable,records=_hinv_write_calls)
elif _hinv_reuse:
    _hinv_impl.GPTQPlus.fasterquant=_hinv_reused
_hinv_init_original=_hinv_impl.GPTQPlus.__init__
def _hinv_init(self,*args,**kwargs):
    _hinv_init_original(self,*args,**kwargs)
    self._hessian_factor_reuse_records=_hinv_reuse_calls
_hinv_impl.GPTQPlus.__init__=_hinv_init
_hinv_stage="stage0" if "--exit_after_precompute" in _hinv_sys.argv else "stage2"
def _hinv_record():
    _hinv_destination=_HinvPath(_hinv_os.environ["RESEARCH_RUN_DIR"])
    _hinv_report=dict(stage=_hinv_stage,early_release=_hinv_release,reuse=_hinv_reuse,isolated_pool=_hinv_pool,
        pool_helper_sha256=_hinv_pool_expected,
        pool_helper_unchanged=(not _hinv_pool_expected or _hinv_hashlib.sha256(_hinv_pool_helper.read_bytes()).hexdigest()==_hinv_pool_expected),
        helper_sha256=_hinv_expected,
        helper_unchanged=_hinv_hashlib.sha256(_hinv_helper.read_bytes()).hexdigest()==_hinv_expected,
        patched_inverse_sha256=_hinv_hashlib.sha256(_hinv_source.encode()).hexdigest(),
        patched_fasterquant_sha256=_hinv_hashlib.sha256(_hinv_reuse_source.encode()).hexdigest(),
        calls=_hinv_calls,reuse_calls=_hinv_reuse_calls,weight_write_calls=_hinv_write_calls,
        stable_weight=_hinv_stable,stable_helper_sha256=_hinv_stable_expected,
        stable_helper_unchanged=(not _hinv_stable_expected or _hinv_hashlib.sha256(_hinv_stable_helper.read_bytes()).hexdigest()==_hinv_stable_expected))
    (_hinv_destination/f"inverse_lifetime_{_hinv_stage}_rank{_hinv_os.environ.get('RANK','0')}.json").write_text(_hinv_json.dumps(_hinv_report,indent=2))
_hinv_atexit.register(_hinv_record)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight",type=Path,required=True)
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--tag-suffix",required=True)
    p.add_argument("--variant",choices=("stable-weight",),default="stable-weight")
    p.add_argument("--models",choices=("qwen17b","qwen06b"),nargs="+",default=["qwen17b","qwen06b"])
    p.add_argument("--prefix-stop-layer",type=int,default=0)
    p.add_argument("--full-pairs",type=int,default=3)
    args=p.parse_args()
    if args.full_pairs<1:
        raise ValueError("positive pair count required")
    if not 0 <= args.prefix_stop_layer < 27:
        raise ValueError("prefix screen must end before final layer27")
    out=args.output_directory.resolve()
    out.mkdir(exist_ok=False)
    jobs,proc=[],None
    helper=ROOT/"tools/hessian_inverse_lifetime_prototype.py"
    helper_sha=sha(helper)
    pool_helper=ROOT/"tools/hessian_inverse_pool_prototype.py"
    pool_sha=None
    stable_helper=ROOT/"tools/hessian_stable_weight_prototype.py"
    stable_sha=sha(stable_helper)
    references=[pair for pair in REFERENCES if pair[0] in args.models]
    tags=[]
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    def execute(label,command):
        nonlocal proc
        (out/"active.json").write_text(json.dumps(dict(label=label,command=command),indent=2))
        print(f"[inverse-followup] starting {label}",flush=True)
        with (out/(label+".log")).open("x") as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            code=proc.wait()
        jobs.append(dict(label=label,returncode=code,finished=time.strftime("%F %T")))
        (out/"jobs.json").write_text(json.dumps(jobs,indent=2))
        print(f"[inverse-followup] {label} exited {code}",flush=True)
        return code
    def link_completed(run,reference,label):
        if not (run/"status.json").exists() or not read(run/"status.json").get("valid"):
            return
        def win(path):
            path=str(path)
            if not path.startswith("/mnt/d/"):
                raise ValueError("housekeeping only within task D drive")
            return "D:/"+path[len("/mnt/d/"):]
        execute("disk-"+label,["/mnt/c/Users/65377/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
            win(ROOT/"tools/deduplicate_completed_prepared_checkpoint.py"),
            "--control-run",win(reference),"--candidate-run",win(run),
            "--output",win(out/("disk-"+label+".json")),"--apply"])
    try:
        while not args.preflight.exists():
            time.sleep(5)
        checked=read(args.preflight)
        if not checked.get("valid") or len(checked.get("models",[]))!=2:
            raise RuntimeError("requires successful actual-dimension CUDA checks for both models")
        if args.variant in {"reuse","isolated-pool","stable-weight"} and any(len(row["api"]["arms"][1].get("reuse_calls",[]))!=4 for row in checked["models"]):
            raise RuntimeError("reuse preflight did not exercise every group")
        if pool_sha:
            if checked.get("pool_source_sha256")!=pool_sha or not checked.get("helper_cases") or not all(x["exact"] for x in checked["helper_cases"]):
                raise RuntimeError("requires bound successful pool retry/fallback preflight")
            (out/"pool_prototype.py").write_bytes(pool_helper.read_bytes())
        if checked.get("stable_weight_source_sha256")!=stable_sha or not checked.get("copy_cast_cases") or not all(x["exact"] for x in checked["copy_cast_cases"]):
            raise RuntimeError("requires bound CUDA stable-write/cast preflight")
        (out/"stable_weight_prototype.py").write_bytes(stable_helper.read_bytes())
        # Check patched method source against actual preflight, not just a flag.
        import torch
        from gptq_utils import gptq_plus_utils as impl
        from tools.hessian_inverse_lifetime_prototype import build_method,build_reusing_fasterquant
        _,inverse_source=build_method(impl)
        if hashlib.sha256(inverse_source.encode()).hexdigest()!=checked["helper_source_sha256"]:
            raise RuntimeError("inverse arithmetic source changed since CUDA validation")
        if args.variant in {"reuse","isolated-pool","stable-weight"}:
            from tools.hessian_stable_weight_prototype import build_stable_fasterquant
            _,reuse_source=build_stable_fasterquant(impl)
            if reuse_source.encode()!=(args.preflight.parent/"patched_fasterquant.py").read_bytes():
                raise RuntimeError("storage reuse source changed since CUDA validation")
        entry=out/"entry.py"
        first_reference=ROOT/"outputs"/f"full_pipeline_profile_{REFERENCES[0][1]}-r1-candidate"
        launch=read(first_reference/"manifest.json")
        raw=read(launch["stage2_source_manifest"])
        base=Path(raw["command"][raw["command"].index("baseline-profile")-1])
        entry.write_text(prefix(helper_sha,pool_sha,stable_sha)+"\n"+base.read_text())
        compile(entry.read_text(),str(entry),"exec")
        (out/"manifest.json").write_text(json.dumps(dict(variant=args.variant,helper_sha256=helper_sha,pool_helper_sha256=pool_sha,stable_helper_sha256=stable_sha,
            entry_sha256=sha(entry),preflight=str(args.preflight),preflight_sha256=sha(args.preflight),
            control="Accepted checkpoint+CE system; not raw upstream baseline",
            candidate="Consumed-H inverse reuse plus stable down_proj weight destination; no private pool",
            config="cal2x1024/micro2/eval2x1024/seed44; unchanged Adam/GPTQ/Fisher"),indent=2))
        for stop_layer,pairs in ((args.prefix_stop_layer,1),(27,args.full_pairs)):
            for model,accepted in references:
                if sha(stable_helper)!=stable_sha:
                    raise RuntimeError("stable weight source changed during queue")
                if sha(helper)!=helper_sha:
                    raise RuntimeError("research helper changed during queue")
                if pool_sha and sha(pool_helper)!=pool_sha:
                    raise RuntimeError("private pool helper changed during queue")
                if shutil.disk_usage(ROOT).free<12*1024**3:
                    raise RuntimeError("fresh paired runs require 12 GiB disk")
                reference=ROOT/"outputs"/f"full_pipeline_profile_{accepted}-r1-candidate"
                original=read(reference/"manifest.json")
                expected=core_sources(original)
                check_sources(expected)
                scope=f"layer{stop_layer}" if stop_layer<27 else "full"
                tag=f"{model}-hinv-{args.variant}-{scope}-{args.tag_suffix}"
                tags.append(tag)
                folder=out/(tag+"-manifests")
                folder.mkdir()
                for side in ("control","candidate"):
                    for stage in ("stage0","stage2"):
                        data=read(original[stage+"_source_manifest"])
                        cmd=data["command"]
                        cmd[cmd.index("baseline-profile")-1]=str(entry)
                        if stage=="stage2":
                            set_option(cmd,"--quant_stop_layer",str(stop_layer))
                            if side=="candidate":
                                cmd.append("--research_hinv_early_release")
                                if args.variant in {"reuse","isolated-pool","stable-weight"}:
                                    cmd.append("--research_hinv_reuse_consumed_h")
                                cmd.append("--research_hinv_stable_weight")
                                if pool_sha:
                                    cmd.append("--research_hinv_isolated_pool")
                        data.update(scope_note="Independent Hessian inverse lifetime/pool component; both arms share accepted checkpoint+CE, same REAL-Q algorithm/calibration. Any stop before layer27 is prefix only.",
                                    explicit_shared_conditions="cal2x1024/refresh microbatch2/eval2x1024/seed44; alpha0 with Adam block-GD; sequential group mode")
                        (folder/(side+"-"+stage+".json")).write_text(json.dumps(data,indent=2))
                command=[sys.executable,"-u",str(ROOT/"tools/run_full_pipeline_memory_campaign.py"),
                    "--tag",tag,"--pairs",str(pairs),"--gpu-indices","0","--timeout-seconds","7200"]
                for side in ("control","candidate"):
                    for stage in ("stage0","stage2"):
                        command.extend([f"--{side}-{stage}-manifest",str(folder/(side+"-"+stage+".json"))])
                code=execute(tag,command)
                check_sources(expected)
                campaign=ROOT/"outputs"/f"memory_campaign_{tag}"
                status=read(campaign/"status.json")
                def record_campaign():
                    execute("record-"+tag,[sys.executable,str(ROOT/"tools/record_completed_memory_followup_20261010.py"),
                        "--wait-for",str(campaign/"status.json"),"--campaign-tags",tag,"--output-dir",str(out/("records-"+tag))])
                if code or not status.get("valid"):
                    record_campaign()
                    raise RuntimeError(f"{tag} failed numerical or measured-memory gate; inspect before more GPU runs")
                for pair in range(1,pairs+1):
                    for side in ("control","candidate"):
                        run=ROOT/"outputs"/f"full_pipeline_profile_{tag}-r{pair}-{side}"
                        actual=read(run/"inverse_lifetime_stage2_rank0.json")
                        expected_calls=4*(stop_layer+1)
                        if (not actual["helper_unchanged"] or len(actual["calls"])!=expected_calls
                                or any(row["enabled"]!=(side=="candidate") for row in actual["calls"])):
                            raise RuntimeError("requested method did not cover all down_proj inverse calls")
                        if (not actual.get("stable_helper_unchanged") or actual.get("stable_helper_sha256")!=stable_sha
                                or len(actual.get("weight_write_calls",[]))!=stop_layer+1
                                or (side=="candidate" and not all(x["pointer_preserved"] for x in actual["weight_write_calls"]))):
                            raise RuntimeError("stable write coverage/source/destination not validated")
                        if pool_sha and (not actual["pool_helper_unchanged"] or actual["pool_helper_sha256"]!=pool_sha):
                            raise RuntimeError("private pool source not bound/unchanged")
                        if side=="candidate" and args.variant in {"reuse","isolated-pool","stable-weight"} and len(actual["reuse_calls"])!=expected_calls:
                            raise RuntimeError("not all consumed groups reused storage")
                        if side=="candidate" and pool_sha and not all(x["private_pool_released"] for x in actual["calls"]):
                            raise RuntimeError("not all solver pools released")
                record_campaign()
                for pair in range(1,pairs+1):
                    for side in ("control","candidate"):
                        run=ROOT/"outputs"/f"full_pipeline_profile_{tag}-r{pair}-{side}"
                        link_completed(run,reference,tag+f"-r{pair}-"+side)
        (out/"status.json").write_text(json.dumps(dict(valid=True,jobs=jobs,campaigns=tags,finished=time.strftime("%F %T")),indent=2))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        (out/"status.json").write_text(json.dumps(dict(valid=False,error=repr(exc),jobs=jobs,campaigns=tags,
                                                     finished=time.strftime("%F %T")),indent=2))
        raise


if __name__=="__main__":
    main()
