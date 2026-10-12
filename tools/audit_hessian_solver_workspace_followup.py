"""Independent completed-repeat audit; waits until the entire GPU queue exits."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_memory_repeatability_20261010 import audit


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue",type=Path,required=True)
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--tag-suffix",required=True)
    p.add_argument("--variant",default="workspace-stable",choices=("workspace-stable",))
    p.add_argument("--models",nargs="+",default=["qwen17b","qwen06b"],choices=("qwen17b","qwen06b"))
    args=p.parse_args()
    out=args.output_directory
    out.mkdir(exist_ok=False)
    print("[inverse-audit] waiting; no GPU or large-file hashing during experiments",flush=True)
    while not (args.queue/"status.json").exists():
        time.sleep(5)
    parent=read(args.queue/"status.json")
    rows=[]
    for model,old_audit in (("qwen17b","cross-repeat-qwen17b-checkpoint-ce-threepairs-20261011-0223.json"),
                            ("qwen06b","cross-repeat-checkpoint-ce-threepairs-20261010-1650.json")):
        if model not in args.models:
            continue
        tag=f"{model}-hinv-{args.variant}-full-{args.tag_suffix}"
        campaign=ROOT/"outputs"/("memory_campaign_"+tag)
        if not (campaign/"status.json").exists() or not read(campaign/"status.json").get("valid"):
            rows.append(dict(model=model,complete_valid_full_campaign=False,
                             note="Not a full positive; missing work is not a measured negative"))
            continue
        print(f"[inverse-audit] hashing completed independent caches: {model}",flush=True)
        result=audit(campaign)
        errors=list(result["errors"])
        reference=read(ROOT/"outputs"/old_audit)["arms"][0]["canonical"]
        for field in ("state_sha256","state_tensors","kl","ppl","layers","cache",
                      "stage0_algorithm_command","stage2_algorithm_command"):
            if result["arms"][0]["canonical"][field]!=reference[field]:
                errors.append("accepted-system-reference:"+field)
        for arm in result["arms"]:
            observed=read(Path(arm["directory"])/"inverse_lifetime_stage2_rank0.json")
            if len(observed["calls"])!=112 or not observed["helper_unchanged"]:
                errors.append("inverse coverage:"+arm["directory"])
            if arm["side"]=="candidate" and args.variant!="early-release" and len(observed["reuse_calls"])!=112:
                errors.append("storage reuse coverage:"+arm["directory"])
            if arm["side"]=="candidate" and not all(x.get("workspace_reuse_enabled") for x in observed["calls"]):
                errors.append("solver workspace coverage:"+arm["directory"])
            if (not observed.get("stable_helper_unchanged") or len(observed.get("weight_write_calls",[]))!=28
                    or (arm["side"]=="candidate" and not all(x["pointer_preserved"] for x in observed["weight_write_calls"]))):
                errors.append("stable weight/source coverage:"+arm["directory"])
            if args.variant=="isolated-pool" and not observed.get("pool_helper_unchanged"):
                errors.append("private pool source changed:"+arm["directory"])
            if args.variant=="isolated-pool" and arm["side"]=="candidate" and not all(x.get("private_pool_released") for x in observed["calls"]):
                errors.append("solver pool coverage:"+arm["directory"])
        result["errors"],result["valid"]=errors,result["valid"] and not errors
        (out/(model+"-cross-repeat.json")).write_text(json.dumps(result,indent=2))
        rows.append(dict(model=model,complete_valid_full_campaign=True,
                         valid=result["valid"],errors=errors,requested_pairs=result["requested_pairs"],
                         all_arms_match_accepted_system_reference=not errors,
                         unique_files_hashed=result["unique_files_hashed"]))
    valid=parent.get("valid",False) and all(row.get("valid",False) for row in rows)
    report=dict(valid=valid,parent_valid=parent.get("valid",False),models=rows,
                finished=time.strftime("%F %T"),scope="Independent full-model cross-repeat/reference and method-coverage audit; no speed/physical smaller-GPU claim")
    (out/"status.json").write_text(json.dumps(report,indent=2))
    marker="<!-- inverse-lifetime-cross-repeat:"+args.tag_suffix+" -->"
    journal=ROOT/"docs/memory_speed_optimization_journal.md"
    if marker not in journal.read_text(encoding="utf-8-sig"):
        with journal.open("a",encoding="utf-8") as f:
            f.write("\n\n"+marker+"\n### AS续记：完整队列退出后的独立复核\n\n"
                    "**理论与方法。** 延续AS的矩阵最后消费者与原布局存储复用。除每对审计之外，重新实际哈希已完成缓存，核对六臂跨重复身份与已接受checkpoint＋CE参考，逐worker核查112次down_proj inverse/复用调用覆盖。仅共享SHA一致的已完成prepared FP硬链接，校准/Fisher统计独立。\n\n"
                    "**实际效果及评价。** 完整独立审计valid="+str(valid)+"；详见下列原始状态。缺失或未完成全层不列为成功；失败不剔除、不重复推广。诊断时间不计加速，正式校准规模和多卡仍待验证。\n\n"
                    "```json\n"+json.dumps(report,ensure_ascii=False,indent=2)+"\n```\n\n"
                    "**反思及证据。** 组件原值与完整资源值分别见AS和对应自动续记；跨重复cache/state/28层/KL/PPL、原参考身份、实际方法覆盖需同时成立。工件"+str(out)+"。\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
