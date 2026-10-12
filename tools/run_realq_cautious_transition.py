"""Four-way Cautious ablation at real refresh and next-GPTQ boundaries."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_realq_adam_direction_snapshot import parse_targets

SOURCES = ["realq/refresh/block_gd.py", "realq/refresh/adam_transition_probe.py",
           "realq/quant/realq_layer.py", "realq/runner/layer_loop.py", "realq/config.py",
           "realq/pipeline.py", "tools/evaluate_adam_transition.py", "tools/run_realq_cautious_transition.py"]
POLICIES = ("adam", "cautious_mask", "cautious_scale", "cautious")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_arm(args, campaign, seed, target, policy):
    label = f"s{seed}-l{target['layer']}-{target['module'].replace('.', '_')}-b{target['block']}-{policy}"
    directory = campaign / label
    directory.mkdir()
    snapshots = directory / "transition"
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               REALQ_ADAM_PROBE_RULE=policy, REALQ_ADAM_PROBE_LAYER=str(target["layer"]),
               REALQ_ADAM_PROBE_MODULE=str(target["module"]), REALQ_ADAM_PROBE_BLOCK=str(target["block"]),
               REALQ_ADAM_TRANSITION_DIR=str(snapshots),
               REALQ_ADAM_TRANSITION_STOP="0" if args.continue_prefix else "1")
    env.pop("REALQ_ADAM_HISTORY_POLICY", None)
    cmd = [sys.executable, "-u", "-m", "realq.ptq", "--model", args.model,
           "--dataset", "wikitext2", "--eval_datasets", "wikitext2", "--nsamples", "8",
           "--seq_len", "64", "--eval_seq_len", "128", "--eval_samples", "16", "--w_bits", "4",
           "--w_clip", "true", "--num_groups", "4", "--blocksize", "128", "--act_order", "true",
           "--rotate", "false", "--group_parallel_quant", "none", "--global_loss_bsz", "1", "--bsz", "1",
           "--hessian_accum_bsz", "8", "--backward_samples", "8", "--backward_bsz", "1",
           "--final_layer_backward_bsz", "1", "--quant_stop_layer", "4", "--log_column_block_loss", "true",
           "--skip_eval", "false", "--skip_kl_ppl_eval", "false", "--seed", str(seed), "--refresh_seed", "0",
           "--output_dir", str(directory / "logs"), "--exp", label]
    started = time.monotonic()
    print(f"[campaign] capture START {label}", flush=True)
    record = {"command": cmd, "seed": seed, "target": target, "policy": policy}
    with (directory / "run.log").open("w") as log:
        proc = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30s", "900", *cmd], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record["capture_returncode"] = proc.returncode
    record["capture_seconds"] = time.monotonic()-started
    log = (directory / "run.log").read_text(errors="replace")
    lines = re.findall(r"\[adam_probe_snapshot\] ([^\r\n]+)", log)
    record["snapshot"] = dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", lines[0])) if len(lines)==1 else {}
    record["prefix_metrics"] = re.findall(r"Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)", log)
    (directory / "status.json").write_text(json.dumps(record, indent=2))
    if proc.returncode or not (snapshots / "COMPLETE").is_file() or len(lines) != 1 or record["snapshot"].get("rule") != policy:
        raise RuntimeError(f"capture failed {label}; see {directory / 'run.log'}")
    print(f"[campaign] capture OK {label}; offline evaluation START", flush=True)
    eval_cmd = [sys.executable, "-u", str(ROOT / "tools/evaluate_adam_transition.py"), "--model", args.model,
                "--directory", str(snapshots), "--eval-samples", str(args.eval_samples)]
    with (directory / "evaluation.log").open("w") as log:
        proc = subprocess.run(eval_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record["evaluation_returncode"] = proc.returncode
    record["total_seconds"] = time.monotonic()-started
    (directory / "status.json").write_text(json.dumps(record, indent=2))
    if proc.returncode:
        raise RuntimeError(f"offline evaluation failed {label}; see {directory / 'evaluation.log'}")
    evaluation = json.loads((snapshots / "evaluation.json").read_text())
    if policy not in evaluation["update"]["exactly_matched_policies"]:
        raise RuntimeError(f"actual update differs from selected policy: {label}")
    record["evaluation"] = evaluation
    record["directory"] = str(directory)
    print(f"[campaign] DONE {label}: " + json.dumps({k: {"loss":v['surrogate']['mean'],"kl":v['kl']} for k,v in evaluation['stages'].items()}), flush=True)
    return record


def audit_pair(base, candidate):
    import torch
    fields = ("w_sha256", "g_sha256", "m_sha256", "v_sha256", "adam_step")
    for field in fields:
        if not base["snapshot"].get(field) or base["snapshot"][field] != candidate["snapshot"].get(field):
            raise RuntimeError(f"pre-state mismatch {field}")
    for field in ("quant_state_hashes",):
        if base["evaluation"]["context"][field] != candidate["evaluation"]["context"][field]:
            raise RuntimeError(f"GPTQ state mismatch: {field}")
    if base["evaluation"]["validation"] != candidate["evaluation"]["validation"]:
        raise RuntimeError("validation sample mismatch")
    if base["evaluation"]["stages"]["before_update"] != candidate["evaluation"]["stages"]["before_update"]:
        raise RuntimeError("before-update loss/KL/output mismatch")
    for file in ("before_update.pt",):
        a=torch.load(Path(base["directory"])/"transition"/file, weights_only=True)
        b=torch.load(Path(candidate["directory"])/"transition"/file, weights_only=True)
        if not torch.equal(a,b):
            raise RuntimeError(f"state tensor mismatch {file}")
    # Full current prefix and reference/loss data must match, not just Adam slices.
    def identical(a,b):
        if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict): return a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)): return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
        return a==b
    ca=torch.load(Path(base["directory"])/"transition/context.pt",weights_only=False)
    cb=torch.load(Path(candidate["directory"])/"transition/context.pt",weights_only=False)
    for key in ("prefix_layers","next_layer","perm"):
        if not identical(ca[key],cb[key]): raise RuntimeError(f"context differs: {key}")
    ra=torch.load(Path(base["directory"])/"transition/refresh.pt",weights_only=False)
    rb=torch.load(Path(candidate["directory"])/"transition/refresh.pt",weights_only=False)
    if not identical(ra,rb): raise RuntimeError("refresh gradient/loss inputs differ")
    qa=torch.load(Path(base["directory"])/"transition/after_next_quant.pt",weights_only=True)
    qb=torch.load(Path(candidate["directory"])/"transition/after_next_quant.pt",weights_only=True)
    start=ca["column_end"]; end=min(start+128, qa.shape[1])
    columns=ca["perm"][start:end] if ca["perm"] is not None else torch.arange(start,end)
    return {"same_pre_state":True,"next_quantized_block_changed_fraction":float((qa[:,columns]!=qb[:,columns]).float().mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--tag",required=True)
    parser.add_argument("--targets",default="0,self_attn.o_proj,1;3,mlp.down_proj,1;3,mlp.down_proj,16")
    parser.add_argument("--seeds",default="42,43")
    parser.add_argument("--policies",default=",".join(POLICIES))
    parser.add_argument("--eval-samples",type=int,default=16)
    parser.add_argument("--continue-prefix",action="store_true")
    args=parser.parse_args()
    policies=args.policies.split(",")
    if policies[0]!="adam" or len(set(policies))!=len(policies) or set(policies)-set(POLICIES):
        parser.error("policies must start with adam and contain unique allowed arms")
    campaign=ROOT/"outputs"/f"realq-adam-transition-{args.tag}"
    campaign.mkdir(exist_ok=False)
    sources={name:sha(ROOT/name) for name in SOURCES}
    summary={"protocol":vars(args),"sources":sources,"results":{},"failure":None}
    try:
        for seed in map(int,args.seeds.split(",")):
            for target in parse_targets(args.targets):
                key=f"s{seed}-l{target['layer']}-{target['module']}-b{target['block']}"
                arms={}
                summary["results"][key]=arms
                for policy in policies:
                    arms[policy]=run_arm(args,campaign,seed,target,policy)
                    if policy!="adam": arms[policy]["audit"]=audit_pair(arms["adam"],arms[policy])
                    (campaign/"partial_summary.json").write_text(json.dumps(summary,indent=2))
    except BaseException as error:
        summary["failure"]=repr(error)
        raise
    finally:
        summary["sources_unchanged"]=all(sha(ROOT/name)==digest for name,digest in sources.items())
        (campaign/"summary.json").write_text(json.dumps(summary,indent=2))
        print(f"[campaign] summary: {campaign/'summary.json'}",flush=True)
    if not summary["sources_unchanged"]: raise RuntimeError("source changed during campaign")


if __name__=="__main__": main()
