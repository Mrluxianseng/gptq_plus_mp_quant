"""Compare BF16 vs FP32 target-linear arithmetic at saved transitions."""
from __future__ import annotations
import argparse,json,math,types
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from realq.attention import configure_attention_backend
from realq.refresh.adam_transition_probe import tensor_hash
from utils import data_utils,model_utils,quant_utils
from utils.loss_utils import tokenwise_kl_from_logits
from utils.reproducibility import configure_reproducibility


@torch.no_grad()
def kl(analyzer,tokens,teacher):
    values=[]
    for i,row in enumerate(tokens):
        h=analyzer.model.model(input_ids=row[None].cuda(),use_cache=False).last_hidden_state[0]
        s=analyzer.get_lm_head()(h).float()
        t=analyzer.get_lm_head()(teacher[i].cuda()).float()
        values.append(tokenwise_kl_from_logits(s,t).double().mean().item())
    return sum(values)/len(values),values


def run(model_path,campaign,targets=None):
    summary=json.loads((campaign/"summary.json").read_text())
    if summary["failure"] or not summary["sources_unchanged"]: raise RuntimeError("Invalid campaign")
    results={}
    for key,arms in summary["results"].items():
        if targets and key not in targets: continue
        print(f"[precision] START {key}",flush=True)
        d=Path(arms["adam"]["directory"])/"transition"
        context=torch.load(d/"context.pt",map_location="cpu",weights_only=False)
        ev=arms["adam"]["evaluation"]["validation"]
        analyzer=model_utils.ModelAnalyzer(model_path,ev["seq_len"],skip_state_dict=True)
        configure_attention_backend(analyzer.model,context["config"]["attention_backend"])
        quant_utils.add_actquant(analyzer)
        analyzer.model.eval().requires_grad_(False).cuda()
        texts=data_utils._get_dataset(analyzer.tokenizer,"wikitext2","validation")
        encoded=analyzer.tokenizer("\n\n".join(texts),return_tensors="pt").input_ids[0]
        tokens=encoded[:ev["windows"]*ev["seq_len"]].reshape(ev["windows"],ev["seq_len"])
        if tensor_hash(tokens)!=ev["tokens_sha256"]: raise RuntimeError("validation tokens differ")
        with torch.no_grad():
            teacher=[analyzer.model.model(input_ids=x[None].cuda(),use_cache=False).last_hidden_state[0].cpu() for x in tokens]
            for i,state in enumerate(context["prefix_layers"]): analyzer.get_layers()[i].load_state_dict(state,strict=True)
            analyzer.get_layers()[context["layer"]+1].load_state_dict(context["next_layer"],strict=True)
        target=analyzer.get_layers()[context["layer"]].get_submodule(context["module"])
        weights={p:torch.load(d/f"{p}.pt",map_location="cpu",weights_only=True) for p in ("before_update","after_update","after_next_quant")}
        baseline={}; high={}
        for stage,w in weights.items():
            target.weight.copy_(w.to(device="cuda",dtype=torch.bfloat16))
            baseline[stage]=kl(analyzer,tokens,teacher)
            expected=arms["adam"]["evaluation"]["stages"][stage]["kl"]
            if abs(baseline[stage][0]-expected)>1e-7:
                raise RuntimeError(f"BF16 control failed to replay {stage}: {baseline[stage][0]} vs {expected}")
        target.weight=torch.nn.Parameter(weights["before_update"].to(device="cuda",dtype=torch.float32),requires_grad=False)
        original_forward=target.forward
        def fp32_forward(module,input):
            out=F.linear(input.float(),module.weight,module.bias)
            return out.to(input.dtype)
        target.forward=types.MethodType(fp32_forward,target)
        for stage,w in weights.items():
            target.weight.copy_(w.to(device="cuda",dtype=torch.float32))
            high[stage]=kl(analyzer,tokens,teacher)
        target.forward=original_forward
        result={"target":key,"bf16":{k:v[0] for k,v in baseline.items()},"fp32_target_linear":{k:v[0] for k,v in high.items()},"changes":{}}
        for stage in ("after_update","after_next_quant"):
            result["changes"][stage]={
                "bf16_delta_vs_before":baseline[stage][0]-baseline["before_update"][0],
                "fp32_delta_vs_before":high[stage][0]-high["before_update"][0],
                "bf16_minus_fp32_delta":(baseline[stage][0]-baseline["before_update"][0])-(high[stage][0]-high["before_update"][0]),
                "bf16_per_window":baseline[stage][1],"fp32_per_window":high[stage][1]}
        results[key]=result
        (campaign/"precision_comparison.json").write_text(json.dumps(results,indent=2))
        print(f"[precision] DONE {key}: "+json.dumps(result["changes"]),flush=True)
        del analyzer,target
        torch.cuda.empty_cache()
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("campaign",type=Path);p.add_argument("--model",required=True);p.add_argument("--target",action="append");a=p.parse_args()
    torch.set_num_threads(4);configure_reproducibility(0,deterministic=True)
    run(a.model,a.campaign,set(a.target) if a.target else None)
