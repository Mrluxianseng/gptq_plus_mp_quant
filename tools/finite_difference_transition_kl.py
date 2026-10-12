"""Finite-difference actual REAL-Q Adam steps on paired calibration samples."""
from __future__ import annotations
import argparse,json,types
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from realq.attention import configure_attention_backend
from realq.refresh.adam_transition_probe import tensor_hash
from utils import model_utils,quant_utils
from utils.loss_utils import tokenwise_kl_from_logits
from utils.reproducibility import configure_reproducibility


def move(x):
    if isinstance(x,torch.Tensor): return x.cuda()
    if isinstance(x,dict): return {k:move(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return type(x)(move(v) for v in x)
    return x

def hidden(x): return x[0] if isinstance(x,tuple) else x

def sample_kw(refresh,i):
    def select(v):
        if isinstance(v,torch.Tensor): return v[i:i+1] if v.ndim and v.shape[0]==refresh["inps"].shape[0] else v
        if isinstance(v,(tuple,list)): return type(v)(select(z) for z in v)
        if isinstance(v,dict): return {k:select(z) for k,z in v.items()}
        return v
    return move(select(refresh["kw"]))

@torch.no_grad()
def objective(analyzer,layer,context,refresh,indices,teacher_logits,fp32_suffix=False):
    losses=[]
    for i in indices:
        x=refresh["inps"][i:i+1].cuda(); kw=sample_kw(refresh,i)
        if fp32_suffix:
            x=x.float()
            def cast_fp32(v):
                if isinstance(v,torch.Tensor): return v.float() if v.is_floating_point() else v
                if isinstance(v,(tuple,list)): return type(v)(cast_fp32(z) for z in v)
                if isinstance(v,dict): return {k:cast_fp32(z) for k,z in v.items()}
                return v
            kw=cast_fp32(kw)
        out=hidden(layer(x,**kw))
        for downstream in analyzer.get_layers()[context["layer"]+1:]: out=hidden(downstream(out,**kw))
        out=analyzer.get_lm_head()(analyzer.get_layernorm_before_head()(out)).float()
        losses.append(tokenwise_kl_from_logits(out,teacher_logits[i]).mean().double().item())
    return sum(losses)/len(losses)

def setup(model_path,campaign,key):
    summary=json.loads((campaign/"summary.json").read_text())
    if summary["failure"] or not summary["sources_unchanged"]: raise RuntimeError("invalid campaign")
    arms=summary["results"][key]; d=Path(arms["adam"]["directory"])/"transition"
    context=torch.load(d/"context.pt",map_location="cpu",weights_only=False)
    refresh=torch.load(d/"refresh.pt",map_location="cpu",weights_only=False)
    before=torch.load(d/"before_update.pt",map_location="cpu",weights_only=True)
    after=torch.load(d/"after_update.pt",map_location="cpu",weights_only=True)
    analyzer=model_utils.ModelAnalyzer(model_path,128,skip_state_dict=True)
    configure_attention_backend(analyzer.model,context["config"]["attention_backend"])
    quant_utils.add_actquant(analyzer); analyzer.model.eval().requires_grad_(False).cuda()
    for i,state in enumerate(context["prefix_layers"]): analyzer.get_layers()[i].load_state_dict(state,strict=True)
    analyzer.get_layers()[context["layer"]+1].load_state_dict(context["next_layer"],strict=True)
    layer=analyzer.get_layers()[context["layer"]]
    target=layer.get_submodule(context["module"])
    if before.shape!=target.weight.shape: raise RuntimeError("weight shape mismatch")
    target.weight.copy_(before.to(target.weight))
    with torch.no_grad():
        teacher_logits_bf16={}
        for i in refresh["selected"]:
            ref=refresh["fp_target"][i:i+1].cuda(); kw=sample_kw(refresh,i)
            for downstream in analyzer.get_layers()[context["layer"]+1:]: ref=hidden(downstream(ref,**kw))
            teacher_logits_bf16[i]=analyzer.get_lm_head()(analyzer.get_layernorm_before_head()(ref)).float().detach()
    token_info=arms["adam"]["evaluation"]["validation"]
    direction=before-after
    perm,start=context["perm"],context["column_end"]
    # Assert the saved update is zero on the frozen prefix and finite.
    if perm is not None and torch.count_nonzero(direction[:,perm][:,:start]): raise RuntimeError("frozen prefix delta")
    return arms,d,context,refresh,before,direction,analyzer,layer,target,teacher_logits_bf16

def evaluate_target(model_path,campaign,key,scales):
    arms,d,context,refresh,before,direction,analyzer,layer,target,teacher_logits_bf16=setup(model_path,campaign,key)
    selected=refresh["selected"]
    bf16_points={}
    for t in scales:
        target.weight.copy_((before-t*direction).to(target.weight))
        bf16_points[str(t)]=objective(analyzer,layer,context,refresh,selected,teacher_logits_bf16)
    # Smooth FP32 reference: convert the entire target block, remaining
    # transformer suffix, final norm and LM head. Inputs/reference activations
    # are the same saved refresh tensors, promoted from BF16 to FP32.
    for downstream in analyzer.get_layers()[context["layer"]:]: downstream.float()
    analyzer.get_layernorm_before_head().float(); analyzer.get_lm_head().float()
    fp32_parameter=torch.nn.Parameter(before.cuda().float(),requires_grad=False)
    target.module.weight=fp32_parameter
    target.weight=fp32_parameter
    with torch.no_grad():
        teacher_logits_fp32={}
        for i in selected:
            ref=refresh["fp_target"][i:i+1].cuda().float(); kw=sample_kw(refresh,i)
            def cast_fp32(v):
                if isinstance(v,torch.Tensor): return v.float() if v.is_floating_point() else v
                if isinstance(v,(tuple,list)): return type(v)(cast_fp32(z) for z in v)
                if isinstance(v,dict): return {k:cast_fp32(z) for k,z in v.items()}
                return v
            kw=cast_fp32(kw)
            for downstream in analyzer.get_layers()[context["layer"]+1:]: ref=hidden(downstream(ref,**kw))
            teacher_logits_fp32[i]=analyzer.get_lm_head()(analyzer.get_layernorm_before_head()(ref)).detach()
    def set_scale(t): target.weight.copy_((before-t*direction).cuda())
    points={}
    for t in scales:
        set_scale(t); points[str(t)]=objective(analyzer,layer,context,refresh,selected,teacher_logits_fp32,fp32_suffix=True)
    set_scale(0.0); target.weight.requires_grad_(True)
    grad_total=torch.zeros_like(before,dtype=torch.float64)
    with torch.enable_grad():
        for i in selected:
            x=refresh["inps"][i:i+1].cuda()
            def select(v):
                if isinstance(v,torch.Tensor): return v[i:i+1] if v.ndim and v.shape[0]==refresh["inps"].shape[0] else v
                if isinstance(v,(tuple,list)): return type(v)(select(z) for z in v)
                if isinstance(v,dict): return {k:select(z) for k,z in v.items()}
                return v
            kw=sample_kw(refresh,i)
            def cast_fp32(v):
                if isinstance(v,torch.Tensor): return v.float() if v.is_floating_point() else v
                if isinstance(v,(tuple,list)): return type(v)(cast_fp32(z) for z in v)
                if isinstance(v,dict): return {k:cast_fp32(z) for k,z in v.items()}
                return v
            kw=cast_fp32(kw); x=x.float(); ref=teacher_logits_fp32[i]
            out=hidden(layer(x,**kw))
            for downstream in analyzer.get_layers()[context["layer"]+1:]: out=hidden(downstream(out,**kw))
            out=analyzer.get_lm_head()(analyzer.get_layernorm_before_head()(out)).float()
            loss=tokenwise_kl_from_logits(out,ref).mean()/len(selected)
            if not loss.requires_grad:
                raise RuntimeError(f"FP32 autograd graph missing: grad_enabled={torch.is_grad_enabled()} weight_requires_grad={target.weight.requires_grad} output_grad_fn={out.grad_fn}")
            grad,=torch.autograd.grad(loss,target.weight)
            grad_total.add_(grad.detach().cpu().double())
            del x,kw,ref,out,loss,grad
    adam_directional_derivative=-float((grad_total*direction.double()).sum())
    result={"target":key,"precision":"entire target transformer block + downstream suffix + norm + LM head in FP32",
        "samples":selected,"f_t_fp32":points,"f_t_bf16":bf16_points,"adam_autograd_directional_derivative_dL_dt":adam_directional_derivative,
        "central_difference_h":{},"finite_difference_minus_autograd":{},
        "stored_refresh_proxy_loss":refresh["actual_loss"]}
    for h in (0.25,0.5,1.0):
        fd=(points[str(h)]-points[str(-h)])/(2*h)
        result["central_difference_h"][str(h)]=fd
        result["finite_difference_minus_autograd"][str(h)]=fd-adam_directional_derivative
    result["candidate_actual_step_kl_change_fp32path"]=points["1.0"]-points["0.0"]
    result["candidate_actual_step_kl_change_bf16path"]=bf16_points["1.0"]-bf16_points["0.0"]
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("campaign",type=Path);p.add_argument("--model",required=True);p.add_argument("--target",action="append",required=True);a=p.parse_args()
    configure_reproducibility(0,deterministic=True);torch.set_num_threads(4)
    scales=[-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0]
    result_path=a.campaign/"finite_difference_kl.json"
    all_results=json.loads(result_path.read_text()) if result_path.exists() else {}
    for key in a.target:
        print(f"[finite-difference] START {key}",flush=True)
        all_results[key]=evaluate_target(a.model,a.campaign,key,scales)
        result_path.write_text(json.dumps(all_results,indent=2))
        print(f"[finite-difference] DONE {key}: "+json.dumps(all_results[key]),flush=True)
