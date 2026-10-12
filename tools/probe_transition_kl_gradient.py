"""Diagnostic-only true downstream KL gradient at saved REAL-Q states.

Never called by quantization. No optimizer changes or extra production backward.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from realq.attention import configure_attention_backend
from realq.refresh.adam_transition_probe import tensor_hash
from realq.refresh.fisher_loss import fisher_mse_loss
from utils import data_utils, model_utils, quant_utils
from utils.loss_utils import tokenwise_kl_from_logits
from utils.reproducibility import configure_reproducibility


def to_device(value,device):
    if isinstance(value,torch.Tensor): return value.to(device)
    if isinstance(value,dict): return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return type(value)(to_device(v,device) for v in value)
    return value


def probe(model_path, target, arms):
    started=time.monotonic()
    directory=Path(arms["adam"]["directory"])/"transition"
    evaluation=arms["adam"]["evaluation"]
    context=torch.load(directory/"context.pt",map_location="cpu",weights_only=False)
    refresh=torch.load(directory/"refresh.pt",map_location="cpu",weights_only=False)
    before=torch.load(directory/"before_update.pt",map_location="cpu",weights_only=True)
    analyzer=model_utils.ModelAnalyzer(model_path,128,skip_state_dict=True)
    configure_attention_backend(analyzer.model,context["config"]["attention_backend"])
    quant_utils.add_actquant(analyzer)
    analyzer.model.eval().requires_grad_(False).cuda()
    torch.cuda.reset_peak_memory_stats()
    texts=data_utils._get_dataset(analyzer.tokenizer,"wikitext2","validation")
    v=evaluation["validation"]
    encoded=analyzer.tokenizer("\n\n".join(texts),return_tensors="pt").input_ids[0]
    tokens=encoded[:v["windows"]*v["seq_len"]].reshape(v["windows"],v["seq_len"])
    if tensor_hash(tokens)!=v["tokens_sha256"]:
        raise RuntimeError("validation token mismatch")
    with torch.no_grad():
        teacher=[analyzer.model.model(input_ids=row[None].cuda(),use_cache=False).last_hidden_state[0].cpu() for row in tokens]
        for i,state in enumerate(context["prefix_layers"]):
            analyzer.get_layers()[i].load_state_dict(state,strict=True)
        analyzer.get_layers()[context["layer"]+1].load_state_dict(context["next_layer"],strict=True)
        module=analyzer.get_layers()[context["layer"]].get_submodule(context["module"])
        module.weight.copy_(before.to(module.weight))

    # Paired calibration audit: the proxy and KL use the same stored refresh
    # inputs and sample indices. The teacher suffix starts at REAL-Q's cached
    # FP layer output; the student suffix starts at the current quantized
    # layer output, matching the project's paired teacher/student construction.
    layer=analyzer.get_layers()[context["layer"]]
    def sample_kw(index):
        def take(value):
            if isinstance(value,torch.Tensor):
                return value[index:index+1] if value.ndim and value.shape[0]==refresh["inps"].shape[0] else value
            if isinstance(value,(tuple,list)): return type(value)(take(x) for x in value)
            if isinstance(value,dict): return {k:take(v) for k,v in value.items()}
            return value
        return take(refresh["kw"])
    def hidden_of(output): return output[0] if isinstance(output,tuple) else output
    def suffix_logits(hidden,kw):
        for downstream in analyzer.get_layers()[context["layer"]+1:]:
            hidden=hidden_of(downstream(hidden,**kw))
        hidden=analyzer.get_layernorm_before_head()(hidden)
        return analyzer.get_lm_head()(hidden).float()
    calibration_gradient=torch.zeros_like(before,dtype=torch.float64)
    calibration_kl=[]
    module.weight.requires_grad_(True)
    for index in refresh["selected"]:
        x=refresh["inps"][index:index+1].cuda()
        fp_target=refresh["fp_target"][index:index+1].cuda()
        kw=to_device(sample_kw(index),"cuda")
        with torch.no_grad():
            ref_logits=suffix_logits(fp_target,kw).detach()
        with torch.enable_grad():
            q_out=hidden_of(layer(x,**kw))
            student_logits=suffix_logits(q_out,kw)
            per_token=tokenwise_kl_from_logits(student_logits,ref_logits)
            loss=per_token.mean()
            calibration_kl.append(loss.detach().double().item())
            if not loss.requires_grad:
                raise RuntimeError(f"calibration KL graph missing; target weight requires_grad={module.weight.requires_grad}, output grad_fn={q_out.grad_fn}")
            grad,=torch.autograd.grad(loss,module.weight)
        calibration_gradient.add_(grad.detach().cpu().double(),alpha=1/len(refresh["selected"]))
        del x,fp_target,kw,ref_logits,q_out,student_logits,per_token,loss,grad
    module.weight.requires_grad_(False)

    # Surrogate gradient on exactly the same selected refresh samples.
    proxy_gradient=torch.zeros_like(before,dtype=torch.float64)
    module.weight.requires_grad_(True)
    with torch.enable_grad():
        for index in refresh["selected"]:
            x=refresh["inps"][index:index+1].cuda()
            kw=to_device(sample_kw(index),"cuda")
            output=hidden_of(layer(x,**kw))
            loss=fisher_mse_loss(output,refresh["fp_target"][index:index+1].cuda(),
                refresh["fisher"].cuda(),a_loss_ratio=refresh["a_loss_ratio"])
            alpha=refresh["slide_alpha"]
            if alpha is not None and alpha<1:
                next_out=hidden_of(analyzer.get_layers()[context["layer"]+1](output,**kw))
                next_loss=fisher_mse_loss(next_out,refresh["next_fp_target"][index:index+1].cuda(),
                    refresh["next_fisher"].cuda(),a_loss_ratio=refresh["a_loss_ratio"])
                loss=alpha*loss+(1-alpha)*next_loss
            grad,=torch.autograd.grad(loss,module.weight)
            proxy_gradient.add_(grad.detach().cpu().double(),alpha=1/len(refresh["selected"]))
            del x,kw,output,loss,grad
    perm,start=context["perm"],context["column_end"]
    def active(x): return x[:,perm][:,start:] if perm is not None else x[:,start:]
    pg,kg=active(proxy_gradient),active(calibration_gradient)
    calibration_alignment={
        "same_refresh_selected_samples":list(refresh["selected"]),
        "mean_calibration_downstream_kl":sum(calibration_kl)/len(calibration_kl),
        "calibration_downstream_kl_per_sample":calibration_kl,
        "surrogate_true_kl_active_suffix_cosine":float((pg*kg).sum()/(pg.norm()*kg.norm()).clamp_min(1e-30)),
        "surrogate_gradient_norm":float(pg.norm()),
        "true_kl_gradient_norm":float(kg.norm()),
        "true_kl_gradient_dot_nominal_adam_update":float((kg*refresh["adam_update"].double()).sum()),
    }
    module.weight.requires_grad_(True)
    gradient=torch.zeros_like(before,dtype=torch.float64)
    per_window=[]
    for i,row in enumerate(tokens):
        hidden=analyzer.model.model(input_ids=row[None].cuda(),use_cache=False).last_hidden_state[0]
        logits=analyzer.get_lm_head()(hidden).float()
        with torch.no_grad():
            ref_logits=analyzer.get_lm_head()(teacher[i].cuda()).float()
        token_loss=tokenwise_kl_from_logits(logits,ref_logits)
        per_window.append(token_loss.detach().double().mean().item())
        grad,=torch.autograd.grad(token_loss.mean(),module.weight)
        gradient.add_(grad.detach().cpu().double(),alpha=1/len(tokens))
        del hidden,logits,ref_logits,token_loss,grad
    reference=evaluation["stages"]["before_update"]
    replay_difference=max(abs(a-b) for a,b in zip(per_window,reference["per_window_kl"]))
    if replay_difference>1e-7:
        raise RuntimeError(f"KL-gradient forward replay mismatch: {replay_difference}")
    perm,start=context["perm"],context["column_end"]
    def suffix(x):
        return x[:,perm][:,start:] if perm is not None else x[:,start:]
    kg=suffix(gradient)
    sg=refresh["raw_gradient"].double()
    d=refresh["adam_update"].double()
    conflict=d*refresh["clipped_gradient"].double()<0
    result={"target":target,"kl":sum(per_window)/len(per_window),
            "same_calibration_input_alignment":calibration_alignment,
            "gradient_dtype":str(module.weight.dtype),"cpu_accumulation_dtype":"float64",
            "max_kl_forward_replay_difference":replay_difference,
            "cosine_surrogate_true_kl_active_suffix":float((kg*sg).sum()/(kg.norm()*sg.norm()).clamp_min(1e-30)),
            "true_kl_g_dot_nominal_adam_d":float((kg*d).sum()),
            "true_kl_g_dot_nominal_adam_d_on_surrogate_conflicts":float((kg[conflict]*d[conflict]).sum()),
            "arms":{}}
    for policy,arm in arms.items():
        after=torch.load(Path(arm["directory"])/"transition/after_update.pt",map_location="cpu",weights_only=True)
        dfp32=before.double()-after.double()
        dbf16=before.bfloat16().double()-after.bfloat16().double()
        actual=arm["evaluation"]["stages"]["after_update"]["kl"]-reference["kl"]
        result["arms"][policy]={"predicted_kl_change_fp32_delta":-float((gradient*dfp32).sum()),
            "predicted_kl_change_effective_bf16_delta":-float((gradient*dbf16).sum()),
            "actual_kl_change":actual}
    result["peak_allocated"]=torch.cuda.max_memory_allocated()
    result["seconds"]=time.monotonic()-started
    torch.save(gradient.float(),directory/"diagnostic_true_kl_gradient.pt")
    del analyzer,module,context,refresh,teacher
    import gc
    gc.collect();torch.cuda.empty_cache()
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    parser.add_argument("--model",required=True)
    parser.add_argument("--target",action="append",help="Restrict to exact preregistered target keys")
    args=parser.parse_args()
    torch.set_num_threads(4)
    configure_reproducibility(0,deterministic=True)
    summary=json.loads((args.campaign/"summary.json").read_text())
    if summary["failure"] or not summary["sources_unchanged"]: raise RuntimeError("campaign invalid")
    results={}
    for target,arms in summary["results"].items():
        if args.target and target not in args.target: continue
        print(f"[kl-gradient] START {target}",flush=True)
        results[target]=probe(args.model,target,arms)
        (args.campaign/"true_kl_gradient_analysis.json").write_text(json.dumps(results,indent=2))
        print(f"[kl-gradient] DONE {target}: "+json.dumps(results[target]),flush=True)
