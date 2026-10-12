"""CUDA equivalence and memory preflight using actual Qwen down_proj inputs.

Actual first-block weights and WikiText tokens; uniform saliency is a component
fixture, not the REAL-Q Fisher trajectory. Full model/cache/quality validation
is mandatory afterwards. No local component peak is called a full-process gain.
"""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
from gptq_utils import gptq_plus_utils as impl
from tools.hessian_inverse_lifetime_prototype import build_method, build_reusing_fasterquant, install_down_proj
from utils.data_utils import get_tokens
from utils.quant_utils import WeightQuantizer
from utils.reproducibility import configure_reproducibility


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def helper_pair(h_cpu, proposed, *, need=False, inplace=False, damping=.01, increment=.0015):
    helper = impl.GPTQPlus.__new__(impl.GPTQPlus)
    original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
    h = h_cpu.cuda()
    # Warm solver workspaces before either measurement; no warm result retained.
    with torch.no_grad():
        warm = original(helper, h.clone(), percdamp=damping, damp_auto_increment=increment,
                        need_hinv_init=need, inplace_damping=inplace)
        del warm
    records = []
    for label, method in (("control", original), ("candidate", proposed)):
        working = h.clone()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with torch.no_grad():
            inv, upper, damp, fallback = method(helper, working, percdamp=damping,
                damp_auto_increment=increment, need_hinv_init=need, inplace_damping=inplace)
        torch.cuda.synchronize()
        records.append(dict(side=label, inverse=None if inv is None else digest(inv),
                            upper=digest(upper), upper_stride=list(upper.stride()),
                            working_after=digest(working), damping=damp, fallback=fallback,
                            allocated_baseline_bytes=before,
                            allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                            reserved_peak_bytes=torch.cuda.max_memory_reserved()))
        del inv, upper, working
    fields = ("inverse", "upper", "upper_stride", "working_after", "damping", "fallback")
    exact = all(records[0][key] == records[1][key] for key in fields)
    del h
    torch.cuda.empty_cache()
    return dict(dim=h_cpu.shape[0], dtype=str(h_cpu.dtype), need_inverse=need,
                inplace_damping=inplace, exact=exact, arms=records)


def actual_first_block(model_path, samples, length):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = "sdpa"
    layer = Qwen3DecoderLayer(config, 0).to(dtype=torch.bfloat16)
    loaded, embedding = {}, None
    for filename in sorted(model_path.glob("*.safetensors")):
        with safe_open(filename, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith("model.layers.0."):
                    loaded[key[len("model.layers.0."):]] = source.get_tensor(key)
                elif key == "model.embed_tokens.weight":
                    embedding = source.get_tensor(key)
    if embedding is None:
        raise ValueError("missing real token embedding")
    layer.load_state_dict(loaded, strict=True)
    del loaded
    weight = layer.mlp.down_proj.weight.detach().clone()
    layer = layer.cuda().eval()
    layer.requires_grad_(False)
    rotary = Qwen3RotaryEmbedding(config).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokens = get_tokens("wikitext2", "train", tokenizer, length, samples, seed=44)
    captured = []
    handle = layer.mlp.down_proj.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu()))
    with torch.no_grad():
        for ids in tokens:
            x = torch.nn.functional.embedding(ids.unsqueeze(0), embedding).cuda()
            pos = torch.arange(length, device="cuda").unsqueeze(0)
            positions = rotary(x, pos)
            mask = torch.full((length,length), float("-inf"), device="cuda", dtype=torch.float32).triu(1)[None,None]
            value = layer(x, attention_mask=mask, position_ids=pos,
                          position_embeddings=positions, use_cache=False)
            del value, x, pos, positions, mask
    handle.remove()
    del layer, rotary, embedding, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return weight, torch.cat(captured, dim=0), tokens


def api_pair(weight, inputs, proposed, out, *, factor_reuse=False):
    original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
    original_fasterquant = impl.GPTQPlus.fasterquant
    reused, reuse_source = build_reusing_fasterquant(impl)
    if factor_reuse:
        (out/"patched_fasterquant.py").write_text(reuse_source)
    arms = []
    for enabled in (False, True):
        configure_reproducibility(44, deterministic=True)
        layer = torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,device="cuda",dtype=weight.dtype)
        layer.weight.data.copy_(weight)
        saliency = torch.ones(inputs.shape[0],inputs.shape[1],4)
        worker = impl.GPTQPlus(layer, saliency, torch.zeros(weight.shape),4,0.,0.)
        worker.layer_idx, worker.layer_name = 0,"mlp.down_proj"
        worker.quantizer = WeightQuantizer()
        worker.quantizer.configure(4,perchannel=True,sym=True)
        activation = inputs.cuda()
        worker.add_batch(activation,None)
        worker.finalize_hessian()
        del activation
        calls = []
        reuse_calls = []
        worker._hessian_factor_reuse_records = reuse_calls
        install_down_proj(impl,enabled=enabled,records=calls)
        impl.GPTQPlus.fasterquant = reused if enabled and factor_reuse else original_fasterquant
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        try:
            worker.fasterquant(blocksize=128,groupsize=-1,actorder=True,
                              enable_gradient_update=False,inner_kernel_impl="torch",
                              g_update_mode="block_gd",group_parallel_mode="none")
            torch.cuda.synchronize()
            arms.append(dict(side="candidate" if enabled else "control", weight=digest(layer.weight),
                quantizer={name:digest(value) for name,value in worker.quantizer.state_dict().items()},
                allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                reserved_peak_bytes=torch.cuda.max_memory_reserved(), calls=calls,reuse_calls=reuse_calls))
        finally:
            impl.GPTQPlus._compute_hessian_inverse_with_fallback=original
            impl.GPTQPlus.fasterquant=original_fasterquant
        if len(calls)!=4:
            raise RuntimeError("not all four down-projection inverse calls were observed")
        if enabled and factor_reuse and len(reuse_calls)!=4:
            raise RuntimeError("not all four consumed Hessian groups were reused")
        del layer, worker
        gc.collect()
        torch.cuda.empty_cache()
    return dict(exact=arms[0]["weight"]==arms[1]["weight"] and arms[0]["quantizer"]==arms[1]["quantizer"],
                arms=arms,scope="Actual down_proj API, real WikiText FFN activations, uniform saliency; no Adam or full model quality proof")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--model-paths",type=Path,nargs="+",required=True)
    p.add_argument("--seq-len",type=int,default=256)
    p.add_argument("--factor-reuse",action="store_true")
    args=p.parse_args()
    out=args.output_directory
    out.mkdir(exist_ok=False)
    configure_reproducibility(44,deterministic=True)
    torch.set_num_threads(1)
    proposed,source=build_method(impl)
    (out/"patched_inverse.py").write_text(source)
    cases,models=[],[]
    result=dict(valid=False,helper_cases=cases,models=models,torch=str(torch.__version__),
                helper_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                scope="CUDA component preflight, not a full-pipeline memory/quality claim")
    try:
        logging.disable(logging.WARNING)
        for dtype in (torch.float32,torch.float64):
            x=torch.randn(40,32,dtype=dtype)
            h=x.T@x
            for need in (False,True):
                for inplace in (False,True):
                    cases.append(helper_pair(h,proposed,need=need,inplace=inplace))
                    if not cases[-1]["exact"]:
                        raise RuntimeError("small helper mismatch")
        for kind in ("indefinite","nonfinite"):
            h=torch.eye(32)
            h[0,0]=-2 if kind=="indefinite" else float("nan")
            for need in (False,True):
                row=helper_pair(h,proposed,need=need,damping=.98,increment=.01)
                row["case"]=kind
                cases.append(row)
                if not row["exact"]:
                    raise RuntimeError("fallback mismatch")
        print("[inverse-lifetime] 12 CUDA helper/retained-inverse/retry/fallback cases exact",flush=True)
        for path in args.model_paths:
            print(f"[inverse-lifetime] real first-block calibration: {path}",flush=True)
            weight,inputs,tokens=actual_first_block(path,2,args.seq_len)
            x=inputs.cuda().reshape(-1,inputs.shape[-1]).float()
            h=(x.T@x)/x.shape[0]
            del x
            dead=torch.diagonal(h)==0
            h[dead,dead]=1
            h_cpu=h.cpu()
            del h,dead
            row=dict(model=str(path),input_shape=list(inputs.shape),input_sha256=digest(inputs),
                     tokens_sha256=[digest(value) for value in tokens])
            row["inverse"]=helper_pair(h_cpu,proposed)
            if not row["inverse"]["exact"]:
                raise RuntimeError("actual-dimension inverse mismatch")
            row["api"]=api_pair(weight,inputs,proposed,out,factor_reuse=args.factor_reuse)
            models.append(row)
            if not row["api"]["exact"]:
                raise RuntimeError("real down_proj quantization mismatch")
            print(json.dumps(dict(model=str(path),inverse=row["inverse"],api=row["api"])),flush=True)
            del weight,inputs,h_cpu,tokens
        result["valid"]=True
    finally:
        (out/"status.json").write_text(json.dumps(result,indent=2))
    print("[inverse-lifetime] component preflight passed",flush=True)


if __name__=="__main__":
    main()
