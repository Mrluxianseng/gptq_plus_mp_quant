"""Real decoder-layer value/forward/pool lifetime preflight, not full acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.replay_weight_compact_pool_prototype import prepare, finish


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    args.output_directory.mkdir(exist_ok=False)
    report=dict(valid=False,scope='Real FP16 decoder component copies and output exactness; no quantization/global peak/speed claim',models=[])
    try:
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
        if any(s.strip().isdigit() for s in gpu.stdout.splitlines()):
            raise RuntimeError('another CUDA compute process active')
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(44)
        report['torch']=str(torch.__version__)
        report['helper_sha256']=hashlib.sha256((ROOT/'tools/replay_weight_compact_pool_prototype.py').read_bytes()).hexdigest()
        for model_name,index in [('Qwen3-0.6B',25),('Qwen3-1.7B',27)]:
            model=AutoModelForCausalLM.from_pretrained('/mnt/d/llamaModels/'+model_name,
                torch_dtype=torch.float16,local_files_only=True,attn_implementation='eager')
            layer=model.model.layers[index].cuda().eval()
            ids={n:id(p) for n,p in layer.named_parameters()}
            originals={n:p.detach().cpu().clone() for n,p in layer.named_parameters()}
            h=torch.randn(1,32,model.config.hidden_size,device='cuda',dtype=torch.float16)
            pos=torch.arange(32,device='cuda')[None,:]
            rotary=model.model.rotary_emb.cuda()
            with torch.no_grad():
                emb=rotary(h,pos)
                mask=torch.full((1,1,32,32),torch.finfo(h.dtype).min,device='cuda',dtype=h.dtype).triu(1)
                kwargs=dict(attention_mask=mask,position_ids=pos,position_embeddings=emb,use_cache=False)
                output=layer(h,**kwargs)
                reference=(output[0] if isinstance(output,(tuple,list)) else output).cpu()
                del output
                records=[]
                ticket=prepare(torch,layer,index,enabled=True,records=records)
                values_equal=all(torch.equal(p.detach().cpu(),originals[n]) for n,p in layer.named_parameters())
                object_ids_equal=all(id(p)==ids[n] for n,p in layer.named_parameters())
                output=layer(h,**kwargs)
                result=(output[0] if isinstance(output,(tuple,list)) else output).cpu()
                del output
                forward_exact=torch.equal(reference,result)
                layer.cpu()
                finish(torch,ticket)
                del ticket
                copied_cpu_exact=all(torch.equal(p.detach(),originals[n]) for n,p in layer.named_parameters())
                released=records[0]['released'] and records[0]['active_pool_bytes_after_layer_release']==0
            report['models'].append(dict(model=model_name,layer=index,values_exact=values_equal,
                parameter_ids_preserved=object_ids_equal,forward_exact=forward_exact,
                cpu_release_values_exact=copied_cpu_exact,pool_released=released,records=records))
            if not all((values_equal,object_ids_equal,forward_exact,copied_cpu_exact,released)):
                raise RuntimeError('real decoder compact-pool component failed')
            del layer,model,h,pos,emb,mask,rotary,kwargs,originals,reference,result
            print(model_name+' component exact and pool released',flush=True)
        report['valid']=True
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        report['finished']=time.strftime('%F %T')
        with (args.output_directory/'status.json').open('x') as stream:json.dump(report,stream,indent=2)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
