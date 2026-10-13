"""Native server CPU regression, without installing pytest or touching CUDA."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace,MethodType

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(output):
    import torch
    from transformers import AutoModelForCausalLM,Qwen3Config
    from lm_eval.models.huggingface import HFLM
    from tools.qa_score_lifetime_prototype import build_method as rows
    from tools.qa_score_no_kv_prototype import build_method as no_kv
    from tools.audit_qa_score_lifetime_component import memory_mib
    cases=[]
    torch.set_num_threads(2)
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        for vocab in (79,4096):
            partial=[]
            fake=SimpleNamespace(AUTO_MODEL_CLASS=AutoModelForCausalLM,logits_cache=True,
                batch_size=2,max_length=32,rank=0,device='cpu',
                cache_hook=SimpleNamespace(add_partial=lambda *args:partial.append(args)))
            fake._select_cont_toks=MethodType(HFLM._select_cont_toks,fake)
            def model_call(inp,**kwargs):
                return torch.sin(inp.float().unsqueeze(-1)*0.2+torch.arange(vocab).float()*0.03).to(dtype)
            fake._model_call=model_call
            requests=[(('ctx','yes'),[1,2,3],[4]),(('ctx','no'),[1,2,3],[5]),
                (('multitok','many'),[8,9,10,11],[1,2,3]),(('truncate','answer'),list(range(1,61)),[6]),
                (None,[1,3,5,7],[2,4]),(('last','answer'),[2],[4])]
            c=HFLM._loglikelihood_tokens(fake,requests,disable_tqdm=True)
            old=list(partial);partial.clear()
            n=rows(HFLM._loglikelihood_tokens)(fake,requests,disable_tqdm=True)
            assert c==n and old==partial
            cases.append('real HFLM scoring '+str(dtype)+' vocab='+str(vocab))
    torch.manual_seed(1)
    config=Qwen3Config(vocab_size=79,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,use_cache=True)
    model=AutoModelForCausalLM.from_config(config).eval()
    fake=SimpleNamespace(model=model,AUTO_MODEL_CLASS=AutoModelForCausalLM)
    for dtype in (torch.float32,torch.bfloat16):
        model.to(dtype)
        inp=torch.tensor([[1,2,3,4],[5,6,7,8]])
        c=HFLM._model_call(fake,inp)
        n=no_kv(HFLM._model_call)(fake,inp)
        assert torch.equal(c,n) and model.config.use_cache is True
        cases.append('actual tiny Qwen3 KV exact '+str(dtype))
    try: no_kv(lambda:None)
    except (OSError,RuntimeError,IndentationError): pass
    else: raise AssertionError('Wrong model_call source admitted')
    assert memory_mib(' 16692 MiB ')==16692 and memory_mib('10802')==10802
    for value in ('N/A','19.1 GB','-2'):
        try: memory_mib(value)
        except ValueError: pass
        else: raise AssertionError('Invalid memory admitted')
    assert not torch.cuda.is_initialized()
    result=dict(valid=True,cases=cases,cuda_initialized=False,
        actual_likelihood_source_sha256=hashlib.sha256(inspect.getsource(HFLM._loglikelihood_tokens).encode()).hexdigest(),
        actual_model_call_source_sha256=hashlib.sha256(inspect.getsource(HFLM._model_call).encode()).hexdigest())
    with Path(output).open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main(sys.argv[1])
