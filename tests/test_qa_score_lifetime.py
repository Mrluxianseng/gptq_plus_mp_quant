import inspect
from types import SimpleNamespace,MethodType
import pytest
import torch
from transformers import AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM

from tools.qa_score_lifetime_prototype import build_method,build_source


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('vocab',[79,4096])
def test_real_library_scoring_cache_padding_and_truncation_are_exact(dtype,vocab):
    partial=[]
    fake=SimpleNamespace(AUTO_MODEL_CLASS=AutoModelForCausalLM,logits_cache=True,
        batch_size=2,max_length=32,rank=0,device='cpu',
        cache_hook=SimpleNamespace(add_partial=lambda *args:partial.append(args)))
    fake._select_cont_toks=MethodType(HFLM._select_cont_toks,fake)
    def model_call(inp,**kwargs):
        return torch.sin(inp.float().unsqueeze(-1)*0.2+torch.arange(vocab).float()*0.03).to(dtype)
    fake._model_call=model_call
    requests=[(('context-a','yes'),[1,2,3],[4]),(('context-a','no'),[1,2,3],[5]),
              (('context-b','many'),[8,9,10,11],[1,2,3]),
              (('truncation','answer'),list(range(1,61)),[6]),
              (None,[1,3,5,7],[2,4]),(('context-c','answer'),[2],[4])]
    control=HFLM._loglikelihood_tokens(fake,requests,disable_tqdm=True)
    old_partial=list(partial);partial.clear()
    candidate=build_method(HFLM._loglikelihood_tokens)(fake,requests,disable_tqdm=True)
    assert candidate==control and partial==old_partial
    assert len(candidate)==len(requests)
    assert not torch.cuda.is_initialized()


def test_source_contract_change_and_double_install_rejected():
    original=inspect.getsource(HFLM._loglikelihood_tokens)
    changed=build_source(original)
    assert 'del multi_logits' in changed
    assert changed.index('logits = self._select_cont_toks')<changed.index('logits = F.log_softmax')
    with pytest.raises(RuntimeError):build_source(changed)
    with pytest.raises(RuntimeError):build_source(original.replace('dim=-1','dim=0'))
