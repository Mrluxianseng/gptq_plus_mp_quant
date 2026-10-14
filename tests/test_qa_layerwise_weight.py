from types import SimpleNamespace
import pytest
import torch
from transformers import AutoModelForCausalLM,Qwen3Config
from lm_eval.models.huggingface import HFLM
from tools.qa_layerwise_weight_prototype import LayerwiseWeights
from tools.qa_scoring_cache_offload_prototype import build_method


def model():
    torch.manual_seed(1)
    cfg=Qwen3Config(vocab_size=79,hidden_size=128,intermediate_size=256,num_hidden_layers=28,
        num_attention_heads=16,num_key_value_heads=8,head_dim=8,use_cache=True)
    return AutoModelForCausalLM.from_config(cfg).eval()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_actual_28_layers_identity_logits_and_cache_constructor(dtype):
    m=model().to(dtype);inp=torch.arange(16*8).reshape(16,8)%79
    captures=[]
    ticket=LayerwiseWeights(m,enabled=False,device='cpu',capture=lambda i,x:captures.append(x.detach().clone()))
    fake=SimpleNamespace(model=m,AUTO_MODEL_CLASS=AutoModelForCausalLM)
    c=HFLM._model_call(fake,inp);ticket.close();old=captures[:];captures.clear()
    ticket=LayerwiseWeights(m,enabled=True,device='cpu',capture=lambda i,x:captures.append(x.detach().clone()))
    n=build_method(HFLM._model_call,offloading=False)(fake,inp)
    assert torch.equal(c,n) and len(old)==len(captures)==28 and all(torch.equal(a,b) for a,b in zip(old,captures))
    assert len(ticket.records)==28 and all(ticket.calls) and m.config.use_cache is True
    ticket.close();assert not ticket.handles
    assert not torch.cuda.is_initialized()


def test_dispatch_and_training_are_rejected_before_moving_parameters():
    m=model();m.model.layers[0]._hf_hook=object()
    with pytest.raises(RuntimeError,match='dispatch'): LayerwiseWeights(m,enabled=True,device='cpu')
    del m.model.layers[0]._hf_hook;m.train()
    with pytest.raises(RuntimeError,match='eval'): LayerwiseWeights(m,enabled=True,device='cpu')
