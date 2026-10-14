import pytest
import torch
from tools.qa_mapped_head_natural_batch import NaturalBatchMappedHeadStorage,build_batch_method
from test_qa_mapped_head_storage import Library,Pool,CUDA


@pytest.mark.parametrize('batch',[1,7,16])
def test_natural_tail_preserves_original_full_linear_and_all_requests(batch):
    head=torch.nn.Linear(64,256,bias=False).bfloat16().eval()
    value=torch.randn(batch,2,64,dtype=torch.bfloat16)
    library=Library();cuda=CUDA()
    ticket=NaturalBatchMappedHeadStorage(head,configured_batch_size=16,enabled=True,
        expected_shape=(256,64),library=library,cuda=cuda,require_cuda=False,
        pool_factory=lambda:Pool(library,batch*2*256*2))
    with torch.no_grad(): expected=torch.nn.functional.linear(value,head.weight);output=head(value)
    assert torch.equal(expected,output) and output.shape[0]==batch
    ticket.pool.active=0;del output;ticket.release_batch();report=ticket.close()
    assert report['configured_batch_size']==16 and report['actual_batch_sizes']==[batch]
    assert report['final_host_live']==0 and not torch.cuda.is_initialized()


def test_no_user_batch_reduction_or_invalid_input_allowed():
    head=torch.nn.Linear(64,256,bias=False).bfloat16().eval();library=Library()
    with pytest.raises(RuntimeError,match='remain 16'):
        NaturalBatchMappedHeadStorage(head,configured_batch_size=8,enabled=False,
            expected_shape=(256,64),library=library,pool_factory=lambda:None,require_cuda=False)
    ticket=NaturalBatchMappedHeadStorage(head,enabled=False,expected_shape=(256,64),library=library,
        pool_factory=lambda:None,require_cuda=False)
    for batch in (0,17):
        with pytest.raises(RuntimeError,match='natural'): head(torch.randn(batch,2,64,dtype=torch.bfloat16))
    assert ticket.calls==0;ticket.close()


def test_actual_hflm_batch16_seventeen_requests_keeps_tail_full_bytes(tmp_path):
    from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen3Config
    from lm_eval.models.huggingface import HFLM
    from tools.qa_score_lifetime_prototype import build_method
    from tools.qa_complete_observer import CompleteQAObserver,validate_observer_pair
    torch.manual_seed(1)
    cfg=Qwen3Config(vocab_size=79,hidden_size=128,intermediate_size=256,num_hidden_layers=28,
        num_attention_heads=16,num_key_value_heads=8,head_dim=8)
    model=AutoModelForCausalLM.from_config(cfg).bfloat16().eval()
    tokenizer=AutoTokenizer.from_pretrained('/mnt/d/llamaModels/Qwen3-0.6B',local_files_only=True)
    hflm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=16,device='cpu')
    requests=[(('natural-tail',str(i)),[1,i+2,3],[4]) for i in range(17)]
    original=HFLM._loglikelihood_tokens;results=[]
    try:
        HFLM._loglikelihood_tokens=build_method(original)
        with CompleteQAObserver(model,HFLM,tmp_path/'control') as observer:
            with observer.task('cpu_natural_tail'):
                results.append(hflm._loglikelihood_tokens(requests,disable_tqdm=True))
        ticket=NaturalBatchMappedHeadStorage(model.lm_head,enabled=False,expected_shape=(79,128),
            library=Library(),pool_factory=lambda:None,require_cuda=False)
        HFLM._loglikelihood_tokens=build_batch_method(original,ticket.release_batch)
        with CompleteQAObserver(model,HFLM,tmp_path/'candidate') as observer:
            with observer.task('cpu_natural_tail'):
                results.append(hflm._loglikelihood_tokens(requests,disable_tqdm=True))
        report=ticket.close()
        assert report['actual_batch_sizes']==[16,1] and report['configured_batch_size']==16
        assert len(results[0])==len(results[1])==17 and results[0]==results[1]
        assert validate_observer_pair(tmp_path/'control',tmp_path/'candidate',['cpu_natural_tail'])['valid']
    finally: HFLM._loglikelihood_tokens=original
    assert not torch.cuda.is_initialized()
