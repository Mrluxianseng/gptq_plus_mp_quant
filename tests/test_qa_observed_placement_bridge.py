import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from transformers import AutoModelForCausalLM,Qwen3Config,AutoTokenizer
from lm_eval.models.huggingface import HFLM
from tools.qa_observed_placement_bridge import ObservedPlacementBridge,PeakCarry
from tools.qa_complete_observer import validate_observer_pair
from tools.audit_memory_boundary_points import boundaries,compare


class FakeCUDA:
    def __init__(self): self.rows=[[0,0,0,0] for _ in range(4)]
    def device_count(self): return 4
    def memory_allocated(self,i): return self.rows[i][0]
    def memory_reserved(self,i): return self.rows[i][1]
    def max_memory_allocated(self,i): return self.rows[i][2]
    def max_memory_reserved(self,i): return self.rows[i][3]
    def synchronize(self,i): pass
    def reset_peak_memory_stats(self,i): self.rows[i][2:]=self.rows[i][:2]


def fixture(fail=False):
    torch.manual_seed(1)
    cfg=Qwen3Config(vocab_size=79,hidden_size=128,intermediate_size=256,num_hidden_layers=28,
        num_attention_heads=16,num_key_value_heads=8,head_dim=8)
    model=AutoModelForCausalLM.from_config(cfg).bfloat16().eval()
    tokenizer=AutoTokenizer.from_pretrained('/mnt/d/llamaModels/Qwen3-0.6B',local_files_only=True)
    cuda=FakeCUDA()
    memory=SimpleNamespace(_process_profile_peaks={0:(900,1000),1:(1300,2000)})
    memory.get_process_cuda_memory_peaks=lambda device=None:memory._process_profile_peaks.get(
        0 if device is None else int(str(device).split(':')[1]),(0,0))
    def evaluate(hflm,*,tasks,**kwargs):
        cuda.rows[3]=[0,0,5000,6000] # A higher QA peak on another device; already freed.
        requests=[((tasks[0],str(i)),[1,i+2,3],[4]) for i in range(16)]
        hflm._loglikelihood_tokens(requests,disable_tqdm=True)
        if fail: raise ValueError('original evaluator failure')
        return dict(results={tasks[0]:dict(acc=0.5)})
    evaluation=SimpleNamespace(simple_evaluate=evaluate)
    def qa_eval(model,tokenizer,batch=16):
        hflm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=batch,device='cpu')
        result={}
        with torch.no_grad():
            for name in ('fixture_a','fixture_b'):
                result[name]=evaluation.simple_evaluate(hflm,tasks=[name])['results'][name]['acc']
        return result
    calls=[]
    def distribute(model): calls.append(id(model))
    return model,tokenizer,SimpleNamespace(qa_eval=qa_eval),SimpleNamespace(distribute_model=distribute),evaluation,memory,cuda,calls


def test_actual_hflm_composed_pair_preserves_bytes_and_peak_footer(tmp_path):
    source_methods=(HFLM._model_call,HFLM._loglikelihood_tokens)
    for enabled,side in ((False,'control'),(True,'candidate')):
        model,tok,ev,dist,evaluation,memory,cuda,calls=fixture()
        original_eval,original_dist,original_simple=ev.qa_eval,dist.distribute_model,evaluation.simple_evaluate
        footer_getter=memory.get_process_cuda_memory_peaks # Pre-existing import alias also must work.
        bridge=ObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=enabled,
            destination=tmp_path/side,expected_tasks=('fixture_a','fixture_b'),validation_only=True,
            device='cpu',exclusive_peak_owner=True)
        dist.distribute_model(model)
        assert calls==([] if enabled else [id(model)])
        assert ev.qa_eval(model,tok,16)==dict(fixture_a=0.5,fixture_b=0.5)
        record=json.loads((tmp_path/side/'qa_observed_placement_rank0.json').read_text())
        assert record['completed'] and record['formal_enabled'] is False
        assert record['memory']['visible_device_count']==4
        assert footer_getter('cuda:0')==(900,1000) and footer_getter('cuda:1')==(1300,2000)
        assert footer_getter('cuda:3')==(5000,6000)
        assert (HFLM._model_call,HFLM._loglikelihood_tokens)==source_methods
        assert evaluation.simple_evaluate is original_simple
        bridge.restore()
        assert ev.qa_eval is original_eval and dist.distribute_model is original_dist
        assert footer_getter('cuda:3')==(5000,6000)
    assert validate_observer_pair(tmp_path/'control/qa_complete',tmp_path/'candidate/qa_complete',
        ['fixture_a','fixture_b'])['valid']
    paths=[list((tmp_path/side/'qa_stage2/memory_profile').rglob('*.jsonl')) for side in ('control','candidate')]
    assert compare(*(boundaries(p) for p in paths))['valid']
    assert not torch.cuda.is_initialized()


def test_original_failure_restores_methods_and_keeps_prior_and_qa_peaks(tmp_path):
    model,tok,ev,dist,evaluation,memory,cuda,calls=fixture(fail=True)
    original=(HFLM._model_call,HFLM._loglikelihood_tokens)
    original_simple=evaluation.simple_evaluate
    bridge=ObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=True,
        destination=tmp_path,expected_tasks=('fixture_a','fixture_b'),validation_only=True,
        device='cpu',exclusive_peak_owner=True)
    dist.distribute_model(model)
    with pytest.raises(ValueError,match='original evaluator failure'): ev.qa_eval(model,tok,16)
    record=json.loads((tmp_path/'qa_observed_placement_rank0.json').read_text())
    assert record['completed'] is False and record['memory']['status']=='failed'
    assert memory.get_process_cuda_memory_peaks('cuda:3')==(5000,6000)
    assert memory.get_process_cuda_memory_peaks('cuda:1')==(1300,2000)
    assert (HFLM._model_call,HFLM._loglikelihood_tokens)==original
    assert evaluation.simple_evaluate is original_simple
    assert not any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.model.layers)
    bridge.restore()


def test_prepared_interface_cannot_enable_formal_or_modify_unsupported_owner(tmp_path):
    model,tok,ev,dist,evaluation,memory,cuda,calls=fixture()
    original=ev.qa_eval
    with pytest.raises(RuntimeError,match='validation-only'):
        ObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=True,destination=tmp_path,
            expected_tasks=('fixture_a',),validation_only=False,device='cpu',exclusive_peak_owner=True)
    memory._process_profile_peaks=None
    with pytest.raises(RuntimeError,match='Unsupported'):
        ObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=True,destination=tmp_path,
            expected_tasks=('fixture_a',),validation_only=True,device='cpu',exclusive_peak_owner=True)
    assert ev.qa_eval is original


def test_accumulator_protocol_failure_not_silently_accepted():
    owner=SimpleNamespace(_process_profile_peaks={0:('bad',0)})
    with pytest.raises(RuntimeError,match='Invalid original'):
        PeakCarry(owner,SimpleNamespace(peaks=[[1,2]]))
