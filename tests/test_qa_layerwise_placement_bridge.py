from types import SimpleNamespace
import json
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config
from lm_eval.models.huggingface import HFLM
from tools.qa_layerwise_placement_bridge import PlacementBridge, validate_model


def model():
    torch.manual_seed(1)
    config = Qwen3Config(vocab_size=79, hidden_size=128, intermediate_size=256,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=8)
    result = AutoModelForCausalLM.from_config(config).bfloat16().eval()
    result.model.layers[27].register_buffer('placement_fixture_buffer', torch.arange(12).float())
    return result


def modules(fail=False):
    distributed = []
    logits = []
    def distribute(m):
        distributed.append(id(m))
        return 'original-placement-result'
    def qa_eval(m, tokenizer, batch=16):
        fake = SimpleNamespace(model=m, AUTO_MODEL_CLASS=AutoModelForCausalLM)
        with torch.no_grad():
            value = HFLM._model_call(fake, torch.arange(batch*8).reshape(batch,8)%79)
        logits.append(value.detach().clone())
        if fail:
            raise ValueError('original evaluator failed')
        return dict(cpu_fixture_forward_sum=float(value.float().sum()))
    return SimpleNamespace(qa_eval=qa_eval, logits=logits), SimpleNamespace(distribute_model=distribute), distributed


def test_actual_full_forward_preserved_and_original_dispatch_delegated(tmp_path):
    m=model()
    score, call = HFLM._loglikelihood_tokens, HFLM._model_call
    outputs=[]
    for enabled in (False, True):
        ev, dist, calls = modules()
        original_ev, original_dist = ev.qa_eval, dist.distribute_model
        bridge=PlacementBridge(ev,dist,enabled=enabled,destination=tmp_path/str(enabled),device='cpu')
        assert dist.distribute_model(m) == (None if enabled else 'original-placement-result')
        assert calls == ([] if enabled else [id(m)])
        ev.qa_eval(m,None,16)
        outputs.append(ev.logits[-1])
        assert HFLM._model_call is call and HFLM._loglikelihood_tokens is score
        record=json.loads(bridge.path.read_text())
        assert record['completed'] and record['after_layout']==record['placement']['before_layout']
        assert record['cache_offload_cuda_effective'] is False
        if enabled:
            assert len(record['weight_records'])==28 and record['parameter_identity_preserved']
        bridge.restore()
        assert ev.qa_eval is original_ev and dist.distribute_model is original_dist
    assert torch.equal(*outputs)
    assert torch.equal(m.model.layers[27].placement_fixture_buffer,torch.arange(12).float())
    assert not torch.cuda.is_initialized()


def test_failure_restores_scoring_and_hooks_and_saves_failure(tmp_path):
    m=model();ev,dist,_=modules(fail=True)
    old=(HFLM._model_call,HFLM._loglikelihood_tokens)
    bridge=PlacementBridge(ev,dist,enabled=True,destination=tmp_path,device='cpu')
    dist.distribute_model(m)
    with pytest.raises(ValueError,match='original evaluator failed'): ev.qa_eval(m,None)
    assert (HFLM._model_call,HFLM._loglikelihood_tokens)==old
    assert not any(layer._forward_hooks or layer._forward_pre_hooks for layer in m.model.layers)
    record=json.loads(bridge.path.read_text())
    assert record['completed'] is False and len(record['weight_records'])==28
    bridge.restore()


def test_meta_dispatch_duplicate_and_wrong_model_rejected(tmp_path):
    m=model();m.model.layers[0]._hf_hook=object()
    with pytest.raises(RuntimeError,match='before accelerate'): validate_model(m)
    del m.model.layers[0]._hf_hook
    with torch.device('meta'):
        empty=AutoModelForCausalLM.from_config(m.config).bfloat16().eval()
    with pytest.raises(RuntimeError,match='meta state'): validate_model(empty)
    ev,dist,_=modules();bridge=PlacementBridge(ev,dist,enabled=True,destination=tmp_path,device='cpu')
    with pytest.raises(RuntimeError,match='preceding placement'): ev.qa_eval(m,None)
    with pytest.raises(RuntimeError,match='already installed'):
        PlacementBridge(ev,dist,enabled=True,destination=tmp_path,device='cpu')
    bridge.restore()


def test_cleanup_failure_is_saved_and_original_error_remains(monkeypatch,tmp_path):
    from tools import qa_layerwise_placement_bridge as bridge_module
    real=bridge_module.LayerwiseWeights
    class BrokenClose(real):
        def close(self):
            super().close()
            raise RuntimeError('fixture cleanup failed')
    monkeypatch.setattr(bridge_module,'LayerwiseWeights',BrokenClose)
    for fail in (False,True):
        m=model();ev,dist,_=modules(fail=fail)
        bridge=PlacementBridge(ev,dist,enabled=True,destination=tmp_path/str(fail),device='cpu')
        dist.distribute_model(m)
        expected=ValueError if fail else RuntimeError
        with pytest.raises(expected,match='original evaluator failed' if fail else 'fixture cleanup failed'):
            ev.qa_eval(m,None)
        record=json.loads(bridge.path.read_text())
        assert not record['completed'] and record['cleanup_error_type']=='RuntimeError'
        assert record['parameter_identity_preserved'] is False
        bridge.restore()


def test_existing_installer_and_nonprimitive_metrics_rejected(tmp_path):
    ev,dist,_=modules();ev._qa_scoring_lifetime_installed=True
    with pytest.raises(RuntimeError,match='Do not stack'):
        PlacementBridge(ev,dist,enabled=True,destination=tmp_path,device='cpu')
    ev._qa_scoring_lifetime_installed=False
    ev.qa_eval=lambda *args,**kw: dict(invalid=torch.ones(1))
    bridge=PlacementBridge(ev,dist,enabled=True,destination=tmp_path,device='cpu')
    m=model();dist.distribute_model(m)
    with pytest.raises(RuntimeError,match='JSON primitives'): ev.qa_eval(m,None)
    assert json.loads(bridge.path.read_text())['completed'] is False
    bridge.restore()


@pytest.mark.parametrize('hidden,intermediate,heads',[(1024,3072,16),(2048,6144,16)])
def test_actual_rotation_wrapper_meta_layout_preserved_and_meta_not_runnable(hidden,intermediate,heads):
    from utils import model_utils,rotation_utils
    from tools.qa_layerwise_placement_bridge import layout
    cfg=Qwen3Config(vocab_size=151936,hidden_size=hidden,intermediate_size=intermediate,
        num_hidden_layers=28,num_attention_heads=heads,num_key_value_heads=8,head_dim=128,
        architectures=['Qwen3ForCausalLM'])
    with torch.device('meta'):
        m=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.bfloat16).eval()
    analyzer=model_utils.ModelAnalyzer.__new__(model_utils.ModelAnalyzer)
    analyzer.model=m;analyzer.model_arch='Qwen3ForCausalLM'
    assert len(layout(m))==311
    rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
    state=layout(m)
    buffers=[(name,value) for name,value in state.items() if name.endswith('.had_K')]
    assert len(state)==535 and len(buffers)==28
    assert all(value==[[12,12],'torch.float32'] for name,value in buffers)
    with pytest.raises(RuntimeError,match='meta state'): validate_model(m)
    assert not torch.cuda.is_initialized()
