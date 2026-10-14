import ast
import functools
import inspect
import json
import textwrap
import types
import pytest
import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from tools.qa_control_head_observer import ControlHeadObserver
from tools.qa_mapped_observed_placement_bridge_v2 import bridge_source, head_ticket, MappedObservedPlacementBridge
from tools import qa_mapped_observed_placement_bridge as ancestor
from tools.qa_complete_observer import validate_observer_pair
from test_qa_observed_placement_bridge import fixture


def make_head():
    return torch.nn.Linear(12, 37, bias=False, dtype=torch.bfloat16).eval()


@pytest.mark.parametrize('sequential', [False, True])
def test_real_accelerate_forward_full_bytes_natural_batches_and_hook_preservation(sequential):
    head = make_head()
    add_hook_to_module(head, AlignDevicesHook(execution_device='cpu', io_same_device=True))
    if sequential:
        add_hook_to_module(head, AlignDevicesHook(execution_device='cpu'), append=True)
    forward, hook, weight = head.forward, head._hf_hook, head.weight
    existing_pre = head.register_forward_pre_hook(lambda module, inputs: None)
    existing_post = head.register_forward_hook(lambda module, inputs, output: None)
    before_handles = (set(head._forward_pre_hooks), set(head._forward_hooks))
    before_weight = weight.detach().clone()
    phases = []
    observer = ControlHeadObserver(head, expected_shape=(37,12), require_cuda=False, mark=phases.append)
    for batch in (16, 1, 7):
        x = torch.arange(batch*3*12).reshape(batch,3,12).to(torch.bfloat16)/100
        with torch.no_grad():
            reference = torch.nn.functional.linear(x, weight)
            actual = head(x)
        assert torch.equal(reference.view(torch.uint8), actual.view(torch.uint8))
        observer.release_batch()
    result = observer.close()
    assert result['actual_batch_sizes'] == [16,1,7]
    assert result['forward_chain']['kind'] == 'accelerate_align_devices'
    assert len(phases) == 12
    assert head.forward is forward and head._hf_hook is hook and head.weight is weight
    assert (set(head._forward_pre_hooks),set(head._forward_hooks)) == before_handles
    assert torch.equal(weight.view(torch.uint8), before_weight.view(torch.uint8))
    existing_pre.remove(); existing_post.remove()


def test_original_forward_error_remains_primary_and_own_handles_removed():
    head = make_head()
    add_hook_to_module(head, AlignDevicesHook(execution_device='cpu'))
    observer = ControlHeadObserver(head, expected_shape=(37,12), require_cuda=False)
    forward, hook = head.forward, head._hf_hook
    # A genuine downstream module hook fails before this observer: no model math replacement.
    failure = head.register_forward_pre_hook(lambda m,i: (_ for _ in ()).throw(ValueError('original hook failure')),
                                             prepend=True)
    with pytest.raises(ValueError, match='original hook failure'):
        head(torch.zeros(16,3,12,dtype=torch.bfloat16))
    failure.remove()
    observer.close()
    assert head.forward is forward and head._hf_hook is hook
    assert not head._forward_pre_hooks and not head._forward_hooks


def test_reject_unreviewed_offload_tampered_chain_and_mapped_candidate():
    head = make_head()
    original = head.forward
    head.forward = lambda x: original(x)
    with pytest.raises(RuntimeError, match='Unreviewed'):
        ControlHeadObserver(head, expected_shape=(37,12), require_cuda=False)
    head = make_head()
    add_hook_to_module(head, AlignDevicesHook(execution_device='cpu'))
    head._hf_hook.offload = True
    with pytest.raises(RuntimeError, match='nonoffloading'):
        ControlHeadObserver(head, expected_shape=(37,12), require_cuda=False)
    head._hf_hook.offload = False
    observer = ControlHeadObserver(head, expected_shape=(37,12), require_cuda=False)
    head._old_forward = types.MethodType(lambda self,x:x, head)
    with pytest.raises(RuntimeError, match='Unreviewed'):
        head(torch.zeros(16,3,12,dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match='Unreviewed'):
        observer.close()
    assert not head._forward_pre_hooks and not head._forward_hooks
    head = make_head(); add_hook_to_module(head, AlignDevicesHook(execution_device='cpu'))
    with pytest.raises(RuntimeError, match='original Linear forward'):
        head_ticket(head, enabled=True, configured_batch_size=16, expected_shape=(37,12),
                    library=None, cuda=torch.cuda, require_cuda=False, pool_factory=lambda:object(), mark=lambda p:None)


def test_actual_bridge_source_only_changes_ticket_factory():
    original = ast.parse(textwrap.dedent(inspect.getsource(ancestor.MappedObservedPlacementBridge)))
    new = ast.parse(bridge_source())
    for node in ast.walk(new):
        if isinstance(node, ast.Name) and node.id == '_qa_admitted_head_ticket':
            node.id = 'NaturalBatchMappedHeadStorage'
    assert ast.dump(original,include_attributes=False) == ast.dump(new,include_attributes=False)


def test_real_hflm_all_tasks_full_bytes_with_hooked_control_and_original_footer(tmp_path):
    from lm_eval.models.huggingface import HFLM
    original = (HFLM._model_call,HFLM._loglikelihood_tokens)
    for enabled, side in ((False,'control'),(True,'candidate')):
        model,tok,ev,dist,evaluation,memory,cuda,calls = fixture()
        original_dist = dist.distribute_model
        captured = {}
        if not enabled:
            def dispatch(model):
                result = original_dist(model)
                add_hook_to_module(model.lm_head, AlignDevicesHook(execution_device='cpu',io_same_device=True))
                captured.update(forward=model.lm_head.forward,hook=model.lm_head._hf_hook,weight=model.lm_head.weight)
                return result
            dist.distribute_model = dispatch
        bridge = MappedObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=enabled,
            mapped_enabled=False,library=None,destination=tmp_path/side,
            expected_tasks=('fixture_a','fixture_b'),validation_only=True,device='cpu',exclusive_peak_owner=True)
        dist.distribute_model(model)
        assert ev.qa_eval(model,tok,16) == dict(fixture_a=0.5,fixture_b=0.5)
        record = json.loads((tmp_path/side/'qa_mapped_head_rank0.json').read_text())
        assert record['completed'] and record['formal_enabled'] is False
        assert record['head']['actual_batch_sizes'] == [16,16]
        assert memory.get_process_cuda_memory_peaks('cuda:3') == (5000,6000)
        if not enabled:
            assert model.lm_head.forward is captured['forward']
            assert model.lm_head._hf_hook is captured['hook'] and model.lm_head.weight is captured['weight']
        bridge.restore()
        assert (HFLM._model_call,HFLM._loglikelihood_tokens) == original
    assert validate_observer_pair(tmp_path/'control/qa_complete',tmp_path/'candidate/qa_complete',
                                  ['fixture_a','fixture_b'])['valid']
    assert not torch.cuda.is_initialized()
