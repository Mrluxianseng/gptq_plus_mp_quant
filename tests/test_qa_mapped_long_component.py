import ast
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tools.run_qa_mapped_long_component import source,expected_phases
from tools.run_qa_long_stream_fingerprint import source as original
from tools.audit_qa_mapped_long_completed import head_checks


def test_real_raw_source_keeps_computation_and_single_wrappers():
    a,b=ast.parse(original()),ast.parse(source())
    def selected(tree,name):
        matches=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name]
        assert len(matches)==1
        return ast.dump(matches[0],include_attributes=False)
    for name in ('state','capture','measured_call','observe_forward','observe_output'):
        assert selected(a,name)==selected(b,name)
    text=source()
    assert text.count('MappedHeadStorage(model.lm_head')==1
    assert 'wrap_model' not in text and 'hadamard_backend' not in text
    compile(text,'actual_raw_long','exec')


def test_missing_frozen_anchor_refused():
    with patch('tools.run_qa_mapped_long_component.previous_source',return_value=original().replace('    torch.manual_seed(1)','    torch.manual_seed(2)')):
        try: source()
        except RuntimeError: pass
        else: raise AssertionError('Missing source anchor was accepted')


def test_every_new_phase_preserves_per_forward_layer_identity():
    phases=expected_phases(3)
    assert len(phases)==356
    assert all(f'forward{f}_layer{i}_{s}' in phases for f in (1,2,3) for i in range(28)
        for s in ('load_start','compute_start','offload_start','end'))
    assert 'head3_storage_end' in phases and 'mapped_head_pool_release' in phases


def test_actual_head_payload_and_missing_private_accounting_refused():
    import hashlib,inspect
    cfg=SimpleNamespace(hidden_size=8,vocab_size=32)
    payload=16*12*32*2
    record=dict(call=1,enabled=True,completed=True,input_shape=[16,12,8],output_shape=[16,12,32],
        output_stride=[384,32,1],dtype='torch.bfloat16',output_payload_bytes=payload,accounting_complete=True,
        private_active=payload,private_reserved=payload,native_allocated=payload,snapshot_native_allocated=payload,
        native_reserved=payload,snapshot_native_reserved=payload,mapped_host_live=payload,private_segments_in_native=True)
    row=dict(logits=[dict(shape=[16,12,32],stride=[384,32,1])],mapped_head=dict(enabled=True,calls=1,records=[record],
        original_forward_sha256=hashlib.sha256(inspect.getsource(torch.nn.Linear.forward).encode()).hexdigest(),
        released_active=0,final_host_live=0,allocations=1,releases=1,peak_callback_host_bytes=payload))
    assert all(head_checks(row,cfg,True).values())
    for field,value in [('output_payload_bytes',payload-2),('private_segments_in_native',False),('mapped_host_live',0),('snapshot_native_reserved',0)]:
        bad=deepcopy(row);bad['mapped_head']['records'][0][field]=value
        assert not all(head_checks(bad,cfg,True).values())
    bad=deepcopy(row);bad['mapped_head']['final_host_live']=payload
    assert not all(head_checks(bad,cfg,True).values())
