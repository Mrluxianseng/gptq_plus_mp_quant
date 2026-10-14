import copy
import pytest
from tools.qa_storage_payload_budget import payload,observed


def fixture():
    cfg=dict(vocab_size=8,num_hidden_layers=28,num_key_value_heads=2,head_dim=2,hidden_size=4)
    cache=[dict(shape=[16,2,3,2],value_shape=[16,2,3,2],dtype='torch.bfloat16',value_dtype='torch.bfloat16') for _ in range(28)]
    row=dict(batch_size=16,logits=[dict(shape=[16,3,8],dtype='torch.bfloat16')],cache_outputs=[cache],
        mapped_head=dict(records=[dict(completed=True,input_shape=[16,3,4],output_shape=[16,3,8],dtype='torch.bfloat16',output_payload_bytes=768)]))
    return cfg,row


def test_full_head_and_both_kv_tensors_arithmetic_natural_batch():
    cfg,row=fixture();result=observed(cfg,row)[0]
    assert result['full_logits_payload_bytes']==768
    assert result['one_layer_kv_payload_bytes']==768 and result['all_layers_logical_kv_payload_bytes']==21504
    assert payload(cfg,1,3)['full_logits_payload_bytes']==48
    assert result['object_bytes_are_not_peaks']


def test_changed_payload_precision_missing_kv_or_batch_refused():
    cfg,row=fixture()
    for change in ('payload','precision','kv','batch'):
        bad=copy.deepcopy(row)
        if change=='payload':bad['mapped_head']['records'][0]['output_payload_bytes']=767
        if change=='precision':bad['logits'][0]['dtype']='torch.float32'
        if change=='kv':bad['cache_outputs'][0].pop()
        if change=='batch':bad['batch_size']=8
        with pytest.raises(ValueError):observed(cfg,bad)
    with pytest.raises(ValueError):payload(cfg,17,3)
