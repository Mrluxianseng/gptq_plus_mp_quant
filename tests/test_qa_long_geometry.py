from types import SimpleNamespace
from tools.audit_qa_long_completed import geometry_checks


def fixture():
    config = SimpleNamespace(hidden_size=128, head_dim=16, num_key_value_heads=8)
    row = dict(logits=[dict(shape=[16, 1027, 151936])],
        layers=[dict(layout=[[[16, 1027, 128], 'torch.bfloat16', [1027*128, 128, 1]]]) for _ in range(28)],
        cache_outputs=[[dict(shape=[16, 8, 1027, 16], value_shape=[16, 8, 1027, 16],
            dtype='torch.bfloat16', value_dtype='torch.bfloat16', stride=[131456, 16432, 16, 1],
            value_stride=[131456, 16432, 16, 1]) for _ in range(28)]])
    return row, config


def test_nonempty_complete_geometry():
    row, config = fixture()
    assert all(geometry_checks(row, config).values())


def test_changed_layer_sequence_and_partial_cache_rejected():
    row, config = fixture()
    row['layers'][27]['layout'][0][0][1] = 1024
    assert not geometry_checks(row, config)['layer_output_geometry']
    row, config = fixture()
    row['cache_outputs'][0].pop()
    assert not geometry_checks(row, config)['actual_kv_geometry']


def test_value_stride_and_empty_forward_rejected():
    row, config = fixture()
    row['cache_outputs'][0][7]['value_stride'][1] = 1
    assert not geometry_checks(row, config)['actual_kv_geometry']
    row, config = fixture()
    row['logits'] = []
    assert not all(geometry_checks(row, config).values())
