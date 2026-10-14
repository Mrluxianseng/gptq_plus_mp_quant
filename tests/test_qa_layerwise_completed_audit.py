from copy import deepcopy
from tools.audit_qa_layerwise_completed import coverage, SIDES


def fixture():
    rows = {}
    for side in SIDES:
        enabled = side in ('layerwise', 'layerwise_cache')
        cache = [dict(device='cpu' if side == 'layerwise_cache' and i else 'cuda:0',
            shape=[16, 8, 128, 128], dtype='torch.bfloat16', stride=[131072, 16384, 128, 1],
            value_device='cpu' if side == 'layerwise_cache' and i else 'cuda:0',
            value_shape=[16, 8, 128, 128], value_dtype='torch.bfloat16',
            value_stride=[131072, 16384, 128, 1]) for i in range(28)]
        rows[side] = dict(logits=[{}], cache_outputs=[cache],
            weight_records=[dict(forward=1, layer=i, parameter_bytes=1024,
                enabled=enabled, final_device='cpu' if enabled else 'cuda:0') for i in range(28)],
            forward_info=[dict(use_cache=True, explicit_cache=side == 'layerwise_cache',
                offloading=side == 'layerwise_cache')])
    return rows


def test_complete_offloaded_cache_layout_and_method_records():
    assert all(coverage(fixture()).values())


def test_duplicate_layer_and_changed_value_layout_are_rejected():
    rows = fixture()
    rows['layerwise']['weight_records'][27] = deepcopy(rows['layerwise']['weight_records'][0])
    assert not coverage(rows)['layerwise_weight_coverage']
    rows = fixture()
    rows['layerwise_cache']['cache_outputs'][0][8]['value_stride'][1] = 1
    assert not coverage(rows)['layerwise_cache_kv_layout']


def test_missing_cache_and_disabled_original_semantics_are_rejected():
    rows = fixture()
    rows['layerwise_cache']['cache_outputs'] = []
    rows['control']['forward_info'][0]['use_cache'] = False
    checks = coverage(rows)
    assert not checks['layerwise_cache_cache_calls']
    assert not checks['control_forward_calls']
