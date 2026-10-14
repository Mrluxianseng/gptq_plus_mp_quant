"""Independent completed feasibility audit and broad peak attribution.

No absent long-input reference is inferred from the shorter paired validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_qa_layerwise_long_feasibility import source
from tools.audit_qa_layerwise_prefetch import prefetch_coverage
from tools.audit_qa_memory_rows_completed import sha
from tools.audit_completed_memory_pair_capacity import intervals
from tools.audit_memory_boundary_points import boundaries


def geometry_checks(row, config):
    calls = len(row['logits'])
    layers = row['layers']
    cache = row['cache_outputs']
    head_dim = config.head_dim
    return dict(
        layer_output_geometry=len(layers) == 28 and calls > 0 and all(
            len(layer['layout']) == calls and all(
                layout[0] == [16, logit['shape'][1], config.hidden_size]
                and layout[1] == 'torch.bfloat16'
                and layout[2] == [logit['shape'][1] * config.hidden_size, config.hidden_size, 1]
                for layout, logit in zip(layer['layout'], row['logits'])) for layer in layers),
        actual_kv_geometry=len(cache) == calls and calls > 0 and all(len(out) == 28 and all(
            item['shape'] == item['value_shape'] == [16, config.num_key_value_heads, logit['shape'][1], head_dim]
            and item['dtype'] == item['value_dtype'] == 'torch.bfloat16'
            and item['stride'] == item['value_stride']
            for item in out) for out, logit in zip(cache, row['logits'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    namespace = dict(__name__='qa_long_independent', __file__=str(ROOT/'tools/run_qa_layerwise_long_feasibility.py'))
    exec(compile(source(), 'qa_long_independent', 'exec'), namespace)
    bindings = json.loads((args.root/'source_bindings.json').read_text())
    result = namespace['feasibility'](args.root, bindings)
    row = json.loads((args.root/'layerwise_cache.json').read_text())
    from transformers import AutoConfig, AutoModelForCausalLM
    import torch
    config = AutoConfig.from_pretrained(row['model'], local_files_only=True)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    result['checks'].update(geometry_checks(row, config))
    payload = [sum(p.numel()*p.element_size() for p in layer.parameters()) for layer in model.model.layers]
    result['checks']['actual_layer_payload'] = all(x['parameter_bytes'] == payload[x['layer']] for x in row['weight_records'])
    cache = prefetch_coverage(args.root)
    result['checks']['official_prefetch_source_contract'] = cache['valid'] and cache['values_observed']
    table = intervals([Path(p) for p in row['memory_profile_files']])
    points = boundaries(row['memory_profile_files'])
    calls = len(row['logits'])
    phases = {'gptq_fwrd_setup', 'runtime_install', 'qa_scoring_start', 'qa_scoring_finished'}
    for call in range(1, calls+1):
        phases.update(f'forward{call}_{suffix}' for suffix in ('model_start', 'logits_ready', 'scoring'))
        phases.update(f'forward{call}_layer{i}_{suffix}' for i in range(28)
            for suffix in ('load_start', 'compute_start', 'offload_start', 'end'))
    result['checks']['exact_phase_identity_coverage'] = {key[3] for key in table} == phases and len(table) == len(phases)
    result['checks']['point_identity_complete'] = len(points) == len(phases)*4 and all(
        key[0:3] == ('stage2', 'evaluation', 0) and key[4] == 0
        and type(value) is int and value >= 0 for key, value in points.items())
    result['checks']['each_interval_bounds_consistent'] = all(
        entry['sampled_peak_'+metric+'_bytes'] >= max(entry[point+'_'+metric+'_bytes'] for point in ('start', 'end'))
        for entry in table.values() for metric in ('allocated', 'reserved'))
    maximum = max(x['sampled_peak_allocated_bytes'] for x in table.values())
    result['peak_attribution'] = dict(max_allocated_intervals=[dict(identity=list(k),
        phase=x['phase'], next_phase=x['next_phase'], peak_allocated_bytes=x['sampled_peak_allocated_bytes'],
        peak_reserved_bytes=x['sampled_peak_reserved_bytes']) for k, x in table.items()
        if x['sampled_peak_allocated_bytes'] == maximum],
        full_logits_bytes_by_forward=[x['shape'][0]*x['shape'][1]*x['shape'][2]*2 for x in row['logits']],
        note='A decoder-layer-end interval also contains norm, LM head and model-call completion; not a pure GEMM peak measurement.')
    result['prefetch_source'] = cache
    result['boundary_identity_sha256'] = hashlib.sha256(json.dumps(sorted(points.items())).encode()).hexdigest()
    result['artifact_sha256'] = {p.name:sha(p) for p in args.root.iterdir() if p.is_file()}
    result['auditor_sha256'] = sha(__file__)
    result['valid'] = all(result['checks'].values())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(valid=result['valid'], checks=result['checks'], peak_attribution=result['peak_attribution']), indent=2))
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
