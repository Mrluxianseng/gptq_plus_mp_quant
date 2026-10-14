"""Pair the accepted layerwise/KV method against bounded LM-head workspace."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_qa_layerwise_component_v2 import source as previous_source


HEAD_AUDIT = '''
def audit(root, bindings):
    result = base_audit(root, bindings)
    c = json.loads((root / 'control.json').read_text())
    n = json.loads((root / 'layerwise_cache.json').read_text())
    calls = len(c['logits'])
    checks = result['reports']['layerwise_cache']['checks']
    for name, row in (('control', c), ('candidate', n)):
        expected = name == 'candidate'
        records = row['head_records']
        checks[name + '_head_coverage'] = len(records) == calls and all(
            r['forward'] == i + 1 and r['enabled'] == expected and r['chunk_rows'] == 256
            and r['input_device'] == 'cuda:0' and r['input_dtype'] == 'torch.bfloat16'
            and r['output_dtype'] == 'torch.bfloat16'
            and r['output_device'] == ('cpu' if expected else 'cuda:0')
            and r['output_shape'] == row['logits'][i]['shape']
            and r['chunks'] == ((r['input_shape'][0] * r['input_shape'][1] + 255) // 256 if expected else 1)
            for i, r in enumerate(records))
        identities = [(r['forward'], r['layer']) for r in row['weight_records']]
        checks[name + '_all_layers_offloaded'] = (len(identities) == calls * 28
            and set(identities) == {(f, i) for f in range(1, calls + 1) for i in range(28)}
            and all(r['enabled'] and r['final_device'] == 'cpu' for r in row['weight_records']))
        checks[name + '_original_cache_offloaded'] = (len(row['forward_info']) == calls
            and all(r['use_cache'] and r['explicit_cache'] and r['offloading'] for r in row['forward_info']))
    checks['kv_layout_exact'] = (len(c['cache_outputs']) == len(n['cache_outputs']) == calls
        and all(len(a) == len(b) == 28 and a == b for a, b in zip(c['cache_outputs'], n['cache_outputs'])))
    device_ids = []
    for name in ('control', 'layerwise_cache'):
        with (root / (name + '_nvml.csv')).open() as stream:
            samples = list(csv.reader(stream))
        ids = {(x[2].strip(), x[3].strip()) for x in samples if len(x) == 4}
        checks[name + '_nvml_identity'] = bool(samples) and all(len(x) == 4 for x in samples) and len(ids) == 1 and next(iter(ids))[0] == '0' and next(iter(ids))[1].startswith('GPU-')
        device_ids.append(ids)
    checks['same_nvml_device'] = device_ids[0] == device_ids[1]
    result['reports']['layerwise_cache']['valid'] = all(checks.values())
    result['valid'] = all(checks.values())
    result['scope'] = 'bounded raw BF16 model LM-head screen; control already uses rows/layerwise/KV offload'
    return result

'''


def source():
    text = previous_source()
    replacements = {
        "SIDES=('control','rows','layerwise','layerwise_cache')": "SIDES=('control','layerwise_cache')",
        "enabled=args.arm in ('layerwise','layerwise_cache')": "enabled=True",
        "    original_score=HFLM._loglikelihood_tokens;original_call=HFLM._model_call":
        "    from tools.qa_lm_head_storage_prototype import CpuLogitsHead,gpu_scored_rows_method\n    head_ticket=CpuLogitsHead(model.lm_head,enabled=args.arm=='layerwise_cache',mark=profile.mark)\n    original_score=HFLM._loglikelihood_tokens;original_call=HFLM._model_call",
        "    if args.arm!='control': HFLM._loglikelihood_tokens=rows_method(original_score)":
        "    HFLM._loglikelihood_tokens=gpu_scored_rows_method(original_score) if args.arm=='layerwise_cache' else rows_method(original_score)",
        "    if args.arm=='layerwise_cache': HFLM._model_call=cache_method(original_call)":
        "    HFLM._model_call=cache_method(original_call)",
        "        ticket.close();handle.remove();": "        head_ticket.close();ticket.close();handle.remove();",
        "        weight_records=ticket.records,": "        head_records=head_ticket.records,weight_records=ticket.records,",
        "def audit(root,bindings):": "def base_audit(root,bindings):",
        "def main():": HEAD_AUDIT + "def main():",
        "source=[Path(__file__),Path(__file__).with_name('run_qa_layerwise_component.py'),":
        "source=[Path(__file__),Path(__file__).with_name('run_qa_layerwise_component.py'),Path(__file__).with_name('run_qa_layerwise_component_v2.py'),Path(__file__).with_name('qa_lm_head_storage_prototype.py'),",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen component contract changed: ' + old)
        text = text.replace(old, new)
    compile(text, 'qa_lm_head_component', 'exec')
    return text


if __name__ == '__main__':
    # Preserve independent rerun capability without invoking a GPU worker.
    import argparse
    import json
    import sys
    if '--audit-root' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--audit-root', type=Path, required=True)
        parser.add_argument('--audit-output', type=Path, required=True)
        args = parser.parse_args()
        namespace = dict(__name__='qa_lm_head_audit', __file__=__file__)
        exec(compile(source(), 'qa_lm_head_component', 'exec'), namespace)
        result = namespace['audit'](args.audit_root, json.loads((args.audit_root/'source_bindings.json').read_text()))
        with args.audit_output.open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(dict(valid=result['valid'], checks=result['reports']['layerwise_cache']['checks']), indent=2))
        if not result['valid']:
            raise SystemExit(1)
    else:
        exec(compile(source(), 'qa_lm_head_component', 'exec'), dict(__name__='__main__', __file__=__file__))
