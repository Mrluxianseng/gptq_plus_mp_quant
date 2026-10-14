"""Fresh read-only audit of complete component coverage and K/V layout."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_qa_layerwise_prefetch import prefetch_coverage
from tools.run_qa_layerwise_component import audit, SIDES
from tools.audit_qa_memory_rows_completed import sha


def coverage(rows):
    control = rows['control']
    calls = len(control['logits'])
    expected_ids = {(forward, layer) for forward in range(1, calls + 1) for layer in range(28)}
    control_payload = {(r['forward'], r['layer']): r['parameter_bytes'] for r in control['weight_records']}
    checks = {}
    for side, row in rows.items():
        records = row['weight_records']
        ids = [(r['forward'], r['layer']) for r in records]
        enabled = side in ('layerwise', 'layerwise_cache')
        checks[side + '_weight_coverage'] = (calls > 0 and len(ids) == len(expected_ids)
            and set(ids) == expected_ids and all(r['parameter_bytes'] > 0
                and r['parameter_bytes'] == control_payload.get((r['forward'], r['layer']))
                and r['enabled'] == enabled
                and r['final_device'] == ('cpu' if enabled else 'cuda:0') for r in records))
        checks[side + '_cache_calls'] = (len(row['cache_outputs']) == calls
            and all(len(out) == 28 for out in row['cache_outputs']))
        checks[side + '_forward_calls'] = (len(row['forward_info']) == calls and all(
            x['use_cache'] and x['explicit_cache'] == (side == 'layerwise_cache')
            and x['offloading'] == (side == 'layerwise_cache') for x in row['forward_info']))
        observed_values = all('value_device' in x for out in row['cache_outputs'] for x in out)
        if observed_values:
            checks[side + '_kv_layout'] = checks[side + '_cache_calls'] and all(
                all(x[field] == y[field] for field in (
                    'shape', 'dtype', 'stride', 'value_shape', 'value_dtype', 'value_stride'))
                and x['shape'] == x['value_shape'] and x['dtype'] == x['value_dtype']
                and x['value_device'] == x['device']
                and x['device'] == ('cpu' if side == 'layerwise_cache' and layer != 0 else 'cuda:0')
                for out, reference in zip(row['cache_outputs'], control['cache_outputs'])
                for layer, (x, y) in enumerate(zip(out, reference)))
        else:
            checks[side + '_key_layout'] = checks[side + '_cache_calls'] and all(
                all(x[field] == y[field] for field in ('shape', 'dtype', 'stride'))
                and x['device'] == ('cpu' if side == 'layerwise_cache' and layer != 0 else 'cuda:0')
                for out, reference in zip(row['cache_outputs'], control['cache_outputs'])
                for layer, (x, y) in enumerate(zip(out, reference)))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bindings = json.loads((args.root / 'source_bindings.json').read_text())
    base = audit(args.root, bindings)
    prefetch = prefetch_coverage(args.root)
    rows = {side: json.loads((args.root / (side + '.json')).read_text()) for side in SIDES}
    checks = coverage(rows)
    checks['prefetch_source_and_device'] = prefetch['valid']
    for side, report in base['reports'].items():
        report['checks']['actual_cache_device_coverage'] = (
            prefetch['valid'] if side == 'layerwise_cache' else report['checks']['actual_cache_device_coverage'])
        checks[side + '_numerics_source_telemetry_global'] = all(report['checks'].values())
    device_ids = []
    for side in SIDES:
        with (args.root / (side + '_nvml.csv')).open() as stream:
            samples = list(csv.reader(stream))
        identities = {(x[2].strip(), x[3].strip()) for x in samples if len(x) == 4}
        checks[side + '_nvml_complete_identity'] = (bool(samples) and all(len(x) == 4 for x in samples)
            and len(identities) == 1 and next(iter(identities))[0] == '0'
            and next(iter(identities))[1].startswith('GPU-'))
        device_ids.append(identities)
    checks['same_physical_device'] = all(x == device_ids[0] for x in device_ids)
    result = dict(valid=all(checks.values()), checks=checks, reports=base['reports'],
        prefetch_coverage=prefetch, nvml_identity=[sorted(x) for x in device_ids],
        artifact_sha256={p.name: sha(p) for p in args.root.iterdir() if p.is_file()},
        auditor_sha256=sha(__file__), scope=base['scope'], nvml_is_sampled=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(valid=result['valid'], checks=checks,
        values_observed=prefetch['values_observed']), indent=2))
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
