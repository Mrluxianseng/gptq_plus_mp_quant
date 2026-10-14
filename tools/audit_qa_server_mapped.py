"""Read-only complete QA storage admission for actual server artifacts."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.qa_server_mapped_validation import sha


def audit(control, candidate, expected_tasks, bindings, *, expected_devices=4):
    from tools.qa_complete_observer import validate_observer_pair, method_record
    from tools.audit_memory_boundary_points import boundaries, compare
    from tools.audit_completed_memory_pair_capacity import intervals
    from tools.qa_score_lifetime_prototype import build_method as rows
    from tools.qa_mapped_head_natural_batch import build_batch_method
    from tools.qa_mapped_observed_placement_bridge import control_batch_method
    from tools.qa_scoring_cache_offload_prototype import build_method as cache_method
    from lm_eval.models.huggingface import HFLM
    roots = [Path(control), Path(candidate)]
    def read(root, name):
        return json.loads((root/name).read_text())
    states = [read(r, 'qa_complete/qa_complete_summary.json') for r in roots]
    bridges = [read(r, 'qa_observed_placement_rank0.json') for r in roots]
    placements = [read(r, 'qa_layerwise_bridge_rank0.json') for r in roots]
    heads = [read(r, 'qa_mapped_head_rank0.json') for r in roots]
    caches = [read(r, 'qa_actual_cache_rank0.json') for r in roots]
    numerical = validate_observer_pair(*(r/'qa_complete' for r in roots), expected_tasks)
    checks = dict(numerical=numerical['valid'], source_identity=bool(bindings) and all(sha(p)==h for p,h in bindings.items()),
        complete_wrapped_state=all(s['before']['count']==535 and sum(n.endswith('.had_K') for n in s['before']['layout'])==28 for s in states),
        complete_bridges=all(r['completed'] for r in bridges),
        control_native=placements[0]['placement']['mode']=='original-distribute-model' and not placements[0]['enabled'],
        candidate_layerwise=placements[1]['placement']['mode']=='single-device-layerwise' and placements[1]['enabled'],
        complete_head=all(r['completed'] for r in heads),
        complete_cache=all(r['completed'] for r in caches),
        all_visible_devices=all(len(r['memory']['devices'])==expected_devices for r in bridges))
    call_count = states[0]['forward_count']
    records = [r['head']['records'] for r in heads]
    checks['actual_full_head_calls'] = all(len(r)==call_count for r in records)
    checks['head_geometry_equal'] = all(all(c[k]==n[k] for k in ('input_shape','output_shape','output_stride','dtype','output_payload_bytes')) for c,n in zip(*records))
    checks['mapped_accounting'] = all(r['accounting_complete'] and r['private_segments_in_native']
        and r['private_active']>=r['output_payload_bytes'] and r['mapped_host_live']==r['private_reserved']
        and r['native_allocated']==r['snapshot_native_allocated']
        and r['native_reserved']==r['snapshot_native_reserved'] for r in records[1])
    checks['no_control_mapping'] = not heads[0]['mapped_enabled'] and all(not r['enabled'] for r in records[0])
    head = heads[1]['head']
    checks['candidate_pool_release'] = (len(head['batch_releases'])==call_count
        and all(r['active_before_release']==0 and r['host_live_after_release']==0 for r in head['batch_releases'])
        and head['final_host_live']==0 and head['allocations']==head['releases'])
    cache_shapes = []
    for side, record in enumerate(caches):
        calls = record['calls']
        checks[f'cache_coverage_{side}'] = len(calls)==call_count and all(r['use_cache'] and len(r['layers'])==28 for r in calls)
        if side:
            checks['official_offload_selected'] = all(r['cache_type']=='DynamicCache' and r['cache_offloading'] for r in calls)
        cache_shapes.append([[{k:v for k,v in layer.items() if not k.endswith('_device')} for layer in r['layers']] for r in calls])
    checks['original_KV_geometry'] = cache_shapes[0]==cache_shapes[1]
    expected_scores = [control_batch_method(HFLM._loglikelihood_tokens, lambda:None),
                       build_batch_method(HFLM._loglikelihood_tokens, lambda:None)]
    expected_calls = [HFLM._model_call, cache_method(HFLM._model_call, offloading=True)]
    for side in range(2):
        for kind, methods in [('score',expected_scores),('call',expected_calls)]:
            checks[f'actual_selected_{kind}_{side}'] = states[side]['selected_'+kind]['selected_code_sha256']==method_record(methods[side])['selected_code_sha256']
    paths = [sorted((r/'qa_stage2/memory_profile').rglob('*.jsonl')) for r in roots]
    points = compare(*(boundaries(p) for p in paths))
    tables = [intervals(p) for p in paths]
    missing = sorted(tables[0].keys() ^ tables[1].keys())
    regressions = []
    for key in sorted(tables[0].keys() & tables[1].keys()):
        for metric in ('allocated','reserved'):
            values = [t[key]['sampled_peak_'+metric+'_bytes'] for t in tables]
            if values[1]>values[0]:
                regressions.append(dict(identity=list(key),metric=metric,control=values[0],candidate=values[1]))
    checks['complete_interval_identities'] = bool(tables[0]) and not missing and len(paths[0])==len(paths[1])==expected_devices
    checks['complete_independent_points'] = bool(points['matched_values']) and not points['missing']
    for metric in ('allocated','reserved'):
        values = [max(d['peak_'+metric+'_bytes'] for d in b['memory']['devices']) for b in bridges]
        checks['global_qa_'+metric] = values[1]<=values[0]
    return dict(valid=all(checks.values()),checks=checks,numerical=numerical,points=points,
        interval_count=len(tables[0]),interval_missing=[list(k) for k in missing],interval_regressions=regressions,
        strict_local_nonincrease=points['valid'] and not regressions,
        qa_device_memory=[r['memory']['devices'] for r in bridges],
        callback_host_peak_bytes=heads[1]['callback_host_peak_bytes'],
        native_cuda_counters_include_mapped_host=True,
        scope='Complete server QA mathematical and allocator audit; NVML and overall formal pipeline acceptance are separate')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--bindings',type=Path,required=True);p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.control,a.candidate,a.tasks,json.loads(a.bindings.read_text()))
    with a.output.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(dict(valid=r['valid'],checks=r['checks'],interval_count=r['interval_count'],point_count=r['points']['matched_values']),indent=2))
    if not r['valid']:raise SystemExit(1)
