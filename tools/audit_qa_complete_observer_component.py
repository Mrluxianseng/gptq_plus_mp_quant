"""Fresh completed-component audit; does not run GPU workers or alter old reports."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_complete_observer_component import source
from tools.qa_complete_observer import validate_observer_pair
from tools.audit_qa_layerwise_completed import coverage
from tools.audit_qa_layerwise_prefetch import prefetch_coverage
from tools.audit_qa_memory_rows_completed import sha
from tools.audit_memory_boundary_points import boundaries
from tools.audit_completed_memory_pair_capacity import intervals


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    namespace=dict(__name__='qa_complete_independent',__file__=str(ROOT/'tools/run_qa_complete_observer_component.py'))
    exec(compile(source(),'qa_complete_independent','exec'),namespace)
    bindings=json.loads((args.root/'source_bindings.json').read_text())
    result=namespace['audit'](args.root,bindings)
    rows={side:json.loads((args.root/(side+'.json')).read_text()) for side in ('control','layerwise_cache')}
    numerical=validate_observer_pair(args.root/'control_complete',args.root/'layerwise_cache_complete',['synthetic_component'])
    extra=coverage(rows);prefetch=prefetch_coverage(args.root)
    extra['kv_actual_geometry_and_prefetch']=prefetch['valid'] and prefetch['values_observed']
    extra['complete_new_numerical_observer']=numerical['valid']
    buffers={name:value for name,value in rows['control']['before']['layout'].items() if name.endswith('.had_K')}
    extra['wrapper_535_state_28_had']=(rows['control']['before']['count']==535 and len(buffers)==28
        and all(value==[[12,12],'torch.float32'] for value in buffers.values()))
    for side,row in rows.items():
        summary=json.loads((args.root/(side+'_complete/qa_complete_summary.json')).read_text())
        extra[side+'_new_state_matches_original']=summary['before']==row['before'] and summary['after']==row['after']
        paths=[Path(p) for p in row['memory_profile_files']]
        table=intervals(paths);point=boundaries(paths)
        extra[side+'_all_device_phase_identity']=bool(table) and len(point)==4*len(table) and all(
            r['phase'].startswith('qa_device'+str(r['device_index'])+'__') and r['peak_measurement_mode']=='exact'
            and r['rank']==0 and r['stage']=='stage2' and r['logkind']=='evaluation' for r in table.values())
    result.update(independent_checks=extra,prefetch_source=prefetch,numerical_observer=numerical,
        valid=result['valid'] and all(extra.values()),auditor_sha256=sha(__file__),
        scope='new observer integration on unrotated-base online-wrapper fixture; not formal QA/REAL-Q quality')
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=extra,
        nvml=result['reports']['layerwise_cache']['nvml_peak_mib']),indent=2))
    if not result['valid']: raise SystemExit(1)


if __name__=='__main__': main()
