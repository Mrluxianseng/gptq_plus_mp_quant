"""Read-only completed wrapper-placement audit; not formal model-quality proof."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_rotation_placement_component_v2 import source
from tools.audit_qa_layerwise_completed import coverage
from tools.audit_qa_layerwise_prefetch import prefetch_coverage
from tools.audit_qa_memory_rows_completed import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    namespace=dict(__name__='qa_rotation_independent',__file__=str(ROOT/'tools/run_qa_rotation_placement_component_v2.py'))
    exec(compile(source(),'qa_rotation_independent','exec'),namespace)
    bindings=json.loads((args.root/'source_bindings.json').read_text())
    result=namespace['audit'](args.root,bindings)
    rows={side:json.loads((args.root/(side+'.json')).read_text()) for side in ('control','layerwise_cache')}
    extra=coverage(rows)
    prefetch=prefetch_coverage(args.root)
    extra['actual_offloaded_kv_and_source']=prefetch['valid'] and prefetch['values_observed']
    buffers={name:value for name,value in rows['control']['before']['layout'].items() if name.endswith('.had_K')}
    extra['complete_wrapper_states_and_buffers']=(rows['control']['before']['count']==535 and len(buffers)==28
        and all(value==[[12,12],'torch.float32'] for value in buffers.values()))
    result['independent_checks']=extra
    result['prefetch_source']=prefetch
    result['valid']=result['valid'] and all(extra.values())
    result['artifact_sha256']={f.name:sha(f) for f in args.root.iterdir() if f.is_file()}
    result['auditor_sha256']=sha(__file__)
    result['scope']='activation-runtime placement fixture with unrotated base weights; NOT formal REAL-Q quality or full QA'
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],extra_checks=extra,
        checks=result['reports']['layerwise_cache']['checks'],nvml_peak_mib=result['reports']['layerwise_cache']['nvml_peak_mib']),indent=2))
    if not result['valid']: raise SystemExit(1)


if __name__=='__main__': main()
