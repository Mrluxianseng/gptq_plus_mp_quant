"""Independent composed head/535state/input/full-numerics/device/peak audit."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_mapped_observed_bridge_component import namespace
from tools.audit_qa_mapped_natural_tail_completed import head_checks
from tools.audit_qa_mapped_long_batch_completed import batch_checks
from tools.audit_qa_memory_rows_completed import sha
from tools.audit_completed_memory_pair_capacity import intervals
from tools.audit_memory_boundary_points import boundaries


def expected_phases(calls):
    phases={'qa_setup','qa_task1_start','qa_task1_end','qa_mapped_storage_cleanup','qa_bridge_cleanup_complete'}
    for call in range(1,calls+1):
        phases.update(f'qa_forward{call}_{s}' for s in ('model_start','logits_ready','scoring'))
        phases.update(f'qa_forward{call}_layer{i}_observed' for i in range(28))
        phases.update(f'forward{call}_layer{i}_{s}' for i in range(28) for s in ('load_start','compute_start','offload_start','end'))
        phases.update(f'head{call}_storage_{s}' for s in ('start','end','release_start','release_end'))
    return phases


def audit(root):
    result=namespace()['audit'](root)
    rows=[json.loads((root/(s+'.json')).read_text()) for s in ('control','candidate')]
    from transformers import AutoConfig
    config=AutoConfig.from_pretrained(rows[0]['model'],local_files_only=True)
    checks={}
    checks['same_early_capabilities']=rows[0]['mapped_capabilities']==rows[1]['mapped_capabilities'] and all(v==1 for v in rows[0]['mapped_capabilities'].values())
    for side,row in zip(('control','candidate'),rows):
        mapped=json.loads((root/side/'qa_mapped_head_rank0.json').read_text())
        summary=json.loads((root/side/'qa_complete/qa_complete_summary.json').read_text())
        events=[json.loads(line) for line in (root/side/'qa_complete/qa_complete_events.jsonl').read_text().splitlines()]
        logits=[r for r in events if r['kind']=='logits'];inputs=[r for r in events if r['kind']=='model_input']
        head=dict(mapped['head'],peak_callback_host_bytes=mapped['callback_host_peak_bytes'])
        adaptation=dict(mapped_head=head,logits=logits)
        checks.update({side+'_'+key:value for key,value in head_checks(adaptation,config,side=='candidate').items()})
        checks.update({side+'_'+key:value for key,value in batch_checks(adaptation,side=='candidate').items()})
        checks[side+'_completed']=mapped['completed'] is True and mapped['formal_enabled'] is False
        checks[side+'_full_requests_tail']=len(row['scores'])==65 and summary['forward_count']==4 and len(inputs)==len(logits)==4 and head['configured_batch_size']==16
        checks[side+'_actual_inputs']=head['actual_batch_sizes']==[r['shape'][0] for r in inputs]==[16,16,16,1] and inputs[-1]['shape']==[1,3]
        checks[side+'_input_logit_geometry']=all(r['shape']==[i['shape'][0],i['shape'][1],config.vocab_size]
            and r['dtype']=='torch.bfloat16' for r,i in zip(logits,inputs))
        buffers={k:v for k,v in summary['before']['layout'].items() if k.endswith('.had_K')}
        checks[side+'_all28_Had_buffers']=len(buffers)==28 and all(v==[[12,12],'torch.float32'] for v in buffers.values())
        bridge=json.loads((root/side/'qa_observed_placement_rank0.json').read_text())
        table=intervals([Path(p) for p in bridge['memory']['memory_profile_files']]);points=boundaries(bridge['memory']['memory_profile_files'])
        phases={'qa_device0__'+p for p in expected_phases(4)}
        checks[side+'_every_phase']={k[3] for k in table}==phases and len(table)==len(phases)==593
        checks[side+'_every_point']=len(points)==len(phases)*4 and all(k[:3]==('stage2','evaluation',0) and k[4]==0 for k in points)
        checks[side+'_bounds']=all(r['sampled_peak_'+m+'_bytes']>=max(r[p+'_'+m+'_bytes'] for p in ('start','end')) for r in table.values() for m in ('allocated','reserved'))
    result['independent_composition_checks']=checks
    result['valid']=result['valid'] and all(checks.values())
    result['auditor_sha256']=sha(__file__)
    result['native_cuda_counters_include_mapped_host']=True
    result['scope']='complete65request native-control versus mapped composed QA compatibility fixture, unrotatedbase535/torch_fallback/onevisibleGPU; NOT formal REAL-Q/fullQA/multidevice'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.root)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=result['checks'],composition=result['independent_composition_checks'],nvml=result['nvml_peak_mib']),indent=2))
    if not result['valid']: raise SystemExit(1)
