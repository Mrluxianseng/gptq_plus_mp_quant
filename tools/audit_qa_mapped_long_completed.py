"""Independent complete raw311 long mapped-output audit; no GPU execution."""
import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_mapped_long_component import NUMERICAL_KEYS,expected_phases
from tools.audit_qa_memory_rows_completed import sha
from tools.audit_memory_boundary_points import boundaries,compare
from tools.audit_completed_memory_pair_capacity import intervals
from tools.audit_qa_long_completed import geometry_checks
from tools.audit_qa_layerwise_prefetch import prefetch_coverage
from tools.storage_process_device_peaks import valid_record


def head_checks(row,config,enabled):
    head=row['mapped_head'];records=head['records'];logits=row['logits']
    import torch
    original=hashlib.sha256(inspect.getsource(torch.nn.Linear.forward).encode()).hexdigest()
    checks=dict(head_enabled=head['enabled'] is enabled,
        original_full_forward=head['original_forward_sha256']==original,
        calls=head['calls']==len(records)==len(logits)>0,
        geometry=len(records)==len(logits) and all(r['call']==i+1 and r['enabled'] is enabled and r['completed'] is True
            and r['input_shape']==[16,logit['shape'][1],config.hidden_size] and r['output_shape']==logit['shape']
            and r['output_stride']==logit['stride'] and r['dtype']=='torch.bfloat16'
            and r['output_payload_bytes']==2*16*logit['shape'][1]*config.vocab_size
            for i,(r,logit) in enumerate(zip(records,logits))))
    if enabled:
        checks['all_accounting']=bool(records) and all(r['accounting_complete'] is True
            and r['private_active']>=r['output_payload_bytes']
            and r['native_allocated']==r['snapshot_native_allocated']>=r['private_active']
            and r['native_reserved']==r['snapshot_native_reserved']>=r['private_reserved']>=r['private_active']
            and r['mapped_host_live']==r['private_reserved'] and r['private_segments_in_native'] is True for r in records)
        checks['release']=head['released_active']==head['final_host_live']==0 and head['allocations']==head['releases']>0
        checks['host_peak']=head['peak_callback_host_bytes']>=max(r['mapped_host_live'] for r in records)
    else: checks['no_mapped_host']=head['peak_callback_host_bytes']==0
    return checks


def audit(root):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    bindings=json.loads((root/'source_bindings.json').read_text())
    admission=json.loads((root/'admission.json').read_text());reference=Path(admission['reference'])
    old=json.loads((reference/'layerwise_cache.json').read_text())
    c,n=[json.loads((root/side/'layerwise_cache.json').read_text()) for side in ('control','candidate')]
    config=AutoConfig.from_pretrained(c['model'],local_files_only=True)
    with torch.device('meta'): model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16)
    layout={key:[list(value.shape),str(value.dtype)] for key,value in model.state_dict().items()}
    payload=[sum(p.numel()*p.element_size() for p in layer.parameters()) for layer in model.model.layers]
    checks={key:c[key]==n[key]==old[key] for key in NUMERICAL_KEYS}
    checks.update(source_sha=all(sha(path)==digest for path,digest in bindings.items()),
        fresh_reference_admission=json.loads((root/'reference_readmission.json').read_text())['valid'] is True
            and sha(root/'reference_readmission.json')==admission['reference_readmission_sha256']
            and sha(reference/'layerwise_cache.json')==admission['reference_record_sha256'],
        reference_sources=all(sha(p)==h for p,h in json.loads((reference/'source_bindings.json').read_text()).items()),
        full_original_state=c['before']==c['after']==n['before']==n['after'] and c['before']['layout']==layout and c['before']['count']==len(layout)==311,
        map_capabilities=c['mapped_attributes']==n['mapped_attributes'] and all(x==1 for x in n['mapped_attributes'].values()))
    tables=[];point_tables=[];hosts=[];nvml=[];device_ids=[]
    for side,row in zip(('control','candidate'),(c,n)):
        calls=len(row['logits']);phases=expected_phases(calls)
        checks[side+'_calls_scores']=calls==3 and len(row['scores'])==64 and all(math.isfinite(x[0]) and type(x[1]) is bool for x in row['scores'])
        checks[side+'_batch_bf16']=row['batch_size']==16 and all(x['shape'][0]==16 and x['shape'][2]==config.vocab_size and x['dtype']=='torch.bfloat16' and len(x['sha256'])==64 for x in row['logits'])
        checks[side+'_all_layers']=len(row['layers'])==28 and all(x['layer']==i and x['count']==calls and len(x['layout'])==calls and len(x['sha256'])==64 for i,x in enumerate(row['layers']))
        checks.update({side+'_'+k:v for k,v in geometry_checks(row,config).items()})
        ids=[(r['forward'],r['layer']) for r in row['weight_records']]
        checks[side+'_weights']=len(ids)==calls*28 and set(ids)=={(f,i) for f in range(1,calls+1) for i in range(28)} and all(r['enabled'] and r['final_device']=='cpu' and r['parameter_bytes']==payload[r['layer']] for r in row['weight_records'])
        checks[side+'_cache']=all(r['use_cache'] and r['explicit_cache'] and r['offloading'] for r in row['forward_info']) and len(row['forward_info'])==calls
        prefetch=prefetch_coverage(root/side)
        checks[side+'_official_kv']=prefetch['valid'] and prefetch['values_observed']
        checks.update({side+'_head_'+k:v for k,v in head_checks(row,config,side=='candidate').items()})
        checks[side+'_all_device_peaks']=valid_record(row['memory'],1) and row['memory']['peak_allocated_bytes']>0
        table=intervals([Path(p) for p in row['memory_profile_files']]);points=boundaries(row['memory_profile_files'])
        checks[side+'_exact_phases']={key[3] for key in table}==phases and len(table)==len(phases) and all(r['peak_measurement_mode']=='exact' for r in table.values())
        checks[side+'_points']=len(points)==len(phases)*4 and all(k[:3]==('stage2','evaluation',0) and k[4]==0 and type(v) is int and v>=0 for k,v in points.items())
        checks[side+'_peak_bounds']=all(r['sampled_peak_'+m+'_bytes']>=max(r[p+'_'+m+'_bytes'] for p in ('start','end')) for r in table.values() for m in ('allocated','reserved'))
        tables.append(table);point_tables.append(points)
        with (root/(side+'_nvml.csv')).open() as stream: samples=list(csv.reader(stream))
        checks[side+'_nvml_coverage']=bool(samples) and all(len(s)==4 and s[2].strip()=='0' for s in samples)
        nvml.append(max(float(s[1].strip().split()[0]) for s in samples));device_ids.append({s[3].strip() for s in samples})
        host=json.loads((root/(side+'_host_rss.json')).read_text());hosts.append(host)
        checks[side+'_host_observed']=host['sample_count']>0 and host['max_sampled_worker_rss_bytes']>0
    point=compare(*point_tables)
    checks['interval_identity']=bool(tables[0]) and tables[0].keys()==tables[1].keys()
    checks['point_identity']=bool(point['matched_values']) and not point['missing']
    checks['nvml_global']=nvml[1]<=nvml[0] and len(device_ids[0])==1 and device_ids[0]==device_ids[1]
    for metric in ('allocated','reserved'): checks['global_'+metric]=n['memory']['peak_'+metric+'_bytes']<=c['memory']['peak_'+metric+'_bytes']
    regressions=[dict(identity=list(k),metric=m,control=tables[0][k]['sampled_peak_'+m+'_bytes'],candidate=tables[1][k]['sampled_peak_'+m+'_bytes']) for k in tables[0].keys()&tables[1].keys() for m in ('allocated','reserved') if tables[1][k]['sampled_peak_'+m+'_bytes']>tables[0][k]['sampled_peak_'+m+'_bytes']]
    return dict(valid=all(checks.values()),checks=checks,memory=[c['memory'],n['memory']],nvml_peak_mib=nvml,
        host_observations=hosts,mapped_host_peak_bytes=n['mapped_head']['peak_callback_host_bytes'],
        head_records=n['mapped_head'],interval_count=len(tables[0]),points=point,interval_regressions=regressions,
        strict_local_nonincrease=not regressions and point['valid'],native_cuda_counters_include_mapped_host=True,
        actual_forward_shapes=[x['shape'] for x in n['logits']],reference=str(reference),nvml_is_sampled=True,
        artifact_sha256={str(p):sha(p) for p in (root/'control/layerwise_cache.json',root/'candidate/layerwise_cache.json')},
        auditor_sha256=sha(__file__),scope='raw311 synthetic-token long component, fresh paired same best method plus mapped whole-head output; not formal QA or server capacity')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.root)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=result['checks'],nvml=result['nvml_peak_mib'],host=result['host_observations']),indent=2))
    if not result['valid']: raise SystemExit(1)
