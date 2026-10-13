"""Independent read-only audit of completed control/rows QA workers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_qa_score_lifetime_component import memory_mib
from tools.storage_process_device_peaks import valid_record


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''): digest.update(data)
    return digest.hexdigest()


def audit(root):
    c,n=[json.loads((root/(side+'.json')).read_text()) for side in ('control','rows')]
    binding=json.loads((root/'source_bindings.json').read_text())
    equal_keys=('mode','model','batch_size','model_use_cache','tasks','metrics','request_counts',
        'request_sha256','score_sha256','model_state_before','model_state_after',
        'likelihood_source_sha256','model_call_source_sha256')
    checks={key+'_exact':c[key]==n[key] for key in equal_keys}
    checks.update(arm_identity=c['arm']=='control' and n['arm']=='rows',
        all_weights_unchanged=c['model_state_before']==c['model_state_after']==n['model_state_before']==n['model_state_after'],
        sources_unchanged=bool(binding) and all(sha(path)==digest for path,digest in binding.items()),
        all_device_allocator_complete=all(valid_record(x['memory'],1) for x in (c,n)),
        batch_unchanged=c['batch_size']==16,
        nonempty_score_coverage=bool(c['request_counts']) and all(type(v) is int and v>0 for v in c['request_counts']),
        scoring_cache_preserved=bool(c['forward_use_cache']) and c['forward_use_cache']==n['forward_use_cache'] and all(c['forward_use_cache']))
    if c['mode']=='full':
        checks['ten_complete_tasks']=(len(c['tasks'])==len(set(c['tasks']))==10 and set(c['metrics'])==set(c['tasks']) and all(bool(x) for x in c['metrics'].values()))
    else:
        checks['short_complete_coverage']=c['metrics']['passages']==32 and c['metrics']['requests']==len(c['metrics']['scores'])==64
    peaks=[];counts=[];identities=[]
    for side in ('control','rows'):
        with (root/(side+'_nvml.csv')).open() as stream: rows=list(csv.reader(stream))
        if not rows or any(len(row)!=4 for row in rows): raise RuntimeError('Missing/incomplete NVML samples')
        peaks.append(max(memory_mib(row[1]) for row in rows));counts.append(len(rows))
        identities.append({(row[2].strip(),row[3].strip()) for row in rows})
    checks['same_physical_gpu']=identities[0]==identities[1] and len(identities[0])==1
    checks['sampled_device_peak_nonincreasing']=peaks[1]<=peaks[0]
    for key in ('peak_allocated_bytes','peak_reserved_bytes'):
        checks[key+'_nonincreasing']=n['memory'][key]<=c['memory'][key]
    return dict(valid=all(checks.values()),checks=checks,scope=c['scope'],
        tasks=c['tasks'],scored_requests=sum(c['request_counts']),model_forward_calls=len(c['forward_use_cache']),
        control=c['memory'],candidate=n['memory'],nvml_peak_mib=peaks,nvml_sample_count=counts,
        nvml_is_sampled=True,seconds=[c['seconds'],n['seconds']],
        artifacts={path.name:sha(path) for path in root.iterdir() if path.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
    if not result['valid']: raise SystemExit(1)
