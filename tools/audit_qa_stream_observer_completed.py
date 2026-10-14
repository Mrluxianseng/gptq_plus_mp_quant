"""Compare complete stream-observer replay with the preserved same-workload run."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_long_stream_fingerprint import source
from tools.audit_qa_memory_rows_completed import sha
from tools.audit_memory_boundary_points import boundaries,compare
from tools.audit_completed_memory_pair_capacity import intervals
from tools.audit_qa_score_lifetime_component import memory_mib


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    readmission=args.root/'reference_readmission.json'
    with (args.root/'reference_readmission.log').open('x') as stream:
        checked=subprocess.run([sys.executable,'tools/audit_qa_long_completed.py','--root',str(args.reference),'--output',str(readmission)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT)
    if checked.returncode: raise RuntimeError('Reference independently rejected; no comparison success')
    namespace=dict(__name__='qa_stream_observer_audit',__file__=str(ROOT/'tools/run_qa_long_stream_fingerprint.py'))
    exec(compile(source(),'qa_stream_observer_audit','exec'),namespace)
    base=namespace['feasibility'](args.root,json.loads((args.root/'source_bindings.json').read_text()))
    c=json.loads((args.reference/'layerwise_cache.json').read_text());n=json.loads((args.root/'layerwise_cache.json').read_text())
    keys=('model','batch_size','max_context_tokens','max_continuation_tokens','before','after',
        'scores','request_sha256','logits','layers','weight_records','forward_info','cache_outputs',
        'parameter_identity_preserved','model_use_cache','likelihood_source_sha256','model_call_source_sha256')
    checks={k:c[k]==n[k] for k in keys}
    checks['new_complete_feasibility']=base['valid']
    cp,np=boundaries(c['memory_profile_files']),boundaries(n['memory_profile_files'])
    points=compare(cp,np)
    ci,ni=intervals([Path(f) for f in c['memory_profile_files']]),intervals([Path(f) for f in n['memory_profile_files']])
    checks['interval_identity_complete']=bool(ci) and ci.keys()==ni.keys()
    checks['point_identity_complete']=bool(points['matched_values']) and not points['missing']
    peaks=[];identities=[]
    for root in (args.reference,args.root):
        with (root/'layerwise_cache_nvml.csv').open() as stream: rows=list(csv.reader(stream))
        peaks.append(max(memory_mib(x[1]) for x in rows));identities.append({(x[2].strip(),x[3].strip()) for x in rows})
    checks['same_physical_gpu']=identities[0]==identities[1] and len(identities[0])==1
    checks['nvml_nonincrease']=peaks[1]<=peaks[0]
    for metric in ('peak_allocated_bytes','peak_reserved_bytes'): checks[metric]=n['memory'][metric]<=c['memory'][metric]
    regressions=[dict(identity=list(k),control=ci[k],candidate=ni[k]) for k in ci.keys()&ni.keys()
        if any(ni[k]['sampled_peak_'+m+'_bytes']>ci[k]['sampled_peak_'+m+'_bytes'] for m in ('allocated','reserved'))]
    host=[json.loads((root/'layerwise_cache_host_rss.json').read_text()) for root in (args.reference,args.root)]
    result=dict(valid=all(checks.values()),checks=checks,complete_run=base,nvml_peak_mib=peaks,
        host_observations=host,interval_count=len(ci),interval_regressions=regressions,points=points,
        source_artifact_sha256={str(f):sha(f) for f in (args.reference/'layerwise_cache.json',args.root/'layerwise_cache.json',readmission)},
        scope='observer equivalence/overhead only; same previously accepted QA storage method, no new QA compute memory benefit',
        auditor_sha256=sha(__file__))
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=checks,nvml_peak_mib=peaks,host_observations=host),indent=2))
    if not result['valid']: raise SystemExit(1)


if __name__=='__main__': main()
