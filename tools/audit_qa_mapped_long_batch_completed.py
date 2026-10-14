"""Fresh complete long batch-lifetime audit; preserves the v1 rejection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def batch_checks(row,enabled):
    head=row['mapped_head'];releases=head.get('batch_releases',[])
    checks=dict(batch_release_count=len(releases)==head['calls']>0,
        batch_release_identity=all(r['call']==i+1 and r['enabled'] is enabled for i,r in enumerate(releases)))
    if enabled:
        checks['each_batch_active_and_host_zero']=bool(releases) and all(
            r['active_before_release']==r['host_live_after_release']==0
            and r['allocations']==r['releases']>0 for r in releases)
        checks['pool_created_per_batch']=head['allocations']==head['releases']==len(releases)
    return checks


def module_source():
    text=(ROOT/'tools/audit_qa_mapped_long_completed.py').read_text()
    old='from tools.run_qa_mapped_long_component import NUMERICAL_KEYS,expected_phases'
    if text.count(old)!=1: raise RuntimeError('Frozen auditor import changed')
    text=text.replace(old,'from tools.run_qa_mapped_long_batch_component import NUMERICAL_KEYS,expected_phases')
    anchor="        checks.update({side+'_head_'+k:v for k,v in head_checks(row,config,side=='candidate').items()})"
    if text.count(anchor)!=1: raise RuntimeError('Frozen auditor head coverage changed')
    text=text.replace(anchor,anchor+"\n        checks.update({side+'_batch_'+k:v for k,v in batch_checks(row,side=='candidate').items()})")
    compile(text,'qa_mapped_long_batch_audit','exec')
    return text


_scope=dict(__name__='qa_mapped_long_batch_audit',__file__=__file__,batch_checks=batch_checks)
exec(compile(module_source(),'qa_mapped_long_batch_audit','exec'),_scope)
audit=_scope['audit']
if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.root)
    result['scope']='raw311 synthetic long-input paired batch-last-consumer mapped output, not formal QA or server capacity'
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=result['checks'],nvml=result['nvml_peak_mib'],host=result['host_observations']),indent=2))
    if not result['valid']: raise SystemExit(1)
