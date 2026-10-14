"""Long-input single-arm feasibility. No long-input control is fabricated."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_qa_layerwise_component_v2 import source as previous_source


FEASIBILITY = '''
def feasibility(root, bindings):
    import math
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from tools.audit_memory_boundary_points import boundaries
    from tools.audit_completed_memory_pair_capacity import intervals
    from tools.storage_process_device_peaks import valid_record
    n=json.loads((root/'layerwise_cache.json').read_text())
    with torch.device('meta'):
        m=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(n['model'],local_files_only=True),torch_dtype=torch.bfloat16)
    expected={k:[list(v.shape),str(v.dtype)] for k,v in m.state_dict().items()}
    calls=len(n['logits'])
    admission=json.loads((root/'short_admission.json').read_text())
    admission_path=Path(admission['path'])/'independent_complete_kv_audit.json'
    ids=[(r['forward'],r['layer']) for r in n['weight_records']]
    ic=intervals([Path(p) for p in n['memory_profile_files']])
    point=boundaries(n['memory_profile_files'])
    with (root/'layerwise_cache_nvml.csv').open() as stream: samples=list(csv.reader(stream))
    host=json.loads((root/'layerwise_cache_host_rss.json').read_text())
    checks=dict(short_admission_unchanged=sha(admission_path)==admission['sha256'] and json.loads(admission_path.read_text()).get('valid') is True,
        source_model_library_sha=all(sha(p)==h for p,h in bindings.items()),
        host_observation=host['sample_count']>0 and host['max_sampled_worker_rss_bytes']>0,
        complete_state_unchanged=n['before']==n['after'] and n['before']['layout']==expected and n['before']['count']==len(expected),
        original_parameters=n['parameter_identity_preserved'],
        batch_precision=n['batch_size']==16 and all(x['dtype']=='torch.bfloat16' for x in n['logits']),
        complete_nonempty_logits=calls>0 and all(x['shape'][0]==16 and x['shape'][2]==m.config.vocab_size and bool(x['sha256']) for x in n['logits']),
        complete_nonempty_28_layers=len(n['layers'])==28 and calls>0 and all(x['layer']==i and x['count']==calls and len(x['layout'])==calls and bool(x['sha256']) for i,x in enumerate(n['layers'])),
        scores_finite=len(n['scores'])==64 and all(math.isfinite(x[0]) and type(x[1]) is bool for x in n['scores']),
        weights_coverage=len(ids)==calls*28 and set(ids)=={(f,i) for f in range(1,calls+1) for i in range(28)} and all(r['enabled'] and r['final_device']=='cpu' and r['parameter_bytes']>0 for r in n['weight_records']),
        kv_semantics=len(n['forward_info'])==calls and all(r['use_cache'] and r['explicit_cache'] and r['offloading'] for r in n['forward_info']),
        kv_placement=len(n['cache_outputs'])==calls and all(len(out)==28 and all(x['device']==x['value_device']==('cuda:0' if i==0 else 'cpu') and x['shape']==x['value_shape'] and x['dtype']==x['value_dtype']=='torch.bfloat16' for i,x in enumerate(out)) for out in n['cache_outputs']),
        all_device_peaks=valid_record(n['memory'],1) and n['memory']['peak_allocated_bytes']>0,
        complete_exact_intervals=bool(ic) and len(ic)==calls*116+1 and all(x.get('peak_measurement_mode')=='exact' for x in ic.values()),
        independent_points=bool(point) and len(point)==len(ic)*4,
        nvml_identity=bool(samples) and all(len(x)==4 and x[2].strip()=='0' for x in samples) and len({x[3].strip() for x in samples})==1)
    return dict(valid=all(checks.values()), checks=checks, memory=n['memory'],
        nvml_peak_mib=max(memory_mib(x[1]) for x in samples), interval_count=len(ic), point_count=len(point),
        scope='single-arm synthetic long-input feasibility; NO long-input numerical control or paired capacity claim',
        short_admission=admission, host=host, nvml_is_sampled=True)

'''


def source():
    text=previous_source()
    replacements={
        "SIDES=('control','rows','layerwise','layerwise_cache')":"SIDES=('layerwise_cache',)",
        "choices=(128,256)":"choices=(1024,1270)",
        "default=128);p.add_argument('--arm'":"default=1024);p.add_argument('--arm'",
        "                relay=LiveFileRelay(args.output/(side+'.log'))":"                host_max=0;host_samples=0\n                relay=LiveFileRelay(args.output/(side+'.log'))",
        "                while proc.poll() is None: relay.forward();time.sleep(0.5)":"                while proc.poll() is None:\n                    relay.forward()\n                    try:\n                        rss=[x for x in Path('/proc/'+str(proc.pid)+'/status').read_text().splitlines() if x.startswith('VmRSS:')]\n                        if rss: host_max=max(host_max,int(rss[0].split()[1])*1024);host_samples+=1\n                    except FileNotFoundError: pass\n                    time.sleep(0.5)\n                with (args.output/(side+'_host_rss.json')).open('x') as f: json.dump(dict(pid=proc.pid,max_sampled_worker_rss_bytes=host_max,sample_count=host_samples,sample_interval_seconds=0.5,scope='single worker sampled VmRSS; not total host/PSS or continuous maximum'),f,indent=2)",
        "def main():":FEASIBILITY+"def main():",
        "result=audit(args.output,bindings)":"result=feasibility(args.output,bindings)",
        "print(json.dumps({side:dict(valid=r['valid'],checks=r['checks'],nvml_peak_mib=r['nvml_peak_mib']) for side,r in result['reports'].items()},indent=2))":"print(json.dumps(result,indent=2))",
        "source=[Path(__file__),Path(__file__).with_name('run_qa_layerwise_component.py'),":"source=[Path(__file__),Path(__file__).with_name('run_qa_layerwise_component_v2.py'),Path(__file__).with_name('run_qa_layerwise_component.py'),",
        "    args.output.mkdir(parents=True,exist_ok=False)":"    short_root=Path('outputs/qa-layerwise-local-'+('qwen06' if '0.6B' in args.model else 'qwen17')+'-ctx256-20261014-v2')\n    admission=json.loads((short_root/'independent_complete_kv_audit.json').read_text())\n    if admission.get('valid') is not True: raise RuntimeError('Short component admission absent/rejected')\n    old_bindings=json.loads((short_root/'source_bindings.json').read_text())\n    if any(sha(p)!=h for p,h in old_bindings.items()): raise RuntimeError('Short admission source changed')\n    args.output.mkdir(parents=True,exist_ok=False)\n    (args.output/'short_admission.json').write_text(json.dumps(dict(path=str(short_root),sha256=sha(short_root/'independent_complete_kv_audit.json')),indent=2))",
    }
    for old,new in replacements.items():
        if text.count(old)!=1: raise RuntimeError('Frozen feasibility contract changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_long_feasibility','exec')
    return text


if __name__=='__main__':
    import argparse,json
    if '--audit-root' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--audit-root',type=Path,required=True);p.add_argument('--audit-output',type=Path,required=True)
        args=p.parse_args();namespace=dict(__name__='qa_long_audit',__file__=__file__)
        exec(compile(source(),'qa_long_feasibility','exec'),namespace)
        result=namespace['feasibility'](args.audit_root,json.loads((args.audit_root/'source_bindings.json').read_text()))
        with args.audit_output.open('x') as f: json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
        if not result['valid']: raise SystemExit(1)
    else:
        exec(compile(source(),'qa_long_feasibility','exec'),dict(__name__='__main__',__file__=__file__))
