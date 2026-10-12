"""Read completed experiment artifacts; never infer success from partial logs."""
import json
import argparse
from pathlib import Path
import re

parser=argparse.ArgumentParser()
parser.add_argument('--root',default='outputs/resume_20260927_1216')
root=Path(parser.parse_args().root)
rows={}
for mode in ('baseline-profile','prefix-fast'):
    folder=root/mode
    status=json.loads((folder/'status.json').read_text())
    if status['returncode']!=0 or not status['sources_unchanged']:
        raise RuntimeError(f'{mode} did not complete cleanly')
    metrics=json.loads((folder/(mode+'_metrics.json')).read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    log=(folder/'run.log').read_text(errors='replace')
    match=re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)',log)
    if not match: raise RuntimeError(f'{mode} missing final exact evaluation')
    command=manifest['command']
    rows[mode]={**metrics,**status,'kl':float(match[1]),'ppl':float(match[2]),
                'quantization_arguments':command[command.index('--model'):],
                'source_hashes':manifest['hashes']}
a,b=rows['baseline-profile'],rows['prefix-fast']
assert a['quantization_arguments']==b['quantization_arguments'],'configuration mismatch'
for p in ('gptq_utils/gptq_plus_utils.py','process_args.py'):
    assert a['source_hashes'][p]==b['source_hashes'][p],'quantization source mismatch'
result={'arms':rows,'final_state_equal':a['state_sha256']==b['state_sha256'],
        'exact_quality_equal':(a['kl'],a['ppl'])==(b['kl'],b['ppl']),
        'quantization_speedup':a['quantization_seconds']/b['quantization_seconds'],
        'quantization_time_reduction_pct':100*(1-b['quantization_seconds']/a['quantization_seconds']),
        'process_time_reduction_pct':100*(1-b['wall_seconds']/a['wall_seconds']),
        'peak_allocated_delta_mib':(b['peak_allocated']-a['peak_allocated'])/1024**2,
        'single_pair_only':True,
        'limitations':['One timing pair on one model/configuration; no statistical stability claim.',
                       'Prototype targets down_proj only and disables reuse outside supported paths.',
                       'Process timing includes identical state hashing overhead; quantization timing excludes hashing.']}
(root/'comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))
