"""Follow the clean performance pair with two fixed-config correction arms."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.chdir('/mnt/d/gptq_plus_realq')
root=Path('outputs/resume_20260927_1216').resolve()
deadline=time.monotonic()+7200
while not (root/'prefix-fast/status.json').exists():
    baseline_status=root/'baseline-profile/status.json'
    if baseline_status.exists() and json.loads(baseline_status.read_text())['returncode']:
        raise RuntimeError('baseline failed; stop followups')
    if time.monotonic()>deadline: raise TimeoutError('performance pair did not finish')
    time.sleep(10)
for arm in ('baseline-profile','prefix-fast'):
    s=json.loads((root/arm/'status.json').read_text())
    assert s['returncode']==0 and s['sources_unchanged'],f'{arm} failed'

subprocess.run(['/home/uky/gptq_plus/.venv/bin/python','tools/compare_research_results.py'],check=True)
comparison=json.loads((root/'comparison.json').read_text())
if not comparison['final_state_equal'] or not comparison['exact_quality_equal']:
    raise RuntimeError('cache equivalence failed; investigate before next direction')

base=json.loads((root/'baseline-profile/manifest.json').read_text())['command']
for label,mode,extra in (
    ('cross-control','baseline-profile',['--quant_stop_layer','4']),
    ('cross-corrected','cross-fixed',['--quant_stop_layer','4','--cross_module_compensate'])):
    folder=root/label
    folder.mkdir(exist_ok=False)
    entry=folder/'entry.py'
    shutil.copyfile('tools/overnight_ptq_probe.py',entry)
    command=base.copy()
    old_entry=next(i for i,v in enumerate(command) if v.endswith('/entry.py'))
    command[old_entry]=str(entry)
    command[old_entry+1]=mode
    command[command.index('--exp')+1]='resume_20260927_'+label
    command+=extra
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',RESEARCH_RUN_DIR=str(folder))
    env.pop('REPLAY_IMPL_PATH',None)
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in ('gptq_utils/gptq_plus_utils.py','process_args.py',str(entry))}
    (folder/'manifest.json').write_text(json.dumps({'command':command,'hashes':hashes},indent=2))
    print('START',label,time.strftime('%F %T'),flush=True)
    started=time.monotonic()
    with (folder/'run.log').open('w') as logfile:
        result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s','3600',*command],
                              env=env,stdout=logfile,stderr=subprocess.STDOUT)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    status={'returncode':result.returncode,'sources_unchanged':unchanged,
            'wall_seconds':time.monotonic()-started,'finished':time.strftime('%F %T')}
    (folder/'status.json').write_text(json.dumps(status,indent=2))
    print('END',label,json.dumps(status),flush=True)
    if result.returncode or not unchanged: raise RuntimeError('correction arm failed; stop')
