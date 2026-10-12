"""Repeat full baseline/cache timing with execution order reversed."""
import hashlib, json, os, shutil, subprocess, time
from pathlib import Path

os.chdir('/mnt/d/gptq_plus_realq')
root=Path('outputs/resume_20260927_reverse_order').resolve()
root.mkdir(parents=True,exist_ok=False)
manifest=json.loads(Path('outputs/resume_20260927_1216/baseline-profile/manifest.json').read_text())
base=manifest['command']
base_entry=next(i for i,x in enumerate(base) if x.endswith('/entry.py'))
originals=('gptq_utils/gptq_plus_utils.py','process_args.py')

# Repeat the same pair but run the cache arm first to expose order and device-state effects.
for label,mode in (('prefix-fast','prefix-fast'),('baseline-profile','baseline-profile')):
    folder=root/label
    folder.mkdir()
    entry=folder/'entry.py'
    cache=folder/'frozen_prefix_replay.py'
    shutil.copyfile('tools/overnight_ptq_probe.py',entry)
    shutil.copyfile('gptq_utils/frozen_prefix_replay.py',cache)
    command=base.copy()
    command[base_entry]=str(entry)
    command[base_entry+1]=mode
    command[command.index('--exp')+1]='resume_20260927_reverse_'+label
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (*originals,str(entry),str(cache))}
    (folder/'manifest.json').write_text(json.dumps({'command':command,'hashes':hashes},indent=2))
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',RESEARCH_RUN_DIR=str(folder),
        REPLAY_IMPL_PATH=str(cache))
    start=time.monotonic()
    print('START',label,time.strftime('%F %T'),flush=True)
    with (folder/'run.log').open('w') as log:
        result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s','3600',*command],
            env=env,stdout=log,stderr=subprocess.STDOUT)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    status={'returncode':result.returncode,'wall_seconds':time.monotonic()-start,
        'sources_unchanged':unchanged,'finished':time.strftime('%F %T')}
    (folder/'status.json').write_text(json.dumps(status,indent=2))
    print('END',label,json.dumps(status),flush=True)
    if result.returncode or not unchanged:
        raise RuntimeError(f'{label} failed or sources changed; stop the repeat pair')

subprocess.run(['/home/uky/gptq_plus/.venv/bin/python','tools/compare_research_results.py',
                '--root',str(root)],check=True)
