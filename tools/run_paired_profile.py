"""Run the paired refresh diagnostic on the same five-layer audit budget."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.chdir('/mnt/d/gptq_plus_realq')
parser=argparse.ArgumentParser()
parser.add_argument('--tag',required=True)
parser.add_argument('--skip-eval',action='store_true')
opts=parser.parse_args()
if not opts.tag.replace('-','').isalnum():
    raise ValueError('tag must contain only letters, digits, hyphens')
root=Path('outputs/resume_20260927_1216/paired-profile-'+opts.tag).resolve()
root.mkdir(parents=True,exist_ok=False)
source=json.loads(Path('outputs/resume_20260927_1216/baseline-profile/manifest.json').read_text())
command=source['command'].copy()
entry=Path('/mnt/d/gptq_plus_realq/tools/profile_prefix_refresh.py')
old=next(i for i,x in enumerate(command) if x.endswith('/entry.py'))
command[old]=str(entry)
command.pop(old+1) # mode is only an argument for overnight_ptq_probe.py
command[command.index('--exp')+1]='resume_20260927_paired_profile'
command += ['--quant_stop_layer','4']
if opts.skip_eval:
    command += ['--skip_eval']
cache=Path('/mnt/d/gptq_plus_realq/gptq_utils/frozen_prefix_replay.py')
runenv=os.environ.copy()
runenv.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
              PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
              RESEARCH_RUN_DIR=str(root),REPLAY_IMPL_PATH=str(cache))
hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in ('gptq_utils/gptq_plus_utils.py','process_args.py',str(entry),str(cache))}
manifest={'command':command,'hashes':hashes}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
started=time.monotonic()
with (root/'run.log').open('w') as log:
    result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s','3600',*command],
                          env=runenv,stdout=log,stderr=subprocess.STDOUT)
unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
status={'returncode':result.returncode,'wall_seconds':time.monotonic()-started,
        'sources_unchanged':unchanged,'finished':time.strftime('%F %T')}
(root/'status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status),flush=True)
if result.returncode or not unchanged:
    raise RuntimeError('paired refresh run failed or its code changed')
