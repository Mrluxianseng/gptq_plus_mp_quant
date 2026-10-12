"""Bounded serial validation campaign, with frozen entrypoints and exit records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('phase',choices=['verify','pair','cross'])
args=parser.parse_args()
os.chdir('/mnt/d/gptq_plus_realq')
root=Path('outputs/resume_20260927_1216').resolve()
root.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
common=['--model','/mnt/d/llamaModels/Qwen3-0.6B','--dataset','wikitext2',
    '--nsamples','32','--seq_len','256','--w_method','gptq_plus','--w_bits','4','--w_clip',
    '--num_groups','4','--blocksize','128','--act_order','--kl_topk','-1','--bsz','1',
    '--final_layer_stats_bsz','4','--alpha','0.0','--enable_gptq_plus','0',
    '--backward_samples','32','--backward_bsz','4','--final_layer_backward_bsz','4',
    '--g_update_mode','block_gd','--grad_lr','5e-5','--grad_optimizer','adam',
    '--grad_refresh_loss','fisher_diag_mse','--global_loss','--global_loss_bsz','1',
    '--grad_clip','5e-5','--final_layer_grad_clip','5e-4','--final_layer_grad_lr','1e-6',
    '--group_parallel_quant','none','--eval_seq_len','1024','--eval_datasets','wikitext2']

def run(mode,extra=()):
    run_dir=root/mode
    run_dir.mkdir(exist_ok=False)
    entry=run_dir/'entry.py'
    replay=run_dir/'frozen_prefix_replay.py'
    shutil.copyfile('tools/overnight_ptq_probe.py',entry)
    shutil.copyfile('gptq_utils/frozen_prefix_replay.py',replay)
    run_env=env.copy()
    run_env.update(RESEARCH_RUN_DIR=str(run_dir),REPLAY_IMPL_PATH=str(replay))
    command=[sys.executable,'-u','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1',
        '--rdzv_endpoint=localhost:29428',str(entry),mode,'--exp','resume_20260927_'+mode,
        *common,*extra]
    files=['gptq_utils/gptq_plus_utils.py','process_args.py',str(entry),str(replay)]
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    (run_dir/'manifest.json').write_text(json.dumps({'command':command,'hashes':hashes},indent=2))
    print('START',mode,time.strftime('%F %T'),flush=True)
    started=time.monotonic()
    with (run_dir/'run.log').open('w') as logfile:
        try:
            result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s','3600',*command],
                                  env=run_env,stdout=logfile,stderr=subprocess.STDOUT)
            rc=result.returncode
        except Exception as exc:
            rc=-1
            logfile.write(repr(exc))
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    status={'mode':mode,'returncode':rc,'wall_seconds':time.monotonic()-started,
            'sources_unchanged':unchanged,'finished':time.strftime('%F %T')}
    (run_dir/'status.json').write_text(json.dumps(status,indent=2))
    print('END',json.dumps(status),flush=True)
    if rc or not unchanged:
        raise RuntimeError('run failed or source changed; stop and audit')

if args.phase=='verify':
    run('prefix-verify',['--quant_stop_layer','1','--nsamples','4','--backward_samples','4','--skip_eval'])
elif args.phase=='pair':
    status=json.loads((root/'prefix-verify/status.json').read_text())
    assert status['returncode']==0 and status['sources_unchanged']
    run('baseline-profile')
    run('prefix-fast')
else:
    run('cross-fixed',['--cross_module_compensate','--quant_stop_layer','4'])
