"""Smoke-check, then compare legacy vs algebra-corrected TS_BLOCKWISE end-to-end.

The treatment is residual row-error transport. All quality arms share num_groups=1,
since the current blockwise implementation explicitly requires that structure.
"""
import hashlib, json, os, shutil, subprocess, time
import argparse
from pathlib import Path

os.chdir('/mnt/d/gptq_plus_realq')
parser=argparse.ArgumentParser()
parser.add_argument('--smoke-only',action='store_true')
parser.add_argument('--smoke-tag',default='initial')
parser.add_argument('--quality-dir',default='outputs/resume_20260927_ts_quality')
parser.add_argument('--eval-samples',type=int,default=8)
parser.add_argument('--corrected-confirmation',action='store_true',
                    help='Run only matched plain/corrected arms for a larger evaluation subset.')
opts=parser.parse_args()
smoke_root='outputs/resume_20260927_ts_smoke-'+opts.smoke_tag
root=Path(smoke_root if opts.smoke_only
          else opts.quality_dir).resolve()
root.mkdir(parents=True,exist_ok=False)
source=json.loads(Path('outputs/resume_20260927_1216/baseline-profile/manifest.json').read_text())
base=source['command']
entry_index=next(i for i,x in enumerate(base) if x.endswith('/entry.py'))
orig=('gptq_utils/gptq_plus_utils.py','process_args.py','ptq.py','utils/data_utils.py')

def run(label,mode,extra,ts_env=False,timeout=3600):
    folder=root/label
    folder.mkdir()
    entry=folder/'entry.py'
    shutil.copyfile('tools/overnight_ptq_probe.py',entry)
    command=base.copy()
    command[entry_index]=str(entry)
    command[entry_index+1]=mode
    command[command.index('--exp')+1]='resume_20260927_ts_'+label
    # The implementation's blockwise row metric is only defined for a single
    # shared row block; keep this fixed across the plain and TS quality arms.
    command[command.index('--num_groups')+1]='1'
    # TS_BLOCKWISE currently lives in fasterquant's group-parallel kernel.
    # With `none`, the legacy path silently ignores two_sided_metric, so make
    # the algorithm path explicit and identical for every quality arm.
    command[command.index('--group_parallel_quant')+1]='rank'
    # The final transformer layer switches every refresh to full KL. In the
    # rank group-parallel path this costs minutes per 128-column block, so the
    # first quality comparison leaves only that final layer in FP. All three
    # arms share this stop point; this is an explicitly partial-W4 result.
    command += ['--quant_stop_layer','26']
    command += ['--eval_samples',str(opts.eval_samples)]
    command+=extra
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (*orig,str(entry))}
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',RESEARCH_RUN_DIR=str(folder))
    if ts_env:
        env['TS_BLOCKWISE']='1'
    manifest={'command':command,'hashes':hashes,'TS_BLOCKWISE':ts_env}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    started=time.monotonic()
    print('START',label,time.strftime('%F %T'),flush=True)
    with (folder/'run.log').open('w') as log:
        result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s',f'{timeout}',*command],
            env=env,stdout=log,stderr=subprocess.STDOUT)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    status={'returncode':result.returncode,'wall_seconds':time.monotonic()-started,
        'sources_unchanged':unchanged,'finished':time.strftime('%F %T')}
    (folder/'status.json').write_text(json.dumps(status,indent=2))
    print('END',label,json.dumps(status),flush=True)
    if result.returncode or not unchanged:
        raise RuntimeError(f'{label} failed or source changed; stop quality campaign')

# Small configuration is only a numerical/8-GB smoke; it is not a quality result.
run('smoke','ts-fixed',['--two_sided_rounding','1','--two_sided_modules','mlp.down_proj',
    '--quant_stop_layer','1','--nsamples','4','--backward_samples','4','--skip_eval'],
    ts_env=True,timeout=900)

if opts.smoke_only:
    print('TS_BLOCKWISE numerical/VRAM smoke passed; full quality arms not run.',flush=True)
    raise SystemExit(0)

# Matched 27/28-layer 32x256 evaluation. The plain arm is the num_groups=1
# control; the two-sided arms differ only in the verified outgoing-error
# transform. Layer 27 remains FP in every arm to avoid the rank-path KL cost.
tsflags=['--two_sided_rounding','1','--two_sided_modules','mlp.down_proj']
run('plain-g1','baseline-profile',[],timeout=3600)
if not opts.corrected_confirmation:
    run('ts-legacy','baseline-profile',tsflags,ts_env=True,timeout=3600)
run('ts-corrected','ts-fixed',tsflags,ts_env=True,timeout=3600)

def metrics(label):
    folder=root/label
    status=json.loads((folder/'status.json').read_text())
    assert status['returncode']==0 and status['sources_unchanged']
    m=json.loads((folder/(('baseline-profile' if label in ('plain-g1','ts-legacy') else 'ts-fixed')+'_metrics.json')).read_text())
    log=(folder/'run.log').read_text(errors='replace')
    import re
    x=re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)',log)
    if not x: raise RuntimeError(f'{label} missing exact quality evaluation')
    return {**m,**status,'kl':float(x[1]),'ppl':float(x[2]),
        'two_sided_blockwise_calls':log.count('two_sided_blockwise layer=')}

arm_names=('plain-g1','ts-corrected') if opts.corrected_confirmation else ('plain-g1','ts-legacy','ts-corrected')
result={'smoke_status':json.loads((root/'smoke/status.json').read_text()),
        'arms':{n:metrics(n) for n in arm_names},
        'scope':f'27/28 layers are W4 and final layer 27 stays FP in every arm; evaluation uses the first {opts.eval_samples} fixed-length WikiText2 test windows. This is a screening/partial-W4 result, not a full-test or full-W4 claim. Interpret TS against matched G=1 control. The default G=4 run is context, not a matched comparator.'}
(root/'quality_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
