"""Recover only the incomplete third hardware pair after the host reboot.

Completed first/second pairs are independently readmitted, never rerun.
Original interrupted files and absent final statuses remain unchanged.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_replay_stats_full_followup_v2 as base
from tools.audit_completed_memory_pair_capacity import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-directory',type=Path,required=True)
    args=p.parse_args();out=args.output_directory.resolve();out.mkdir(exist_ok=False)
    original=ROOT/'outputs/qwen06-hardware-capacity-followup-20261012-v1'
    old=ROOT/'outputs/replay-stats-full-followup-20261012-v2'
    proc=reader=None
    report=dict(valid=False,scope='Interrupted repeat3 recovery only, unchanged stable+stats method; original host-reboot evidence preserved',completed_pairs=[])
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        bound=base.read(original/'source_bindings.json');base.check_bound_sources(bound)
        entry=out/'entry.py';entry.write_bytes((original/'entry.py').read_bytes())
        if base.digest(entry)!=base.digest(old/'entry.py'):raise RuntimeError('original raw method changed')
        for path in (entry,Path(__file__).resolve()):bound[str(path)]=base.digest(path)
        base.save_new(out/'source_bindings.json',bound);base.check_bound_sources(bound)
        prior=(('qwen06b-replay-stats-full-20261012-v2-p1',old/'source_bindings.json'),
               ('qwen06b-replay-stats-hardware-20261012-v1-p2',original/'source_bindings.json'))
        for repeat,(tag,bindings) in enumerate(prior,1):
            value=audit(tag,bindings,method='stable');path=out/f'independent_completed_pair_readmission{repeat}.json'
            base.save_new(path,value)
            if not value['capacity_valid']:raise RuntimeError('completed prior pair readmission failed')
            report['completed_pairs'].append(dict(repeat=repeat,tag=tag,capacity_valid=True,
                strict_valid=value['strict_interval_point_valid'],audit=str(path)))
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if any(s.strip().isdigit() for s in gpu.splitlines()):raise RuntimeError('another CUDA process active')
        tag='qwen06b-replay-stats-hardware-20261012-recovery-v2-p3'
        folder=out/(tag+'-manifests');folder.mkdir()
        for side in ('control','candidate'):
            for stage in ('stage0','stage2'):
                value=base.read(original/'qwen06b-replay-stats-hardware-20261012-v1-p3-manifests'/(side+'-'+stage+'.json'))
                command=value['command'];command[command.index('baseline-profile')-1]=str(entry)
                value['scope_note']='Fresh recovery of host-reboot interrupted repeat3, original stable+stats math/workload; not a completed failed-method rerun.'
                base.save_new(folder/(side+'-'+stage+'.json'),value)
        for stage in ('stage0','stage2'):
            commands=[base.read(folder/(s+'-'+stage+'.json'))['command'] for s in ('control','candidate')]
            if base.normalized_command(commands[0])!=base.normalized_command(commands[1]):raise RuntimeError('algorithm/data mismatch')
        command=[sys.executable,'-u',str(ROOT/'tools/run_replay_stats_memory_campaign.py'),'--tag',tag,'--pairs','1','--gpu-indices','0','--timeout-seconds','7200']
        for side in ('control','candidate'):
            for stage in ('stage0','stage2'):command.extend([f'--{side}-{stage}-manifest',str(folder/(side+'-'+stage+'.json'))])
        (out/'active.json').write_text(json.dumps(dict(label=tag,command=command)))
        reader=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/stream_memory_research_progress.py'),
            '--queues',str(out),'--output',str(ROOT/'logs/realq-system-restart-20261010-1408.log'),'--append'],cwd=ROOT,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        base.check_bound_sources(bound)
        with (out/(tag+'.log')).open('x') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);code=proc.wait()
        value=audit(tag,out/'source_bindings.json',method='stable');path=out/'independent_capacity_repeat3.json'
        base.save_new(path,value)
        if not value['capacity_valid']:raise RuntimeError('recovered full hardware capacity gate failed')
        report['completed_pairs'].append(dict(repeat=3,tag=tag,capacity_valid=True,
            strict_valid=value['strict_interval_point_valid'],audit=str(path),legacy_campaign_returncode=code))
        report['valid']=True
    except BaseException as exc:
        report['error']=repr(exc)
        if proc and proc.poll() is None:base.stop_process_tree(proc.pid);proc.wait()
        raise
    finally:
        report['finished']=time.strftime('%F %T');base.save_new(out/'status.json',report)
        if reader:reader.wait(timeout=15)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
