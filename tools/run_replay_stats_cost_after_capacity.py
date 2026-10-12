"""CPU-only wait for final independent capacity acceptance, then serial cost.

No CUDA is started from an incomplete status or a launch-only source audit.
The cost controller independently re-audits all six memory pairs itself.
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
from tools.audit_completed_memory_pair_capacity import digest,read
from tools.run_replay_stats_full_followup_v2 import save_new,stop_process_tree
from tools.run_replay_stats_low_overhead_cost import check_bound


def final_admission(followup, audit):
    return (followup.get('valid') is True and len(followup.get('completed_pairs',[]))==3
            and [x.get('repeat') for x in followup['completed_pairs']]==[1,2,3]
            and all(x.get('capacity_valid') is True for x in followup['completed_pairs'])
            and audit.get('capacity_valid') is True and bool(audit.get('checks'))
            and all(x is True for x in audit['checks'].values())
            and audit.get('method')=='stable'
            and audit.get('tag')=='qwen06b-replay-stats-hardware-20261012-v1-p3')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capacity-followup',type=Path,required=True)
    p.add_argument('--independent-final-audit',type=Path,required=True)
    p.add_argument('--waiter-output',type=Path,required=True)
    p.add_argument('--cost-output',type=Path,required=True)
    args=p.parse_args();out=args.waiter_output.resolve();out.mkdir(exist_ok=False)
    proc=None;report=dict(valid=False,scope='CPU waiting only until complete native final admission')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        bound=read(ROOT/'outputs/qwen06-hardware-capacity-followup-20261012-v1/source_bindings.json')
        for name in ('run_replay_stats_cost_after_capacity.py','run_replay_stats_low_overhead_cost.py',
                     'timing_completed_prepared_disk.py','prepare_replay_stats_low_overhead_entry.py',
                     'run_low_overhead_speed_campaign.py','realq_low_overhead_speed_entry.py',
                     'deduplicate_completed_prepared_checkpoint.py'):
            path=ROOT/'tools'/name;bound[str(path)]=digest(path)
        save_new(out/'waiter_source_bindings.json',bound);check_bound(bound)
        print('CPU waiting for completed independent third-pair capacity admission; no CUDA work',flush=True)
        while not (args.capacity_followup/'status.json').exists():time.sleep(5)
        final=read(args.capacity_followup/'status.json')
        if final.get('valid') is not True:raise RuntimeError('capacity followup failed; no timing launch')
        while not args.independent_final_audit.exists():time.sleep(5)
        check_bound(bound);independent=read(args.independent_final_audit)
        if not final_admission(final,independent):raise RuntimeError('independent final admission incomplete')
        save_new(out/'admission.json',dict(valid=True,capacity_status_sha256=digest(args.capacity_followup/'status.json'),
                 independent_final_audit_sha256=digest(args.independent_final_audit)))
        command=[sys.executable,'-u',str(ROOT/'tools/run_replay_stats_low_overhead_cost.py'),
                 '--capacity-followup',str(args.capacity_followup.resolve()),'--output-directory',str(args.cost_output.resolve())]
        proc=subprocess.Popen(command,cwd=ROOT);code=proc.wait()
        check_bound(bound);report['cost_returncode']=code
        result=read(args.cost_output/'status.json')
        report['valid']=code==0 and result.get('valid') is True
        report['cost_status_sha256']=digest(args.cost_output/'status.json')
        if not report['valid']:raise RuntimeError('cost verification failed; inspect preserved artifacts')
    except BaseException as exc:
        report['error']=repr(exc)
        if proc and proc.poll() is None:stop_process_tree(proc.pid);proc.wait()
        raise
    finally:
        report['finished']=time.strftime('%F %T');save_new(out/'status.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
