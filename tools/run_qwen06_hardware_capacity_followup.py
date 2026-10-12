"""Complete two fresh repeats under human clarified global capacity criterion.

The original completed stable+stats 0.6B pair is repeat1 after independent
capacity reclassification. New repeat2/3 keep that exact method, not the size
policy or compact-pool methods. Original strict gate outputs are preserved.
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-first-capacity-audit',type=Path,required=True)
    parser.add_argument('--wait-for-compact-audit',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();out=args.output_directory.resolve();out.mkdir(exist_ok=False)
    proc=reader=None
    report=dict(valid=False,scope='Fresh repeat2/3 original stable+stats method under human global hardware criterion; local strict diagnostics preserved',completed_pairs=[])
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        waiter_sources={str(path):base.digest(path) for path in (Path(__file__).resolve(),ROOT/'tools/audit_completed_memory_pair_capacity.py')}
        base.save_new(out/'waiter_source_bindings.json',waiter_sources)
        first=base.read(args.accepted_first_capacity_audit)
        if not first['capacity_valid'] or first['method']!='stable' or first['tag']!='qwen06b-replay-stats-full-20261012-v2-p1':
            raise RuntimeError('independent original-method first-pair hardware acceptance required')
        report['accepted_first_audit_sha256']=base.digest(args.accepted_first_capacity_audit)
        print('CPU waiting for completed independent compact audit; no CUDA work',flush=True)
        while not args.wait_for_compact_audit.exists():time.sleep(5)
        base.check_bound_sources(waiter_sources)
        compact=base.read(args.wait_for_compact_audit)
        # A new numerical/source/infrastructure failure must be explained first.
        # Only resource differences alone can close this optional branch.
        if not all(compact['checks'].values()) or compact['method']!='compact':
            raise RuntimeError('compact quality/source/cache/method/telemetry incomplete; inspect bug before proceeding')
        compact_finished=args.wait_for_compact_audit.parent/'status.json'
        while not compact_finished.exists():time.sleep(5)
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
        if any(s.strip().isdigit() for s in gpu.stdout.splitlines()):raise RuntimeError('another CUDA process active')
        old=ROOT/'outputs/replay-stats-full-followup-20261012-v2'
        bound=base.read(old/'source_bindings.json');base.check_bound_sources(bound)
        entry=out/'entry.py';entry.write_bytes((old/'entry.py').read_bytes())
        if base.digest(entry)!=base.digest(old/'entry.py'):raise RuntimeError('raw original method entry changed')
        for path in (entry,Path(__file__).resolve(),ROOT/'tools/audit_completed_memory_pair_capacity.py'):
            bound[str(path)]=base.digest(path)
        base.save_new(out/'source_bindings.json',bound);base.check_bound_sources(bound)
        # Fresh independent bytes/cache/method audit again before repeat2.
        initial=audit(first['tag'],old/'source_bindings.json',method='stable')
        base.save_new(out/'independent_original_pair_readmission.json',initial)
        if not initial['capacity_valid']:raise RuntimeError('first-pair readmission failed')
        reader=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/stream_memory_research_progress.py'),'--queues',str(out),'--output',str(ROOT/'logs/realq-system-restart-20261010-1408.log'),'--append'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        report['completed_pairs'].append(dict(repeat=1,tag=first['tag'],capacity_valid=True,strict_valid=initial['strict_interval_point_valid'],audit=str(out/'independent_original_pair_readmission.json')))
        original_manifests=old/'qwen06b-replay-stats-full-20261012-v2-p1-manifests'
        for repeat in (2,3):
            base.check_bound_sources(bound)
            tag=f'qwen06b-replay-stats-hardware-20261012-v1-p{repeat}'
            folder=out/(tag+'-manifests');folder.mkdir()
            for side in ('control','candidate'):
                for stage in ('stage0','stage2'):
                    data=base.read(original_manifests/(side+'-'+stage+'.json'));command=data['command'];command[command.index('baseline-profile')-1]=str(entry)
                    data['scope_note']='Exact original stable+stats method; fresh repeat under user global hardware acceptance, local regressions retained as diagnostics.'
                    base.save_new(folder/(side+'-'+stage+'.json'),data)
            for stage in ('stage0','stage2'):
                commands=[base.read(folder/(side+'-'+stage+'.json'))['command'] for side in ('control','candidate')]
                if base.normalized_command(commands[0])!=base.normalized_command(commands[1]):raise RuntimeError('algorithm/data command mismatch')
            cmd=[sys.executable,'-u',str(ROOT/'tools/run_replay_stats_memory_campaign.py'),'--tag',tag,'--pairs','1','--gpu-indices','0','--timeout-seconds','7200']
            for side in ('control','candidate'):
                for stage in ('stage0','stage2'):cmd.extend([f'--{side}-{stage}-manifest',str(folder/(side+'-'+stage+'.json'))])
            (out/'active.json').write_text(json.dumps(dict(label=tag,command=cmd)))
            with (out/(tag+'.log')).open('x') as stream:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);code=proc.wait()
            # Legacy campaign returns1 for local resource regressions. This
            # never replaces independent complete numerical/global acceptance.
            completed=audit(tag,out/'source_bindings.json',method='stable')
            audit_path=out/f'independent_capacity_repeat{repeat}.json';base.save_new(audit_path,completed)
            if not completed['capacity_valid']:raise RuntimeError('complete global capacity gate failed; no later repeats')
            report['completed_pairs'].append(dict(repeat=repeat,tag=tag,capacity_valid=True,
                strict_valid=completed['strict_interval_point_valid'],legacy_campaign_returncode=code,audit=str(audit_path)))
            (out/'partial.json').write_text(json.dumps(report,indent=2))
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
