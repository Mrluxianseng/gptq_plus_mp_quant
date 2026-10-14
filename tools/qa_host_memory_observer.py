"""Read-only Linux process-tree RSS/PSS telemetry; no CUDA/model imports.

Per-process reads are sequential, not an atomic tree snapshot or a continuous
peak proof. Missing/reused processes and unavailable limits remain explicit.
Locked/VmLck/VmPin never substitute for CUDA allocator callback pinned bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

FIELDS=('Rss','Pss','Shared_Clean','Shared_Dirty','Private_Clean','Private_Dirty','Swap','SwapPss','Locked')


def kb_fields(text,required=()):
    result={}
    for line in text.splitlines():
        if ':' not in line:continue
        key,value=line.split(':',1);parts=value.split()
        if len(parts)==2 and parts[1]=='kB':
            if key in result or not parts[0].isdigit():raise ValueError('Invalid/duplicate memory field '+key)
            result[key]=int(parts[0])*1024
    if not set(required).issubset(result):raise ValueError('Missing byte fields: '+str(set(required)-set(result)))
    return result


def identity(proc,pid):
    text=(proc/str(pid)/'stat').read_text()
    prefix,separator,tail=text.rpartition(')')
    parts=tail.split()
    if not separator or len(parts)<20 or int(prefix.split('(',1)[0])!=pid:
        raise ValueError('Malformed process identity')
    return dict(pid=pid,ppid=int(parts[1]),start_ticks=int(parts[19]))


def descendants(table,pid):
    result={pid};pending=[pid]
    while pending:
        parent=pending.pop()
        for row in table.values():
            if row['ppid']==parent and row['pid'] not in result:
                result.add(row['pid']);pending.append(row['pid'])
    return {i:table[i] for i in sorted(result) if i in table}


def scan(proc):
    rows={};errors=[]
    for path in proc.iterdir():
        if not path.name.isdigit():continue
        try:rows[int(path.name)]=identity(proc,int(path.name))
        except (OSError,ValueError) as exc:errors.append(dict(pid=int(path.name),error_type=type(exc).__name__))
    return rows,errors


def capacity(proc,pid,cgroup):
    result={}
    try:result['meminfo_bytes']=kb_fields((proc/'meminfo').read_text(),('MemTotal','MemAvailable','SwapTotal','SwapFree'))
    except (OSError,ValueError) as exc:result['meminfo_error']=repr(exc)
    try:
        paths=[line.split(':',2)[2] for line in (proc/str(pid)/'cgroup').read_text().splitlines() if line.startswith('0::')]
        if len(paths)!=1:raise ValueError('Unified cgroup path unavailable')
        base=cgroup.resolve();target=(base/paths[0].lstrip('/')).resolve()
        if not target.is_relative_to(base):raise ValueError('Unified cgroup path escapes mount')
        result['cgroup_v2_path']=paths[0];result['cgroup_v2_ancestors']=[]
        while target.is_relative_to(base):
            row=dict(path=str(target),values={},unavailable={})
            for field in ('memory.current','memory.max','memory.swap.current','memory.swap.max'):
                try:
                    raw=(target/field).read_text().strip()
                    if raw!='max' and not raw.isdigit():raise ValueError('Invalid cgroup memory value')
                    row['values'][field]=raw if raw=='max' else int(raw)
                except (OSError,ValueError) as exc:row['unavailable'][field]=type(exc).__name__
            result['cgroup_v2_ancestors'].append(row)
            if target==base:break
            target=target.parent
    except (OSError,ValueError) as exc:result['cgroup_error']=repr(exc)
    return result


class HostTreeObserver:
    def __init__(self,pid,proc=Path('/proc'),cgroup=Path('/sys/fs/cgroup')):
        if type(pid) is not int or pid<=0:raise ValueError('Explicit positive root PID required')
        self.proc,self.cgroup,self.pid=Path(proc),Path(cgroup),pid
        self.root_identity=identity(self.proc,pid)

    def sample(self):
        result=dict(monotonic_start_ns=time.monotonic_ns(),root_identity=self.root_identity,
            processes=[],errors=[],atomic_snapshot=False,continuous_peak_proof=False)
        before,errors=scan(self.proc);tree=descendants(before,self.pid)
        result['scan_errors']=errors
        if before.get(self.pid)!=self.root_identity:
            result.update(complete=False,termination='root_exited_or_identity_changed',totals_bytes=None)
            return result
        for pid,expected in tree.items():
            try:
                if identity(self.proc,pid)!=expected:raise ValueError('Process identity changed before read')
                memory=kb_fields((self.proc/str(pid)/'smaps_rollup').read_text(),FIELDS)
                status=kb_fields((self.proc/str(pid)/'status').read_text())
                if identity(self.proc,pid)!=expected:raise ValueError('Process identity changed during read')
                result['processes'].append(dict(identity=expected,smaps_rollup_bytes=memory,
                    status_bytes={key:status.get(key) for key in ('VmRSS','VmHWM','VmLck','VmPin','VmSwap')}))
            except (OSError,ValueError) as exc:
                result['errors'].append(dict(identity=expected,error_type=type(exc).__name__,error=str(exc)))
        after,after_errors=scan(self.proc)
        result['scan_errors']+=after_errors
        result['tree_identity_stable']=tree==descendants(after,self.pid) and after.get(self.pid)==self.root_identity
        result['complete']=bool(tree) and not result['errors'] and not result['scan_errors'] and result['tree_identity_stable']
        result['totals_bytes']={key:sum(row['smaps_rollup_bytes'][key] for row in result['processes']) for key in FIELDS}
        result['capacity']=capacity(self.proc,self.pid,self.cgroup)
        result['monotonic_end_ns']=time.monotonic_ns()
        return result


def collect(pid,output,interval=0.5,seconds=None):
    if interval<0.1 or (seconds is not None and seconds<=0):raise ValueError('Invalid observation duration/interval')
    output=Path(output);summary_path=output.with_suffix('.summary.json')
    if output.exists() or summary_path.exists():raise FileExistsError('Preserve existing evidence; choose a fresh output')
    observer=HostTreeObserver(pid);output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary=dict(formal_enabled=False,samples=0,complete_samples=0,partial_samples=0,
        max_complete_sample_totals_bytes={key:0 for key in FIELDS},source_sha256=source_sha,
        scope='Sequential Linux process-tree sampled accounting, not total-host/continuous peak/server capacity proof')
    try:
        with output.open('x') as stream:
            while True:
                row=observer.sample();stream.write(json.dumps(row)+'\n');stream.flush()
                summary['samples']+=1
                if row['complete']:
                    summary['complete_samples']+=1
                    for key in FIELDS:summary['max_complete_sample_totals_bytes'][key]=max(summary['max_complete_sample_totals_bytes'][key],row['totals_bytes'][key])
                else:summary['partial_samples']+=1
                if row.get('termination'):
                    summary['termination']=row['termination'];break
                if seconds is not None and time.monotonic()-started>=seconds:
                    summary['termination']='requested_observation_window_ended';break
                time.sleep(interval)
    except BaseException as exc:
        summary.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        summary['source_unchanged']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==source_sha
        summary['jsonl_sha256']=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
        with summary_path.open('x') as stream:json.dump(summary,stream,indent=2)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval',type=float,default=0.5);p.add_argument('--seconds',type=float);args=p.parse_args()
    result=collect(os.getpid() if args.pid=='self' else int(args.pid),args.output,args.interval,args.seconds)
    print(json.dumps(result,indent=2))
