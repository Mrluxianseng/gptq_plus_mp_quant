"""Bounded telemetry-file hashing and explicit Linux runtime identity.

Keep v1 and its CPU evidence untouched. No CUDA/model/data imports or runtime
QA enablement. Collected PSS is still sampled/sequential process accounting.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from tools import qa_host_memory_observer as ancestor


def bounded_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class HostTreeObserver(ancestor.HostTreeObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.boot_id=self.read_boot()
        self.runtime=dict(boot_id=self.boot_id,uname=dict(zip(
            ('sysname','nodename','release','version','machine'),os.uname())),
            proc_root=str(self.proc),cgroup_root=str(self.cgroup))

    def read_boot(self):
        try:return (self.proc/'sys/kernel/random/boot_id').read_text().strip()
        except OSError:return None

    def sample(self):
        row=super().sample()
        if self.boot_id is not None and self.read_boot()!=self.boot_id:
            row.update(complete=False,termination='boot_identity_changed')
        return row


def collect_source():
    text=inspect.getsource(ancestor.collect)
    changes={
        "source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()":
        "source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()\n    ancestor_sha=_bounded_sha(_ancestor_file)",
        'summary=dict(formal_enabled=False,samples=0,complete_samples=0,partial_samples=0,':
        'summary=dict(formal_enabled=False,runtime_identity=observer.runtime,ancestor_source_sha256=ancestor_sha,samples=0,complete_samples=0,partial_samples=0,',
        "        summary['jsonl_sha256']=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None":
        "        summary['jsonl_sha256']=_bounded_sha(output) if output.exists() else None\n        summary['ancestor_source_unchanged']=_bounded_sha(_ancestor_file)==ancestor_sha",
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError('Frozen host observer anchor changed')
        text=text.replace(old,new)
    compile(text,'qa_host_pss_observer_v2','exec')
    return text


scope=dict(ancestor.__dict__,__file__=__file__,HostTreeObserver=HostTreeObserver,
    _bounded_sha=bounded_sha,_ancestor_file=Path(ancestor.__file__))
exec(compile(collect_source(),'qa_host_pss_observer_v2','exec'),scope)
collect=scope['collect']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval',type=float,default=0.5);p.add_argument('--seconds',type=float);args=p.parse_args()
    result=collect(os.getpid() if args.pid=='self' else int(args.pid),args.output,args.interval,args.seconds)
    print(json.dumps(result,indent=2))
