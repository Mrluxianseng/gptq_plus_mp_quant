"""Track stable PID/start/boot identity across normal nohup reparenting."""
import inspect
from pathlib import Path
import textwrap
from tools import qa_host_memory_observer as raw
from tools import qa_host_memory_observer_v2 as previous


def identity_key(row):
    return None if row is None else (row['pid'],row['start_ticks'])


def sample_source():
    text=textwrap.dedent(inspect.getsource(raw.HostTreeObserver.sample))
    changes={
        'before.get(self.pid)!=self.root_identity':'_identity_key(before.get(self.pid))!=_identity_key(self.root_identity)',
        'after.get(self.pid)==self.root_identity':'_identity_key(after.get(self.pid))==_identity_key(self.root_identity)',
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError('Frozen process identity anchor changed')
        text=text.replace(old,new)
    return text


sample_scope=dict(raw.__dict__,_identity_key=identity_key)
exec(compile(sample_source(),'qa_host_pss_stable_identity_v3','exec'),sample_scope)


class HostTreeObserver(previous.HostTreeObserver):
    def sample(self):
        row=sample_scope['sample'](self)
        if self.boot_id is not None and self.read_boot()!=self.boot_id:
            row.update(complete=False,termination='boot_identity_changed')
        return row


def collect_source():
    text=previous.collect_source()
    changes={
        'ancestor_sha=_bounded_sha(_ancestor_file)':
        'ancestor_sha=_bounded_sha(_ancestor_file)\n    observer_sha=_bounded_sha(_observer_v2_file)',
        'summary=dict(formal_enabled=False,runtime_identity=observer.runtime,':
        'summary=dict(formal_enabled=False,observer_v2_source_sha256=observer_sha,runtime_identity=observer.runtime,',
        "summary['ancestor_source_unchanged']=_bounded_sha(_ancestor_file)==ancestor_sha":
        "summary['ancestor_source_unchanged']=_bounded_sha(_ancestor_file)==ancestor_sha\n        summary['observer_v2_source_unchanged']=_bounded_sha(_observer_v2_file)==observer_sha",
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError('Frozen host collector provenance anchor changed')
        text=text.replace(old,new)
    return text


scope=dict(previous.scope,__file__=__file__,HostTreeObserver=HostTreeObserver,
    _observer_v2_file=Path(previous.__file__))
exec(compile(collect_source(),'qa_host_pss_stable_collect_v3','exec'),scope)
collect=scope['collect']
bounded_sha=previous.bounded_sha
