"""CPU-only admission for reviewed exact-counter owners; not formal acceptance.

Never resets, synchronizes, queries CUDA or carries live tensors. Known native
and QA profilers must be inactive before/after this exclusive QA lease. Their
full source hashes must match an explicit controller binding manifest.
"""
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
from utils.memory_utils import CudaPhaseMemoryProfiler
from tools.qa_visible_device_phase_profile import VisibleDevicePhases

_lock=threading.Lock()
_owner=None


def required_sources():
    return {str(Path(path).resolve()) for path in (
        __file__,inspect.getfile(CudaPhaseMemoryProfiler),inspect.getfile(VisibleDevicePhases))}


def verify_sources(bindings):
    if not isinstance(bindings,dict) or not required_sources().issubset(bindings):
        raise RuntimeError('Explicit reviewed peak-owner source bindings required')
    for path in required_sources():
        expected=bindings[path]
        if not isinstance(expected,str) or len(expected)!=64 or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Bound peak-owner source changed: '+path)


def active_profiles():
    result=[]
    for obj in gc.get_objects():
        if isinstance(obj,CudaPhaseMemoryProfiler):
            if getattr(obj,'enabled',False) and getattr(obj,'peak_mode',None)=='exact' and getattr(obj,'phase',None) is not None:
                result.append(dict(kind='native-exact',object_id=id(obj),phase=str(obj.phase),
                                   path=str(getattr(obj,'path',None)),device=str(getattr(obj,'device',None))))
        elif isinstance(obj,VisibleDevicePhases) and not getattr(obj,'closed',True):
            result.append(dict(kind='qa-visible-exact',object_id=id(obj),phase=str(obj.phase),
                               path=str(obj.root),visible_device_count=int(obj.count)))
    return sorted(result,key=lambda row:row['object_id'])


class CounterAdmission:
    def __init__(self,destination,*,bindings,validation_only):
        if validation_only is not True:
            raise RuntimeError('Counter admission preparation is validation-only')
        self.path=Path(destination)
        if self.path.exists(): raise FileExistsError('Counter evidence exists; choose a fresh path')
        self.bindings=dict(bindings)
        self.entered=False
        self.record=dict(schema='qa-counter-admission-v1',formal_enabled=False,completed=False,
            scope='CPU known-profiler ownership/source check only; not full numerical/memory/NVML acceptance',
            process_id=os.getpid(),sources={path:self.bindings.get(path) for path in required_sources()})

    def write(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('x') as stream:json.dump(self.record,stream,indent=2,allow_nan=False)

    def __enter__(self):
        global _owner
        try:
            verify_sources(self.bindings)
            with _lock:
                if _owner is not None: raise RuntimeError('QA exact-counter lease already owned')
                active=active_profiles();self.record['entry_active_profiles']=active
                if active: raise RuntimeError('Existing exact profiler must finish before QA counter ownership')
                _owner=self;self.entered=True
            return self
        except BaseException as exc:
            self.record.update(error_type=type(exc).__name__,error=str(exc))
            self.write();raise

    def __exit__(self,kind,error,traceback):
        global _owner
        cleanup=None
        try:
            verify_sources(self.bindings)
            active=active_profiles();self.record['exit_active_profiles']=active
            if active: raise RuntimeError('QA exact profiler remains active after ownership handoff')
            self.record['completed']=error is None
        except BaseException as exc:
            cleanup=exc
            self.record.update(completed=False,cleanup_error_type=type(exc).__name__,cleanup_error=str(exc))
        finally:
            with _lock:
                if _owner is not self:
                    cleanup=cleanup or RuntimeError('QA exact-counter ownership changed')
                    self.record.update(completed=False,ownership_changed=True)
                else: _owner=None
            if error is not None:
                self.record.update(completed=False,error_type=type(error).__name__,error=str(error))
            self.entered=False
            self.write()
        if cleanup is not None and error is None: raise cleanup
        return False
