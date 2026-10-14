import hashlib
import io
import json
from pathlib import Path
import types
import pytest
import torch
from utils import memory_utils
from tools.qa_visible_device_phase_profile import VisibleDevicePhases
from tools.qa_peak_counter_admission import CounterAdmission,required_sources
from tools.qa_mapped_observed_placement_bridge_v3 import MappedObservedPlacementBridge
from tools.qa_complete_observer import validate_observer_pair
from test_qa_observed_placement_bridge import fixture


def bindings():return {path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in required_sources()}


def native_metadata():
    profiler=memory_utils.CudaPhaseMemoryProfiler.__new__(memory_utils.CudaPhaseMemoryProfiler)
    profiler.enabled=True;profiler.peak_mode='exact';profiler.phase='original_quantization'
    profiler.path='native_cpu_fixture';profiler.device=torch.device('cuda:0')
    return profiler


def test_known_native_active_interval_refused_then_explicit_finish_admitted(tmp_path):
    native=native_metadata()
    with pytest.raises(RuntimeError,match='must finish'):
        with CounterAdmission(tmp_path/'refused.json',bindings=bindings(),validation_only=True): pass
    report=json.loads((tmp_path/'refused.json').read_text())
    assert not report['completed'] and report['entry_active_profiles'][0]['phase']=='original_quantization'
    native.phase=None # CPU metadata fixture of the native finish protocol, not a real CUDA run.
    with CounterAdmission(tmp_path/'admitted.json',bindings=bindings(),validation_only=True): pass
    assert json.loads((tmp_path/'admitted.json').read_text())['completed']


def test_nested_lease_source_change_and_formal_request_refused(tmp_path):
    with CounterAdmission(tmp_path/'outer.json',bindings=bindings(),validation_only=True):
        with pytest.raises(RuntimeError,match='already owned'):
            with CounterAdmission(tmp_path/'inner.json',bindings=bindings(),validation_only=True): pass
    bad=bindings();bad[next(iter(bad))]='0'*64
    with pytest.raises(RuntimeError,match='source changed'):
        with CounterAdmission(tmp_path/'source.json',bindings=bad,validation_only=True): pass
    with pytest.raises(RuntimeError,match='validation-only'):
        CounterAdmission(tmp_path/'formal.json',bindings=bindings(),validation_only=False)


def test_original_exception_preserved_and_active_exit_evidence(tmp_path):
    native=None
    with pytest.raises(ValueError,match='original eval error'):
        with CounterAdmission(tmp_path/'failed.json',bindings=bindings(),validation_only=True):
            native=native_metadata()
            raise ValueError('original eval error')
    report=json.loads((tmp_path/'failed.json').read_text())
    assert not report['completed'] and report['error_type']=='ValueError'
    assert report['cleanup_error_type']=='RuntimeError' and report['exit_active_profiles']
    native.phase=None
    with CounterAdmission(tmp_path/'later.json',bindings=bindings(),validation_only=True): pass


def test_active_qa_group_refused_without_cuda_queries(tmp_path):
    from test_qa_visible_device_phase_profile import FakeCUDA
    cuda=FakeCUDA()
    group=VisibleDevicePhases(cuda,tmp_path/'active',exclusive_peak_owner=True)
    queries=(list(cuda.synchronized),list(cuda.reset))
    with pytest.raises(RuntimeError,match='must finish'):
        with CounterAdmission(tmp_path/'group_refused.json',bindings=bindings(),validation_only=True): pass
    assert (cuda.synchronized,cuda.reset)==queries
    group.finish('completed')


def test_real_hflm_cpu_full_bytes_and_prior_peak_footer_with_exclusive_owner(tmp_path):
    from lm_eval.models.huggingface import HFLM
    original=(HFLM._model_call,HFLM._loglikelihood_tokens)
    for enabled,side in ((False,'control'),(True,'candidate')):
        model,tok,ev,dist,evaluation,memory,cuda,calls=fixture()
        bridge=MappedObservedPlacementBridge(ev,dist,evaluation,memory,cuda,enabled=enabled,mapped_enabled=False,
            library=None,destination=tmp_path/side,expected_tasks=('fixture_a','fixture_b'),
            validation_only=True,device='cpu',exclusive_peak_owner=True,counter_source_bindings=bindings())
        dist.distribute_model(model)
        assert ev.qa_eval(model,tok,16)==dict(fixture_a=0.5,fixture_b=0.5)
        report=json.loads((tmp_path/side/'qa_counter_owner_rank0.json').read_text())
        assert report['completed'] and report['entry_active_profiles']==report['exit_active_profiles']==[]
        assert not report['formal_enabled'] and memory.get_process_cuda_memory_peaks('cuda:3')==(5000,6000)
        bridge.restore();assert (HFLM._model_call,HFLM._loglikelihood_tokens)==original
    assert validate_observer_pair(tmp_path/'control/qa_complete',tmp_path/'candidate/qa_complete',['fixture_a','fixture_b'])['valid']
    assert not torch.cuda.is_initialized()
