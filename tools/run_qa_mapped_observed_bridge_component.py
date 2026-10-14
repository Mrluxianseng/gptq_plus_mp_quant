"""Versioned native-control/full-observer/mapped placement compatibility fixture."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def source():
    text=(ROOT/'tools/run_qa_observed_bridge_component.py').read_text()
    changes={
        '    from tools.qa_observed_placement_bridge import ObservedPlacementBridge':
        '    from tools.qa_mapped_observed_placement_bridge import MappedObservedPlacementBridge as ObservedPlacementBridge\n    from tools.qa_mapped_head_storage_prototype import prepare_runtime',
        '    if torch.cuda.device_count()!=1:':'''    _mapped_library,_mapped_capabilities=prepare_runtime(os.environ['QA_MAPPED_HEAD_LIBRARY'])
    _mapped_allocator=(torch.cuda.memory.CUDAPluggableAllocator(os.environ['QA_MAPPED_HEAD_LIBRARY'],
        'qa_mapped_alloc','qa_mapped_free') if args.arm=='candidate' else None)
    if torch.cuda.device_count()!=1:''',
        '    scores=[];forward_info=[];cache_outputs=[]':
        "    requests.append((('natural-tail-unique','choice'),[11,12,13],[14]))\n    scores=[];forward_info=[];cache_outputs=[]",
        "        validation_only=True,exclusive_peak_owner=True)":
        "        validation_only=True,exclusive_peak_owner=True,mapped_enabled=args.arm=='candidate',\n        library=_mapped_library,allocator=_mapped_allocator)",
        "        parameter_identity_preserved=spec==identity,config_use_cache=model.config.use_cache,":
        "        mapped_capabilities=_mapped_capabilities,parameter_identity_preserved=spec==identity,config_use_cache=model.config.use_cache,",
        "scope='native-dispatch composed observer/storage bridge synthetic online-wrapper fixture; NOT REAL-Q/full QA'":
        "scope='native-control mapped/full-observer online-wrapper fixture with65requests/natural-tail; unrotatedbase/torch_fallback, NOT REAL-Q/full QA'",
        "    files=[Path(__file__),*map(Path,":
        "    (args.output/'worker_source.py').write_text(_generator())\n    files=[Path(__file__),args.output/'worker_source.py',ROOT/'tools/run_qa_observed_bridge_component.py',ROOT/'tools/qa_mapped_observed_placement_bridge.py',ROOT/'tools/qa_mapped_head_storage_prototype.py',ROOT/'tools/qa_mapped_head_batch_lifetime.py',ROOT/'tools/qa_mapped_head_natural_batch.py',ROOT/'tools/audit_qa_mapped_observed_bridge_completed.py',*map(Path,",
        "    (args.output/'source_bindings.json').write_text(json.dumps({str(p.resolve()):sha(p) for p in files},indent=2))":'''    _probe=ROOT/'outputs/qa-mapped-host-pool-probe-20261014-v1'
    _probe_status=json.loads((_probe/'status.json').read_text())
    _bindings=json.loads((_probe/'source_bindings.json').read_text())
    if not _probe_status['valid'] or not _probe_status['native_accounting_complete'] or not all(sha(p)==h for p,h in _bindings.items()):
        raise RuntimeError('Mapped runtime/source admission failed')
    _bindings.update({str(p.resolve()):sha(p) for p in files})
    (args.output/'source_bindings.json').write_text(json.dumps(_bindings,indent=2))''',
        "                worker_process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)":
        "                if not all(sha(p)==h for p,h in _bindings.items()): raise RuntimeError('Bound source changed')\n                worker_process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,QA_MAPPED_HEAD_LIBRARY=str(_probe/'mapped_host_probe.so')))",
        '    result=audit(args.output)':
        '    from tools.audit_qa_mapped_observed_bridge_completed import audit as _complete_audit\n    result=_complete_audit(args.output)',
    }
    for old,new in changes.items():
        if text.count(old)!=1: raise RuntimeError('Frozen native bridge source anchor changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_mapped_observed_bridge_fixture','exec')
    return text


def namespace():
    scope=dict(__name__='qa_mapped_observed_bridge_fixture',__file__=__file__,_generator=source)
    exec(compile(source(),'qa_mapped_observed_bridge_fixture','exec'),scope)
    return scope


if __name__=='__main__': namespace()['main']()
