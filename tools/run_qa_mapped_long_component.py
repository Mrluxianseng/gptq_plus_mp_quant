"""Fresh paired raw-model long QA output-placement component, not formal QA."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_long_stream_fingerprint import source as previous_source
from tools.audit_qa_memory_rows_completed import sha
from tools.storage_method_log_stream import LiveFileRelay


def source():
    text=previous_source()
    changes={
        '    torch.manual_seed(1)':'''    from tools.qa_mapped_head_storage_prototype import prepare_runtime,MappedHeadStorage
    _mapped_library,_mapped_attributes=prepare_runtime(os.environ['QA_MAPPED_HEAD_LIBRARY'])
    _mapped_enabled=os.environ.get('QA_MAPPED_HEAD_ENABLED')=='1'
    torch.manual_seed(1)''',
        "    profile.mark('runtime_install')":'''    profile.mark('runtime_install')
    _mapped_pool=None
    _mapped_allocator=None
    if _mapped_enabled:
        _mapped_allocator=torch.cuda.memory.CUDAPluggableAllocator(os.environ['QA_MAPPED_HEAD_LIBRARY'],'qa_mapped_alloc','qa_mapped_free')
        _mapped_pool=torch.cuda.MemPool(_mapped_allocator.allocator())
    _head_storage=MappedHeadStorage(model.lm_head,enabled=_mapped_enabled,
        expected_shape=(model.config.vocab_size,model.config.hidden_size),library=_mapped_library,
        pool=_mapped_pool,mark=profile.mark)
    del _mapped_pool
    _head_record=None
    _primary_error=None''',
        "    except BaseException:\n        profile.finish(status='failed');raise":
        "    except BaseException as _error:\n        _primary_error=_error\n        profile.finish(status='failed');raise",
        '        ticket.close();handle.remove();output_handle.remove();HFLM._model_call=original_call;HFLM._loglikelihood_tokens=original_score':'''        try:
            if _primary_error is None:
                profile.mark('mapped_head_pool_release')
                _head_record=_head_storage.close()
                _head_record['peak_callback_host_bytes']=int(_mapped_library.qa_peak_bytes())
            else:
                for _h in _head_storage.handles: _h.remove()
                _partial=dict(completed=False,records=_head_storage.records,
                    error_type=type(_primary_error).__name__,error=str(_primary_error),
                    mapped_host_live=int(_mapped_library.qa_live_bytes()),
                    mapped_host_peak=int(_mapped_library.qa_peak_bytes()),
                    allocator_error=int(_mapped_library.qa_last_error()))
                with (args.output/'mapped_failure.json').open('x') as _f: json.dump(_partial,_f,indent=2)
        finally:
            ticket.close();handle.remove();output_handle.remove();HFLM._model_call=original_call;HFLM._loglikelihood_tokens=original_score''',
        '        weight_records=ticket.records,':
        '        mapped_head=_head_record,mapped_attributes=_mapped_attributes,weight_records=ticket.records,',
    }
    for old,new in changes.items():
        if text.count(old)!=1: raise RuntimeError('Frozen raw long source anchor changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_mapped_long_component','exec')
    return text


def namespace():
    data=dict(__name__='qa_mapped_long_worker',__file__=str(Path(__file__).resolve()))
    exec(compile(source(),'qa_mapped_long_component','exec'),data)
    return data


def expected_phases(calls):
    phases={'gptq_fwrd_setup','runtime_install','qa_scoring_start','qa_scoring_finished','mapped_head_pool_release'}
    for call in range(1,calls+1):
        phases.update(f'forward{call}_{suffix}' for suffix in ('model_start','logits_ready','scoring'))
        phases.update(f'forward{call}_layer{i}_{suffix}' for i in range(28)
            for suffix in ('load_start','compute_start','offload_start','end'))
        phases.update(f'head{call}_storage_{suffix}' for suffix in ('start','end'))
    return phases


NUMERICAL_KEYS=('model','batch_size','max_context_tokens','max_continuation_tokens','before','after',
    'scores','request_sha256','logits','layers','weight_records','forward_info','cache_outputs',
    'parameter_identity_preserved','model_use_cache','likelihood_source_sha256','model_call_source_sha256')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length',type=int,choices=(1024,1270),default=1270)
    p.add_argument('--reference',type=Path);p.add_argument('--side',choices=('control','candidate'))
    args=p.parse_args()
    if args.side:
        args.arm='layerwise_cache';args.output=args.output/args.side
        os.environ['QA_MAPPED_HEAD_ENABLED']='1' if args.side=='candidate' else '0'
        return namespace()['worker'](args)
    if args.reference is None: p.error('--reference required for fresh independent admission')
    args.output.mkdir(parents=True,exist_ok=False)
    reference=json.loads((args.reference/'layerwise_cache.json').read_text())
    if reference['model']!=args.model or reference['max_context_tokens']!=args.length:
        raise RuntimeError('Reference workload/model mismatch')
    with (args.output/'reference_readmission.log').open('x') as log:
        result=subprocess.run([sys.executable,'tools/audit_qa_long_completed.py','--root',str(args.reference),
            '--output',str(args.output/'reference_readmission.json')],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Fresh raw long reference rejected')
    probe=ROOT/'outputs/qa-mapped-host-pool-probe-20261014-v1'
    status=json.loads((probe/'status.json').read_text())
    bindings=json.loads((args.reference/'source_bindings.json').read_text())
    bindings.update(json.loads((probe/'source_bindings.json').read_text()))
    if not status['valid'] or not status['native_accounting_complete']:
        raise RuntimeError('Mapped runtime probe absent/rejected')
    files=[Path(__file__),ROOT/'tools/audit_qa_mapped_long_completed.py',
        ROOT/'tools/qa_mapped_head_storage_prototype.py',ROOT/'tools/run_qa_long_stream_fingerprint.py',
        ROOT/'tools/audit_qa_long_completed.py',ROOT/'tools/audit_qa_layerwise_completed.py',
        ROOT/'tools/audit_qa_layerwise_prefetch.py']
    bindings.update({str(path.resolve()):sha(path) for path in files})
    if not all(sha(path)==digest for path,digest in bindings.items()):
        raise RuntimeError('Admission source/model/library changed')
    (args.output/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    (args.output/'worker_source.py').write_text(source())
    bindings[str((args.output/'worker_source.py').resolve())]=sha(args.output/'worker_source.py')
    (args.output/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    (args.output/'admission.json').write_text(json.dumps(dict(reference=str(args.reference.resolve()),
        reference_record_sha256=sha(args.reference/'layerwise_cache.json'),
        reference_readmission_sha256=sha(args.output/'reference_readmission.json'),
        same_observer_map_flag_and_phase_identity=True,formal_enabled=False),indent=2))
    env=dict(os.environ,QA_MAPPED_HEAD_LIBRARY=str(probe/'mapped_host_probe.so'),CUDA_VISIBLE_DEVICES='0',
        REALQ_PHASE_PEAK_MODE='exact',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for side in ('control','candidate'):
        if not all(sha(path)==digest for path,digest in bindings.items()): raise RuntimeError('Bound source changed')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Another GPU worker is active')
        (args.output/side).mkdir()
        with (args.output/(side+'_nvml.csv')).open('x') as samples,(args.output/(side+'.log')).open('x') as log:
            sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,index,uuid','--format=csv,noheader','-lms','100'],stdout=samples)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--model',args.model,'--output',str(args.output),
                '--length',str(args.length),'--side',side]
            maximum=0;count=0;worker=None
            try:
                worker=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                relay=LiveFileRelay(args.output/(side+'.log'))
                while worker.poll() is None:
                    relay.forward()
                    try:
                        rss=[line for line in Path('/proc/'+str(worker.pid)+'/status').read_text().splitlines() if line.startswith('VmRSS:')]
                        if rss: maximum=max(maximum,int(rss[0].split()[1])*1024);count+=1
                    except FileNotFoundError: pass
                    time.sleep(0.5)
                relay.forward()
            finally:
                sampler.terminate();sampler.wait(timeout=10)
                with (args.output/(side+'_host_rss.json')).open('x') as stream:
                    json.dump(dict(pid=None if worker is None else worker.pid,max_sampled_worker_rss_bytes=maximum,
                        sample_count=count,sample_interval_seconds=0.5,
                        scope='single-worker sampled VmRSS, includes load and hashing, not total host/PSS or continuous peak'),stream,indent=2)
            print(side,'exit',worker.returncode,flush=True)
            if worker.returncode: raise RuntimeError('Worker failed; original source/log/partial evidence preserved')
    from tools.audit_qa_mapped_long_completed import audit
    report=audit(args.output)
    with (args.output/'status.json').open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(dict(valid=report['valid'],checks=report['checks'],nvml=report['nvml_peak_mib']),indent=2))
    if not report['valid']: raise RuntimeError('Complete raw long mapped component rejected')


if __name__=='__main__': main()
