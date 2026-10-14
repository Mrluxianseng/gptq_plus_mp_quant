"""Short exact whole-head mapped-output screen atop accepted QA storage.

Both sides keep complete GPU GEMM/batch16/BF16/KV/scoring. Base weights remain
unrotated online-wrapper fixtures. No formal QA/REAL-Q quality claim.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_complete_observer_component import source as previous_source


def source():
    text=previous_source()
    replacements={
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
        '        ticket.close();handle.remove();output_handle.remove();HFLM._model_call=original_call;HFLM._loglikelihood_tokens=original_score':
        '''        try:
            if _primary_error is None:
                profile.mark('mapped_head_pool_release')
                _head_record=_head_storage.close()
            else:
                # A failing forward traceback may still own an output. Do not
                # free its pool or replace the original error with cleanup.
                for _h in _head_storage.handles: _h.remove()
                _partial=dict(completed=False,records=_head_storage.records,
                    error_type=type(_primary_error).__name__,error=str(_primary_error),
                    mapped_host_live=_mapped_library.qa_live_bytes())
                with (args.output/'mapped_failure.json').open('x') as _f: json.dump(_partial,_f,indent=2)
        finally:
            ticket.close();handle.remove();output_handle.remove();HFLM._model_call=original_call;HFLM._loglikelihood_tokens=original_score''',
        'weight_records=ticket.records,':
        'mapped_head=_head_record,mapped_attributes=_mapped_attributes,weight_records=ticket.records,',
        '        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())':
        '        from tools.qa_tensor_fingerprint_stream import update_tensor_bytes\n        update_tensor_bytes(digest,value)',
        "        row['digest'].update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())":
        "        from tools.qa_tensor_fingerprint_stream import update_tensor_bytes\n        update_tensor_bytes(row['digest'],value)",
        '        value=result.detach().cpu().contiguous()':'        value=result.detach()',
        "sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()":
        "sha256=__import__('tools.qa_tensor_fingerprint_stream',fromlist=['tensor_sha256']).tensor_sha256(value)",
    }
    for old,new in replacements.items():
        if text.count(old)!=1: raise RuntimeError('Frozen mapped-head source anchor changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_mapped_head_screen','exec')
    return text


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def namespace():
    data=dict(__name__='qa_mapped_component',__file__=str(Path(__file__).resolve()))
    exec(compile(source(),'qa_mapped_head_screen','exec'),data)
    return data


def audit(root):
    from tools.qa_complete_observer import validate_observer_pair
    from tools.audit_memory_boundary_points import boundaries,compare
    from tools.audit_completed_memory_pair_capacity import intervals
    from tools.audit_qa_layerwise_completed import coverage
    from tools.storage_process_device_peaks import valid_record
    bindings=json.loads((root/'source_bindings.json').read_text())
    c,n=[json.loads((root/side/'layerwise_cache.json').read_text()) for side in ('control','candidate')]
    checks={key:c[key]==n[key] for key in ('model','batch_size','max_context_tokens','max_continuation_tokens',
        'before','after','scores','request_sha256','logits','layers','weight_records','forward_info','cache_outputs',
        'parameter_identity_preserved','model_use_cache','likelihood_source_sha256','model_call_source_sha256','hadamard_backend')}
    checks.update(states_unchanged=c['before']==c['after']==n['before']==n['after'],
        full_535=c['before']['count']==535,sources=all(sha(p)==h for p,h in bindings.items()),
        batch_bf16=c['batch_size']==16 and all(row['dtype']=='torch.bfloat16' for row in c['logits']),
        complete_peaks=valid_record(c['memory'],1) and valid_record(n['memory'],1))
    extra={key:value for key,value in coverage({'control':c,'layerwise_cache':n}).items()
           if key.startswith('layerwise_cache_')}
    # The baseline deliberately has the same accepted layerwise/cache method.
    # Keep the legacy checker unchanged; its all-resident control gates do not
    # apply here. Exact paired records plus the candidate gates cover both.
    checks['same_best_method_coverage']=all(extra.values()) and checks['weight_records'] and checks['forward_info'] and checks['cache_outputs']
    numerical=validate_observer_pair(root/'control/layerwise_cache_complete',root/'candidate/layerwise_cache_complete',['synthetic_component'])
    checks['new_complete_observer']=numerical['valid']
    heads=[r['mapped_head'] for r in (c,n)];calls=len(c['logits'])
    checks['head_coverage']=(heads[0]['enabled'] is False and heads[1]['enabled'] is True
        and all(h['calls']==len(h['records'])==calls and all(r['completed'] for r in h['records']) for h in heads)
        and all(r['enabled'] and r['accounting_complete'] for r in heads[1]['records']))
    checks['head_original_forward']=heads[0]['original_forward_sha256']==heads[1]['original_forward_sha256']
    checks['pool_release']=(heads[1]['released_active']==0 and heads[1]['final_host_live']==0
        and heads[1]['allocations']==heads[1]['releases'] and heads[1]['allocations']>0)
    checks['mapping_capabilities']=c['mapped_attributes']==n['mapped_attributes'] and all(v==1 for v in n['mapped_attributes'].values())
    paths=[[Path(p) for p in row['memory_profile_files']] for row in (c,n)]
    tables=[intervals(p) for p in paths];point=compare(*(boundaries(p) for p in paths))
    checks['interval_complete']=bool(tables[0]) and tables[0].keys()==tables[1].keys()
    checks['point_complete']=bool(point['matched_values']) and not point['missing']
    checks['exact_all_intervals']=all(row['peak_measurement_mode']=='exact' for table in tables for row in table.values())
    regressions=[dict(key=list(key),metric=metric,control=tables[0][key]['sampled_peak_'+metric+'_bytes'],
        candidate=tables[1][key]['sampled_peak_'+metric+'_bytes']) for key in tables[0].keys()&tables[1].keys()
        for metric in ('allocated','reserved') if tables[1][key]['sampled_peak_'+metric+'_bytes']>tables[0][key]['sampled_peak_'+metric+'_bytes']]
    for metric in ('allocated','reserved'): checks['global_'+metric]=n['memory']['peak_'+metric+'_bytes']<=c['memory']['peak_'+metric+'_bytes']
    nvml=[];ids=[]
    for side in ('control','candidate'):
        samples=list(csv.reader((root/(side+'_nvml.csv')).open()))
        nvml.append(max(float(row[1].strip().split()[0]) for row in samples));ids.append({row[3].strip() for row in samples})
    checks['nvml_global']=nvml[1]<=nvml[0] and len(ids[0])==1 and ids[0]==ids[1]
    return dict(valid=all(checks.values()),checks=checks,numerical=numerical,coverage=extra,
        memory=[c['memory'],n['memory']],nvml_peak_mib=nvml,interval_count=len(tables[0]),
        points=point,interval_regressions=regressions,strict_local_nonincrease=not regressions and point['valid'],
        native_cuda_counters_include_mapped_host=True,
        scope='short unrotated-base online-wrapper synthetic QA component; not formal REAL-Q or full QA',nvml_is_sampled=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('--output',type=Path)
    parser.add_argument('--side',choices=('control','candidate'));parser.add_argument('--audit-root',type=Path)
    parser.add_argument('--audit-output',type=Path);args=parser.parse_args()
    if args.audit_root:
        result=audit(args.audit_root)
        with args.audit_output.open('x') as stream: json.dump(result,stream,indent=2)
        print(json.dumps(result,indent=2))
        if not result['valid']: raise SystemExit(1)
        return
    if not args.model or args.output is None: parser.error('--model/--output required')
    if args.side:
        os.environ['QA_MAPPED_HEAD_ENABLED']='1' if args.side=='candidate' else '0'
        data=namespace()
        args.arm='layerwise_cache';args.length=128;args.output=args.output/args.side
        return data['worker'](args)
    args.output.mkdir(parents=True,exist_ok=False)
    probe=ROOT/'outputs/qa-mapped-host-pool-probe-20261014-v1'
    status=json.loads((probe/'status.json').read_text());probe_bindings=json.loads((probe/'source_bindings.json').read_text())
    if not status['valid'] or not status['native_accounting_complete'] or not all(sha(p)==h for p,h in probe_bindings.items()):
        raise RuntimeError('Mapped runtime/source admission rejected')
    binary=probe/'mapped_host_probe.so'
    files=[Path(__file__),ROOT/'tools/qa_mapped_head_storage_prototype.py',ROOT/'tools/run_qa_complete_observer_component.py',
        ROOT/'tools/run_qa_rotation_placement_component_v2.py',ROOT/'tools/run_qa_rotation_placement_component.py',
        ROOT/'tools/run_qa_layerwise_component_v2.py',ROOT/'tools/run_qa_layerwise_component.py',
        ROOT/'tools/qa_complete_observer.py',ROOT/'tools/qa_visible_device_phase_profile.py',ROOT/'tools/qa_tensor_fingerprint_stream.py',
        ROOT/'tools/qa_layerwise_weight_prototype.py',ROOT/'tools/qa_score_lifetime_prototype.py',ROOT/'tools/qa_scoring_cache_offload_prototype.py',
        ROOT/'utils/memory_utils.py',ROOT/'utils/rotation_utils.py',ROOT/'utils/quant_utils.py',ROOT/'utils/model_utils.py',ROOT/'utils/hadamard_utils.py',ROOT/'utils/had_mat.pt']
    from lm_eval.models.huggingface import HFLM
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3 import modeling_qwen3
    import inspect
    files += [Path(inspect.getfile(HFLM)),Path(inspect.getfile(DynamicCache)),Path(modeling_qwen3.__file__)]
    files += [p for p in Path(args.model).iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.txt')]
    bindings=dict(probe_bindings,**{str(p.resolve()):sha(p) for p in files})
    (args.output/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    env=dict(os.environ,QA_MAPPED_HEAD_LIBRARY=str(binary),CUDA_VISIBLE_DEVICES='0')
    for side in ('control','candidate'):
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Another GPU worker is active')
        (args.output/side).mkdir()
        with (args.output/(side+'_nvml.csv')).open('x') as samples,(args.output/(side+'.log')).open('x') as log:
            sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,index,uuid','--format=csv,noheader','-lms','100'],stdout=samples)
            try:
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--model',args.model,'--output',str(args.output),'--side',side]
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            finally:
                sampler.terminate();sampler.wait()
            print(side,'exit',result.returncode,flush=True)
            if result.returncode: raise RuntimeError('Worker failed; source/log/partial evidence retained')
    result=audit(args.output)
    with (args.output/'status.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
    if not result['valid']: raise RuntimeError('Completed mapped-head screen rejected')


if __name__=='__main__': main()
