"""Bounded local real-model QA component, batch16 and original BF16/SDPA/KV.

Synthetic token requests test execution exactness, not benchmark accuracy.
Full model/logits/layer fingerprints are CPU digests; no model copies saved.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_qa_memory_rows_completed import sha
from tools.audit_qa_score_lifetime_component import memory_mib
from tools.storage_method_log_stream import LiveFileRelay

SIDES=('control','rows','layerwise','layerwise_cache')


def state(model):
    import torch
    digest=hashlib.sha256();layout={}
    for name,value in model.state_dict().items():
        layout[name]=[list(value.shape),str(value.dtype)]
        digest.update(name.encode());digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(sha256=digest.hexdigest(),layout=layout,count=len(layout))


def worker(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from tools.qa_layerwise_weight_prototype import LayerwiseWeights
    from tools.qa_scoring_cache_offload_prototype import build_method as cache_method
    from tools.qa_score_lifetime_prototype import build_method as rows_method
    from tools.storage_process_device_peaks import collect_peaks
    from utils.memory_utils import CudaPhaseMemoryProfiler,get_process_cuda_memory_peaks
    torch.manual_seed(1)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    before=state(model)
    profile=CudaPhaseMemoryProfiler(str(args.output/(args.arm+'_stage2/memory_profile')),'cuda:0',tag='evaluation')
    profile.mark('runtime_install')
    layers=[dict(digest=hashlib.sha256(),count=0,layout=[]) for _ in model.model.layers]
    def capture(index,output):
        value=output[0] if isinstance(output,(tuple,list)) else output
        if not isinstance(value,torch.Tensor): raise RuntimeError('Missing decoder output')
        row=layers[index];row['count']+=1;row['layout'].append([list(value.shape),str(value.dtype),list(value.stride())])
        row['digest'].update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    ticket=LayerwiseWeights(model,enabled=args.arm in ('layerwise','layerwise_cache'),device='cuda:0',
        mark=profile.mark,capture=capture)
    original_score=HFLM._loglikelihood_tokens;original_call=HFLM._model_call
    if args.arm!='control': HFLM._loglikelihood_tokens=rows_method(original_score)
    if args.arm=='layerwise_cache': HFLM._model_call=cache_method(original_call)
    selected_call=HFLM._model_call;logits=[];forward_info=[];cache_outputs=[]
    def observe_forward(module,inputs,kwargs):
        cache=kwargs.get('past_key_values')
        forward_info.append(dict(use_cache=bool(kwargs.get('use_cache',module.config.use_cache)),
            explicit_cache=cache is not None,offloading=bool(getattr(cache,'offloading',False))))
    handle=model.register_forward_pre_hook(observe_forward,with_kwargs=True)
    def observe_output(module,inputs,output):
        cache=output.past_key_values
        if cache is None: raise RuntimeError('Original KV semantics were not preserved')
        cache_outputs.append([dict(device=str(layer.keys.device),shape=list(layer.keys.shape),
            dtype=str(layer.keys.dtype),stride=list(layer.keys.stride())) for layer in cache.layers])
    output_handle=model.register_forward_hook(observe_output)
    def measured_call(self,*a,**kw):
        index=len(logits)+1;profile.mark(f'forward{index}_model_start')
        result=selected_call(self,*a,**kw)
        profile.mark(f'forward{index}_logits_ready')
        value=result.detach().cpu().contiguous()
        logits.append(dict(shape=list(value.shape),dtype=str(value.dtype),stride=list(result.stride()),
            sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()))
        profile.mark(f'forward{index}_scoring')
        return result
    HFLM._model_call=measured_call
    hflm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=16,device='cuda:0')
    rng=random.Random(20261014);requests=[]
    for index in range(32):
        context=[rng.randrange(32,model.config.vocab_size-1000) for _ in range(rng.randint(args.length//2,args.length))]
        for choice in range(2):
            cont=[rng.randrange(32,model.config.vocab_size-1000) for _ in range(1 if index%2==0 else 4)]
            requests.append(((f'component-context-{index}',f'choice-{choice}'),context,cont))
    profile.mark('qa_scoring_start');start=time.monotonic()
    try:
        scores=hflm._loglikelihood_tokens(requests,disable_tqdm=True)
        profile.mark('qa_scoring_finished')
        elapsed=time.monotonic()-start
    except BaseException:
        profile.finish(status='failed');raise
    finally:
        ticket.close();handle.remove();output_handle.remove();HFLM._model_call=original_call;HFLM._loglikelihood_tokens=original_score
    profile.finish(status='completed')
    memory=collect_peaks(torch.cuda,get_process_cuda_memory_peaks)
    after=state(model)
    result=dict(arm=args.arm,scope='bounded synthetic-token real Qwen3 scoring; not full benchmark/PTQ',
        model=args.model,batch_size=16,max_context_tokens=args.length,max_continuation_tokens=4,
        before=before,after=after,scores=scores,
        request_sha256=hashlib.sha256(json.dumps(requests).encode()).hexdigest(),
        logits=logits,layers=[dict(layer=i,count=x['count'],layout=x['layout'],sha256=x['digest'].hexdigest()) for i,x in enumerate(layers)],
        weight_records=ticket.records,forward_info=forward_info,cache_outputs=cache_outputs,parameter_identity_preserved=True,
        model_use_cache=model.config.use_cache,
        likelihood_source_sha256=hashlib.sha256(inspect.getsource(original_score).encode()).hexdigest(),
        model_call_source_sha256=hashlib.sha256(inspect.getsource(original_call).encode()).hexdigest(),
        memory=memory,memory_profile_files=[profile.path],seconds=elapsed)
    with (args.output/(args.arm+'.json')).open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k in ('arm','scope','memory','seconds','model_use_cache')},indent=2),flush=True)


def audit(root,bindings):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from tools.audit_memory_boundary_points import boundaries,compare as points
    from tools.audit_completed_memory_pair_capacity import intervals
    from tools.storage_process_device_peaks import valid_record
    c=json.loads((root/'control.json').read_text());reports={}
    with torch.device('meta'):
        expected=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(c['model'],local_files_only=True),torch_dtype=torch.bfloat16)
    expected_layout={k:[list(v.shape),str(v.dtype)] for k,v in expected.state_dict().items()}
    for side in SIDES[1:]:
        n=json.loads((root/(side+'.json')).read_text())
        checks={k:c[k]==n[k] for k in ('model','batch_size','max_context_tokens','max_continuation_tokens','before','after',
            'scores','request_sha256','logits','layers','model_use_cache','likelihood_source_sha256','model_call_source_sha256')}
        checks['states_unchanged']=c['before']==c['after']==n['before']==n['after']
        checks['complete_model_layout']=c['before']['layout']==expected_layout and c['before']['count']==len(expected_layout)
        calls=len(c['logits'])
        checks['complete_28_nonempty_layers']=len(c['layers'])==28 and all(x['count']==calls and calls>0 for x in c['layers'])
        checks['method_coverage']=len(n['weight_records'])==calls*28 and all(x['enabled']==(side!='rows') for x in n['weight_records'])
        checks['original_cache_semantics']=len(n['forward_info'])==calls and all(x['use_cache'] and x['offloading']==(side=='layerwise_cache') for x in n['forward_info'])
        checks['actual_cache_device_coverage']=(len(n['cache_outputs'])==calls and
            all(len(output)==28 and all(layer['device']==('cpu' if side=='layerwise_cache' else 'cuda:0') for layer in output) for output in n['cache_outputs']))
        checks['identity']=c['parameter_identity_preserved'] and n['parameter_identity_preserved']
        checks['sources_model_files_unchanged']=all(sha(p)==h for p,h in bindings.items())
        checks['complete_device_peaks']=valid_record(c['memory'],1) and valid_record(n['memory'],1)
        peaks=[]
        for arm in ('control',side):
            with (root/(arm+'_nvml.csv')).open() as stream: rows=list(csv.reader(stream))
            peaks.append(max(memory_mib(row[1]) for row in rows))
        checks['global_nvml']=peaks[1]<=peaks[0]
        for key in ('peak_allocated_bytes','peak_reserved_bytes'): checks[key]=n['memory'][key]<=c['memory'][key]
        boundary=points(boundaries(c['memory_profile_files']),boundaries(n['memory_profile_files']))
        ic,in_=intervals([Path(p) for p in c['memory_profile_files']]),intervals([Path(p) for p in n['memory_profile_files']])
        checks['intervals_complete']=bool(ic) and ic.keys()==in_.keys()
        checks['points_complete']=bool(boundary['matched_values']) and not boundary['missing']
        checks['exact_interval_peaks']=all(r.get('peak_measurement_mode')=='exact' for table in (ic,in_) for r in table.values())
        regressions=[dict(key=list(k),control=ic[k],candidate=in_[k]) for k in ic.keys()&in_.keys() if any(in_[k]['sampled_peak_'+m+'_bytes']>ic[k]['sampled_peak_'+m+'_bytes'] for m in ('allocated','reserved'))]
        reports[side]=dict(valid=all(checks.values()),checks=checks,nvml_peak_mib=peaks,
            control=c['memory'],candidate=n['memory'],interval_count=len(ic),interval_regressions=regressions,
            points=boundary,strict_local_nonincrease=not regressions and boundary['valid'])
    return dict(valid=all(x['valid'] for x in reports.values()),reports=reports,scope=c['scope'],nvml_is_sampled=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length',type=int,choices=(128,256),default=128);p.add_argument('--arm',choices=SIDES)
    args=p.parse_args()
    if args.arm: return worker(args)
    args.output.mkdir(parents=True,exist_ok=False)
    source=[Path(__file__),ROOT/'tools/qa_layerwise_weight_prototype.py',ROOT/'tools/qa_scoring_cache_offload_prototype.py',
        ROOT/'tools/qa_score_lifetime_prototype.py',ROOT/'tools/storage_process_device_peaks.py',ROOT/'tools/audit_memory_boundary_points.py',
        ROOT/'tools/audit_completed_memory_pair_capacity.py',ROOT/'tools/storage_method_log_stream.py',ROOT/'utils/memory_utils.py',
        ROOT/'tools/audit_qa_memory_rows_completed.py',ROOT/'tools/audit_qa_score_lifetime_component.py']
    source += [x for x in Path(args.model).iterdir() if x.is_file() and x.suffix in ('.json','.safetensors','.txt')]
    bindings={str(x.resolve()):sha(x) for x in source}
    (args.output/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    for side in SIDES:
        if any(sha(path)!=h for path,h in bindings.items()): raise RuntimeError('Bound source/model changed')
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if any(x.strip().isdigit() for x in pids.splitlines()): raise RuntimeError('Another GPU compute process is active')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',REALQ_PHASE_PEAK_MODE='exact',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        print('Starting '+side+' '+args.model,flush=True)
        with (args.output/(side+'.log')).open('x') as log,(args.output/(side+'_nvml.csv')).open('x') as nvml:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,index,uuid','--format=csv,noheader,nounits','--loop-ms=100'],stdout=nvml,stderr=subprocess.DEVNULL)
            try:
                command=[sys.executable,'-u',__file__,'--model',args.model,'--output',str(args.output),'--length',str(args.length),'--arm',side]
                proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                relay=LiveFileRelay(args.output/(side+'.log'))
                while proc.poll() is None: relay.forward();time.sleep(0.5)
                relay.forward()
                if proc.returncode: raise subprocess.CalledProcessError(proc.returncode,command)
            finally: monitor.terminate();monitor.wait(timeout=10)
    result=audit(args.output,bindings)
    with (args.output/'summary.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({side:dict(valid=r['valid'],checks=r['checks'],nvml_peak_mib=r['nvml_peak_mib']) for side,r in result['reports'].items()},indent=2))
    if not result['valid']: raise SystemExit('One or more component methods rejected; preserve all evidence')


if __name__=='__main__': main()
