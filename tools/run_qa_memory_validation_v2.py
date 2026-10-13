"""Server-only serial short/full QA validation; no PTQ or model copies."""
from __future__ import annotations
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_qa_score_lifetime_component import memory_mib
from tools.storage_process_device_peaks import valid_record


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''): digest.update(data)
    return digest.hexdigest()


def state_sha(model):
    import torch
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def install(side):
    from lm_eval.models.huggingface import HFLM
    from tools.qa_score_lifetime_prototype import build_method as scored_rows
    from tools.qa_score_no_kv_prototype import build_method as no_kv
    original=HFLM._loglikelihood_tokens
    model_call=HFLM._model_call
    if side!='control': HFLM._loglikelihood_tokens=scored_rows(original)
    if side=='rows_no_kv': HFLM._model_call=no_kv(model_call)
    return original,model_call


def worker(args):
    import torch
    import lm_eval
    from lm_eval import tasks
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from tools.storage_process_device_peaks import collect_peaks
    from utils.memory_utils import get_process_cuda_memory_peaks
    torch.manual_seed(1)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa').eval().to('cuda:0')
    before=state_sha(model)
    original,model_call=install(args.arm)
    forward_cache=[]
    def observe_forward(module,inputs,kwargs):
        forward_cache.append(bool(kwargs.get('use_cache',module.config.use_cache)))
    handle=model.register_forward_pre_hook(observe_forward,with_kwargs=True)
    input_digest=hashlib.sha256();output_digest=hashlib.sha256();counts=[]
    selected=HFLM._loglikelihood_tokens
    def observed(self,requests,*a,**kw):
        input_digest.update(json.dumps(requests,ensure_ascii=False).encode())
        result=selected(self,requests,*a,**kw)
        output_digest.update(json.dumps(result).encode())
        counts.append(len(result))
        return result
    HFLM._loglikelihood_tokens=observed
    hflm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=16,device='cuda:0')
    task_dir=ROOT/'datasets/lm_eval_configs/tasks'
    manager=tasks.TaskManager(include_path=str(task_dir),include_defaults=True) if task_dir.is_dir() else tasks.TaskManager(include_defaults=True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    if args.mode=='short':
        task=tasks.get_task_dict(['boolq'],task_manager=manager)['boolq']
        docs=sorted(list(task.validation_docs()),key=lambda doc:len(task.doc_to_text(doc)),reverse=True)[:32]
        requests=[]
        for doc in docs:
            ctx=task.fewshot_context(doc,num_fewshot=0)
            for request in task.construct_requests(doc,ctx):
                text,cont=request.args
                context_tokens,cont_tokens=hflm._encode_pair(text,cont)
                requests.append(((text,cont),context_tokens,cont_tokens))
        scores=hflm._loglikelihood_tokens(requests,disable_tqdm=True)
        metrics={'scores':scores,'passages':len(docs),'requests':len(requests)}
        task_names=['boolq-short']
    else:
        from utils.eval_utils import _resolve_paper_qa_tasks
        from lm_eval import utils as eval_utils
        task_names=_resolve_paper_qa_tasks(eval_utils.pattern_match,manager.all_tasks)
        metrics={}
        for name in task_names:
            print('QA task started: '+name,flush=True)
            hflm.batch_size_per_gpu=16
            metrics[name]=lm_eval.simple_evaluate(hflm,tasks=[name],task_manager=manager)['results']
            print('QA task completed: '+name,flush=True)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    memory=collect_peaks(torch.cuda,get_process_cuda_memory_peaks)
    handle.remove()
    after=state_sha(model)
    result=dict(arm=args.arm,mode=args.mode,model=args.model,batch_size=16,
        model_use_cache=model.config.use_cache,model_state_before=before,model_state_after=after,
        tasks=task_names,metrics=metrics,request_counts=counts,forward_use_cache=forward_cache,
        request_sha256=input_digest.hexdigest(),score_sha256=output_digest.hexdigest(),
        likelihood_source_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        model_call_source_sha256=hashlib.sha256(inspect.getsource(model_call).encode()).hexdigest(),
        seconds=elapsed,memory=memory,scope='original BF16 model QA; not REAL-Q quantized model or full PTQ')
    with (args.output/(args.arm+'.json')).open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','request_counts')},indent=2),flush=True)


def compare(root,bindings):
    control=json.loads((root/'control.json').read_text())
    reports={}
    for side in ('rows','rows_no_kv'):
        candidate=json.loads((root/(side+'.json')).read_text())
        checks={key:control[key]==candidate[key] for key in ('mode','model','batch_size','model_use_cache',
            'model_state_before','model_state_after','tasks','metrics','request_counts','request_sha256',
            'score_sha256','likelihood_source_sha256','model_call_source_sha256')}
        checks['unchanged_weights']=control['model_state_before']==control['model_state_after']==candidate['model_state_before']==candidate['model_state_after']
        checks['sources_unchanged']=all(sha(path)==digest for path,digest in bindings.items())
        checks['complete_device_peaks']=all(valid_record(x['memory'],1) for x in (control,candidate))
        for key in ('peak_allocated_bytes','peak_reserved_bytes'):
            checks[key]=candidate['memory'][key]<=control['memory'][key]
        peaks=[];counts=[]
        for arm in ('control',side):
            with (root/(arm+'_nvml.csv')).open() as stream:
                values=[memory_mib(row[1]) for row in csv.reader(stream) if len(row)>1]
            peaks.append(max(values));counts.append(len(values))
        checks['sampled_nvml']=peaks[1]<=peaks[0]
        checks['coverage']=bool(control['request_counts']) and all(n>0 for n in control['request_counts']) and len(control['tasks'])==(1 if control['mode']=='short' else 10)
        checks['actual_forward_coverage']=(bool(control['forward_use_cache']) and
            len(control['forward_use_cache'])==len(candidate['forward_use_cache']) and
            all(control['forward_use_cache']) and
            all(value==(side!='rows_no_kv') for value in candidate['forward_use_cache']))
        reports[side]=dict(valid=all(checks.values()),checks=checks,nvml_peak_mib=peaks,nvml_sample_count=counts,
            control=control['memory'],candidate=candidate['memory'],seconds=[control['seconds'],candidate['seconds']])
    evidence={path.name:sha(path) for path in root.iterdir() if path.is_file()}
    return dict(valid=all(x['valid'] for x in reports.values()),reports=reports,evidence_sha256=evidence,
        scope=control['scope'],nvml_is_sampled=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--mode',choices=('short','full'),default='short')
    parser.add_argument('--arm',choices=('control','rows','rows_no_kv'))
    args=parser.parse_args()
    if args.arm: return worker(args)
    args.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),ROOT/'tools/qa_score_lifetime_prototype.py',ROOT/'tools/qa_score_no_kv_prototype.py',
        ROOT/'tools/storage_process_device_peaks.py',ROOT/'tools/audit_qa_score_lifetime_component.py',
        ROOT/'utils/eval_utils.py',ROOT/'utils/memory_utils.py',Path(args.model)/'config.json']
    bindings={str(path.resolve()):sha(path) for path in paths}
    (args.output/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    for side in ('control','rows','rows_no_kv'):
        if any(sha(path)!=digest for path,digest in bindings.items()): raise RuntimeError('Bound source changed')
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if any(line.strip().isdigit() for line in pids.splitlines()): raise RuntimeError('Another GPU job is running')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_DATASETS_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        print(f'Starting {args.mode} {side}: {args.model}',flush=True)
        with (args.output/(side+'.log')).open('x') as log, (args.output/(side+'_nvml.csv')).open('x') as nvml:
            monitor=subprocess.Popen(['nvidia-smi',f'--id={args.gpu}','--query-gpu=timestamp,memory.used,index,uuid','--format=csv,noheader,nounits','--loop-ms=100'],stdout=nvml,stderr=subprocess.DEVNULL)
            try:
                subprocess.run([sys.executable,'-u',__file__,'--model',args.model,'--output',str(args.output),
                    '--mode',args.mode,'--arm',side],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            finally:
                monitor.terminate();monitor.wait(timeout=10)
    result=compare(args.output,bindings)
    with (args.output/'summary.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2),flush=True)
    if not result['valid']: raise SystemExit('Exactness or component capacity rejected')


if __name__=='__main__': main()
