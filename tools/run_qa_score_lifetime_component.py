"""Serial short GPU BoolQ scoring test on the original model, not full PTQ.

Source models and datasets remain read-only; no prepared copy or new calibration.
Tests 32 validation passages with longest text, batch16, both answer choices, original
library request/cache implementation. Paired scoring must match exactly.
"""
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


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def worker(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from lm_eval import tasks
    from lm_eval.models.huggingface import HFLM
    from tools.qa_score_lifetime_prototype import build_method
    from tools.storage_process_device_peaks import collect_peaks
    from utils.memory_utils import get_process_cuda_memory_peaks
    torch.manual_seed(1)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').eval().to('cuda:0')
    hflm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=16,device='cuda:0')
    manager=tasks.TaskManager(include_defaults=True)
    task=tasks.get_task_dict(['boolq'],task_manager=manager)['boolq']
    docs=sorted(list(task.validation_docs()),key=lambda doc:len(task.doc_to_text(doc)),reverse=True)[:32]
    requests=[]
    request_meta=[]
    for doc in docs:
        context=task.fewshot_context(doc,num_fewshot=0)
        instances=task.construct_requests(doc,context)
        for instance in instances:
            text,continuation=instance.args
            encoded_context,encoded_cont=hflm._encode_pair(text,continuation)
            requests.append(((text,continuation),encoded_context,encoded_cont))
            request_meta.append([len(encoded_context),len(encoded_cont)])
    original=HFLM._loglikelihood_tokens
    (args.output/f'{args.arm}_likelihood_source.py').write_text(inspect.getsource(original))
    if args.arm=='candidate': HFLM._loglikelihood_tokens=build_method(original)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    scores=hflm._loglikelihood_tokens(requests,disable_tqdm=True)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    memory=collect_peaks(torch.cuda,get_process_cuda_memory_peaks)
    state=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        state.update(name.encode())
        state.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        state.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    result=dict(arm=args.arm,scope='32 real BoolQ passages on original BF16 model; not quantized model/full QA or formal PTQ',
        batch_size=16,passages=len(docs),requests=len(requests),request_lengths=request_meta,
        request_sha256=hashlib.sha256(json.dumps(requests).encode()).hexdigest(),scores=scores,
        likelihood_source_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        model=args.model,model_use_cache=model.config.use_cache,seconds=elapsed,
        vocab_size=model.config.vocab_size,model_state_sha256=state.hexdigest(),
        memory=memory)
    with (args.output/f'{args.arm}.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('scores','request_lengths')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--arm',choices=('control','candidate'))
    args=parser.parse_args()
    if args.arm: return worker(args)
    args.output.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(__file__),ROOT/'tools/qa_score_lifetime_prototype.py',
                  ROOT/'tools/storage_process_device_peaks.py',ROOT/'utils/memory_utils.py',
                  Path(args.model)/'config.json']
    bindings={str(path.resolve()):sha256(path) for path in source_paths}
    with (args.output/'source_bindings.json').open('x') as f: json.dump(bindings,f,indent=2)
    for side in ('control','candidate'):
        if any(sha256(path)!=digest for path,digest in bindings.items()):
            raise RuntimeError('Bound component source changed')
        processes=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if any(line.strip().isdigit() for line in processes.splitlines()): raise RuntimeError('GPU has another compute process')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_DATASETS_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        print(f'Starting {side} component; log {args.output/side}.log',flush=True)
        with (args.output/f'{side}.log').open('x') as log, (args.output/f'{side}_nvml.csv').open('x') as nvml:
            monitor=subprocess.Popen(['nvidia-smi',f'--id={args.gpu}','--query-gpu=timestamp,memory.used,index,uuid','--format=csv,noheader','--loop-ms=100'],stdout=nvml,stderr=subprocess.DEVNULL)
            try:
                subprocess.run([sys.executable,'-u',__file__,'--model',args.model,'--output',str(args.output),'--arm',side],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    c,n=[json.loads((args.output/f'{side}.json').read_text()) for side in ('control','candidate')]
    same=all(c[k]==n[k] for k in ('request_sha256','scores','likelihood_source_sha256','model_use_cache','requests','request_lengths','model_state_sha256'))
    sources_unchanged=all(sha256(path)==digest for path,digest in bindings.items())
    nvml_peaks=[]
    for side in ('control','candidate'):
        samples=list(csv.reader((args.output/f'{side}_nvml.csv').open()))
        nvml_peaks.append(max(int(row[1]) for row in samples if len(row)>1))
    allocated_ok=n['memory']['peak_allocated_bytes']<=c['memory']['peak_allocated_bytes']
    reserved_ok=n['memory']['peak_reserved_bytes']<=c['memory']['peak_reserved_bytes']
    result=dict(valid=same and sources_unchanged and allocated_ok and reserved_ok and nvml_peaks[1]<=nvml_peaks[0],exact_scores=same,sources_unchanged=sources_unchanged,control=c['memory'],candidate=n['memory'],
                nvml_peak_mib=nvml_peaks,nvml_is_sampled=True,
                allocated_nonincreasing=n['memory']['peak_allocated_bytes']<=c['memory']['peak_allocated_bytes'],
                reserved_nonincreasing=n['memory']['peak_reserved_bytes']<=c['memory']['peak_reserved_bytes'],
                scope=c['scope'])
    with (args.output/'summary.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)
    if not result['valid']: raise SystemExit('Scoring or component memory failed; do not integrate optimization')


if __name__=='__main__': main()
