"""Serial four-visible-device server runtime interface admission.

The bounded synthetic wrapper test is not the formal QA result. Formal
quantization and all ten QA tasks run only in the subsequent guarded launcher.
"""
import argparse
import csv
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
from tools.qa_server_mapped_validation import sha, verify_runtime


def sources(model):
    import torch,transformers,lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers.cache_utils import DynamicCache
    from accelerate import hooks
    files = [*ROOT.glob('tools/qa_*.py'), ROOT/'tools/audit_qa_server_mapped.py', Path(__file__),
        *ROOT.glob('utils/*.py'), Path(model)/'config.json', inspect.getfile(HFLM),
        inspect.getfile(DynamicCache), inspect.getfile(torch.cuda.memory.MemPool),
        inspect.getfile(hooks), inspect.getfile(torch.nn.Linear)]
    files += [p for p in Path(model).rglob('*') if p.is_file()]
    return {str(p.resolve()):sha(p) for p in files}


def worker(a):
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from tools.qa_mapped_head_storage_prototype import prepare_runtime
    from tools.qa_server_mapped_validation import install
    from utils import model_utils,rotation_utils,dist_utils,memory_utils,hadamard_utils
    binary,_=verify_runtime(a.runtime_root)
    library,capabilities=prepare_runtime(binary)
    if torch.cuda.device_count()!=4:raise RuntimeError('Server admission requires all four visible devices')
    torch.manual_seed(1)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa').eval()
    analyzer=model_utils.ModelAnalyzer.__new__(model_utils.ModelAnalyzer)
    analyzer.model=model;analyzer.model_arch='Qwen3ForCausalLM'
    rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    requests=[];rng=random.Random(20261014)
    for i in range(32):
        context=[rng.randrange(32,model.config.vocab_size-1000) for _ in range(rng.randint(64,128))]
        for choice in range(2):
            continuation=[rng.randrange(32,model.config.vocab_size-1000) for _ in range(1 if i%2==0 else 4)]
            requests.append(((f'context-{i}',f'choice-{choice}'),context,continuation))
    requests.append((('unique-natural-tail','choice'),[11,12,13],[14]))
    def evaluate(model,tokenizer,batch_size):
        hflm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=batch_size)
        return lm_eval.simple_evaluate(hflm,tasks=['synthetic_interface'])
    def simple(hflm,*,tasks):
        return dict(scores=hflm._loglikelihood_tokens(requests,disable_tqdm=True))
    from types import SimpleNamespace
    eval_module=SimpleNamespace(qa_eval=evaluate)
    original=lm_eval.simple_evaluate;lm_eval.simple_evaluate=simple
    bindings=json.loads((a.output/'source_bindings.json').read_text())
    destination=a.output/a.arm
    bridge=install(eval_module,dist_utils,lm_eval,memory_utils,torch,enabled=a.arm=='candidate',
        destination=destination,expected_tasks=['synthetic_interface'],runtime_root=a.runtime_root,bindings=bindings)
    before={n:id(p) for n,p in model.named_parameters(remove_duplicate=False)}
    try:
        dist_utils.distribute_model(model)
        metrics=eval_module.qa_eval(model,tokenizer,16)
        row=dict(completed=True,arm=a.arm,metrics=metrics,runtime=capabilities,
            parameter_identity=before=={n:id(p) for n,p in model.named_parameters(remove_duplicate=False)},
            backend='torch_fallback' if hadamard_utils.fast_hadamard_transform is None else 'fast_hadamard_transform',
            device_map=getattr(model,'hf_device_map',None),scope='Unrotated535 synthetic65 server interface only; NOT formal QA quality')
        with (a.output/(a.arm+'.json')).open('x') as f:json.dump(row,f,indent=2)
    finally:
        lm_eval.simple_evaluate=original
        if bridge.bridge.pending_model_id is None:bridge.restore()


def admission(root):
    from tools.audit_qa_server_mapped import audit
    r=audit(root/'control',root/'candidate',['synthetic_interface'],json.loads((root/'source_bindings.json').read_text()))
    peaks=[]
    for side in ('control','candidate'):
        device={}
        with (root/(side+'_nvml.csv')).open() as f:
            for row in csv.reader(f):
                if len(row)!=4:continue
                value=float(row[1]);index=row[2].strip();uuid=row[3].strip()
                slot=device.setdefault(index,dict(uuid=uuid,peak_mib=0,samples=0))
                if slot['uuid']!=uuid:raise RuntimeError('NVML physical identity changed')
                slot['peak_mib']=max(slot['peak_mib'],value);slot['samples']+=1
        peaks.append(device)
    r['nvml']=peaks
    r['checks']['all_NVML_identity']=set(peaks[0])==set(peaks[1])=={'0','1','2','3'} and all(peaks[0][k]['uuid']==peaks[1][k]['uuid'] for k in peaks[0])
    r['checks']['global_NVML']=max(v['peak_mib'] for v in peaks[1].values())<=max(v['peak_mib'] for v in peaks[0].values())
    rows=[json.loads((root/(s+'.json')).read_text()) for s in ('control','candidate')]
    r['checks']['actual_server_backend']=all(x['backend']=='fast_hadamard_transform' and x['parameter_identity'] for x in rows)
    r['valid']=all(r['checks'].values())
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=('control','candidate'));a=p.parse_args()
    if a.arm:return worker(a)
    a.output.mkdir(parents=True,exist_ok=False)
    _,bindings=verify_runtime(a.runtime_root);bindings.update(sources(a.model))
    (a.output/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    result=dict(valid=False,scope='Server synthetic interface admission only')
    try:
        for side in ('control','candidate'):
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('Another GPU job exists')
            with (a.output/(side+'.log')).open('x') as f,(a.output/(side+'_nvml.csv')).open('x') as n:
                monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,index,uuid','--format=csv,noheader,nounits','--loop-ms=100'],stdout=n)
                try:
                    subprocess.run([sys.executable,'-u',__file__,'--model',a.model,'--runtime-root',str(a.runtime_root),
                        '--output',str(a.output),'--arm',side],stdout=f,stderr=subprocess.STDOUT,check=True)
                finally:monitor.terminate();monitor.wait(timeout=15)
        result=admission(a.output)
    except BaseException as exc:
        result.update(valid=False,error_type=type(exc).__name__,error=str(exc));raise
    finally:
        with (a.output/'status.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=result['checks']),indent=2))
    if not result['valid']:raise SystemExit(1)


if __name__=='__main__':main()
