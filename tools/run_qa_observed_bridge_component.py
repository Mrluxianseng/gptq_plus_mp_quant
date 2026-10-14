"""Native-dispatch versus composed storage bridge: bounded GPU validation only.

Unrotated base weights plus real online wrappers, synthetic tokens, batch16.
No formal launcher, downloads or benchmark accuracy claim.
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


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def worker(args):
    from types import SimpleNamespace
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from tools.qa_observed_placement_bridge import ObservedPlacementBridge
    from tools.storage_process_device_peaks import collect_peaks
    from utils import dist_utils,memory_utils,model_utils,rotation_utils,hadamard_utils
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Local composed probe requires exactly one visible device')
    torch.manual_seed(1)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).eval()
    analyzer=model_utils.ModelAnalyzer.__new__(model_utils.ModelAnalyzer)
    analyzer.model=model;analyzer.model_arch='Qwen3ForCausalLM'
    rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    spec={n:(id(p),tuple(p.shape),str(p.dtype)) for n,p in model.named_parameters(remove_duplicate=False)}
    rng=random.Random(20261014);requests=[]
    for index in range(32):
        context=[rng.randrange(32,model.config.vocab_size-1000) for _ in range(rng.randint(64,128))]
        for choice in range(2):
            continuation=[rng.randrange(32,model.config.vocab_size-1000) for _ in range(1 if index%2==0 else 4)]
            requests.append(((f'component-context-{index}',f'choice-{choice}'),context,continuation))
    scores=[];forward_info=[];cache_outputs=[]
    def inputs(module,args,kwargs):
        cache=kwargs.get('past_key_values')
        forward_info.append(dict(use_cache=bool(kwargs.get('use_cache',module.config.use_cache)),
            explicit_cache=cache is not None,offloading=bool(getattr(cache,'offloading',False))))
    def outputs(module,args,result):
        cache=result.past_key_values
        if cache is None: raise RuntimeError('Original KV missing')
        cache_outputs.append([dict(key_device=str(layer.keys.device),value_device=str(layer.values.device),
            key_shape=list(layer.keys.shape),value_shape=list(layer.values.shape),
            key_stride=list(layer.keys.stride()),value_stride=list(layer.values.stride()),
            key_dtype=str(layer.keys.dtype),value_dtype=str(layer.values.dtype)) for layer in cache.layers])
    handles=[model.register_forward_pre_hook(inputs,with_kwargs=True),model.register_forward_hook(outputs)]
    def simple_evaluate(hflm,**kwargs):
        result=hflm._loglikelihood_tokens(requests,disable_tqdm=True)
        scores.extend(result)
        # Primitive fixture metadata only; not accuracy on a dataset.
        return dict(results={kwargs['tasks'][0]:dict(request_count=len(result))})
    evaluation=SimpleNamespace(simple_evaluate=simple_evaluate)
    def qa_eval(model,tokenizer,batch):
        hflm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=batch,device='cuda:0')
        return evaluation.simple_evaluate(hflm,tasks=['synthetic_component'])['results']
    ev=SimpleNamespace(qa_eval=qa_eval)
    destination=args.output/args.arm
    bridge=ObservedPlacementBridge(ev,dist_utils,evaluation,memory_utils,torch.cuda,
        enabled=args.arm=='candidate',destination=destination,expected_tasks=('synthetic_component',),
        validation_only=True,exclusive_peak_owner=True)
    started=time.monotonic()
    try:
        dist_utils.distribute_model(model)
        metrics=ev.qa_eval(model,tokenizer,16)
        memory=collect_peaks(torch.cuda,memory_utils.get_process_cuda_memory_peaks)
    finally:
        for handle in handles: handle.remove()
        bridge.restore()
    identity={n:(id(p),tuple(p.shape),str(p.dtype)) for n,p in model.named_parameters(remove_duplicate=False)}
    record=dict(arm=args.arm,model=args.model,batch_size=16,scope='native-dispatch composed observer/storage bridge synthetic online-wrapper fixture; NOT REAL-Q/full QA',
        parameter_identity_preserved=spec==identity,config_use_cache=model.config.use_cache,
        hadamard_backend='torch_fallback' if hadamard_utils.fast_hadamard_transform is None else 'fast_hadamard_transform',
        native_device_map=getattr(model,'hf_device_map',None),scores=scores,forward_info=forward_info,
        cache_outputs=cache_outputs,request_sha256=hashlib.sha256(json.dumps(requests).encode()).hexdigest(),
        metrics=metrics,memory=memory,seconds=time.monotonic()-started)
    with (args.output/(args.arm+'.json')).open('x') as stream: json.dump(record,stream,indent=2)
    print(json.dumps({k:record[k] for k in ('arm','memory','seconds')},indent=2),flush=True)


def audit(root):
    from tools.qa_complete_observer import validate_observer_pair
    from tools.audit_memory_boundary_points import boundaries,compare
    from tools.audit_completed_memory_pair_capacity import intervals
    from tools.storage_process_device_peaks import valid_record
    bindings=json.loads((root/'source_bindings.json').read_text())
    rows=[json.loads((root/(side+'.json')).read_text()) for side in ('control','candidate')]
    bridges=[json.loads((root/side/'qa_observed_placement_rank0.json').read_text()) for side in ('control','candidate')]
    placements=[json.loads((root/side/'qa_layerwise_bridge_rank0.json').read_text()) for side in ('control','candidate')]
    states=[json.loads((root/side/'qa_complete/qa_complete_summary.json').read_text()) for side in ('control','candidate')]
    numerical=validate_observer_pair(root/'control/qa_complete',root/'candidate/qa_complete',['synthetic_component'])
    checks=dict(numerical=numerical['valid'],sources=all(sha(p)==h for p,h in bindings.items()),
        completed=all(b['completed'] and b['formal_enabled'] is False for b in bridges),
        identity=all(r['parameter_identity_preserved'] and r['config_use_cache'] for r in rows),
        batch=all(r['batch_size']==16 for r in rows),
        native_control=placements[0]['placement']['mode']=='original-distribute-model' and rows[0]['native_device_map'] is not None,
        candidate_layerwise=placements[1]['placement']['mode']=='single-device-layerwise' and placements[1]['enabled'] is True,
        full_535=all(s['before']['count']==535 and len([n for n in s['before']['layout'] if n.endswith('.had_K')])==28 for s in states))
    for key in ('model','scores','request_sha256','metrics','hadamard_backend'):
        checks[key]=rows[0][key]==rows[1][key]
    calls=states[0]['forward_count']
    config=json.loads((Path(rows[0]['model'])/'config.json').read_text())
    weights=placements[1]['weight_records']
    checks['weights']=len(weights)==28*calls and {(r['forward'],r['layer']) for r in weights}=={
        (f,i) for f in range(1,calls+1) for i in range(28)} and all(r['enabled'] and r['final_device']=='cpu' for r in weights)
    checks['kv']=True
    for side,row in enumerate(rows):
        events=[json.loads(line) for line in (root/('control' if side==0 else 'candidate')/'qa_complete/qa_complete_events.jsonl').read_text().splitlines()]
        shapes=[event['shape'] for event in events if event['kind']=='model_input']
        checks['kv'] &= len(row['forward_info'])==len(row['cache_outputs'])==calls
        checks['kv'] &= all(r['use_cache'] and r['offloading']==bool(side) and r['explicit_cache']==bool(side) for r in row['forward_info'])
        for shape,output in zip(shapes,row['cache_outputs']):
            checks['kv'] &= len(output)==28
            for index,kv in enumerate(output):
                device='cpu' if side and index!=0 else 'cuda:0'
                checks['kv'] &= kv['key_device']==kv['value_device']==device and kv['key_dtype']==kv['value_dtype']=='torch.bfloat16'
                checks['kv'] &= kv['key_shape']==kv['value_shape'] and kv['key_stride']==kv['value_stride']
                checks['kv'] &= kv['key_shape']==[shape[0],config['num_key_value_heads'],shape[1],config['head_dim']]
    paths=[[Path(p) for p in b['memory']['memory_profile_files']] for b in bridges]
    tables=[intervals(p) for p in paths];point=compare(*(boundaries(p) for p in paths))
    checks['interval_complete']=bool(tables[0]) and tables[0].keys()==tables[1].keys()
    checks['points_complete']=bool(point['matched_values']) and not point['missing']
    checks['phase_identity']=all(r['phase'].startswith(f"qa_device{r['device_index']}__") and r['rank']==0
        and r['stage']=='stage2' and r['logkind']=='evaluation' and r['peak_measurement_mode']=='exact' for table in tables for r in table.values())
    regressions=[dict(identity=list(key),metric=metric,control=tables[0][key]['sampled_peak_'+metric+'_bytes'],
        candidate=tables[1][key]['sampled_peak_'+metric+'_bytes']) for key in tables[0].keys()&tables[1].keys()
        for metric in ('allocated','reserved') if tables[1][key]['sampled_peak_'+metric+'_bytes']>tables[0][key]['sampled_peak_'+metric+'_bytes']]
    checks['peak_coverage']=all(valid_record(row['memory'],1) for row in rows)
    checks['peak_carry']=all(all(row['memory'][metric]>=b['memory'][metric] for metric in ('peak_allocated_bytes','peak_reserved_bytes')) for row,b in zip(rows,bridges))
    checks['global_allocator']=all(rows[1]['memory'][metric]<=rows[0]['memory'][metric] for metric in ('peak_allocated_bytes','peak_reserved_bytes'))
    nvml=[];uuids=[]
    for side in ('control','candidate'):
        samples=list(csv.reader((root/(side+'_nvml.csv')).open()))
        nvml.append(max(float(row[1].strip().split()[0]) for row in samples));uuids.append({row[3].strip() for row in samples})
    checks['nvml']=nvml[1]<=nvml[0] and len(uuids[0])==1 and uuids[0]==uuids[1]
    return dict(valid=all(checks.values()),checks=checks,numerical=numerical,nvml_peak_mib=nvml,
        memory=[row['memory'] for row in rows],interval_count=len(tables[0]),points=point,interval_regressions=regressions,
        strict_local_nonincrease=not regressions and point['valid'],scope=rows[0]['scope'],nvml_is_sampled=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('--output',type=Path)
    parser.add_argument('--arm',choices=('control','candidate'))
    parser.add_argument('--audit-root',type=Path);parser.add_argument('--audit-output',type=Path)
    args=parser.parse_args()
    if args.audit_root:
        result=audit(args.audit_root)
        with args.audit_output.open('x') as stream: json.dump(result,stream,indent=2)
        print(json.dumps(result,indent=2))
        if not result['valid']: raise SystemExit(1)
        return
    if not args.model or args.output is None: parser.error('--model and --output required')
    if args.arm: return worker(args)
    args.output.mkdir(parents=True,exist_ok=False)
    from lm_eval.models.huggingface import HFLM
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3 import modeling_qwen3
    from utils import hadamard_utils
    if hadamard_utils.fast_hadamard_transform is not None:
        raise RuntimeError('This local prepared probe only admits the previously tested torch_fallback')
    files=[Path(__file__),*map(Path,('tools/qa_observed_placement_bridge.py','tools/qa_layerwise_placement_bridge.py',
        'tools/qa_complete_observer.py','tools/qa_tensor_fingerprint_stream.py','tools/qa_task_boundary_observer.py',
        'tools/qa_visible_device_phase_profile.py','tools/qa_layerwise_weight_prototype.py','tools/qa_score_lifetime_prototype.py',
        'tools/qa_scoring_cache_offload_prototype.py','tools/storage_process_device_peaks.py','tools/audit_memory_boundary_points.py',
        'tools/audit_completed_memory_pair_capacity.py','utils/dist_utils.py','utils/memory_utils.py','utils/model_utils.py',
        'utils/rotation_utils.py','utils/quant_utils.py','utils/hadamard_utils.py','utils/had_mat.pt')),
        Path(inspect.getfile(HFLM)),Path(inspect.getfile(DynamicCache)),Path(modeling_qwen3.__file__)]
    files += [p for p in Path(args.model).iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.txt')]
    (args.output/'source_bindings.json').write_text(json.dumps({str(p.resolve()):sha(p) for p in files},indent=2))
    hidden=json.loads((Path(args.model)/'config.json').read_text())['hidden_size']
    if hidden not in (1024,2048): raise RuntimeError('Unknown local model architecture')
    reference=Path('outputs')/('qa-complete-observer-rotation-'+('qwen06' if hidden==1024 else 'qwen17')+'-ctx128-20261014-v1')
    audit_command=[sys.executable,'tools/audit_qa_complete_observer_component.py','--root',str(reference),
        '--output',str(args.output/'reference_readmission.json')]
    with (args.output/'reference_readmission.log').open('x') as log:
        subprocess.run(audit_command,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    for side in ('control','candidate'):
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Another GPU worker is active')
        with (args.output/(side+'_nvml.csv')).open('x') as nvml, (args.output/(side+'.log')).open('x') as log:
            sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,index,uuid','--format=csv,noheader','-lms','100'],stdout=nvml)
            try:
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--model',args.model,'--output',str(args.output),'--arm',side]
                worker_process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                code=worker_process.wait()
            finally:
                sampler.terminate();sampler.wait()
            print(side,'exit',code,flush=True)
            if code: raise RuntimeError('Worker failed; original log retained')
    result=audit(args.output)
    with (args.output/'status.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
    if not result['valid']: raise RuntimeError('Completed bridge component rejected')


if __name__=='__main__': main()
