"""CPU-only object-byte arithmetic from observed full outputs and model config.

Object bytes are neither allocator/NVML peaks nor a hardware admission. The
observed cache total includes all logical K/V tensors, not only CPU-resident
ones. Forecast rows are hypothetical, not executed QA requests.
"""
import argparse
import hashlib
import json
from pathlib import Path


def payload(config,batch,length):
    fields=('vocab_size','num_hidden_layers','num_key_value_heads','head_dim','hidden_size')
    if (type(batch) is not int or not 1<=batch<=16 or type(length) is not int or length<=0
            or any(type(config.get(key)) is not int or config[key]<=0 for key in fields)):
        raise ValueError('Requires natural batch1..16, positive length and explicit original geometry')
    one_kv=2*batch*length*config['num_key_value_heads']*config['head_dim']*2
    return dict(actual_batch_size=batch,input_tokens=length,bf16_element_bytes=2,
        full_logits_payload_bytes=batch*length*config['vocab_size']*2,
        one_layer_kv_payload_bytes=one_kv,
        all_layers_logical_kv_payload_bytes=config['num_hidden_layers']*one_kv,
        one_hidden_state_payload_bytes=batch*length*config['hidden_size']*2,
        object_bytes_are_not_peaks=True)


def observed(config,row):
    if row['batch_size']!=16:raise ValueError('Configured batch16 changed')
    outputs=row['logits'];heads=row['mapped_head']['records'];caches=row['cache_outputs']
    if not outputs or not len(outputs)==len(heads)==len(caches):raise ValueError('Missing per-forward output/cache records')
    results=[]
    for index,(output,head,cache) in enumerate(zip(outputs,heads,caches),1):
        b,l,v=output['shape'];budget=payload(config,b,l)
        if (v!=config['vocab_size'] or output['dtype']!='torch.bfloat16' or not head['completed']
                or head['output_shape']!=output['shape'] or head['dtype']!='torch.bfloat16'
                or head['input_shape']!=[b,l,config['hidden_size']]
                or head['output_payload_bytes']!=budget['full_logits_payload_bytes']):
            raise ValueError('Full original head payload/geometry mismatch')
        expected=[b,config['num_key_value_heads'],l,config['head_dim']]
        if len(cache)!=config['num_hidden_layers'] or any(item['shape']!=expected or item['value_shape']!=expected
            or item['dtype']!='torch.bfloat16' or item['value_dtype']!='torch.bfloat16' for item in cache):
            raise ValueError('Full actual K/V payload geometry missing')
        results.append(dict(forward=index,**budget))
    return results


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(root):
    root=Path(root)
    rows=[read(root/side/'layerwise_cache.json') for side in ('control','candidate')]
    if rows[0]['model']!=rows[1]['model']:raise ValueError('Different model geometry')
    cfg_path=(Path(rows[0]['model'])/'config.json').resolve();config=read(cfg_path)
    bindings=read(root/'source_bindings.json')
    if bindings.get(str(cfg_path))!=sha(cfg_path):raise ValueError('Model config changed from recorded source binding')
    budgets=[observed(config,row) for row in rows]
    if budgets[0]!=budgets[1]:raise ValueError('Paired observed payloads differ')
    independent=root/'independent_completed_audit-20261014-v1.json'
    reference=read(independent)
    return dict(model=rows[0]['model'],config_sha256=sha(cfg_path),configured_batch=16,
        payload_metadata_checks_passed=True,observed_forward_payloads=budgets[0],
        forecasts=[dict(execution_tested=False,input_kind='hypothetical shape, not formal task distribution',
                        **payload(config,16,length)) for length in (131,1235,2048,config['max_position_embeddings'])],
        independent_acceptance_reference=str(independent),independent_acceptance_sha256=sha(independent),
        recorded_prior_acceptance=reference['valid'],new_gpu_acceptance_claimed=False,
        artifact_sha256={str(root/side/'layerwise_cache.json'):sha(root/side/'layerwise_cache.json') for side in ('control','candidate')},
        scope='Exact byte arithmetic and geometry-only check; original full independent audit linked, not a fresh full re-audit/QA peak/CPU capacity proof')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=dict(reports=[report(root) for root in args.root],formal_enabled=False,
                source_sha256=sha(__file__),new_gpu_execution=False)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
