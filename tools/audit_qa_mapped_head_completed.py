"""Fresh mapped-output component audit; reads full bytes/coverage, never runs GPU."""
import argparse
import inspect
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_mapped_head_component import audit,sha
from tools.qa_complete_observer import validate_observer_pair


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.root)
    rows=[json.loads((args.root/side/'layerwise_cache.json').read_text()) for side in ('control','candidate')]
    config=json.loads((Path(rows[0]['model'])/'config.json').read_text())
    name='qwen06' if config['hidden_size']==1024 else 'qwen17'
    reference=ROOT/('outputs/qa-complete-observer-rotation-'+name+'-ctx128-20261014-v1')
    checks={}
    for index,side in enumerate(('control','candidate')):
        row=rows[index]
        numerical=validate_observer_pair(reference/'layerwise_cache_complete',
            args.root/side/'layerwise_cache_complete',['synthetic_component'])
        checks[side+'_prior_accepted_full_bytes']=numerical['valid']
        buffers={k:v for k,v in row['before']['layout'].items() if k.endswith('.had_K')}
        checks[side+'_535_expected_buffers']=len(buffers)==28 and all(v==[[12,12],'torch.float32'] for v in buffers.values())
        checks[side+'_head_real_geometry']=(row['mapped_head']['calls']==len(row['logits'])
            and all(record['input_shape']==[logit['shape'][0],logit['shape'][1],config['hidden_size']]
                and record['output_shape']==logit['shape'] and record['output_stride']==logit['stride']
                and record['output_payload_bytes']==2*logit['shape'][0]*logit['shape'][1]*config['vocab_size']
                for record,logit in zip(row['mapped_head']['records'],row['logits'])))
        checks[side+'_kv_actual_geometry']=len(row['cache_outputs'])==len(row['logits']) and all(
            len(cache)==28 and all(kv['shape']==kv['value_shape']==[logit['shape'][0],config['num_key_value_heads'],logit['shape'][1],config['head_dim']]
                and kv['dtype']==kv['value_dtype']=='torch.bfloat16' and kv['stride']==kv['value_stride']
                and kv['device']==kv['value_device']==('cuda:0' if layer==0 else 'cpu')
                for layer,kv in enumerate(cache)) for cache,logit in zip(row['cache_outputs'],row['logits']))
    records=rows[1]['mapped_head']['records']
    checks['each_native_private_accounting_recomputed']=all(record['private_active']>=record['output_payload_bytes']
        and record['native_allocated']==record['snapshot_native_allocated']>=record['private_active']
        and record['native_reserved']==record['snapshot_native_reserved']>=record['private_reserved']
        and record['mapped_host_live']==record['private_reserved']
        and record['private_segments_in_native'] is True for record in records)
    bindings=json.loads((reference/'source_bindings.json').read_text())
    checks['prior_accepted_sources_unchanged']=all(sha(path)==digest for path,digest in bindings.items())
    import torch,hashlib
    head_source=hashlib.sha256(inspect.getsource(torch.nn.Linear.forward).encode()).hexdigest()
    checks['original_linear_forward_source']=all(row['mapped_head']['original_forward_sha256']==head_source for row in rows)
    result['independent_checks']=checks
    result['valid']=result['valid'] and all(checks.values())
    result['auditor_sha256']=sha(__file__)
    result['scope']='complete short synthetic online-wrapper mapped-head component; NOT formal QA/REAL-Q or server capacity'
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=checks,nvml=result['nvml_peak_mib']),indent=2))
    if not result['valid']: raise SystemExit(1)


if __name__=='__main__': main()
