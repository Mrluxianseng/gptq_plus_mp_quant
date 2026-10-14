"""Independent completed audit of official circular KV prefetch placement."""
import argparse
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_layerwise_component import audit
from tools.audit_qa_memory_rows_completed import sha


def prefetch_coverage(root):
    from transformers import AutoConfig
    from transformers.cache_utils import Cache,DynamicCache,CacheLayerMixin
    row=json.loads((root/'layerwise_cache.json').read_text())
    cfg=AutoConfig.from_pretrained(row['model'],local_files_only=True)
    cache=DynamicCache(config=cfg)
    update=inspect.getsource(Cache.update);prefetch=inspect.getsource(Cache.prefetch)
    offload=inspect.getsource(CacheLayerMixin.offload);layer_prefetch=inspect.getsource(CacheLayerMixin.prefetch)
    supported=(not any(cache.is_sliding) and len(cache.layers)==28 and
        update.count('self.prefetch(layer_idx + 1, self.only_non_sliding)')==1 and
        prefetch.count('self.layers[layer_idx].prefetch()')==1 and
        'self.is_sliding.index(False)' in prefetch and
        'self.keys.to("cpu", non_blocking=True)' in offload and
        'self.values.to("cpu", non_blocking=True)' in offload)
    calls=len(row['logits']);outputs=row['cache_outputs']
    keys=(len(outputs)==calls and all(len(output)==28 and all(
        layer['device']==('cuda:0' if index==0 else 'cpu')
        for index,layer in enumerate(output)) for output in outputs))
    observed_values=all('value_device' in layer for output in outputs for layer in output)
    values=(not observed_values or all(layer['value_device']==layer['device'] for output in outputs for layer in output))
    return dict(valid=supported and keys and values,supported_source_contract=supported,
        observed_keys_correct=keys,values_observed=observed_values,values_correct_if_observed=values,
        expected='layer0 prefetched to cuda:0 for the next decode; layers1-27 CPU',
        source_snapshot_after_run=dict(update=update,prefetch=prefetch,offload=offload,layer_prefetch=layer_prefetch),
        library_path=inspect.getfile(Cache),library_sha256_after_run=sha(inspect.getfile(Cache)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();bindings=json.loads((args.root/'source_bindings.json').read_text())
    original=audit(args.root,bindings)
    coverage=prefetch_coverage(args.root)
    checks=dict(original['reports']['layerwise_cache']['checks'])
    checks['actual_cache_device_coverage']=coverage['valid']
    result=dict(valid=all(checks.values()) and all(original['reports'][s]['valid'] for s in ('rows','layerwise')),
        layerwise_cache_checks=checks,prefetch_coverage=coverage,original_report=original,
        note='Original all-CPU check/status preserved. v1 keys observed; value placement is source-derived unless v2 value_device exists.',
        artifact_sha256={p.name:sha(p) for p in args.root.iterdir() if p.is_file()})
    with args.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=checks,values_observed=coverage['values_observed']),indent=2))
    if not result['valid']: raise SystemExit(1)


if __name__=='__main__': main()
