"""Fresh full-byte audit of a new natural-tail workload, never old-input equality."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_qa_mapped_long_batch_completed import module_source as previous_module,batch_checks


def geometry_checks(row,config):
    logits=row['logits'];layers=row['layers'];cache=row['cache_outputs'];calls=len(logits)
    return dict(layer_output_geometry=len(layers)==28 and calls>0 and all(
        len(layer['layout'])==calls and all(layout[0]==[logit['shape'][0],logit['shape'][1],config.hidden_size]
            and layout[1]=='torch.bfloat16' and layout[2]==[logit['shape'][1]*config.hidden_size,config.hidden_size,1]
            for layout,logit in zip(layer['layout'],logits)) for layer in layers),
        actual_kv_geometry=len(cache)==calls and calls>0 and all(len(out)==28 and all(
            item['shape']==item['value_shape']==[logit['shape'][0],config.num_key_value_heads,logit['shape'][1],config.head_dim]
            and item['dtype']==item['value_dtype']=='torch.bfloat16' and item['stride']==item['value_stride']
            for item in out) for out,logit in zip(cache,logits)))


def natural_batch_checks(row):
    head=row['mapped_head'];logits=row['logits'];sizes=[x['shape'][0] for x in logits]
    return dict(configured_batch16=row['batch_size']==head['configured_batch_size']==16,
        original_natural_batch_shapes=sizes==head['actual_batch_sizes']==[16,16,16,1],
        no_deleted_requests=len(row['scores'])==65,
        actual_tail_input=bool(logits) and logits[-1]['shape'][0:2]==[1,3])


def module_source():
    text=previous_module()
    changes={
        'from tools.run_qa_mapped_long_batch_component import NUMERICAL_KEYS,expected_phases':
        'from tools.run_qa_mapped_natural_tail_component import NUMERICAL_KEYS,expected_phases',
        'from tools.audit_qa_long_completed import geometry_checks':'# Natural-tail geometry supplied by this versioned audit.',
        'c[key]==n[key]==old[key] for key in NUMERICAL_KEYS':'c[key]==n[key] for key in NUMERICAL_KEYS',
        "calls==3 and len(row['scores'])==64":"calls==4 and len(row['scores'])==65",
        "x['shape'][0]==16 and x['shape'][2]==config.vocab_size":"1<=x['shape'][0]<=16 and x['shape'][2]==config.vocab_size",
        "r['input_shape']==[16,logit['shape'][1],config.hidden_size]":
        "r['input_shape']==[logit['shape'][0],logit['shape'][1],config.hidden_size]",
        "2*16*logit['shape'][1]*config.vocab_size":"2*logit['shape'][0]*logit['shape'][1]*config.vocab_size",
        "        checks.update({side+'_batch_'+k:v for k,v in batch_checks(row,side=='candidate').items()})":
        "        checks.update({side+'_batch_'+k:v for k,v in batch_checks(row,side=='candidate').items()})\n        checks.update({side+'_natural_'+k:v for k,v in natural_batch_checks(row).items()})",
        "scope='raw311 synthetic-token long component, fresh paired same best method plus mapped whole-head output; not formal QA or server capacity'":
        "scope='raw311 new65 synthetic requests with natural16/16/16/1 tail, fresh paired same QA math; prior long reference is source admission only; not formal QA or server capacity'",
    }
    for old,new in changes.items():
        if text.count(old)!=1: raise RuntimeError('Frozen natural-tail audit anchor changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_mapped_natural_tail_audit','exec')
    return text


_scope=dict(__name__='qa_mapped_natural_tail_audit',__file__=__file__,batch_checks=batch_checks,
    geometry_checks=geometry_checks,natural_batch_checks=natural_batch_checks)
exec(compile(module_source(),'qa_mapped_natural_tail_audit','exec'),_scope)
audit=_scope['audit'];head_checks=_scope['head_checks']
if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.root)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=result['checks'],nvml=result['nvml_peak_mib']),indent=2))
    if not result['valid']: raise SystemExit(1)
