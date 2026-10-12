"""Supplement the frozen FP16-fixture coverage mistake using actual BF16 flow.

Do not edit the original component, runner, audit or frozen waiter. The full
BF16 pair is already complete and exactly matches canonical state/output/KL.
Validate all 28 actual parameter maps against the source model architecture
and config dtype, plus existing native-pool/deferred-cleanup records.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_completed_memory_pair_capacity import audit,read,digest
from tools.run_replay_weight_compact_screen import compact_coverage


def actual_expected(config):
    if config.get('torch_dtype')!='bfloat16':raise RuntimeError('requires source config BF16')
    h=config['hidden_size'];head=config['head_dim'];q=config['num_attention_heads']*head;k=config['num_key_value_heads']*head;m=config['intermediate_size']
    shapes=[('self_attn.q_proj.weight',[q,h]),('self_attn.k_proj.weight',[k,h]),('self_attn.v_proj.weight',[k,h]),('self_attn.o_proj.weight',[h,q]),
        ('self_attn.q_norm.weight',[head]),('self_attn.k_norm.weight',[head]),('mlp.gate_proj.weight',[m,h]),('mlp.up_proj.weight',[m,h]),('mlp.down_proj.weight',[h,m]),
        ('input_layernorm.weight',[h]),('post_attention_layernorm.weight',[h])]
    parameters=[]
    for name,shape in shapes:
        numel=1
        for v in shape:numel*=v
        parameters.append(dict(name=name,bytes=numel*2,shape=shape,stride=[shape[1],1] if len(shape)==2 else [1],dtype='torch.bfloat16'))
    return dict(parameters=parameters,payload_bytes=sum(p['bytes'] for p in parameters),parameter_count=len(parameters))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    tag='qwen06b-replay-weight-compact-screen-20261012-v1';out=ROOT/'outputs/replay-weight-compact-screen-20261012-v1';runs=[ROOT/('outputs/full_pipeline_profile_'+tag+'-r1-'+side) for side in ('control','candidate')]
    # This independently covers exactly the same workspace/reuse/size/dead/stats
    # method. Pool validation is added below, not bypassed by accepting size.
    result=audit(tag,out/'source_bindings.json',method='size')
    config_path=ROOT.parent/'llamaModels/Qwen3-0.6B/config.json';config=read(config_path);expected=actual_expected(config)
    maps=[read(run/'replay_weight_compact_stage2_rank0.json') for run in runs];cleanups=[read(run/'replay_weight_cleanup_stage2_rank0.json') for run in runs]
    covered=compact_coverage(*maps,cleanups,expected)
    result['checks']['actual_bf16_28_compact_maps_and_release_covered']=covered
    result['method']='compact';result['valid']=result['capacity_valid']=result['capacity_valid'] and covered
    result['strict_interval_point_valid']=result['strict_interval_point_valid'] and covered
    result['coverage_spec_correction']=dict(original_controller_status_sha256=digest(out/'status.json'),
        original_component_sha256=digest(ROOT/'outputs/replay-weight-compact-cuda-20261012-v1/status.json'),
        model_config_sha256=digest(config_path),actual_dtype='torch.bfloat16',old_fixture_dtype='torch.float16',
        source='Original model config architecture + full actual BF16 pair, not a modified FP16 fixture',expected=expected,
        note='No solver/storage/math source or old status was changed. Full canonical507/all28/KL/PPL bitexact BF16 result validates actual dtype path; FP16 component alone did not.')
    result['supplemental_audit_source_sha256']=digest(Path(__file__))
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],strict=result['strict_interval_point_valid'],checks=result['checks'],source_count=result['source_count'],peaks=result['full_global_peaks'],interval_regressions=len(result['interval_regressions']),point_regressions=len(result['points']['regressions']))))


if __name__=='__main__':main()
