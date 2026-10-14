"""Independent native-bridge audit, including readmission and selected code identity."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_observed_bridge_component import audit,sha
from tools.qa_complete_observer import validate_observer_pair,method_record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.root)
    control=json.loads((args.root/'control.json').read_text())
    config=json.loads((Path(control['model'])/'config.json').read_text())
    model='qwen06' if config['hidden_size']==1024 else 'qwen17'
    reference=ROOT/('outputs/qa-complete-observer-rotation-'+model+'-ctx128-20261014-v1')
    prior=json.loads((args.root/'reference_readmission.json').read_text())
    bindings=json.loads((reference/'source_bindings.json').read_text())
    checks=dict(reference_readmission=prior['valid'] is True,
        reference_sources_current=all(sha(p)==h for p,h in bindings.items()))
    from lm_eval.models.huggingface import HFLM
    from tools.qa_score_lifetime_prototype import build_method as rows_method
    from tools.qa_scoring_cache_offload_prototype import build_method as cache_method
    expected_calls=(HFLM._model_call,cache_method(HFLM._model_call))
    expected_scores=(HFLM._loglikelihood_tokens,rows_method(HFLM._loglikelihood_tokens))
    for index,side in enumerate(('control','candidate')):
        legacy='control' if index==0 else 'layerwise_cache'
        # Same already-accepted synthetic requests, original full bytes and
        # method, but new placement and profiling interface. Phase counts differ
        # deliberately, so only the within-new-pair memory audit is claimed.
        numerical=validate_observer_pair(reference/(legacy+'_complete'),args.root/side/'qa_complete',['synthetic_component'])
        checks[side+'_prior_complete_numerical']=numerical['valid']
        summary=json.loads((args.root/side/'qa_complete/qa_complete_summary.json').read_text())
        checks[side+'_selected_method_code']=(summary['selected_call']==method_record(expected_calls[index])
            and summary['selected_score']==method_record(expected_scores[index]))
        checks[side+'_535_had_layout']=summary['before']['count']==535 and len([
            name for name,value in summary['before']['layout'].items()
            if name.endswith('.had_K') and value==[[12,12],'torch.float32']])==28
    result['independent_checks']=checks
    result['valid']=result['valid'] and all(checks.values())
    result['auditor_sha256']=sha(__file__)
    result['reference_readmission_sha256']=sha(args.root/'reference_readmission.json')
    result['scope']='native-dispatch composition of same accepted local wrapper fixture; NOT formal QA or REAL-Q quality'
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=checks,nvml=result['nvml_peak_mib']),indent=2))
    if not result['valid']: raise SystemExit(1)


if __name__=='__main__': main()
