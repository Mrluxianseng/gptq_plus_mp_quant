"""Supplement complete composition audit with actual selected method identity."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_qa_mapped_observed_bridge_completed import audit
from tools.qa_complete_observer import method_record
from tools.qa_mapped_observed_placement_bridge import control_batch_method,selected_digest
from tools.qa_mapped_head_batch_lifetime import build_batch_method
from tools.qa_score_lifetime_prototype import build_method as rows_method
from tools.qa_scoring_cache_offload_prototype import build_method as cache_method
from tools.audit_qa_memory_rows_completed import sha


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.root)
    from lm_eval.models.huggingface import HFLM
    scores=(control_batch_method(HFLM._loglikelihood_tokens,lambda:None),build_batch_method(HFLM._loglikelihood_tokens,lambda:None))
    calls=(HFLM._model_call,cache_method(HFLM._model_call))
    installed=(HFLM._loglikelihood_tokens,rows_method(HFLM._loglikelihood_tokens))
    checks={}
    for index,side in enumerate(('control','candidate')):
        summary=json.loads((args.root/side/'qa_complete/qa_complete_summary.json').read_text())
        head=json.loads((args.root/side/'qa_mapped_head_rank0.json').read_text())
        checks[side+'_actual_selected_score']=summary['selected_score']==method_record(scores[index])
        checks[side+'_actual_selected_call']=summary['selected_call']==method_record(calls[index])
        checks[side+'_installed_and_selected_identity']=(head['installed_score_code_sha256']==selected_digest(installed[index])
            and head['selected_score_code_sha256']==selected_digest(scores[index]))
    result['independent_selected_method_checks']=checks
    result['valid']=result['valid'] and all(checks.values())
    result['selected_method_auditor_sha256']=sha(__file__)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=checks,nvml=result['nvml_peak_mib']),indent=2))
    if not result['valid']: raise SystemExit(1)
