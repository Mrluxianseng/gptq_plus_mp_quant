"""Fresh full audit plus installed Accelerate and actual selected-method identity."""
import argparse
import hashlib
import inspect
import json
import marshal
from pathlib import Path
import sys
import types
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_qa_mapped_observed_bridge_completed import audit as ancestor_audit
from tools.qa_complete_observer import method_record
from tools.qa_mapped_observed_placement_bridge import control_batch_method, selected_digest
from tools.qa_mapped_head_batch_lifetime import build_batch_method
from tools.qa_score_lifetime_prototype import build_method as rows_method
from tools.qa_scoring_cache_offload_prototype import build_method as cache_method
from tools.audit_qa_memory_rows_completed import sha


def audit(root):
    result=ancestor_audit(root)
    from accelerate import hooks
    from lm_eval.models.huggingface import HFLM
    import torch
    code=[value for value in hooks.add_hook_to_module.__code__.co_consts
          if isinstance(value,types.CodeType) and value.co_name=='new_forward']
    expected=dict(kind='accelerate_align_devices',
        forward_code_sha256=hashlib.sha256(marshal.dumps(code[0])).hexdigest(),
        original_linear_code_sha256=hashlib.sha256(marshal.dumps(torch.nn.Linear.forward.__code__)).hexdigest(),
        accelerate_source=inspect.getfile(hooks),accelerate_source_sha256=sha(inspect.getfile(hooks)),
        hooks=[dict(type='AlignDevicesHook',execution_device='cuda:0',offload=False,
                    io_same_device=True,no_grad=False,place_submodules=False)])
    control=json.loads((root/'control/qa_mapped_head_rank0.json').read_text())
    checks=dict(actual_accelerate_control=control['head'].get('forward_chain')==expected,
        actual_hook_preserved=all(json.loads((root/(side+'.json')).read_text())['control_hook_preserved']
                                  for side in ('control','candidate')))
    scores=(control_batch_method(HFLM._loglikelihood_tokens,lambda:None),build_batch_method(HFLM._loglikelihood_tokens,lambda:None))
    calls=(HFLM._model_call,cache_method(HFLM._model_call))
    installed=(HFLM._loglikelihood_tokens,rows_method(HFLM._loglikelihood_tokens))
    for index,side in enumerate(('control','candidate')):
        summary=json.loads((root/side/'qa_complete/qa_complete_summary.json').read_text())
        head=json.loads((root/side/'qa_mapped_head_rank0.json').read_text())
        checks[side+'_actual_selected_score']=summary['selected_score']==method_record(scores[index])
        checks[side+'_actual_selected_call']=summary['selected_call']==method_record(calls[index])
        checks[side+'_installed_and_selected_identity']=(head['installed_score_code_sha256']==selected_digest(installed[index])
            and head['selected_score_code_sha256']==selected_digest(scores[index]))
    result['control_align_checks']=checks
    result['valid']=result['valid'] and all(checks.values())
    result['control_align_auditor_sha256']=sha(__file__)
    result['scope']='genuine CUDA AlignDevicesHook control-head observation versus admitted mapped composition, synthetic65/unrotated535/torch_fallback/onevisible; NOT formal QA/multiGPU or new algorithm gain'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.root)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=result['valid'],checks=result['control_align_checks'],nvml=result['nvml_peak_mib']),indent=2))
    if not result['valid']: raise SystemExit(1)
