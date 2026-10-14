"""Preserve v1 failure; derive expected SHA from an actual installed hook function.

marshal serialization of a nested code constant differs before and after its
function exists, even for the same code object. No GPU replay or math change.
"""
import ast
import hashlib
import inspect
import marshal
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import audit_qa_control_align_completed as ancestor


def installed_forward_sha():
    import torch
    from accelerate import hooks
    specimen=torch.nn.Linear(1,1,bias=False,dtype=torch.bfloat16).eval()
    hooks.add_hook_to_module(specimen,hooks.AlignDevicesHook(execution_device='cpu',io_same_device=True))
    return hashlib.sha256(marshal.dumps(specimen.forward.func.__code__)).hexdigest()


def audit_source():
    text=inspect.getsource(ancestor.audit)
    old='hashlib.sha256(marshal.dumps(code[0])).hexdigest()'
    if text.count(old)!=1: raise RuntimeError('Frozen expected-forward SHA anchor changed')
    text=text.replace(old,'_installed_forward_sha()')
    compile(text,'qa_control_align_audit_v2','exec')
    return text


scope=dict(ancestor.__dict__,__file__=__file__,_installed_forward_sha=installed_forward_sha)
exec(compile(audit_source(),'qa_control_align_audit_v2','exec'),scope)
audit=scope['audit']


if __name__=='__main__':
    p=ancestor.argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.root)
    result['expected_sha_derivation']='Actual installed Accelerate CPU-hook function, not an uninstantiated nested code constant'
    result['prior_status_preserved']=ancestor.sha(args.root/'status.json')
    result['v2_source_sha256']=ancestor.sha(__file__)
    with args.output.open('x') as stream: ancestor.json.dump(result,stream,indent=2)
    print(ancestor.json.dumps(dict(valid=result['valid'],checks=result['control_align_checks'],nvml=result['nvml_peak_mib']),indent=2))
    if not result['valid']: raise SystemExit(1)
