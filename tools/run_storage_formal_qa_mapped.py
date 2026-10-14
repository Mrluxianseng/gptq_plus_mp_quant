"""Explicit formal experiment adapter for whole-logit mapped QA storage.

Original quantization, KL/PPL, caches, tasks and hyperparameters are retained.
Validation-only helpers remain opt-in; default formal scripts are unchanged.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_storage_method_formal as formal
from tools import run_storage_formal_global_capacity as global_runner
from tools.qa_mapped_early_runtime_entry import build_entry as add_early
from tools.qa_server_mapped_validation import sha, verify_runtime


SETUP = '''
if _qa_early_selected:
    import lm_eval as _qa_server_lm
    import torch as _qa_server_torch
    from utils import eval_utils as _qa_server_eval, dist_utils as _qa_server_dist, memory_utils as _qa_server_mem
    from tools.qa_server_mapped_validation import install as _qa_server_install
    _qa_server_binding = json.loads(Path(os.environ['QA_SERVER_BINDINGS']).read_text())
    _qa_server_bridge = _qa_server_install(_qa_server_eval,_qa_server_dist,_qa_server_lm,
        _qa_server_mem,_qa_server_torch,enabled=_hinv_release,
        destination=Path(os.environ['RESEARCH_RUN_DIR']),expected_tasks=_qa_server_eval.PAPER_QA_TASKS,
        runtime_root=os.environ['QA_SERVER_RUNTIME_ROOT'],bindings=_qa_server_binding)
'''


def entry(source):
    captures=[n for n in ast.parse(source).body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='_hinv_release' for t in n.targets)]
    anchor="runpy.run_path('ptq.py', run_name='__main__')"
    if len(captures)!=1 or source.count(anchor)!=1:
        raise RuntimeError('Original formal arm capture/entry changed')
    generated,_=add_early(source.replace(anchor,SETUP+'\n'+anchor),validation_only=True)
    return generated


def require_admission(runtime_root, component_roots):
    from tools.run_qa_server_mapped_component import admission
    binary,bindings=verify_runtime(runtime_root)
    expected={'Qwen3-0.6B','Qwen3-1.7B'}
    seen=set()
    for root in component_roots:
        root=Path(root)
        previous=json.loads((root/'status.json').read_text())
        if not previous['valid'] or not admission(root)['valid']:
            raise RuntimeError('Actual server component admission rejected: '+str(root))
        row=json.loads((root/'source_bindings.json').read_text())
        seen.update(Path(p).parent.name for p in row if Path(p).name=='config.json')
        bindings.update(row)
    if seen!=expected:
        raise RuntimeError('Both exact server model admissions required')
    return binary,bindings


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--component-root',type=Path,action='append',required=True)
    known,remaining=p.parse_known_args()
    if '--skip-qa-eval' in remaining:raise RuntimeError('Formal mapped validation requires complete ten-task QA')
    if '--run' not in remaining:raise RuntimeError('Use the explicit guarded --run experiment')
    binary,bindings=require_admission(known.runtime_root,known.component_root)
    # Add only new entry/audit helpers to the formal source manifest.
    extra=list(ROOT.glob('tools/qa_*.py'))+[Path(__file__),ROOT/'tools/audit_qa_server_mapped.py',ROOT/'tools/run_qa_server_mapped_component.py']
    formal.FORMAL_TOOL_SOURCES+=tuple(str(p.relative_to(ROOT)) for p in extra if str(p.relative_to(ROOT)) not in formal.FORMAL_TOOL_SOURCES)
    bindings.update({str(p.resolve()):sha(p) for p in extra})
    # Save the immutable admission manifest outside the new controller root,
    # which the original formal runner must create itself.
    manifest=known.runtime_root/'formal_server_bindings.json'
    with manifest.open('x') as f:json.dump(bindings,f,indent=2)
    os.environ.update(REALQ_QA_STORAGE_VALIDATION='1',QA_MAPPED_HEAD_LIBRARY=str(binary.resolve()),
        QA_MAPPED_HEAD_LIBRARY_SHA256=sha(binary),QA_SERVER_RUNTIME_ROOT=str(known.runtime_root.resolve()),
        QA_SERVER_BINDINGS=str(manifest.resolve()))
    original_entry=formal.build_entry
    formal.build_entry=lambda:entry(original_entry())
    original_audit=global_runner.audit_pair
    def paired(control,candidate,ranks,source_bindings):
        from tools.audit_qa_server_mapped import audit
        from utils.eval_utils import PAPER_QA_TASKS
        result=original_audit(control,candidate,ranks,source_bindings)
        qa=audit(control,candidate,PAPER_QA_TASKS,bindings,expected_devices=ranks)
        result['qa_whole_mapped_storage']=qa
        result['checks']['complete_exact_server_qa']=qa['valid']
        result['valid']=result['capacity_valid']=all(result['checks'].values())
        result['strict_interval_point_valid'] &= qa['strict_local_nonincrease']
        return result
    global_runner.audit_pair=paired
    sys.argv=[sys.argv[0],*remaining]
    global_runner.main()


if __name__=='__main__':main()
