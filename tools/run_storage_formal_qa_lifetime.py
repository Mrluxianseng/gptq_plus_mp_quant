"""Prepared formal adapter; requires server full-QA validation before launch."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_storage_formal_global_capacity as global_runner
from tools import run_storage_method_formal as formal
from tools.qa_scoring_lifetime_install import validate_pair
from tools.qa_score_lifetime_prototype import build_source


def require_full_qa_validation(root,models):
    import hashlib
    import inspect
    from tools.audit_qa_memory_rows_completed import audit
    reports={}
    for name in models:
        path=Path(root)/f'{name}-full'
        try: report=audit(path)
        except (OSError,ValueError,KeyError) as exc:
            raise RuntimeError('Missing or incomplete server full-QA validation: '+str(path)) from exc
        if not report['valid'] or len(report['tasks'])!=10:
            raise RuntimeError('Server full-QA validation rejected: '+str(path))
        if Path(formal.read(path/'control.json')['model']).name!=name:
            raise RuntimeError('Full-QA validation model identity mismatch: '+str(path))
        from lm_eval.models.huggingface import HFLM
        control=formal.read(path/'control.json')
        for key,method in (('likelihood_source_sha256',HFLM._loglikelihood_tokens),
                           ('model_call_source_sha256',HFLM._model_call)):
            if control[key]!=hashlib.sha256(inspect.getsource(method).encode()).hexdigest():
                raise RuntimeError('Installed QA library differs from full-QA validation')
        reports[name]=report
    return reports


def entry(source):
    import ast
    captures=[n for n in ast.parse(source).body if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Name) and t.id=='_hinv_release' for t in n.targets)]
    if len(captures)!=1: raise RuntimeError('Missing original candidate flag capture')
    anchor="runpy.run_path('ptq.py', run_name='__main__')"
    if source.count(anchor)!=1: raise RuntimeError('Unexpected formal entry anchor')
    setup="""from tools.qa_scoring_lifetime_install import install as _install_qa_lifetime
from utils import eval_utils as _qa_lifetime_eval
_install_qa_lifetime(_qa_lifetime_eval,
    enabled=_hinv_release,
    destination=Path(os.environ['RESEARCH_RUN_DIR']),rank=int(os.environ.get('RANK','0')))
"""
    result=source.replace(anchor,setup+anchor)
    compile(result,'formal_qa_lifetime_entry','exec')
    return result


def qa_audit(control,candidate):
    from utils.eval_utils import PAPER_QA_TASKS
    records=[formal.read(root/'qa_scoring_lifetime_rank0.json') for root in (control,candidate)]
    report=validate_pair(*records)
    source_checks=[]
    for root,record in zip((control,candidate),records):
        original=(root/'qa_likelihood_original_rank0.py').read_text()
        selected=(root/'qa_likelihood_selected_rank0.py').read_text()
        source_checks.append(formal.sha(root/'qa_likelihood_original_rank0.py')==record['original_source_sha256'] and
            formal.sha(root/'qa_likelihood_selected_rank0.py')==record['selected_source_sha256'] and
            selected==(build_source(original) if record['enabled'] else original))
    report['checks']['actual_selected_sources']=all(source_checks)
    report['checks']['all_paper_tasks']=set(records[0].get('metrics',{}))==set(PAPER_QA_TASKS)|{'acc_avg'}
    report['valid']=all(report['checks'].values())
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--qa-validation-root',type=Path)
    parser.add_argument('--only',choices=('both','qwen06b','qwen17b'),default='both')
    known,remaining=parser.parse_known_args()
    if '--skip-qa-eval' in remaining:
        raise RuntimeError('QA lifetime adapter requires complete QA evaluation')
    if '--run' in remaining:
        if known.qa_validation_root is None:
            raise RuntimeError('--run requires --qa-validation-root with completed server full-QA evidence')
        models=['Qwen3-0.6B','Qwen3-1.7B']
        if known.only!='both':
            models=['Qwen3-0.6B' if known.only=='qwen06b' else 'Qwen3-1.7B']
        require_full_qa_validation(known.qa_validation_root,models)
    sys.argv=[sys.argv[0],*remaining,'--only',known.only]
    formal.FORMAL_TOOL_SOURCES += ('tools/qa_score_lifetime_prototype.py',
        'tools/qa_scoring_lifetime_install.py','tools/run_storage_formal_qa_lifetime.py',
        'tools/audit_qa_memory_rows_completed.py','tools/audit_qa_score_lifetime_component.py')
    original_entry=formal.build_entry
    formal.build_entry=lambda:entry(original_entry())
    original_audit=global_runner.audit_pair
    def audit_pair(control,candidate,ranks,bindings):
        result=original_audit(control,candidate,ranks,bindings)
        qa=qa_audit(control,candidate)
        result['qa_scoring_lifetime']=qa
        result['checks']['complete_exact_qa_scoring_lifetime']=qa['valid']
        result['valid']=result['capacity_valid']=all(result['checks'].values())
        result['strict_interval_point_valid']=result['strict_interval_point_valid'] and qa['valid']
        return result
    global_runner.audit_pair=audit_pair
    global_runner.main()


if __name__=='__main__': main()
