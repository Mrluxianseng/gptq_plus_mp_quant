import ast
import json
from types import SimpleNamespace,MethodType
import pytest
import torch
from transformers import AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
from tools.qa_scoring_lifetime_install import install,validate_pair
from tools.run_storage_formal_qa_lifetime import entry,qa_audit,require_full_qa_validation
from tools import run_storage_method_formal as formal
from tools import run_storage_formal_qa_lifetime as adapter
from utils.eval_utils import PAPER_QA_TASKS


def module(fail=False):
    def qa_eval(model,tokenizer,*a,**kw):
        fake=SimpleNamespace(AUTO_MODEL_CLASS=AutoModelForCausalLM,logits_cache=True,
            batch_size=2,max_length=16,rank=0,device='cpu',
            cache_hook=SimpleNamespace(add_partial=lambda *a:None))
        fake._select_cont_toks=MethodType(HFLM._select_cont_toks,fake)
        fake._model_call=lambda inp,**kw:torch.sin(inp.float().unsqueeze(-1)+torch.arange(79).float()*0.03)
        requests=[(('ctx','yes'),[1,2,3],[4]),(('ctx','no'),[1,2,3],[5]),(None,[6,7,8],[9,10])]
        values=HFLM._loglikelihood_tokens(fake,requests,disable_tqdm=True)
        if fail: raise ValueError('original QA failure')
        return {**{task:float(values[0][0]) for task in PAPER_QA_TASKS},'acc_avg':float(values[1][0])}
    return SimpleNamespace(qa_eval=qa_eval)


def test_existing_evaluator_scores_and_restoration(tmp_path):
    original=HFLM._loglikelihood_tokens
    model=SimpleNamespace(config=SimpleNamespace(use_cache=True))
    roots=[tmp_path/'control',tmp_path/'candidate']
    records=[]
    for enabled,root in zip((False,True),roots):
        mod=module()
        install(mod,enabled=enabled,destination=root)
        values=mod.qa_eval(model,None,16)
        assert HFLM._loglikelihood_tokens is original
        assert len(values)==11 and model.config.use_cache is True
        records.append(json.loads((root/'qa_scoring_lifetime_rank0.json').read_text()))
        with pytest.raises(RuntimeError,match='evidence already exists'): mod.qa_eval(model,None,16)
        assert HFLM._loglikelihood_tokens is original
        with pytest.raises(RuntimeError,match='already installed'): install(mod,enabled=enabled,destination=root)
    assert validate_pair(*records)['valid']
    assert qa_audit(*roots)['valid']
    assert not torch.cuda.is_initialized()


def test_original_failure_kept_and_method_restored(tmp_path):
    original=HFLM._loglikelihood_tokens
    mod=module(fail=True)
    install(mod,enabled=True,destination=tmp_path)
    with pytest.raises(ValueError,match='original QA failure'):
        mod.qa_eval(SimpleNamespace(config=SimpleNamespace(use_cache=True)),None)
    assert HFLM._loglikelihood_tokens is original
    record=json.loads((tmp_path/'qa_scoring_lifetime_rank0.json').read_text())
    assert record['completed'] is False and record['error_type']=='ValueError'


def test_generated_actual_entry_uses_candidate_flag_and_single_install():
    source=entry(formal.build_entry())
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_install_qa_lifetime']
    assert len(calls)==1
    assert source.index('_install_qa_lifetime(_qa_lifetime_eval')<source.index("runpy.run_path('ptq.py', run_name='__main__')")
    assert 'enabled=_hinv_release' in source


def test_missing_full_validation_refuses_before_gpu(tmp_path):
    with pytest.raises(RuntimeError,match='Missing or incomplete'):
        require_full_qa_validation(tmp_path,['Qwen3-0.6B'])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('enabled',[False,True])
def test_real_prefix_consumed_flag_still_selects_qa_method(monkeypatch,tmp_path,enabled):
    import os
    from pathlib import Path
    from tools import qa_scoring_lifetime_install as installer
    source=entry(formal.build_entry())
    nodes=ast.parse(source).body
    flag=next(n for n in nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_hinv_release' for t in n.targets))
    removal=next(n for n in nodes if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='_hinv_marker')
    setup=next(i for i,n in enumerate(nodes) if isinstance(n,ast.ImportFrom) and n.module=='tools.qa_scoring_lifetime_install')
    calls=[]
    monkeypatch.setattr(installer,'install',lambda module,**kw:calls.append(kw))
    monkeypatch.setenv('RESEARCH_RUN_DIR',str(tmp_path))
    args=['entry','baseline-profile']+(['--research_hinv_solver_workspace'] if enabled else [])
    fake_sys=SimpleNamespace(argv=args)
    tree=ast.Module(body=[flag,removal,*nodes[setup:setup+3]],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),'actual_flag_to_qa_setup','exec'),
         dict(_hinv_sys=fake_sys,sys=fake_sys,os=os,Path=Path))
    assert '--research_hinv_solver_workspace' not in args
    assert len(calls)==1 and calls[0]['enabled'] is enabled


@pytest.mark.parametrize('only,models',[('both',['Qwen3-0.6B','Qwen3-1.7B']),
    ('qwen06b',['Qwen3-0.6B']),('qwen17b',['Qwen3-1.7B'])])
def test_actual_run_checks_selected_models_before_formal_main(monkeypatch,tmp_path,only,models):
    import sys
    events=[]
    monkeypatch.setattr(sys,'argv',['runner','--run','--qa-validation-root',str(tmp_path),'--only='+only])
    monkeypatch.setattr(adapter,'require_full_qa_validation',lambda root,names:events.append(('validate',names)))
    monkeypatch.setattr(adapter.global_runner,'main',lambda:events.append(('launch',list(sys.argv))))
    monkeypatch.setattr(adapter.global_runner,'audit_pair',adapter.global_runner.audit_pair)
    monkeypatch.setattr(formal,'build_entry',formal.build_entry)
    monkeypatch.setattr(formal,'FORMAL_TOOL_SOURCES',formal.FORMAL_TOOL_SOURCES)
    adapter.main()
    assert events[0]==('validate',models) and events[1][0]=='launch'
    assert '--qa-validation-root' not in events[1][1] and events[1][1][-2:]==['--only',only]
    assert not torch.cuda.is_initialized()


def test_no_evidence_or_skipped_qa_never_calls_formal_main(monkeypatch):
    import sys
    launches=[]
    monkeypatch.setattr(adapter.global_runner,'main',lambda:launches.append(True))
    monkeypatch.setattr(sys,'argv',['runner','--run'])
    with pytest.raises(RuntimeError,match='requires --qa-validation-root'): adapter.main()
    monkeypatch.setattr(sys,'argv',['runner','--run','--skip-qa-eval'])
    with pytest.raises(RuntimeError,match='requires complete QA'): adapter.main()
    assert not launches


@pytest.mark.parametrize('case',['valid','wrong_model','changed_library','missing_tasks'])
def test_validation_admission_checks_model_and_actual_library(monkeypatch,tmp_path,case):
    import hashlib
    import inspect
    from tools import audit_qa_memory_rows_completed as auditor
    root=tmp_path/'Qwen3-0.6B-full';root.mkdir()
    control=dict(model='/models/Qwen3-0.6B',
        likelihood_source_sha256=hashlib.sha256(inspect.getsource(HFLM._loglikelihood_tokens).encode()).hexdigest(),
        model_call_source_sha256=hashlib.sha256(inspect.getsource(HFLM._model_call).encode()).hexdigest())
    if case=='wrong_model': control['model']='/models/Qwen3-1.7B'
    if case=='changed_library': control['likelihood_source_sha256']='changed'
    (root/'control.json').write_text(json.dumps(control))
    monkeypatch.setattr(auditor,'audit',lambda path:dict(valid=True,tasks=list(PAPER_QA_TASKS) if case!='missing_tasks' else ['boolq']))
    if case=='valid': assert require_full_qa_validation(tmp_path,['Qwen3-0.6B'])['Qwen3-0.6B']['valid']
    else:
        with pytest.raises(RuntimeError): require_full_qa_validation(tmp_path,['Qwen3-0.6B'])
    assert not torch.cuda.is_initialized()
