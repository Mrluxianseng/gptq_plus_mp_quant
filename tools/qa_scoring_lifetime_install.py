"""Install exact scored-row normalization only around the existing QA evaluator."""
import hashlib
import inspect
import json
from pathlib import Path

from tools.qa_score_lifetime_prototype import build_method,build_source


def install(module,*,enabled,destination,rank=0):
    if getattr(module,'_qa_scoring_lifetime_installed',False):
        raise RuntimeError('QA scoring lifetime wrapper already installed')
    original_qa=module.qa_eval
    destination=Path(destination)

    def qa_eval(model,tokenizer,*args,**kwargs):
        filenames=(f'qa_likelihood_original_rank{rank}.py',f'qa_likelihood_selected_rank{rank}.py',
                   f'qa_scoring_lifetime_rank{rank}.json')
        if any((destination/name).exists() for name in filenames):
            raise RuntimeError('QA evidence already exists; use a fresh experiment directory')
        from lm_eval.models.huggingface import HFLM
        original=HFLM._loglikelihood_tokens
        source=inspect.getsource(original)
        selected=build_method(original) if enabled else original
        selected_source=build_source(source) if enabled else source
        input_digest=hashlib.sha256();score_digest=hashlib.sha256();counts=[]
        config_cache_before=model.config.use_cache
        def observe(self,requests,*a,**kw):
            input_digest.update(json.dumps(requests,ensure_ascii=False).encode())
            scores=selected(self,requests,*a,**kw)
            score_digest.update(json.dumps(scores).encode())
            counts.append(len(scores))
            return scores
        HFLM._loglikelihood_tokens=observe
        record=dict(schema='qa-scored-rows-lifetime-v1',enabled=bool(enabled),rank=int(rank),
            original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            selected_source_sha256=hashlib.sha256(selected_source.encode()).hexdigest(),
            config_use_cache_before=config_cache_before,
            scope='existing full QA; scoring lifetime only; no KV/batch/precision/model-call changes')
        try:
            result=original_qa(model,tokenizer,*args,**kwargs)
            record.update(completed=True,metrics=result)
            return result
        except BaseException as exc:
            record.update(completed=False,error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            HFLM._loglikelihood_tokens=original
            record.update(request_counts=counts,request_sha256=input_digest.hexdigest(),
                score_sha256=score_digest.hexdigest(),config_use_cache_after=model.config.use_cache)
            destination.mkdir(parents=True,exist_ok=True)
            for filename,text in ((f'qa_likelihood_original_rank{rank}.py',source),
                                  (f'qa_likelihood_selected_rank{rank}.py',selected_source)):
                with (destination/filename).open('x',encoding='utf-8',newline='\n') as stream: stream.write(text)
            with (destination/f'qa_scoring_lifetime_rank{rank}.json').open('x',encoding='utf-8') as stream:
                json.dump(record,stream,indent=2)

    module.qa_eval=qa_eval
    module._qa_scoring_lifetime_installed=True


def validate_pair(control,candidate):
    checks=dict(completed=control.get('completed') is True and candidate.get('completed') is True,
        schema=control.get('schema')==candidate.get('schema')=='qa-scored-rows-lifetime-v1',
        method_enabled=control.get('enabled') is False and candidate.get('enabled') is True,
        rank=control.get('rank')==candidate.get('rank')==0,
        coverage=bool(control.get('request_counts')) and all(type(x) is int and x>0 for x in control.get('request_counts',[])),
        full_ten_metrics=isinstance(control.get('metrics'),dict) and len(control['metrics'])==11 and 'acc_avg' in control['metrics'])
    for key in ('original_source_sha256','request_counts','request_sha256','score_sha256','metrics','config_use_cache_before','config_use_cache_after'):
        checks[key+'_exact']=key in control and key in candidate and control[key]==candidate[key]
    checks['config_preserved']=control.get('config_use_cache_before')==control.get('config_use_cache_after')==candidate.get('config_use_cache_before')==candidate.get('config_use_cache_after')
    checks['selected_source_changed']=control.get('original_source_sha256')==control.get('selected_source_sha256') and candidate.get('selected_source_sha256')!=candidate.get('original_source_sha256')
    return dict(valid=all(checks.values()),checks=checks)
