from tools.audit_qa_formal_source_handoff import sources,contract


def test_actual_formal_source_contract_and_missing_native_close_refusal():
    texts,bindings=sources()
    assert len(bindings)==5 and contract(*texts)['valid']
    bad=list(texts);bad[1]=bad[1].replace('memory_profiler.finish(status="quantization_complete")','memory_profiler.mark("not_finished")')
    result=contract(*bad)
    assert not result['valid'] and not result['checks']['native_quant_finish_before_return']


def test_missing_kl_close_reduced_tasks_or_evaluator_limit_refused():
    texts,_=sources()
    bad=list(texts);bad[3]=bad[3].replace('_profile.finish(status="completed")','pass')
    assert not contract(*bad)['valid']
    bad=list(texts);bad[4]=bad[4].replace('task_manager=task_manager)','task_manager=task_manager, limit=1)')
    assert not contract(*bad)['valid']
    bad=list(texts);bad[4]=bad[4].replace('    "boolq",','')
    assert not contract(*bad)['valid']


def test_qa_argument_or_rank_dispatch_handoff_change_refused():
    texts,_=sources()
    bad=list(texts);bad[0]=bad[0].replace('eval_utils.qa_eval(model, tokenizer, args.lm_eval_batch_size)','eval_utils.qa_eval(model, tokenizer, 8)')
    assert not contract(*bad)['valid']
    bad=list(texts);bad[0]=bad[0].replace('            dist_utils.distribute_model(model)','            pass')
    assert not contract(*bad)['valid']
