"""Read-only source-contract preflight, never formal runtime/quality acceptance."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

TASKS=('piqa','hellaswag','arc_easy','arc_challenge','winogrande','lambada_openai',
       'ceval-valid','boolq','openbookqa','social_iqa')


def function(tree,name):
    matches=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(matches)!=1: raise ValueError('Expected unique function '+name)
    return matches[0]


def calls(node,name):
    return [item for item in ast.walk(node) if isinstance(item,ast.Call) and ast.unparse(item.func)==name]


def contract(ptq,gptq,memory,profile,eval_source):
    checks={}
    try:
        main=function(ast.parse(ptq),'main')
        quant=calls(main,'quantize_weights');kl=calls(main,'eval_utils.kl_ppl_eval');qa=calls(main,'eval_utils.qa_eval')
        checks['quant_kl_qa_source_order']=(len(quant)==len(kl)==len(qa)==1 and quant[0].lineno<kl[0].lineno<qa[0].lineno)
        checks['qa_original_arguments']=len(qa)==1 and [ast.unparse(n) for n in qa[0].args]==['model','tokenizer','args.lm_eval_batch_size'] and not qa[0].keywords
        blocks=[node for node in ast.walk(main) if isinstance(node,ast.If) and ast.unparse(node.test)=='run_lm_eval']
        checks['qa_rank0_after_original_dispatch']=len(blocks)==1 and len(blocks[0].body)==2 and [ast.unparse(n) for n in blocks[0].body]==['dist_utils.distribute_model(model)','eval_utils.qa_eval(model, tokenizer, args.lm_eval_batch_size)']
        fw=function(ast.parse(gptq),'gptq_fwrd')
        endings=[node for node in fw.body if isinstance(node,ast.If) and ast.unparse(node.test)=='memory_profiler is not None' and calls(node,'memory_profiler.finish')]
        checks['native_quant_finish_before_return']=(len(endings)==1 and ast.unparse(calls(endings[0],'memory_profiler.finish')[0])=="memory_profiler.finish(status='quantization_complete')" and isinstance(fw.body[-1],ast.Return) and ast.unparse(fw.body[-1].value)=='quantizers' and endings[0].lineno<fw.body[-1].lineno)
        close=function(ast.parse(memory),'_close')
        finals=[node.finalbody for node in ast.walk(close) if isinstance(node,ast.Try) and node.finalbody]
        checks['native_close_marks_inactive']=len(finals)==1 and [ast.unparse(node) for node in finals[0][-2:]]==['self._file.close()','self.phase = None']
        setup=function(ast.parse(profile),'_process_profiled_eval')
        wrapped=[node for node in setup.body if isinstance(node,ast.Try)]
        checks['kl_profiler_closed_normal_and_exception']=len(wrapped)==1 and len(wrapped[0].handlers)==1 and ast.unparse(wrapped[0].handlers[0].type)=='BaseException' and [ast.unparse(node) for node in wrapped[0].handlers[0].body]==["_profile.finish(status='failed')",'raise'] and [ast.unparse(node) for node in wrapped[0].orelse]==["_profile.finish(status='completed')",'return _result']
        tree=ast.parse(eval_source);evaluate=function(tree,'qa_eval')
        task_constants=[node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAPER_QA_TASKS' for t in node.targets)]
        checks['full_ten_task_protocol']=len(task_constants)==1 and ast.literal_eval(task_constants[0].value)==TASKS
        evaluate_calls=calls(evaluate,'lm_eval.simple_evaluate')
        checks['unreduced_original_evaluator_call']=len(evaluate_calls)==1 and [ast.unparse(arg) for arg in evaluate_calls[0].args]==['hflm'] and {k.arg:ast.unparse(k.value) for k in evaluate_calls[0].keywords}=={'tasks':'[task_name]','task_manager':'task_manager'}
        assignments=[ast.unparse(node) for node in ast.walk(evaluate) if isinstance(node,ast.Assign)]
        checks['configured_batch_preserved']='hflm.batch_size_per_gpu = lm_eval_batch_size' in assignments
    except (SyntaxError,ValueError,TypeError,IndexError) as exc:
        checks['source_contract_parse']=False
        return dict(valid=False,checks=checks,error=repr(exc),formal_enabled=False)
    return dict(valid=all(checks.values()),checks=checks,formal_enabled=False,
                scope='Source shape/order/normal-close contract only; not arbitrary control-flow proof or actual GPU/fullQA acceptance')


def sources():
    paths=('ptq.py','gptq_utils/gptq_plus_utils.py','utils/memory_utils.py','tools/run_storage_method_pipeline.py','utils/eval_utils.py')
    texts=[(ROOT/path).read_text() for path in paths]
    setup=[ast.literal_eval(node.value) for node in ast.parse(texts[3]).body if isinstance(node,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='EVALUATION_PROFILE_SETUP' for t in node.targets)]
    if len(setup)!=1: raise ValueError('Expected current formal evaluation profile setup')
    return texts[:3]+setup+[texts[4]],{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    texts,bindings=sources();result=contract(*texts);result['source_sha256']=bindings
    result['auditor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
    if not result['valid']:raise SystemExit(1)
