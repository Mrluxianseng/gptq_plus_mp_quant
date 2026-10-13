"""Move row-independent log_softmax after continuation slicing in real HFLM.

No batch, request ordering, model call, score formula, dtype or cache change.
Only scored rows are normalized; raw full-batch logits die at their last use.
"""
from __future__ import annotations
import ast
import inspect
import textwrap


def build_source(source):
    tree=ast.parse(textwrap.dedent(source))
    functions=[x for x in tree.body if isinstance(x,ast.FunctionDef)]
    if len(functions)!=1 or functions[0].name!='_loglikelihood_tokens':
        raise RuntimeError('Unexpected HFLM likelihood source')
    fn=functions[0]
    assignments=[x for x in ast.walk(fn) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='multi_logits' for t in x.targets)]
    if len(assignments)!=1: raise RuntimeError('Expected one batch-logits normalization')
    assign=assignments[0]; call=assign.value
    if (not isinstance(call,ast.Call) or ast.unparse(call.func)!='F.log_softmax'
        or len(call.args)!=1 or [ast.unparse(k) for k in call.keywords]!=['dim=-1']
        or not isinstance(call.args[0],ast.Call)
        or ast.unparse(call.args[0].func)!='self._model_call'):
        raise RuntimeError('HFLM normalization/model-call contract changed')
    selections=[x for x in ast.walk(fn) if isinstance(x,ast.Assign) and ast.unparse(x).startswith('logits = self._select_cont_toks(')]
    if len(selections)!=1: raise RuntimeError('Expected one continuation slice')
    select=selections[0]
    chunks=[x for x in ast.walk(fn) if isinstance(x,ast.For) and ast.unparse(x.target)=='chunk' and ast.unparse(x.iter)=='chunks']
    if len(chunks)!=1: raise RuntimeError('Batch-loop contract changed')
    batch=chunks[0]
    assign.value=call.args[0]
    class Insert(ast.NodeTransformer):
        def visit_Assign(self,node):
            if node is select:
                return [node,ast.parse('logits = F.log_softmax(logits, dim=-1)').body[0]]
            return node
    Insert().visit(fn)
    batch.body.append(ast.Delete(targets=[ast.Name(id='multi_logits',ctx=ast.Del())]))
    ast.fix_missing_locations(tree)
    result=ast.unparse(tree)+'\n'
    compile(result,'qa_scored_rows_lifetime','exec')
    return result


def build_method(original):
    namespace=dict(original.__globals__)
    source=build_source(inspect.getsource(original))
    exec(compile(source,'qa_scored_rows_lifetime','exec'),namespace)
    return namespace['_loglikelihood_tokens']
