"""Avoid an unused returned KV cache only in HFLM causal scoring forwards."""
import ast
import inspect
import textwrap


def build_method(original):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1 or functions[0].name!='_model_call':
        raise RuntimeError('Unexpected HFLM scoring forward')
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n)=='self.model(inps)']
    if len(calls)!=1: raise RuntimeError('Causal scoring forward contract changed')
    calls[0].keywords.append(ast.keyword(arg='use_cache',value=ast.Constant(False)))
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__)
    exec(compile(tree,'qa_scoring_no_returned_kv','exec'),namespace)
    return namespace['_model_call']
