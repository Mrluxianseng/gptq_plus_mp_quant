"""Keep original KV semantics/layout while using the official cache offloader."""
import ast
import inspect
import textwrap
from transformers.cache_utils import DynamicCache


def build_method(original,*,offloading=True):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1 or functions[0].name!='_model_call': raise RuntimeError('Unexpected scoring source')
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n)=='self.model(inps)']
    if len(calls)!=1: raise RuntimeError('Expected original causal scoring call')
    value=ast.parse('_QaDynamicCache(config=self.model.config, offloading='+repr(bool(offloading))+')',mode='eval').body
    calls[0].keywords.append(ast.keyword(arg='past_key_values',value=value))
    guard=ast.parse("if not self.model.config.use_cache: raise RuntimeError('Cache offload requires original enabled KV semantics')").body[0]
    functions[0].body.insert(0,guard)
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__,_QaDynamicCache=DynamicCache)
    exec(compile(tree,'qa_scoring_cache_storage_offload','exec'),namespace)
    return namespace['_model_call']
