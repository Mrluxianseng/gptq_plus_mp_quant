"""Use the established exactness comparator with an explicit Stage2-only scope.

No fake Stage0 statuses or peaks: omitted precompute is outside this comparison.
"""
import ast
import inspect
import textwrap

from tools import compare_full_pipeline_profiles as base


def compare(control, candidate, *, compare_cache=True):
    for path in (control, candidate):
        if base.read_json(path/'status.json').get('scope') != 'stage2_cached':
            raise ValueError('Cached comparisons require both arms to be Stage2-only')
    tree = ast.parse(textwrap.dedent(inspect.getsource(base.compare)))
    function = tree.body[0]
    matches = [n for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'stages' for t in n.targets)]
    if len(matches) != 1:
        raise RuntimeError('Comparator stage guard changed')
    matches[0].value = ast.Tuple(elts=[ast.Constant('stage2')], ctx=ast.Load())
    class Scope(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 'full_pipeline':
                return ast.copy_location(ast.Constant('stage2_cached'), node)
            return node
    tree = Scope().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(vars(base))
    namespace['GLOBAL_FIELDS'] = tuple(k for k in base.GLOBAL_FIELDS if not k.startswith('stage0_'))
    exec(compile(tree, 'cached_exactness_comparator', 'exec'), namespace)
    result = namespace['compare'](control, candidate, compare_cache=compare_cache)
    result['precompute_measured'] = False
    return result
