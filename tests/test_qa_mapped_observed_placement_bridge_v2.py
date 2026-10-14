"""Preserve v1's test-side indentation failure; correct the source reader only."""
from test_qa_mapped_observed_placement_bridge import *
import textwrap


def test_native_control_ast_only_adds_observer_callback():
    original=ast.parse(textwrap.dedent(inspect.getsource(HFLM._loglikelihood_tokens)))
    generated=ast.parse(control_batch_source(inspect.getsource(HFLM._loglikelihood_tokens)))
    class Remove(ast.NodeTransformer):
        def visit_Expr(self,node):
            return None if ast.unparse(node)=='_qa_release_batch()' else node
    assert ast.dump(original,include_attributes=False)==ast.dump(Remove().visit(generated),include_attributes=False)
