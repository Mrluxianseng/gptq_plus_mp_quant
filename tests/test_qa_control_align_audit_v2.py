import ast
import hashlib
import inspect
import marshal
import torch
from accelerate import hooks
from tools.qa_control_head_observer import forward_chain
from tools.audit_qa_control_align_completed_v2 import installed_forward_sha,audit_source
from tools import audit_qa_control_align_completed as ancestor


def test_expected_sha_uses_real_installed_function_and_preserves_every_other_gate():
    head=torch.nn.Linear(1,1,bias=False,dtype=torch.bfloat16).eval()
    hooks.add_hook_to_module(head,hooks.AlignDevicesHook(execution_device='cpu',io_same_device=True))
    assert forward_chain(head)['forward_code_sha256']==installed_forward_sha()
    original=ast.parse(inspect.getsource(ancestor.audit));new=ast.parse(audit_source())
    class Restore(ast.NodeTransformer):
        def visit_Call(self,node):
            if ast.unparse(node)=='_installed_forward_sha()':
                return ast.parse('hashlib.sha256(marshal.dumps(code[0])).hexdigest()',mode='eval').body
            return self.generic_visit(node)
    assert ast.dump(original,include_attributes=False)==ast.dump(Restore().visit(new),include_attributes=False)
    assert not torch.cuda.is_initialized()
