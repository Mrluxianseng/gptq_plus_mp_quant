import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.prepare_replay_stats_low_overhead_entry import prepare_entry as broken_entry
from tools.prepare_replay_stats_low_overhead_entry_v2 import prepare_entry


class TensorFixture:
    shape=(1,2,3)
    dtype='torch.bfloat16'
    def detach(self):return self
    def contiguous(self):return self
    def view(self,_):return self
    def cpu(self):return self
    def numpy(self):return self
    def tobytes(self):return b'original fixture output'


class LayerFixture:
    def __init__(self):self.hook=None;self.removed=False
    def register_forward_hook(self,hook):
        self.hook=hook
        return SimpleNamespace(remove=lambda:setattr(self,'removed',True))


def evaluation_wrapper(source,destination,*,fail=False):
    tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_layer_trace_eval')
    layers=[LayerFixture() for _ in range(28)]
    analyzer=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
    def original(*args,**kwargs):
        for _ in range(2):
            for layer in layers:layer.hook(layer,(),TensorFixture())
        if fail:raise ValueError('original evaluator fixture failure')
        return (25,.3)
    namespace=dict(hashlib=hashlib,json=json,torch=SimpleNamespace(Tensor=TensorFixture,uint8=object()),
                   destination=destination,_layer_trace_original_eval=original)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_post_timer_evaluation_wrapper','exec'),namespace)
    return namespace['_layer_trace_eval'],analyzer,layers


class FingerprintDestinationTests(unittest.TestCase):
    def test_real_wrapper_runtime_binding_and_all28_file(self):
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)
            old,a,_=evaluation_wrapper(broken_entry(),destination)
            with self.assertRaises(NameError):old(None,a)
            new,a,layers=evaluation_wrapper(prepare_entry(),destination)
            self.assertEqual(new(None,a),(25,.3))
            report=json.loads((destination/'layer_output_fingerprints.json').read_text())
            self.assertEqual(len(report),28)
            self.assertTrue(all(r['forward_count']==2 and len(r['sha256'])==64 and r['shapes'] for r in report))
            self.assertTrue(all(layer.removed for layer in layers))

    def test_original_eval_error_and_all_other_ast_nodes_preserved(self):
        old,new=ast.parse(broken_entry()),ast.parse(prepare_entry())
        oldfunction=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='_layer_trace_eval')
        changed=0
        for n in ast.walk(oldfunction):
            if isinstance(n,ast.Name) and n.id=='root':n.id='destination';changed+=1
        self.assertEqual(changed,1);self.assertEqual(ast.dump(old),ast.dump(new))
        with tempfile.TemporaryDirectory() as folder:
            wrapper,a,layers=evaluation_wrapper(prepare_entry(),Path(folder),fail=True)
            with self.assertRaisesRegex(ValueError,'original evaluator'):wrapper(None,a)
            self.assertTrue(all(layer.removed for layer in layers))


if __name__=='__main__':unittest.main()
