import ast
from pathlib import Path
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import weakref

from tools.activation_replay_stats_install import build_entry, observed_source, storage_metadata, install
from tests.test_activation_replay_stats_lifetime import FIXTURE, Box, ROOT


class ReplayInstallTests(unittest.TestCase):
    def test_installed_function_keeps_live_module_globals(self):
        impl = ModuleType('fixture_replay_module')
        impl.__dict__.update(Box=Box, weakref=weakref, memory_profiler=None,
            torch=SimpleNamespace(is_tensor=lambda x:False,
                                  cuda=SimpleNamespace(memory_allocated=lambda:0)))
        exec(FIXTURE, impl.__dict__)
        records, markers = [], []
        with patch('tools.activation_replay_stats_install.inspect.getsource', return_value=FIXTURE):
            install(impl, enabled=True, records=records)
        self.assertIs(impl.gptq_fwrd.__globals__, impl.__dict__)
        # A wrapper installed AFTER compilation must remain visible at runtime.
        impl.memory_profiler=SimpleNamespace(mark=lambda *a,**kw:markers.append(kw['layer']))
        args=SimpleNamespace(model=Box(),last_consumer=lambda refs:None,
                             replay=lambda refs:None,cleanup=lambda:None)
        self.assertIs(impl.gptq_fwrd(args,None,None,None), args.model)
        self.assertEqual(markers,[0,1])
        self.assertEqual([r['layer'] for r in records],[0,1])
        self.assertTrue(all(r['enabled'] and r['unique_cuda_storage_count']==0 for r in records))

    def test_shared_observation_before_release(self):
        for enabled in (False, True):
            records = []
            class Args:
                model = Box()
                def last_consumer(self, refs):
                    self_outer.assertTrue(all(r() is not None for r in refs))
                def replay(self, refs):
                    self_outer.assertEqual([r() is None for r in refs], [enabled]*3)
                def cleanup(self):
                    pass
            self_outer = self
            def observe(i, g, s, d):
                self.assertTrue(all(x['m'] is not None for x in (g, s, d)))
                records.append(i)
            namespace = dict(Box=Box, weakref=weakref, memory_profiler=None, _replay_stats_record=observe)
            exec(observed_source(FIXTURE, enabled=enabled), namespace)
            namespace['_replay_stats_generated_fwrd'](Args(), None, None, None)
            self.assertEqual(records, [0, 1])

    def test_real_entry_and_real_forward_compile(self):
        raw = (ROOT/'outputs/hessian-workspace-full-followup-20261011-v2/entry.py').read_text()
        result = build_entry(raw, installer_sha='i'*64, prototype_sha='p'*64)
        self.assertLess(result.index('_replay_stats_install(impl'), result.index('original_fwrd = impl.gptq_fwrd'))
        self.assertEqual(result.count("runpy.run_path('ptq.py', run_name='__main__')"), 1)
        source = (ROOT/'gptq_utils/gptq_plus_utils.py').read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name=='gptq_fwrd')
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        body = '\n'.join(source.splitlines()[first-1:node.end_lineno])+'\n'
        for enabled in (False, True):
            generated = observed_source(body, enabled=enabled)
            self.assertTrue(generated.startswith('@torch.no_grad()'))
            compile(generated, 'real_observed_forward', 'exec')
        with self.assertRaisesRegex(RuntimeError, 'entry changed'):
            build_entry(raw.replace('original_fwrd = impl.gptq_fwrd','original_fwrd = other'),installer_sha='i',prototype_sha='p')

    def test_storage_deduplication_contains_no_tensor(self):
        class Tensor:
            is_cuda=True
            device='cuda:0'
            def numel(self): return 16
            def untyped_storage(self): return self
            def data_ptr(self): return 123
            def nbytes(self): return 256
        value=Tensor()
        torch=SimpleNamespace(is_tensor=lambda x:isinstance(x,Tensor))
        result=storage_metadata(torch,{'m':SimpleNamespace(gradients=value)},{'m':[value]}, {'m':value})
        self.assertEqual(result,dict(unique_cuda_storage_count=1,observed_cuda_storage_bytes=256))


if __name__=='__main__':
    unittest.main()
