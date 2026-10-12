import ast
import gc
from types import SimpleNamespace
import unittest
import weakref

import torch

from tools.hessian_dead_subgroup_input_prototype import RELEASE_BLOCK, build_source


class DeadSubgroupInputTests(unittest.TestCase):
    def test_real_sequential_boundary_and_future_use_rejected(self):
        import inspect
        import textwrap
        from gptq_utils import gptq_plus_utils as impl
        source = textwrap.dedent(inspect.getsource(impl.GPTQPlus.fasterquant))
        result = build_source(source)
        self.assertIn("released_private_input_bytes", result)
        anchor = next(line for line in source.splitlines()
                      if line.strip().startswith("# Number of blocks + refreshes,"))
        indent = anchor[:len(anchor)-len(anchor.lstrip())]
        injected = source.replace(anchor, anchor + "\n" + indent + "print(H_sub)", 1)
        with self.assertRaises(RuntimeError):
            build_source(injected)

    def test_last_consumer_release_keeps_factors_and_rejects_alias(self):
        # Execute precisely the injected statements. Observer is weak and
        # cannot artificially keep the dead input alive.
        body = ast.parse(RELEASE_BLOCK).body
        function = ast.FunctionDef(name="run", args=ast.arguments(posonlyargs=[],
            args=[ast.arg(arg=n) for n in ("self", "H_sub", "subgroup_states", "hessian_cpu_offload")],
            kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=body + [ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))],
            decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {}
        exec(compile(module, "test_exact_release_block", "exec"), namespace)
        records = []
        helper = SimpleNamespace(layer_name="mlp.down_proj", layer_idx=27, alpha=0,
                                 _dead_subgroup_input_records=records)
        factor = torch.eye(5)
        private = torch.eye(5)
        weak = weakref.ref(private)
        result = namespace["run"](helper, private, [{"Hinv":factor}], False)
        self.assertNotIn("H_sub", result)
        del private
        gc.collect()
        self.assertIsNone(weak())
        self.assertTrue(torch.equal(factor, torch.eye(5)))
        self.assertEqual(records[0]["released_private_input_bytes"], 100)
        with self.assertRaises(RuntimeError):
            namespace["run"](helper, factor, [{"Hinv":factor}], False)
        # Short-circuit guards must not touch a private input that an existing
        # CPU-offload path has already removed, or unrelated module inputs.
        for name, alpha, hcpu in (("mlp.down_proj", 0, True),
                                  ("mlp.up_proj", 0, False),
                                  ("mlp.down_proj", 1, False)):
            other = SimpleNamespace(layer_name=name, alpha=alpha)
            skipped = namespace["run"](other, None, [], hcpu)
            self.assertIsNone(skipped["H_sub"])


if __name__ == "__main__":
    unittest.main()
