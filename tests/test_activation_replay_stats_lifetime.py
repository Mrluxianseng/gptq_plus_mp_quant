import ast
from pathlib import Path
import unittest
import weakref

from tools.activation_replay_stats_lifetime_prototype import build_source


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = '''def gptq_fwrd(args, analyzer, dataloader, dev):
    for i in range(2):
        if True:
            gptq = {"m": Box()}
            saliency_dict = {"m": Box()}
            gradients_dict = {"m": Box()}
            refs = [weakref.ref(x["m"]) for x in (gptq, saliency_dict, gradients_dict)]
            args.last_consumer(refs)
            if memory_profiler is not None:
                memory_profiler.mark(f"layer_{i}_activation_replay_and_cleanup", layer=i)
            args.replay(refs)
            if True:
                args.cleanup()
                del gptq
                del saliency_dict, gradients_dict
    return args.model
'''


class Box:
    pass


class ReplayLifetimeTests(unittest.TestCase):
    def test_last_consumer_then_replay_and_next_layer(self):
        for enabled in (False, True):
            records = []
            class Args:
                model = Box()
                def last_consumer(self, refs):
                    self_outer.assertTrue(all(x() is not None for x in refs))
                def replay(self, refs):
                    records.append([x() is None for x in refs])
                def cleanup(self):
                    pass
            self_outer = self
            args = Args()
            namespace = dict(Box=Box, weakref=weakref, memory_profiler=None)
            source = build_source(FIXTURE) if enabled else FIXTURE
            exec(compile(source, "fixture", "exec"), namespace)
            result = namespace["gptq_fwrd"](args, None, None, None)
            self.assertIs(result, args.model)
            self.assertEqual(records, [[enabled]*3, [enabled]*3])

    def test_real_source_compile_and_new_consumer_rejected(self):
        source = (ROOT / "gptq_utils/gptq_plus_utils.py").read_text(encoding="utf8")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "gptq_fwrd")
        actual = "\n".join(source.splitlines()[node.lineno-1:node.end_lineno]) + "\n"
        proposed = build_source(actual)
        self.assertEqual(proposed.count("del gptq"), 1)
        self.assertEqual(proposed.count("del saliency_dict, gradients_dict"), 1)
        compile(proposed, "real_replay_lifetime", "exec")
        corrupt = FIXTURE.replace("args.replay(refs)", "args.replay(gradients_dict)")
        with self.assertRaisesRegex(RuntimeError, "downstream consumer"):
            build_source(corrupt)


if __name__ == "__main__":
    unittest.main()
