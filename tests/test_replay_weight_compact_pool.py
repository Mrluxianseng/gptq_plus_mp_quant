"""Guard real forward source placement and unsupported aliases on CPU."""
import ast
from pathlib import Path
import unittest
from tools.activation_replay_stats_install import observed_source
from tools.replay_weight_compact_pool_prototype import inject,build_entry
from tools.run_replay_stats_full_followup_v2 import prepare_entry,digest

ROOT=Path(__file__).resolve().parents[1]


class CompactTests(unittest.TestCase):
    def source(self):
        text=(ROOT/'gptq_utils/gptq_plus_utils.py').read_text(encoding='utf-8')
        tree=ast.parse(text)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gptq_fwrd')
        return ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])

    def test_real_last_consumer_and_release_boundaries(self):
        source=self.source()
        for enabled in (False,True):
            proposed=inject(observed_source(source,enabled=enabled),enabled=enabled)
            self.assertEqual(proposed.count('_replay_compact_prepare(layer, i)'),1)
            self.assertLess(proposed.index('_replay_compact_prepare(layer, i)'),proposed.index('memory_profiler.mark(f"layer_{i}_activation_replay_and_cleanup"'))
            self.assertLess(proposed.index('layer_manager.release_layer(i, layer, update_master=True'),proposed.index('_replay_compact_finish('))
            if enabled:self.assertLess(proposed.index('del saliency_dict, gradients_dict'),proposed.index('_replay_compact_prepare(layer, i)'))
        with self.assertRaises(RuntimeError):inject(observed_source(source,enabled=False),enabled=True)

    def test_fresh_entry_live_globals_and_single_wrapper(self):
        raw=prepare_entry(ROOT/'outputs/hessian-workspace-full-followup-20261011-v2')
        source=build_entry(raw,helper_sha=digest(ROOT/'tools/replay_weight_compact_pool_prototype.py'))
        compile(source,'compact_real_entry','exec')
        self.assertEqual(source.count('original_fwrd = impl.gptq_fwrd'),1)
        self.assertLess(source.index('_compact_install(impl'),source.index('_replay_stats_install(impl'))
        self.assertIn("if '--exit_after_precompute' not in sys.argv:",source)
        self.assertNotIn('# Report allocator peaks',source)


if __name__=='__main__':unittest.main()
