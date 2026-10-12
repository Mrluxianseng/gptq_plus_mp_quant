import copy
from pathlib import Path
import unittest
from tools.run_replay_weight_compact_screen import prepare_entry,compact_coverage
from tools.run_replay_stats_full_followup_v2 import read

ROOT=Path(__file__).resolve().parents[1]


class CompactScreenTests(unittest.TestCase):
    def test_real_source_chain_single_wrapper(self):
        source=prepare_entry(ROOT/'outputs/hessian-workspace-full-followup-20261011-v2')
        compile(source,'compact_screen_entry','exec')
        self.assertEqual(source.count('original_fwrd = impl.gptq_fwrd'),1)
        self.assertLess(source.index('_size_policy_helper='),source.index('_compact_original_builder='))
        self.assertLess(source.index('_compact_cleanup_install(impl'),source.index('_replay_stats_source=_replay_stats_install(impl'))
        self.assertIn("if '--exit_after_precompute' not in sys.argv:",source)

    def test_missing_cleanup_and_storage_alias_coverage_rejected(self):
        comp=read(ROOT/'outputs/replay-weight-compact-cuda-20261012-v1/status.json')
        expected=comp['models'][0]['records'][0]
        rows=[dict(copy.deepcopy(expected),layer=i,rescheduled_existing_cleanup_calls=1) for i in range(28)]
        candidate=dict(enabled=True,helper_unchanged=True,records=rows)
        control=dict(enabled=False,helper_unchanged=True,records=[dict(copy.deepcopy(r),enabled=False,released=False) for r in rows])
        cleanups=[dict(enabled=i==1,pending_layers=[],helper_unchanged=True) for i in range(2)]
        self.assertTrue(compact_coverage(control,candidate,cleanups,expected))
        for field,value in [('active_pool_bytes_after_layer_release',1),('shared_storage_count',2),('rescheduled_existing_cleanup_calls',2)]:
            bad=copy.deepcopy(candidate);bad['records'][25][field]=value
            self.assertFalse(compact_coverage(control,bad,cleanups,expected))
        bad=copy.deepcopy(candidate);bad['records'].pop()
        self.assertFalse(compact_coverage(control,bad,cleanups,expected))


if __name__=='__main__':unittest.main()
