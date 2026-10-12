import copy
import unittest
from tools.run_replay_stats_full_followup import ROOT, prepare_entry, replay_coverage


class ReplayFullTests(unittest.TestCase):
    def test_real_fresh_entry_preserves_stage0_and_measurement(self):
        source = prepare_entry(ROOT / "outputs/hessian-workspace-full-followup-20261011-v2")
        compile(source, "replay_fresh", "exec")
        self.assertEqual(source.count("original_fwrd = impl.gptq_fwrd"), 1)
        self.assertIn("activation_replay_stats_install", source)
        self.assertIn("if '--exit_after_precompute' not in sys.argv:", source)
        self.assertNotIn("hessian_bounded_solver_install", source)
        self.assertNotIn("hessian_all_weight_storage", source)
        self.assertNotIn("# Report allocator peaks", source)

    def test_missing_observation_and_storage_change_rejected(self):
        arms = [dict(enabled=enabled, helpers_unchanged=True, records=[
            dict(layer=i, enabled=enabled, boundary="before_activation_replay", observed_cuda_storage_bytes=1024)
            for i in range(28)]) for enabled in (False, True)]
        self.assertTrue(replay_coverage(*arms))
        missing = copy.deepcopy(arms)
        missing[1]["records"].pop()
        self.assertFalse(replay_coverage(*missing))
        mismatch = copy.deepcopy(arms)
        mismatch[1]["records"][27]["observed_cuda_storage_bytes"] += 4
        self.assertFalse(replay_coverage(*mismatch))
        changed = copy.deepcopy(arms)
        changed[1]["helpers_unchanged"] = False
        self.assertFalse(replay_coverage(*changed))


if __name__ == "__main__":
    unittest.main()
