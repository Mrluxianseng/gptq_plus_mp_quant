"""CPU acceptance guards using completed real metadata; no CUDA allocation."""
import copy
from pathlib import Path
import unittest

from tools.run_weight_size_policy_screen import policy_coverage, prepare_entry
from tools.run_replay_stats_full_followup_v2 import read

ROOT = Path(__file__).resolve().parents[1]


class SizePolicyScreenTests(unittest.TestCase):
    def metadata(self):
        run = ROOT / "outputs/full_pipeline_profile_qwen06b-replay-stats-full-20261012-v2-p1-r1-candidate"
        inverse = read(run / "inverse_lifetime_stage2_rank0.json")
        dead = read(run / "dead_subgroup_input_stage2_rank0.json")
        for row in inverse["weight_write_calls"]:
            row.update(enabled=False, requested_enabled=True,
                       min_stable_bytes=20971520, policy="native-medium-size-placement")
        return inverse, dead

    def test_selected_vs_requested_and_missing_coverage(self):
        inverse, dead = self.metadata()
        self.assertTrue(policy_coverage(inverse, dead, enabled=True))
        wrong = copy.deepcopy(inverse)
        wrong["weight_write_calls"][25]["enabled"] = True
        self.assertFalse(policy_coverage(wrong, dead, enabled=True))
        wrong = copy.deepcopy(inverse)
        wrong["weight_write_calls"][25]["requested_enabled"] = False
        self.assertFalse(policy_coverage(wrong, dead, enabled=True))
        wrong = copy.deepcopy(inverse)
        wrong["calls"].pop()
        self.assertFalse(policy_coverage(wrong, dead, enabled=True))

    def test_authentic_entry_fresh_stage0_and_single_measurement_wrapper(self):
        source = prepare_entry(ROOT / "outputs/hessian-workspace-full-followup-20261011-v2")
        compile(source, "size_policy_screen_entry", "exec")
        self.assertEqual(source.count("original_fwrd = impl.gptq_fwrd"), 1)
        self.assertIn("if '--exit_after_precompute' not in sys.argv:", source)
        self.assertLess(source.index("_dead_original_builder="), source.index("_size_policy_helper="))
        self.assertLess(source.index("_replay_stats_install(impl"), source.index("original_fwrd = impl.gptq_fwrd"))
        self.assertNotIn("# Report allocator peaks", source)
        self.assertNotIn("hessian_chunked_finite", source)


if __name__ == "__main__":
    unittest.main()
