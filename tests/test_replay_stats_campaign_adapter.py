import copy
import json
import unittest
from tools import compare_full_pipeline_profiles as original
from tools.run_replay_stats_memory_campaign import normalized_command, ROOT
from tools.run_replay_stats_full_followup_v2 import prepare_entry


class AdapterTests(unittest.TestCase):
    def test_real_commands_only_storage_switch_is_ignored(self):
        folder = ROOT / "outputs/replay-stats-full-followup-20261012-v1/qwen17b-replay-stats-full-20261012-v1-p1-manifests"
        commands = [json.loads((folder / (side + "-stage2.json")).read_text())["command"] for side in ("control", "candidate")]
        saved = copy.copy(original.SYSTEM_COMMAND_FLAGS)
        self.assertNotEqual(*(original.algorithm_command(c) for c in commands))
        self.assertEqual(*(normalized_command(c) for c in commands))
        self.assertEqual(saved, original.SYSTEM_COMMAND_FLAGS)
        for flag in ("--nsamples", "--blocksize", "--grad_lr", "--seed", "--dataset"):
            changed = list(commands[1])
            changed[changed.index(flag) + 1] = "CHANGED"
            self.assertNotEqual(normalized_command(commands[0]), normalized_command(changed))

    def test_real_full_entry_remains_single_and_fresh(self):
        source = prepare_entry(ROOT / "outputs/hessian-workspace-full-followup-20261011-v2")
        compile(source, "adapter_full", "exec")
        self.assertEqual(source.count("original_fwrd = impl.gptq_fwrd"), 1)
        self.assertIn("activation_replay_stats_install", source)
        self.assertIn("if '--exit_after_precompute' not in sys.argv:", source)
        self.assertNotIn("hessian_bounded_solver_install", source)


if __name__ == "__main__":
    unittest.main()
