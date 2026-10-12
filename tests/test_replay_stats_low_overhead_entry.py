from pathlib import Path
import unittest
from tools.prepare_replay_stats_low_overhead_entry import prepare_entry
from tools.run_low_overhead_speed_campaign import without_profile
from tools.run_replay_stats_full_followup_v2 import normalized_command,read

ROOT=Path(__file__).resolve().parents[1]


class LowOverheadTests(unittest.TestCase):
    def test_actual_source_keeps_math_hooks_and_post_timer_fingerprints(self):
        source=prepare_entry()
        compile(source,'accepted_replay_stats_timing_entry','exec')
        self.assertEqual(source.count('original_fwrd = impl.gptq_fwrd'),1)
        self.assertLess(source.index('_replay_stats_install(impl'),source.index('original_fwrd = impl.gptq_fwrd'))
        self.assertEqual(source.count('runpy.run_path('),1)
        self.assertIn('shared_cpu_method_storage_observers_retained',source)
        self.assertIn('heldout_layer_output_capture_after_timer',source)
        self.assertIn('timing stage2 must consume its verified fresh cache',source)
        self.assertNotIn('CudaPhaseMemoryProfiler',source)
        self.assertNotIn('hessian_weight_size_policy',source)
        self.assertNotIn('compact_pool',source)

    def test_only_profile_controls_removed_from_actual_commands(self):
        folder=ROOT/'outputs/replay-stats-full-followup-20261012-v2/qwen06b-replay-stats-full-20261012-v2-p1-manifests'
        for stage in ('stage0','stage2'):
            commands=[read(folder/(side+'-'+stage+'.json'))['command'] for side in ('control','candidate')]
            minimal=[without_profile(c) for c in commands]
            self.assertEqual(normalized_command(minimal[0]),normalized_command(minimal[1]))
            for old,new in zip(commands,minimal):
                for flag in ('--nsamples','--seq_len','--blocksize','--seed','--grad_lr'):
                    self.assertEqual(old[old.index(flag)+1],new[new.index(flag)+1])
                self.assertNotIn('--enable_quant_profile',new)
                self.assertNotIn('--enable_memory_profile',new)


if __name__=='__main__':unittest.main()
