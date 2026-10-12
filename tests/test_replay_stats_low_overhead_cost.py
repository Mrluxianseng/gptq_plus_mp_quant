import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_replay_stats_low_overhead_cost as cost


class CostAdmissionTests(unittest.TestCase):
    def test_actual_reference_method_mutations_rejected(self):
        reference = cost.ROOT/'outputs/full_pipeline_profile_qwen06b-replay-stats-full-20261012-v2-p1-r1-candidate'
        manifest = cost.read(reference/'manifest.json')
        record = dict(metrics_by_rank=[dict(exact_phase_memory_profile=False, component_event_profile=False,
                      heldout_layer_output_capture_after_timer=True, shared_cpu_method_storage_observers_retained=True)],
                      commands={s:cost.timing.without_profile(manifest[s+'_command']) for s in ('stage0','stage2')})
        names = ['layer_output_fingerprints.json','inverse_lifetime_stage2_rank0.json',
                 'dead_subgroup_input_stage2_rank0.json','replay_stats_lifetime_stage2_rank0.json']
        with tempfile.TemporaryDirectory() as folder, patch.object(cost,'cache_inventory',return_value={'13':'equal'}):
            target=Path(folder)
            originals={n:cost.read(reference/n) for n in names}
            def reset():
                for n,d in originals.items():(target/n).write_text(json.dumps(d))
            reset()
            inv=copy.deepcopy(originals[names[1]])
            inv['weight_write_calls'][0]['before']['pointer']+=4096
            inv['weight_write_calls'][0]['after']['segment_bytes']+=20971520
            (target/names[1]).write_text(json.dumps(inv))
            self.assertTrue(cost.verify_arm(target,reference,record)['valid'])
            for name,mutate in [
                (names[1],lambda d:d['calls'].pop()),
                (names[1],lambda d:d['weight_write_calls'][0].update(pointer_preserved=False)),
                (names[2],lambda d:d['records'][0].update(retained_factors=3)),
                (names[3],lambda d:d['records'][0].update(observed_cuda_storage_bytes=1)),
                (names[0],lambda d:d[27].update(sha256=''))]:
                reset();value=copy.deepcopy(originals[name]);mutate(value)
                (target/name).write_text(json.dumps(value))
                with self.assertRaises(RuntimeError):cost.verify_arm(target,reference,record)
            reset();bad=copy.deepcopy(record);bad['commands']['stage2']+=['--nsamples','8']
            with self.assertRaises(RuntimeError):cost.verify_arm(target,reference,bad)

    def test_bound_timer_adaptation_and_incomplete_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder);entry=target/'entry.py';entry.write_text('# exact worker\n')
            bound={str(entry):cost.digest(entry)}
            arm=cost.adapted_arm(entry,bound)
            self.assertIn(str(entry),arm.__code__.co_consts)
            cost.check_bound(bound)
            entry.write_text('# changed\n')
            with self.assertRaises(RuntimeError):cost.check_bound(bound)
            (target/'status.json').write_text(json.dumps(dict(valid=True,completed_pairs=[])))
            with self.assertRaises(RuntimeError):cost.admit(target,target)


if __name__=='__main__':unittest.main()
