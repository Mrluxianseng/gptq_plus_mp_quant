import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_replay_stats_low_overhead_cost_v2 as cost


class RecoveredCostAdmissionTests(unittest.TestCase):
    def test_actual_recovery_tags_and_bindings_and_incomplete_rejection(self):
        actual=cost.ROOT/'outputs/qwen06-hardware-capacity-recovery-20261012-v2'
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);calls=[]
            def observed(tag,bindings,**options):
                calls.append((tag,str(bindings)))
                return dict(capacity_valid=True)
            with patch.object(cost,'audit',side_effect=observed):
                refs,rows=cost.admit(actual,out)
            self.assertEqual(len(rows),6)
            self.assertEqual(calls[2][0],'qwen06b-replay-stats-hardware-20261012-recovery-v2-p3')
            self.assertEqual(Path(calls[2][1]),actual/'source_bindings.json')
            self.assertIn('qwen06-hardware-capacity-followup-20261012-v1',calls[1][1])
            self.assertNotIn('qwen06b-replay-stats-hardware-20261012-v1-p3',[x[0] for x in calls])
            invalid=cost.read(actual/'status.json');invalid['completed_pairs'][2]['capacity_valid']=False
            with patch.object(cost,'read',return_value=invalid):
                with self.assertRaises(RuntimeError):cost.admit(actual,out)


if __name__=='__main__':unittest.main()
