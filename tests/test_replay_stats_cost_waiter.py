import unittest
from tools.run_replay_stats_cost_after_capacity import final_admission


class CostWaiterTests(unittest.TestCase):
    def test_only_complete_native_admission_allows_cost(self):
        f=dict(valid=True,completed_pairs=[dict(repeat=i,capacity_valid=True) for i in (1,2,3)])
        a=dict(capacity_valid=True,method='stable',tag='qwen06b-replay-stats-hardware-20261012-v1-p3',
               checks=dict(quality=True,cache=True,sources=True,method=True,telemetry=True))
        self.assertTrue(final_admission(f,a))
        for key in a['checks']:
            bad={**a,'checks':{**a['checks'],key:False}}
            self.assertFalse(final_admission(f,bad))
        self.assertFalse(final_admission({**f,'completed_pairs':f['completed_pairs'][:2]},a))
        self.assertFalse(final_admission(f,{**a,'method':'compact'}))
        self.assertFalse(final_admission(f,{**a,'capacity_valid':False}))


if __name__=='__main__':unittest.main()
