import unittest
from tools.audit_completed_memory_pair_capacity import capacity_gate


class HardwareGateTests(unittest.TestCase):
    def test_global_regression_rejected_despite_other_validity(self):
        checks=dict(quality=True,source=True,method=True,complete_points=True)
        self.assertTrue(capacity_gate(checks,dict(allocated=(100,90),reserved=(120,110),nvml=(140,130))))
        for name in ('allocated','reserved','nvml'):
            peaks=dict(allocated=(100,90),reserved=(120,110),nvml=(140,130));peaks[name]=(100,101)
            self.assertFalse(capacity_gate(checks,peaks))

    def test_missing_quality_sources_or_telemetry_rejected(self):
        checks=dict(quality=True,source=True,method=True,complete_points=True)
        for name in checks:
            broken=dict(checks);broken[name]=False
            self.assertFalse(capacity_gate(broken,dict(allocated=(100,90))))
        self.assertFalse(capacity_gate(checks,{}))


if __name__=='__main__':unittest.main()
