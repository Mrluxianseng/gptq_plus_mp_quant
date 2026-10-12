import unittest
from datetime import datetime, timedelta

from tools.analyze_memory_time_integral import integrate


class IntegralTests(unittest.TestCase):
    def samples(self, times, values):
        origin = datetime(2026, 10, 10)
        return [(origin + timedelta(seconds=t), v) for t, v in zip(times, values)]

    def test_unequal_sampling_is_time_weighted(self):
        result = integrate(self.samples([0, .1, .4], [0, 100, 100]), .5)
        self.assertAlmostEqual(result["time_weighted_device_memory_mib"], 87.5)
        self.assertAlmostEqual(result["device_memory_mib_seconds"], 35)

    def test_gap_is_rejected_not_filled(self):
        with self.assertRaisesRegex(ValueError, "sampling interval"):
            integrate(self.samples([0, .1, 2], [10, 10, 10]), .5)

    def test_clock_reversal_is_rejected(self):
        with self.assertRaises(ValueError):
            integrate(self.samples([0, .1, .05], [10, 10, 10]), .5)

    def test_single_sample_is_not_an_average(self):
        with self.assertRaises(ValueError):
            integrate(self.samples([0], [10]), .5)


if __name__ == "__main__":
    unittest.main()
