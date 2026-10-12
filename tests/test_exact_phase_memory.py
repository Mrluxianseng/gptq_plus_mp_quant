"""CUDA regression tests for exact intervals and cumulative peak preservation."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from utils import memory_utils


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ExactPhaseMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_utils._process_profile_peaks.clear()

    def test_freed_transient_is_measured_and_later_eval_peak_is_included(self):
        with patch.dict(os.environ, {"REALQ_PHASE_PEAK_MODE": "exact"}):
            profiler = memory_utils.CudaPhaseMemoryProfiler(self.temp.name, "cuda:0")
        baseline = torch.cuda.memory_allocated()
        transient = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        del transient
        profiler.mark("next")
        self.assertGreaterEqual(profiler.rows[0]["sampled_peak_allocated_bytes"], baseline + 32 * 1024 * 1024)
        carried_peak = profiler.rows[0]["cumulative_peak_allocated_bytes"]
        profiler.finish()
        self.assertGreaterEqual(memory_utils.get_process_cuda_memory_peaks()[0], carried_peak)
        evaluation = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.assertGreaterEqual(memory_utils.get_process_cuda_memory_peaks()[0], baseline + 64 * 1024 * 1024)
        del evaluation
        rows = [json.loads(line) for line in Path(profiler.path).read_text().splitlines()]
        self.assertEqual(rows[0]["peak_measurement_mode"], "exact")

    def test_default_sample_mode_does_not_reset_cuda_counters(self):
        with patch.dict(os.environ, {"REALQ_PHASE_PEAK_MODE": "sampled"}):
            profiler = memory_utils.CudaPhaseMemoryProfiler(self.temp.name, "cuda:0")
        transient = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        del transient
        before = torch.cuda.max_memory_allocated()
        profiler.mark("next")
        self.assertGreaterEqual(torch.cuda.max_memory_allocated(), before)
        profiler.finish()


if __name__ == "__main__":
    unittest.main()
