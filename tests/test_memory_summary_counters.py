"""Protect manuscript accounting from missing or inconsistent telemetry."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_memory_campaign import offload_metrics


class SummaryCountersTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage0.log"
            path.write_text(text)
            return offload_metrics(path)

    def row(self):
        return dict(rank=0, requests=3, requested_bytes=300, copies=2,
                    copied_bytes=200, dedup_hits=1, live_cpu_bytes=0,
                    peak_live_cpu_bytes=200)

    def test_absent_metrics_are_unknown(self):
        self.assertEqual(self.parse("saved 20 tensors\n"), [])

    def test_prefixed_multirank_records(self):
        first, second = self.row(), {**self.row(), "rank": 1}
        text = "\x1b[A[INFO] STAGE0_OFFLOAD_METRICS " + json.dumps(first)
        text += "\r\n[INFO] STAGE0_OFFLOAD_METRICS " + json.dumps(second)
        self.assertEqual([row["rank"] for row in self.parse(text)], [0, 1])

    def test_duplicate_rank_is_not_double_counted(self):
        line = "STAGE0_OFFLOAD_METRICS " + json.dumps(self.row()) + "\n"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.parse(line * 2)

    def test_contradictory_transfer_total_rejected(self):
        row = {**self.row(), "copied_bytes": 301}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.parse("STAGE0_OFFLOAD_METRICS " + json.dumps(row))


if __name__ == "__main__":
    unittest.main()
