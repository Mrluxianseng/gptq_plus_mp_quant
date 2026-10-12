import unittest
import json
from pathlib import Path
import tempfile
from tools.audit_memory_boundary_points import boundaries, compare


class BoundaryTests(unittest.TestCase):
    def test_end_regression_cannot_hide_behind_larger_start(self):
        start, end = ("replay", 0, "start", "reserved"), ("replay", 0, "end", "reserved")
        result = compare({start:100, end:10}, {start:90, end:20})
        self.assertFalse(result["valid"])
        self.assertEqual(result["regressions"][0]["key"], list(end))

    def test_partial_or_missing_cannot_certify_completion(self):
        key = ("inverse", 0, "start", "allocated")
        self.assertFalse(compare({key:100}, {key:90}, partial=True)["valid"])
        self.assertFalse(compare({key:100}, {})["valid"])
        self.assertTrue(compare({key:100}, {key:90})["valid"])

    def test_stream_and_rank_identity_are_not_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for stage, rank in (("stage0", 0), ("stage2", 0), ("stage2", 1)):
                folder = root / ("run_" + stage) / "memory_profile"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"cuda_memory_quantization_rank{rank}_pid99.jsonl"
                path.write_text(json.dumps(dict(phase="init", start_reserved_bytes=10,
                                                 end_reserved_bytes=20)) + "\n")
                paths.append(path)
            result = boundaries(paths)
            self.assertEqual(len(result), 6)
            self.assertEqual(len({k[:3] for k in result}), 3)


if __name__ == "__main__":
    unittest.main()
