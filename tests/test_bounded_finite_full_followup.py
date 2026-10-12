"""Acceptance gate tests: no GPU work is launched by these CPU checks."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.run_bounded_finite_full_followup import ROOT, audit_points, check_bound_sources, prepare_entry
from tools.compare_full_pipeline_profiles import digest


class FullBoundedGateTests(unittest.TestCase):
    def test_end_regression_and_missing_point_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / (side + "_rank0.jsonl") for side in ("control", "candidate")]
            rows = [dict(phase="replay", start_allocated_bytes=100, end_allocated_bytes=10,
                         start_reserved_bytes=100, end_reserved_bytes=10),
                    dict(phase="replay", start_allocated_bytes=90, end_allocated_bytes=20,
                         start_reserved_bytes=90, end_reserved_bytes=20)]
            for path, row in zip(paths, rows):
                path.write_text(json.dumps(row) + "\n")
            status = [dict(memory_profile_files=[str(path)]) for path in paths]
            result = audit_points(*status)
            self.assertFalse(result["valid"])
            self.assertEqual(len(result["regressions"]), 2)
            rows[1]["end_allocated_bytes"] = 5
            rows[1]["end_reserved_bytes"] = 5
            paths[1].write_text(json.dumps(rows[1]) + "\n")
            self.assertTrue(audit_points(*status)["valid"])
            del rows[1]["end_reserved_bytes"]
            paths[1].write_text(json.dumps(rows[1]) + "\n")
            self.assertFalse(audit_points(*status)["valid"])
            self.assertFalse(audit_points({}, {})["valid"])

    def test_source_mutation_blocks_followup(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "entry.py"
            source.write_text("pass\n")
            bound = {str(source): digest(source)}
            check_bound_sources(bound)
            source.write_text("raise RuntimeError()\n")
            with self.assertRaisesRegex(RuntimeError, "bound source changed"):
                check_bound_sources(bound)

    def test_real_entry_allows_fresh_stage0_without_double_profiling(self):
        source = prepare_entry(ROOT / "outputs/hessian-workspace-full-followup-20261011-v2")
        compile(source, "full_bounded_entry", "exec")
        self.assertEqual(source.count("runpy.run_path('ptq.py', run_name='__main__')"), 1)
        self.assertNotIn("# Report allocator peaks", source)
        self.assertNotIn("# Profile final evaluation separately", source)
        self.assertIn('if "--exit_after_precompute" not in sys.argv:', source)
        self.assertIn("_all_enabled=_hinv_stable", source)
        self.assertIn("tools.hessian_bounded_solver_install", source)
        self.assertNotIn('sys.argv.remove("--research_all_weight_stable")', source)
        for filename in ("hessian_bounded_solver_install.py", "hessian_chunked_finite_prototype.py",
                         "hessian_solver_workspace_prototype.py", "hessian_all_weight_storage_prototype.py"):
            self.assertIn(digest(ROOT / "tools" / filename), source)


if __name__ == "__main__":
    unittest.main()
