"""CPU adapter guards; no claim about real allocator history."""
from pathlib import Path
import tempfile
import unittest

from tools.compare_full_pipeline_profiles import digest
from tools.hessian_dead_subgroup_install import build_entry, install_builder


class DeadInputInstallerTests(unittest.TestCase):
    def test_actual_stable_builder_uses_existing_factors_and_guard(self):
        from gptq_utils import gptq_plus_utils as impl
        from tools import hessian_stable_weight_prototype as stable
        original = stable.build_stable_fasterquant
        try:
            restored = install_builder(impl, stable)
            self.assertIs(restored, original)
            method, source = stable.build_stable_fasterquant(impl)
            self.assertEqual(method.__name__, "fasterquant")
            release = source.index("del H_sub", source.index("_dead_h_ptr"))
            block = source.index("# Number of blocks + refreshes")
            self.assertLess(release, block)
            self.assertIn('_weight_destination.copy_(_weight_source)', source)
            self.assertIn('last subgroup input aliases a retained factor', source)
        finally:
            stable.build_stable_fasterquant = original

    def test_real_entry_cpu_generation_and_failed_boundary(self):
        root = Path(__file__).resolve().parents[1]
        raw = (root / "outputs/hessian-workspace-full-followup-20261011-v2/entry.py").read_text()
        hashes = [digest(root / "tools" / filename) for filename in
                  ("hessian_dead_subgroup_install.py", "hessian_dead_subgroup_input_prototype.py")]
        source = build_entry(raw, installer_sha=hashes[0], prototype_sha=hashes[1])
        compile(source, "actual_entry", "exec")
        self.assertIn(hashes[0], source)
        self.assertIn(hashes[1], source)
        self.assertIn('self._dead_subgroup_input_records=_dead_records', source)
        self.assertNotIn('hessian_chunked_finite', source)
        self.assertNotIn('hessian_all_weight_storage', source)
        with self.assertRaises(RuntimeError):
            build_entry(raw.replace('_hinv_stable_expected:', '_renamed_guard:'),
                        installer_sha=hashes[0], prototype_sha=hashes[1])


if __name__ == "__main__":
    unittest.main()
