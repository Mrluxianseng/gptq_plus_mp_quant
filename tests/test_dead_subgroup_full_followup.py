"""CPU coverage checks for both model widths and fresh full entry."""
import unittest

from tools.run_dead_subgroup_full_followup import ROOT, cleanup_coverage, prepare_entry


class DeadFullGateTests(unittest.TestCase):
    def test_two_widths_and_missing_retained_factor_rejected(self):
        for width in (3072, 6144):
            calls = [dict(layer=layer, module="mlp.down_proj", dtype="torch.float32", shape=[width,width])
                     for layer in range(28) for _ in range(4)]
            writes = [dict(enabled=True, pointer_preserved=True) for _ in range(28)]
            inverse = dict(calls=calls, weight_write_calls=writes, helper_unchanged=True,
                           stable_helper_unchanged=True, reuse_calls=[{} for _ in range(112)])
            dead = dict(enabled=True, helpers_unchanged=True, records=[
                dict(layer=layer, released_private_input_bytes=width**2*4, retained_factors=4)
                for layer in range(28)])
            self.assertTrue(cleanup_coverage(inverse,dead,enabled=True))
            dead["records"][27]["retained_factors"] = 3
            self.assertFalse(cleanup_coverage(inverse,dead,enabled=True))
            dead["records"][27]["retained_factors"] = 4
            dead["records"][0]["released_private_input_bytes"] += 4
            self.assertFalse(cleanup_coverage(inverse,dead,enabled=True))
            for row in writes:
                row["enabled"] = False
            dead.update(enabled=False,records=[])
            inverse["reuse_calls"] = []
            self.assertTrue(cleanup_coverage(inverse,dead,enabled=False))
            dead["records"] = [dict(layer=0)]
            self.assertFalse(cleanup_coverage(inverse,dead,enabled=False))

    def test_real_fresh_entry_is_not_finite_or_all_linear(self):
        source=prepare_entry(ROOT / "outputs/hessian-workspace-full-followup-20261011-v2")
        compile(source,"dead_input_fresh_entry","exec")
        self.assertIn("hessian_dead_subgroup_install",source)
        self.assertIn("if '--exit_after_precompute' not in sys.argv:",source)
        self.assertNotIn("hessian_bounded_solver_install",source)
        self.assertNotIn("hessian_all_weight_storage",source)
        self.assertNotIn("# Report allocator peaks",source)
        self.assertEqual(source.count("runpy.run_path('ptq.py', run_name='__main__')"),1)


if __name__ == "__main__":
    unittest.main()
