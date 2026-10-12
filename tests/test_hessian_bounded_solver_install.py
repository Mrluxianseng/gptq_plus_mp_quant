"""CPU real-API integration checks, with production classes restored."""
import hashlib
from pathlib import Path
import unittest

import torch

from tools.hessian_bounded_solver_install import build_entry, install_down_proj


class BoundedInstallerTests(unittest.TestCase):
    def test_opt_in_scope_layout_and_inplace_fallback(self):
        from gptq_utils import gptq_plus_utils as impl
        original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
        try:
            for enabled in (False, True):
                impl.GPTQPlus._compute_hessian_inverse_with_fallback = original
                records = []
                install_down_proj(impl, enabled=enabled, records=records)
                for name in ("mlp.down_proj", "mlp.up_proj"):
                    for inplace in (False, True):
                        helper = impl.GPTQPlus.__new__(impl.GPTQPlus)
                        helper.layer_name, helper.layer_idx = name, 27
                        left_h, right_h = torch.eye(5), torch.eye(5)
                        options = dict(percdamp=.01, damp_auto_increment=.1,
                                       need_hinv_init=False, inplace_damping=inplace)
                        left = original(helper, left_h, **options)
                        right = helper._compute_hessian_inverse_with_fallback(right_h, **options)
                        self.assertEqual(left[2:], right[2:])
                        self.assertTrue(torch.equal(left_h, right_h))
                        self.assertIsNone(right[0])
                        self.assertTrue(torch.equal(left[1], right[1]))
                        self.assertEqual(left[1].stride(), right[1].stride())
                self.assertEqual(len(records), 2)
                self.assertEqual([r["bounded_finite_enabled"] for r in records],
                                 [enabled, False])
        finally:
            impl.GPTQPlus._compute_hessian_inverse_with_fallback = original

    def test_entry_generator_is_bound_and_compiles(self):
        root = Path(__file__).resolve().parents[1]
        # Real archived raw entry is a development input, not production code.
        raw = (root / "outputs/hessian-workspace-full-followup-20261011-v2/entry.py").read_text()
        hashes = [hashlib.sha256((root / f"tools/{name}.py").read_bytes()).hexdigest()
                  for name in ("hessian_bounded_solver_install", "hessian_chunked_finite_prototype",
                               "hessian_solver_workspace_prototype")]
        entry = build_entry(raw, installer_sha=hashes[0], finite_sha=hashes[1], workspace_sha=hashes[2])
        compile(entry, "bounded_finite_entry", "exec")
        self.assertIn(hashes[0], entry)
        self.assertIn(hashes[1], entry)
        with self.assertRaises(RuntimeError):
            build_entry(raw.replace("_hinv_expected=", "_changed_expected="),
                        installer_sha=hashes[0], finite_sha=hashes[1], workspace_sha=hashes[2])


if __name__ == "__main__":
    unittest.main()
