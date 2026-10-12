"""CPU semantic checks; CUDA/actual Hessian measurements are separate."""
import unittest
from unittest.mock import patch

import torch

from tools.hessian_chunked_finite_prototype import all_finite_chunked, build_method


class ChunkedFiniteTests(unittest.TestCase):
    def test_layout_nonfinite_and_input_unchanged(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            for transpose in (False, True):
                for bad in (None, float("nan"), float("inf"), -float("inf")):
                    value = torch.arange(77, dtype=dtype).reshape(7, 11)
                    if bad is not None:
                        value[-1, -1] = bad
                    if transpose:
                        value = value.T
                    before = value.clone()
                    stride = value.stride()
                    self.assertEqual(bool(all_finite_chunked(value, 23)),
                                     bool(torch.isfinite(value).all()))
                    torch.testing.assert_close(value, before, rtol=0, atol=0, equal_nan=True)
                    self.assertEqual(value.stride(), stride)
        self.assertTrue(bool(all_finite_chunked(torch.empty(0, 7))))

    def test_dense_mask_is_bounded_and_no_full_matrix_conversion(self):
        original = torch.isfinite
        sizes = []
        def record(value):
            sizes.append(value.numel())
            return original(value)
        for value in (torch.ones(71, 113), torch.ones(71, 113).T):
            sizes.clear()
            with patch("torch.isfinite", side_effect=record):
                self.assertTrue(bool(all_finite_chunked(value, 509)))
            self.assertLessEqual(max(sizes), 509)
            self.assertEqual(sum(sizes), value.numel())

    def test_real_helper_preserves_factors_damping_and_fallback(self):
        from gptq_utils import gptq_plus_utils as impl
        original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
        helper = impl.GPTQPlus.__new__(impl.GPTQPlus)
        for workspace in (False, True):
            method, _ = build_method(impl, workspace=workspace)
            for h in (torch.eye(5), -torch.eye(5)):
                for keep in (False, True):
                    left_input, right_input = h.clone(), h.clone()
                    options = dict(percdamp=.01, damp_auto_increment=.5, need_hinv_init=keep)
                    left = original(helper, left_input, **options)
                    right = method(helper, right_input, **options)
                    self.assertEqual(left[2:], right[2:])
                    self.assertTrue(torch.equal(left_input, right_input))
                    for a, b in zip(left[:2], right[:2]):
                        if a is None:
                            self.assertIsNone(b)
                        else:
                            self.assertTrue(torch.equal(a, b))
                            self.assertEqual(a.stride(), b.stride())


if __name__ == "__main__":
    unittest.main()
