import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptq_utils.fisher_storage import (
    lower_triangle_indices,
    pack_symmetric_lower,
    unpack_symmetric_lower,
)


class PackedFisherStorageTests(unittest.TestCase):
    def _check_round_trip(self, device):
        torch.manual_seed(4107)
        raw = torch.randn(17, 17, dtype=torch.float32, device=device)
        symmetric = raw + raw.T
        rows, cols = lower_triangle_indices(17)
        packed = pack_symmetric_lower(symmetric, rows, cols)
        restored = unpack_symmetric_lower(packed, 17, rows, cols)

        self.assertEqual(packed.numel(), 17 * 18 // 2)
        self.assertTrue(torch.equal(restored, symmetric))

        delta = torch.randn(5, 17, dtype=torch.float32, device=device)
        expected = ((delta @ symmetric) * delta).sum(dim=-1)
        actual = ((delta @ restored) * delta).sum(dim=-1)
        self.assertTrue(torch.equal(actual, expected))

    def test_cpu_round_trip_and_quadratic_form_are_exact(self):
        self._check_round_trip("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_pack_accepts_cpu_indices_and_is_exact(self):
        self._check_round_trip("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_empirical_gram_is_bitwise_symmetric(self):
        torch.manual_seed(19)
        # Match the paper's Qwen3-0.6B calibration sequence length.
        tokens = torch.randn(2048, 1024, dtype=torch.float32, device="cuda")
        fisher = tokens.T @ tokens
        self.assertTrue(torch.equal(fisher, fisher.T))

    def test_rejects_non_square_and_wrong_packed_size(self):
        with self.assertRaises(ValueError):
            pack_symmetric_lower(torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            unpack_symmetric_lower(torch.zeros(5), 3)


if __name__ == "__main__":
    unittest.main()
