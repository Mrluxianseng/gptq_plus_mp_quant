import unittest

import torch

from gptq_utils.gptq_plus_utils import _deterministic_categorical_labels


class ChunkedCategoricalSamplingTest(unittest.TestCase):
    def test_chunked_sampling_is_reproducible_and_shape_preserving(self):
        logits = torch.zeros(2, 4096, 4, dtype=torch.float32)
        labels_a = _deterministic_categorical_labels(
            logits, [3, 11], base_seed=29, chunk_tokens=64
        )
        labels_b = _deterministic_categorical_labels(
            logits, [3, 11], base_seed=29, chunk_tokens=64
        )
        self.assertEqual(tuple(labels_a.shape), (2, 4096))
        self.assertTrue(torch.equal(labels_a, labels_b))
        self.assertTrue(torch.all((labels_a >= 0) & (labels_a < 4)))

    def test_chunked_sampling_preserves_uniform_categorical_distribution(self):
        logits = torch.zeros(1, 8192, 4, dtype=torch.float32)
        labels = _deterministic_categorical_labels(
            logits, [7], base_seed=101, chunk_tokens=128
        )
        counts = torch.bincount(labels.flatten(), minlength=4).float()
        frequencies = counts / counts.sum()
        self.assertTrue(torch.all(torch.abs(frequencies - 0.25) < 0.035), frequencies)

    def test_zero_chunk_uses_legacy_draw_sequence(self):
        torch.manual_seed(5)
        logits = torch.randn(2, 23, 17)
        actual = _deterministic_categorical_labels(
            logits, [2, 9], base_seed=41, chunk_tokens=0
        )
        expected = torch.empty(2, 23, dtype=torch.long)
        for i, sample_id in enumerate([2, 9]):
            gen = torch.Generator().manual_seed(
                int(1000003 * (41 * 100000 + sample_id) + 17)
            )
            probs = torch.softmax(logits[i].reshape(-1, 17).float(), dim=-1)
            expected[i] = torch.multinomial(probs, 1, generator=gen).reshape(23)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
