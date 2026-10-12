"""Identity, alias, mutation and lifetime risks in activation deduplication."""
import gc
import unittest
import weakref

import torch

from utils.saved_tensor_offload import SavedTensorCpuCache


class SavedCpuCacheTests(unittest.TestCase):
    def test_identical_views_share_copy_without_retaining_original(self):
        cache = SavedTensorCpuCache(0)
        original = torch.arange(32.0).reshape(4, 8)
        first = cache._pack_value(original)
        second = cache._pack_value(original.view_as(original))
        self.assertIs(first, second)
        self.assertEqual(cache.stats["copies"], 1)
        self.assertEqual(cache.stats["dedup_hits"], 1)
        ref = weakref.ref(original)
        del original
        gc.collect()
        self.assertIsNone(ref())
        self.assertTrue(torch.equal(cache.unpack(first), torch.arange(32.0).reshape(4, 8)))

    def test_offsets_and_strides_are_not_conflated(self):
        cache = SavedTensorCpuCache(0)
        original = torch.arange(16.0).reshape(4, 4)
        views = (original, original.t(), original[:2], original[2:])
        payloads = [cache._pack_value(view) for view in views]
        self.assertEqual(cache.stats["copies"], 4)
        for view, payload in zip(views, payloads):
            self.assertTrue(torch.equal(cache.unpack(payload), view))

    def test_contiguous_batch_reshape_reuses_only_the_same_region(self):
        cache = SavedTensorCpuCache(0)
        original = torch.arange(64.0).reshape(2, 4, 8)
        first = cache._pack_value(original)
        flattened = original.reshape(8, 8)
        second = cache._pack_value(flattened)
        self.assertEqual(cache.stats["copies"], 1)
        self.assertEqual(cache.stats["dedup_hits"], 1)
        self.assertTrue(torch.equal(cache.unpack(first), original))
        self.assertTrue(torch.equal(cache.unpack(second), flattened))

    def test_inplace_mutation_produces_a_new_snapshot(self):
        cache = SavedTensorCpuCache(0)
        original = torch.arange(8.0)
        before = cache._pack_value(original)
        original.add_(1)
        after = cache._pack_value(original)
        self.assertIsNot(before, after)
        self.assertTrue(torch.equal(cache.unpack(before), torch.arange(8.0)))
        self.assertTrue(torch.equal(cache.unpack(after), torch.arange(8.0) + 1))

    def test_cache_releases_host_payload_after_last_user(self):
        cache = SavedTensorCpuCache(0)
        original = torch.arange(8.0)
        payload = cache._pack_value(original)
        alias = cache._pack_value(original)
        self.assertEqual(cache.stats["live_cpu_bytes"], 32)
        del payload
        gc.collect()
        self.assertEqual(cache.stats["live_cpu_bytes"], 32)
        del alias
        gc.collect()
        self.assertEqual(cache.stats["live_cpu_bytes"], 0)
        self.assertEqual(len(cache._payloads), 0)
        cache._pack_value(original)
        self.assertEqual(cache.stats["copies"], 2)

    def test_distinct_equal_storages_do_not_alias(self):
        cache = SavedTensorCpuCache(0)
        first, second = torch.arange(8.0), torch.arange(8.0)
        self.assertNotEqual(cache.key(first), cache.key(second))
        self.assertIsNot(cache._pack_value(first), cache._pack_value(second))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_fanout_exact_gradient_and_repeated_unpack(self):
        gradients = []
        for mode in ("none", "copy", "dedup"):
            x = torch.linspace(-1, 1, 4096, device="cuda", requires_grad=True)
            cache = SavedTensorCpuCache(0, deduplicate=mode == "dedup")
            if mode == "none":
                y = x * 1.25
                loss = (y.square() + y.square()).sum()
            else:
                with cache.context():
                    y = x * 1.25
                    loss = (y.square() + y.square()).sum()
                # Retained graph exercises unpack multiple times.
                loss.backward(retain_graph=True)
                x.grad = None
            loss.backward()
            gradients.append(x.grad.detach().cpu())
            if mode == "dedup":
                self.assertGreater(cache.stats["dedup_hits"], 0)
                gc.collect()
                self.assertEqual(cache.stats["live_cpu_bytes"], 0)
        self.assertTrue(torch.equal(gradients[0], gradients[1]))
        self.assertTrue(torch.equal(gradients[0], gradients[2]))


if __name__ == "__main__":
    unittest.main()
