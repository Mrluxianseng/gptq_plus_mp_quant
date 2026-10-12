"""Identity, mutation and lifetime checks, without initializing CUDA."""
import gc
import unittest

import torch

from utils.frozen_parameter_offload_prototype import FrozenParameterRegistry


class RegistryTests(unittest.TestCase):
    def test_transposed_parameter_matches_but_equal_values_do_not(self):
        registry = FrozenParameterRegistry()
        weight = torch.randn(5, 7)
        registry.register(weight, allow_cpu=True)
        self.assertTrue(registry.matches(weight.t()))
        self.assertFalse(registry.matches(weight.clone().t()))

    def test_in_place_mutation_requires_registration_of_new_version(self):
        registry = FrozenParameterRegistry()
        weight = torch.randn(5, 7)
        registry.register(weight, allow_cpu=True)
        weight.add_(1)
        self.assertFalse(registry.matches(weight.t()))
        registry.register(weight, allow_cpu=True)
        self.assertTrue(registry.matches(weight.t()))

    def test_trainable_parameter_is_excluded(self):
        registry = FrozenParameterRegistry()
        weight = torch.randn(5, 7, requires_grad=True)
        registry.register(weight, allow_cpu=True)
        self.assertFalse(registry.matches(weight))
        self.assertFalse(registry.matches(weight.t()))

    def test_registry_does_not_keep_storage_alive(self):
        registry = FrozenParameterRegistry()
        weight = torch.randn(5, 7)
        registry.register(weight, allow_cpu=True)
        del weight
        gc.collect()
        self.assertEqual(len(registry._storages), 0)


if __name__ == "__main__":
    unittest.main()
