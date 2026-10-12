"""No CUDA initialization: validate benchmark scope and reference binding."""
import unittest
from pathlib import Path

from tools.run_low_overhead_speed_campaign import without_profile, bind_reference_audit, validate_cache_identity


class SpeedProtocolTests(unittest.TestCase):
    def test_only_diagnostic_arguments_removed(self):
        command = ["python", "entry.py", "--enable_quant_profile", "--quant_profile_target_layers", "0,27",
                   "--enable_memory_profile", "--grad_lr", "5e-5", "--eval_ce_chunk_tokens", "256"]
        self.assertEqual(without_profile(command), ["python", "entry.py", "--grad_lr", "5e-5", "--eval_ce_chunk_tokens", "256"])

    def test_malformed_profile_argument_rejected(self):
        with self.assertRaises(ValueError):
            without_profile(["--quant_profile_target_layers", "--grad_lr", "5e-5"])

    def test_unrelated_audit_rejected(self):
        audit = dict(scope="full_pipeline", quantization_and_quality_validated=True, full_model_quantization_validated=True,
                     control_directory="outputs/first-control", candidate_directory="outputs/first-candidate")
        with self.assertRaisesRegex(ValueError, "does not belong"):
            bind_reference_audit(audit, Path("outputs/other-control"), Path("outputs/first-candidate"))

    def test_stage0_cannot_authorize_full_speed_claim(self):
        with self.assertRaisesRegex(ValueError, "complete-model"):
            bind_reference_audit(dict(scope="stage0_only", quantization_and_quality_validated=False), Path("c"), Path("n"))

    def test_evaluated_prefix_cannot_authorize_full_speed_claim(self):
        with self.assertRaisesRegex(ValueError, "complete-model"):
            bind_reference_audit(dict(scope="full_pipeline", quantization_and_quality_validated=True,
                                      full_model_quantization_validated=False, quantized_prefix_diagnostic=True),
                                 Path("c"), Path("n"))

    def test_changed_or_incomplete_fresh_cache_rejected(self):
        expected = {"fisher.pt": "fisher-hash", "prepared/model.safetensors": "model-hash"}
        validate_cache_identity(dict(expected), expected)
        for actual in ({}, {"fisher.pt": "fisher-hash"}, {**expected, "fisher.pt": "changed"}):
            with self.subTest(actual=actual), self.assertRaisesRegex(RuntimeError, "differ"):
                validate_cache_identity(actual, expected)
        with self.assertRaises(RuntimeError):
            validate_cache_identity({}, {})


if __name__ == "__main__":
    unittest.main()
