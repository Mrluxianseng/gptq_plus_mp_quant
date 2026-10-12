"""Prevent incomplete or regressed experiments from being accepted as success."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_full_pipeline_profiles import GLOBAL_FIELDS, compare


class ProfileAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.control = self.root / "control"
        self.candidate = self.root / "candidate"
        for side in (self.control, self.candidate):
            side.mkdir()
            (side / "nvml_telemetry.csv").write_text("2026/10/10 01:00:00.000, 100, 8000, 50\n2026/10/10 01:00:00.100, 100, 8000, 50\n")
            files = []
            for stage in ("stage0", "stage2"):
                profile = self.root / f"{side.name}_{stage}" / "cuda_rank0.jsonl"
                profile.parent.mkdir()
                row = {"phase": "evaluation", "sampled_peak_allocated_bytes": 100, "sampled_peak_reserved_bytes": 100}
                profile.write_text(json.dumps(row) + "\n")
                files.append(str(profile))
            status = {"returncode": 0, "sources_unchanged": True,
                      "stage0": {"returncode": 0}, "stage2": {"returncode": 0},
                      "metrics": {"state_sha256": "same-digest", "state_tensors": 1},
                      "kl": 0.1, "ppl": 20.0, "memory_profile_files": files,
                      **dict.fromkeys(GLOBAL_FIELDS, 100)}
            status.update(dict.fromkeys(("stage0_process_peak_allocated_bytes", "stage0_process_peak_reserved_bytes", "stage2_process_peak_allocated_bytes", "stage2_process_peak_reserved_bytes", "full_process_peak_allocated_bytes", "full_process_peak_reserved_bytes"), 100))
            self.write(side, "status.json", status)
            command = ["python", f"/{side.name}/entry.py", "--exp", side.name,
                       "--nsamples", "16", "--quant_stop_layer", "1"]
            manifest = {"source_sha256_at_launch": {f"/{side.name}/entry.py": "same-source"},
                        "stage0_command": command, "stage2_command": command}
            self.write(side, "manifest.json", manifest)
            self.write(side, "layer_output_fingerprints.json", [{"layer": 0, "sha256": "x", "forward_count": 1, "shapes": [[1, 16, 32]]}, {"layer": 1, "sha256": "y", "forward_count": 1, "shapes": [[1, 16, 32]]}])

    @staticmethod
    def write(directory, filename, value):
        (directory / filename).write_text(json.dumps(value))

    def result(self):
        return compare(self.control, self.candidate, root=self.root)

    def test_identical_run_passes_but_claims_no_peak_reduction(self):
        report = self.result()
        self.assertTrue(report["strict_sampled_memory_gate_passed"])
        self.assertEqual(report["nvml_peak_reduction_percent"], 0)

    def test_changed_data_cannot_be_accepted_even_with_matching_outputs(self):
        manifest = json.loads((self.candidate / "manifest.json").read_text())
        command = copy.deepcopy(manifest["stage2_command"])
        command[command.index("--nsamples") + 1] = "32"
        manifest["stage2_command"] = command
        self.write(self.candidate, "manifest.json", manifest)
        self.assertFalse(self.result()["exactness_passed"])

    def test_missing_last_layer_is_rejected_even_if_both_traces_match(self):
        for side in (self.control, self.candidate):
            self.write(side, "layer_output_fingerprints.json", [{"layer": 0, "sha256": "x"}])
        self.assertFalse(self.result()["exactness_passed"])

    def test_prefix_quantization_still_requires_full_evaluation_coverage(self):
        model = self.root / "model"
        model.mkdir()
        self.write(model, "config.json", {"num_hidden_layers": 2})
        for side in (self.control, self.candidate):
            manifest = json.loads((side / "manifest.json").read_text())
            manifest["stage2_command"] += ["--model", str(model)]
            command = manifest["stage2_command"]
            command[command.index("--quant_stop_layer") + 1] = "0"
            self.write(side, "manifest.json", manifest)
        report = self.result()
        self.assertTrue(report["exactness_passed"])
        self.assertTrue(report["quantized_prefix_diagnostic"])
        self.assertFalse(report["full_model_quantization_validated"])
        for side in (self.control, self.candidate):
            trace = json.loads((side / "layer_output_fingerprints.json").read_text())
            self.write(side, "layer_output_fingerprints.json", trace[:1])
        self.assertFalse(self.result()["exactness_passed"])

    def test_empty_forward_hashes_cannot_pass_with_all_layer_ids(self):
        for side in (self.control, self.candidate):
            self.write(side, "layer_output_fingerprints.json", [{"layer": layer, "sha256": "empty", "forward_count": 0, "shapes": []} for layer in range(2)])
        self.assertFalse(self.result()["exactness_passed"])

    def test_phase_regression_is_not_hidden_by_unchanged_global_peaks(self):
        profile = self.root / "candidate_stage2" / "cuda_rank0.jsonl"
        profile.write_text(json.dumps({"phase": "evaluation", "sampled_peak_allocated_bytes": 100, "sampled_peak_reserved_bytes": 102}) + "\n")
        report = self.result()
        self.assertTrue(report["exactness_passed"])
        self.assertTrue(report["global_peaks_no_regression"])
        self.assertFalse(report["strict_sampled_memory_gate_passed"])
        self.assertEqual(report["sampled_phase_regression_count"], 1)

    def test_missing_stage_is_rejected(self):
        status = json.loads((self.candidate / "status.json").read_text())
        status["stage0"] = {"returncode": None}
        self.write(self.candidate, "status.json", status)
        self.assertFalse(self.result()["exactness_passed"])

    def test_missing_metrics_is_rejected_without_crashing(self):
        status = json.loads((self.candidate / "status.json").read_text())
        status["metrics"] = None
        self.write(self.candidate, "status.json", status)
        self.assertFalse(self.result()["exactness_passed"])

    def test_misreported_nvml_peak_is_rejected(self):
        status = json.loads((self.candidate / "status.json").read_text())
        status["max_nvml_memory_used_mib"] = 90
        self.write(self.candidate, "status.json", status)
        self.assertFalse(self.result()["exactness_passed"])

    def test_quantization_only_allocator_data_cannot_pass_complete_gate(self):
        status = json.loads((self.candidate / "status.json").read_text())
        status.pop("stage2_process_peak_allocated_bytes")
        self.write(self.candidate, "status.json", status)
        report = self.result()
        self.assertTrue(report["exactness_passed"])
        self.assertFalse(report["strict_sampled_memory_gate_passed"])

    def test_stage0_only_audits_cache_without_claiming_quality(self):
        for side in (self.control, self.candidate):
            status = json.loads((side / "status.json").read_text())
            status.update(scope="stage0_only", metrics=None, kl=None, ppl=None,
                          stage2={"returncode": None, "skipped": "Stage-0-only diagnostic"})
            status["memory_profile_files"] = status["memory_profile_files"][:1]
            self.write(side, "status.json", status)
            (side / "layer_output_fingerprints.json").unlink()
            (side / "static_cache").mkdir()
            (side / "static_cache" / "fisher.pt").write_bytes(b"same Fisher")
        report = compare(self.control, self.candidate, root=self.root, stage0_only=True)
        self.assertTrue(report["exactness_passed"])
        self.assertTrue(report["strict_sampled_memory_gate_passed"])
        self.assertFalse(report["quantization_and_quality_validated"])
        (self.candidate / "static_cache" / "fisher.pt").write_bytes(b"changed Fisher")
        self.assertFalse(compare(self.control, self.candidate, root=self.root, stage0_only=True)["exactness_passed"])

    def test_stage0_only_cannot_be_used_to_label_full_run(self):
        for side in (self.control, self.candidate):
            (side / "static_cache").mkdir()
            (side / "static_cache" / "fisher.pt").write_bytes(b"same Fisher")
        report = compare(self.control, self.candidate, root=self.root, stage0_only=True)
        self.assertFalse(report["exactness_passed"])

    def test_repeated_phase_cannot_hide_a_single_batch_regression(self):
        for side, peaks in (("control", (100, 50)), ("candidate", (90, 60))):
            profile = self.root / f"{side}_stage0" / "cuda_rank0.jsonl"
            profile.write_text("\n".join(json.dumps({"phase": "backward", "sampled_peak_allocated_bytes": peak, "sampled_peak_reserved_bytes": peak, "peak_measurement_mode": "exact"}) for peak in peaks) + "\n")
        report = self.result()
        self.assertTrue(report["global_peaks_no_regression"])
        self.assertFalse(report["strict_measured_memory_gate_passed"])
        self.assertEqual(report["sampled_phase_regression_count"], 1)

    def test_changed_allocator_environment_is_not_a_matched_pair(self):
        manifest = json.loads((self.candidate / "manifest.json").read_text())
        manifest["runtime_environment"] = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
        self.write(self.candidate, "manifest.json", manifest)
        self.assertFalse(self.result()["exactness_passed"])

    def test_different_monitored_gpu_is_not_a_matched_pair(self):
        for side, uuid in ((self.control, "GPU-a"), (self.candidate, "GPU-b")):
            (side / "nvml_telemetry.csv").write_text(f"2026/10/10 01:00:00.000, 100, 8000, 50, 0, {uuid}\n")
        self.assertFalse(self.result()["exactness_passed"])


if __name__ == "__main__":
    unittest.main()
