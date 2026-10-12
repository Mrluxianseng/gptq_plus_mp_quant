"""Guard formal evaluation scope and multi-rank batch divisibility."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PREPARE = Path(__file__).resolve().parents[1] / "tools/prepare_realq_memory_manifests.py"


class MemoryManifestProtocolTests(unittest.TestCase):
    def prepare(self, hidden, *options):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"hidden_size": hidden, "num_hidden_layers": 28, "model_type": "qwen3"}))
        out = root / "manifests"
        result = subprocess.run([sys.executable, str(PREPARE), "--model-path", str(model),
                                 "--output-dir", str(out), *options], text=True, capture_output=True)
        return out, result

    def test_paper_uses_full_test_split_and_world_divisible_batches(self):
        out, result = self.prepare(1024, "--preset", "paper", "--world-size", "4")
        self.assertEqual(result.returncode, 0, result.stderr)
        protocol = json.loads((out / "protocol.json").read_text())
        self.assertEqual((protocol["nsamples"], protocol["seq_len"], protocol["gradient_samples"]), (2048, 2048, 32))
        self.assertEqual((protocol["eval_samples"], protocol["eval_seq_len"]), (0, 2048))
        self.assertFalse(protocol["exact_paper_table_reproduction_claimed"])
        for path in out.glob("*-stage*.json"):
            command = json.loads(path.read_text())["command"]
            for option in ("--bsz", "--global_loss_bsz", "--backward_bsz", "--final_layer_backward_bsz", "--final_layer_stats_bsz"):
                self.assertEqual(int(command[command.index(option)+1]) % 4, 0)

    def test_paper_rate_is_model_specific_and_pilot_is_labelled(self):
        out, result = self.prepare(2048, "--preset", "paper", "--world-size", "4")
        self.assertEqual(result.returncode, 0, result.stderr)
        protocol = json.loads((out / "protocol.json").read_text())
        self.assertEqual(protocol["grad_lr"], "5e-4")
        out, result = self.prepare(1024)
        self.assertEqual(result.returncode, 0, result.stderr)
        protocol = json.loads((out / "protocol.json").read_text())
        self.assertFalse(protocol["paper_resource_scale"])
        self.assertEqual(protocol["eval_samples"], 2)
        self.assertFalse(protocol["rotation"])

    def test_invalid_global_sample_partition_is_rejected_before_run(self):
        _, result = self.prepare(1024, "--world-size", "4", "--nsamples", "6")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("divisible", result.stderr)

    def test_hinv_flag_requires_supported_parallelism_and_is_candidate_only(self):
        _, result = self.prepare(2048, "--candidate-hinv-downproj-offload", "--group-parallel", "rank")
        self.assertNotEqual(result.returncode, 0)
        out, result = self.prepare(2048, "--candidate-hinv-downproj-offload", "--activation-deduplicate")
        self.assertEqual(result.returncode, 0, result.stderr)
        for side in ("control", "candidate"):
            zero = json.loads((out / f"{side}-stage0.json").read_text())["command"]
            two = json.loads((out / f"{side}-stage2.json").read_text())["command"]
            self.assertEqual("--static_saved_tensor_offload_deduplicate" in zero, side == "candidate")
            self.assertEqual("--stage2_hinv_cpu_offload_modules" in two, side == "candidate")


if __name__ == "__main__":
    unittest.main()
