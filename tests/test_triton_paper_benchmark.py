import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tools.run_triton_paper_benchmark import (
    ENTRY_TEMPLATE,
    WORLD_SIZE,
    benchmark_command,
    parse_gpu_indices,
)


class TritonPaperBenchmarkTest(unittest.TestCase):
    def test_gpu_indices_require_four_distinct_physical_devices(self):
        self.assertEqual(parse_gpu_indices("0, 1,2,3"), ["0", "1", "2", "3"])
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            parse_gpu_indices("0,1,2")
        with self.assertRaisesRegex(ValueError, "distinct"):
            parse_gpu_indices("0,1,1,3")
        with self.assertRaisesRegex(ValueError, "physical GPU indices"):
            parse_gpu_indices("0,1,GPU-uuid,3")

    def test_formal_arms_launch_four_ranks_with_divisible_global_batches(self):
        command = benchmark_command(
            "/venv/bin/python",
            Path("/models/Qwen3-0.6B"),
            Path("/campaign/entry.py"),
            "formal-control",
            "torch",
        )
        self.assertEqual(WORLD_SIZE, 4)
        self.assertIn("--nproc_per_node=4", command)
        for option, expected in (
            ("--nsamples", "256"),
            ("--backward_samples", "32"),
            ("--backward_bsz", "32"),
            ("--refresh_mb", "2"),
            ("--global_loss_bsz", "32"),
            ("--static_fisher_microbatch_bsz", "8"),
            ("--bsz", "128"),
            ("--group_parallel_quant", "rank"),
        ):
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertEqual(command[-1], "torch")
        compile(ENTRY_TEMPLATE, "entry.py", "exec")

    def test_project_argument_validation_accepts_four_rank_formal_batches(self):
        argv = [
            "ptq.py", "--exp", "dp-parser-test", "--model", "/tmp/model",
            "--dataset", "wikitext2", "--nsamples", "256", "--seq_len", "2048",
            "--w_method", "gptq_plus", "--w_bits", "4", "--w_clip",
            "--w_groupsize", "128", "--num_groups", "4", "--blocksize", "128",
            "--act_order", "--rotate", "--rotation_seed", "0", "--refresh_seed", "0",
            "--kl_topk", "-1", "--bsz", "128", "--final_layer_stats_bsz", "16",
            "--hessian_accum_bsz", "128", "--enable_gptq_plus", "0",
            "--backward_samples", "32", "--backward_bsz", "32",
            "--final_layer_backward_bsz", "32", "--refresh_mb", "2",
            "--g_update_mode", "block_gd",
            "--grad_lr", "5e-7", "--grad_optimizer", "adam",
            "--grad_refresh_loss", "fisher_diag_mse", "--global_loss",
            "--loss_slide_window", "--global_loss_bsz", "32",
            "--static_fisher_microbatch_bsz", "8",
            "--grad_clip", "5e-5", "--final_layer_grad_clip", "5e-4",
            "--final_layer_grad_lr", "1e-6", "--group_parallel_quant", "rank",
            "--eval_seq_len", "2048", "--eval_datasets", "wikitext2", "--seed", "1",
            "--gptq_inner_kernel", "triton_fused",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}), patch.object(sys, "argv", argv):
            from process_args import parse_gen

            parsed = parse_gen()
        self.assertEqual(parsed.nsamples, 256)
        self.assertEqual(parsed.backward_samples, 32)
        self.assertEqual(parsed.refresh_mb, 2)
        self.assertEqual(parsed.global_loss_bsz, 32)
        self.assertEqual(parsed.static_fisher_microbatch_bsz, 8)
        self.assertEqual(parsed.group_parallel_quant, "rank")


if __name__ == "__main__":
    unittest.main()
