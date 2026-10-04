from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from subprocess_log_stream import iter_output_chunks
from run_triton_paper_benchmark import benchmark_command


class SubprocessLogStreamTests(unittest.TestCase):
    def test_forwards_carriage_return_progress_as_lines(self) -> None:
        child_code = (
            "import sys,time; "
            "sys.stdout.write('step 1\\r'); sys.stdout.flush(); time.sleep(.05); "
            "sys.stdout.write('step 2\\r\\nfinished\\n'); sys.stdout.flush()"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output = "".join(iter_output_chunks(process.stdout))
        process.stdout.close()
        self.assertEqual(process.wait(), 0)
        self.assertEqual(output, "step 1\nstep 2\nfinished\n")

    def test_memory_flags_are_candidate_only(self) -> None:
        entry = Path("/tmp/entry.py")
        control = benchmark_command("python", Path("/model"), entry, "control", "torch")
        candidate = benchmark_command(
            "python", Path("/model"), entry, "candidate", "triton_fused"
        )
        self.assertNotIn("--static_fisher_activation_checkpointing", control)
        self.assertNotIn("--offload_unused_runtime_modules", control)
        self.assertIn("--static_fisher_activation_checkpointing", candidate)
        self.assertIn("--offload_unused_runtime_modules", candidate)
        self.assertIn("--nproc_per_node=4", candidate)


if __name__ == "__main__":
    unittest.main()
