"""CPU checks for fixture fidelity; never launches a CUDA experiment."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from tools.hessian_bottleneck_capture import cpu_snapshot, file_sha, install_capture, tensor_identity


class CaptureTests(unittest.TestCase):
    def test_snapshot_roundtrip_keeps_bf16_and_column_stride(self):
        with tempfile.TemporaryDirectory() as folder:
            original = torch.arange(20, dtype=torch.bfloat16).reshape(4, 5).T
            snapshot = cpu_snapshot(original)
            self.assertEqual(tensor_identity(snapshot), tensor_identity(original))
            original.add_(1)
            self.assertFalse(torch.equal(snapshot, original))
            target = Path(folder) / "tensor.pt"
            torch.save(snapshot, target)
            loaded = torch.load(target, weights_only=True)
            self.assertEqual(tensor_identity(snapshot), tensor_identity(loaded))

    def test_capture_first_actual_call_only_and_preserve_results(self):
        class Worker:
            def _compute_hessian_inverse_with_fallback(self, H_sub, percdamp=.01,
                    damp_auto_increment=.0015, profile_recorder=None, profile_section=None,
                    log_context="", need_hinv_init=True, inplace_damping=False):
                self.calls += 1
                if inplace_damping:
                    H_sub.add_(1)
                self.result = (H_sub.clone() if need_hinv_init else None,
                               H_sub.T.clone(), percdamp, False)
                return self.result
        impl = SimpleNamespace(GPTQPlus=Worker)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "capture"
            records = install_capture(impl, out, [3, 10], {"fixture": "test-only"})
            worker = Worker()
            worker.calls, worker.layer_idx, worker.layer_name = 0, 3, "mlp.down_proj"
            h = torch.arange(9, dtype=torch.float32).reshape(3, 3)
            before = h.clone()
            result = worker._compute_hessian_inverse_with_fallback(h, .03,
                        need_hinv_init=False, inplace_damping=True)
            self.assertIs(result, worker.result)
            self.assertEqual(worker.calls, 1)
            saved = torch.load(out / records[0]["file"], weights_only=True)
            self.assertTrue(torch.equal(saved["H"], before))
            self.assertTrue(torch.equal(saved["H_after"], h))
            self.assertIsNone(saved["inverse"])
            self.assertEqual(saved["options"]["percdamp"], .03)
            self.assertTrue(saved["options"]["inplace_damping"])
            self.assertEqual(saved["factor"].stride(), result[1].stride())
            worker._compute_hessian_inverse_with_fallback(h)
            worker.layer_idx, worker.layer_name = 10, "mlp.up_proj"
            worker._compute_hessian_inverse_with_fallback(h)
            worker.layer_idx = 7
            worker.layer_name = "mlp.down_proj"
            worker._compute_hessian_inverse_with_fallback(h)
            self.assertEqual(len(records), 1)
            self.assertEqual(worker.calls, 4)
            self.assertEqual(json.loads((out / "index.json").read_text())["records"], records)
            path = out / records[0]["file"]
            self.assertEqual(file_sha(path), records[0]["file_sha256"])
            with path.open("ab") as stream:
                stream.write(b"corruption")
            self.assertNotEqual(file_sha(path), records[0]["file_sha256"])


if __name__ == "__main__":
    unittest.main()
