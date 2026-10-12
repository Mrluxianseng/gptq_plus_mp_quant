import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.loss_utils import (
    chunked_tokenwise_kl_from_logits,
    tokenwise_kl_from_logits,
)


class ChunkedKLTests(unittest.TestCase):
    def test_chunked_tokenwise_kl_matches_loss_and_student_gradient(self):
        for dtype in (torch.float32, torch.bfloat16):
            for topk in (-1, 9):
                with self.subTest(dtype=dtype, topk=topk):
                    self._check_case(dtype, topk)

    def _check_case(self, dtype, topk):
        torch.manual_seed(1729)
        student = (torch.randn(3, 7, 31) * 0.7).to(dtype).requires_grad_(True)
        teacher = (torch.randn(3, 7, 31) * 0.7).to(dtype)
        reference_student = student.detach().clone().requires_grad_(True)

        if topk > 0:
            teacher_reference, indices = teacher.float().topk(
                topk, dim=-1, sorted=False
            )
            student_reference = reference_student.float().gather(-1, indices)
            reference_loss = tokenwise_kl_from_logits(
                student_reference, teacher_reference
            ).mean()
        else:
            reference_loss = tokenwise_kl_from_logits(
                reference_student, teacher
            ).mean()
        reference_grad = torch.autograd.grad(reference_loss, reference_student)[0]

        chunked_loss = chunked_tokenwise_kl_from_logits(
            student, teacher, chunk_tokens=5, topk=topk
        )
        chunked_grad = torch.autograd.grad(chunked_loss, student)[0]

        self.assertTrue(torch.allclose(
            chunked_loss, reference_loss, atol=2e-6, rtol=2e-6
        ))
        tolerance = 2e-6 if dtype == torch.float32 else 2e-3
        self.assertTrue(torch.allclose(
            chunked_grad.float(),
            reference_grad.float(),
            atol=tolerance,
            rtol=tolerance,
        ))


if __name__ == "__main__":
    unittest.main()
