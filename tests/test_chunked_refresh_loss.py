import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptq_utils.gptq_plus_utils import compute_refresh_loss


class ChunkedRefreshLossTests(unittest.TestCase):
    def test_all_kl_refresh_surrogates_match_loss_and_gradient(self):
        torch.manual_seed(20261001)
        teacher = torch.randn(2, 7, 19)
        student0 = (teacher + torch.randn_like(teacher) * 0.15).detach()

        cases = (
            ("kl", None),
            ("residual_kl", None),
            ("refined_residual_kl", torch.zeros(19, 19)),
            ("refined_diag_residual_kl", torch.zeros(7, 19)),
        )
        for loss_type, refined_A in cases:
            with self.subTest(loss_type=loss_type):
                ref_student = student0.clone().requires_grad_(True)
                chunk_student = student0.clone().requires_grad_(True)
                common = {
                    "refresh_loss_type": loss_type,
                    "fp_hidden": teacher,
                    "analyzer": None,
                    "kl_topk": 11,
                    "fp_final_hidden": teacher,
                    "refined_A": refined_A,
                }
                with patch(
                    "gptq_utils.gptq_plus_utils.hidden2logits",
                    side_effect=lambda hidden, _analyzer: hidden,
                ):
                    reference_loss = compute_refresh_loss(
                        out_hidden=ref_student,
                        kl_logit_chunk_tokens=0,
                        **common,
                    )
                    reference_grad = torch.autograd.grad(
                        reference_loss, ref_student
                    )[0]

                    chunked_loss = compute_refresh_loss(
                        out_hidden=chunk_student,
                        kl_logit_chunk_tokens=3,
                        **common,
                    )
                    chunked_grad = torch.autograd.grad(
                        chunked_loss, chunk_student
                    )[0]

                self.assertTrue(torch.allclose(
                    chunked_loss, reference_loss, atol=2e-6, rtol=2e-6
                ))
                self.assertTrue(torch.allclose(
                    chunked_grad, reference_grad, atol=2e-6, rtol=2e-6
                ))


if __name__ == "__main__":
    unittest.main()
