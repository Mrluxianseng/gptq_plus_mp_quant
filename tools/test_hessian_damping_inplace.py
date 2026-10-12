"""Numerical equivalence smoke test for in-place Hessian damping."""
from __future__ import annotations

import torch
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptq_utils.gptq_plus_utils import GPTQPlus


def check_case(hessian: torch.Tensor, *, need_hinv_init: bool) -> None:
    helper = GPTQPlus.__new__(GPTQPlus)
    expected_init, expected, expected_damp, expected_fallback = (
        helper._compute_hessian_inverse_with_fallback(
            hessian.clone(),
            percdamp=0.01,
            need_hinv_init=need_hinv_init,
            inplace_damping=False,
        )
    )
    actual_init, actual, actual_damp, actual_fallback = (
        helper._compute_hessian_inverse_with_fallback(
            hessian.clone(),
            percdamp=0.01,
            need_hinv_init=need_hinv_init,
            inplace_damping=True,
        )
    )
    assert expected_fallback == actual_fallback
    assert expected_damp == actual_damp
    assert torch.equal(expected, actual), (
        f"inverse-Cholesky mismatch: max_abs="
        f"{(expected - actual).abs().max().item():.8g}"
    )
    if expected_init is not None:
        assert actual_init is not None
        assert torch.equal(expected_init, actual_init), (
            f"inverse mismatch: max_abs="
            f"{(expected_init - actual_init).abs().max().item():.8g}"
        )


def main() -> None:
    logging.disable(logging.CRITICAL)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this kernel-path smoke test")
    torch.manual_seed(17)
    device = torch.device("cuda")
    for dim in (32, 256):
        x = torch.randn(dim + 8, dim, device=device, dtype=torch.float32)
        hessian = x.T @ x
        check_case(hessian, need_hinv_init=False)
        check_case(hessian, need_hinv_init=True)

    # An indefinite Hessian exercises repeated damping attempts/fallback.
    indefinite = torch.eye(32, device=device, dtype=torch.float32)
    indefinite[0, 1] = 2.0
    indefinite[1, 0] = 2.0
    check_case(indefinite, need_hinv_init=False)
    print("in-place damping matched the reference bitwise for all smoke cases")


if __name__ == "__main__":
    main()
