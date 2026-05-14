"""Hessian numerical helpers: Cholesky inverse with damp + NaN sanitisation.

Mirrors the old GPTQPlus per-subgroup `_compute_hessian_inverse_with_fallback`
but only for the simple non-sharded path. NUM_GROUPS-aware sharded variants
land in sub-task 6.
"""
from __future__ import annotations

import logging

import torch


def cholesky_inverse_with_damp(
    H: torch.Tensor,
    percdamp: float = 0.01,
    max_doublings: int = 8,
) -> torch.Tensor:
    """Compute the upper-triangular Cholesky factor of inv(H_damped).

    Returns the upper-triangular ``Hinv`` factor used by GPTQ's per-column
    update; ``Hinv[i,i:] @ X = X / (L L^T)[i,i:]``.

    On Cholesky failure we double ``percdamp`` up to ``max_doublings`` times,
    then fall back to identity (logging loudly). Matches old behaviour.
    """
    columns = H.shape[-1]
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    cur_pct = float(percdamp)
    diag_idx = torch.arange(columns, device=H.device)
    for attempt in range(max_doublings + 1):
        # Match old GPTQPlus exactly: keep `damp` as a torch.Tensor (fp32 on
        # the GPU) rather than converting to Python float via .item(). The
        # difference between `pct * tensor.mean(diag)` and `pct * tensor.mean(diag).item()`
        # is one fp64-then-back-to-fp32 round trip, which is enough to flip
        # later quant cells when w_clip's MSE search is also active.
        H_d = H.clone()
        damp = cur_pct * torch.mean(torch.diag(H_d))
        H_d[diag_idx, diag_idx] += damp
        try:
            L = torch.linalg.cholesky(H_d)
        except torch.linalg.LinAlgError:
            if attempt == max_doublings:
                logging.warning(
                    "[realq.hessian] Cholesky failed after %d damp doublings (final pct=%g); "
                    "falling back to identity Hinv.",
                    max_doublings, cur_pct,
                )
                return torch.eye(columns, device=H.device, dtype=H.dtype)
            cur_pct *= 2.0
            continue
        Hinv_init = torch.cholesky_inverse(L)
        return torch.linalg.cholesky(Hinv_init, upper=True)

    raise RuntimeError("cholesky_inverse_with_damp: exhausted retry loop without resolution.")
