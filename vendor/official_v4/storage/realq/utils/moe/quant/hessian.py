"""Hessian numerical helpers: Cholesky inverse with damp + NaN sanitisation.

Mirrors the old GPTQPlus per-subgroup `_compute_hessian_inverse_with_fallback`
but only for the simple non-sharded path. NUM_GROUPS-aware sharded variants
land in sub-task 6.
"""
from __future__ import annotations

import logging

import torch


def cholesky_inverse_batched_with_damp(
    hessians: torch.Tensor,
    percdamp: float = 0.01,
    damp_auto_increment: float = 0.0015,
) -> torch.Tensor:
    """Legacy-compatible batched upper-Cholesky factors of damped inverses.

    This intentionally follows ``GPTQPlus.
    _compute_hessian_inverse_batched_with_fallback`` operation-for-operation.
    A Python loop over groups is mathematically equivalent but selects
    different CUDA kernels and can move borderline weights across a quantizer
    cell, which then compounds through later REAL-Q blocks.
    """
    damp_percent_value = float(percdamp)
    if damp_percent_value <= 0:
        raise ValueError(
            f"`percdamp` must be positive. Got {damp_percent_value}."
        )
    num_groups, columns, _ = hessians.shape
    device = hessians.device
    dtype = hessians.dtype
    diag_idx = torch.arange(columns, device=device)
    eye = torch.eye(columns, device=device, dtype=dtype)
    hinv = torch.empty_like(hessians)
    damp_percent = torch.full(
        (num_groups,), damp_percent_value, device=device, dtype=torch.float32,
    )
    pending = torch.ones(num_groups, device=device, dtype=torch.bool)
    last_info = torch.zeros(num_groups, device=device, dtype=torch.int32)

    while bool(pending.any().item()):
        active_idx = pending.nonzero(as_tuple=False).flatten()
        work = hessians.index_select(0, active_idx).clone()
        active_damp = damp_percent.index_select(0, active_idx).to(dtype)
        diag_mean = torch.diagonal(work, dim1=-2, dim2=-1).mean(dim=1)
        work[:, diag_idx, diag_idx] += (active_damp * diag_mean).unsqueeze(1)
        chol, info = torch.linalg.cholesky_ex(work)

        ok = info == 0
        failed = info != 0
        if bool(ok.any().item()):
            ok_local_idx = ok.nonzero(as_tuple=False).flatten()
            ok_global_idx = active_idx.index_select(0, ok_local_idx)
            inverse = torch.cholesky_inverse(
                chol.index_select(0, ok_local_idx)
            )
            upper, upper_info = torch.linalg.cholesky_ex(inverse, upper=True)
            finite = (
                torch.isfinite(inverse).flatten(1).all(dim=1)
                & torch.isfinite(upper).flatten(1).all(dim=1)
            )
            ok2 = (upper_info == 0) & finite
            if bool(ok2.any().item()):
                ok2_local_idx = ok2.nonzero(as_tuple=False).flatten()
                ok2_global_idx = ok_global_idx.index_select(
                    0, ok2_local_idx
                )
                hinv.index_copy_(
                    0, ok2_global_idx, upper.index_select(0, ok2_local_idx)
                )
                pending.index_fill_(0, ok2_global_idx, False)
            failed_ok = ~ok2
            if bool(failed_ok.any().item()):
                failed_positions = failed_ok.nonzero(
                    as_tuple=False
                ).flatten()
                failed_local_idx = ok_local_idx.index_select(
                    0, failed_positions
                )
                failed[failed_local_idx] = True
                failed_global_idx = ok_global_idx.index_select(
                    0, failed_positions
                )
                last_info.index_copy_(
                    0,
                    failed_global_idx,
                    upper_info.index_select(0, failed_positions).to(
                        torch.int32
                    ),
                )

        failed_positions = failed.nonzero(as_tuple=False).flatten()
        failed_global_idx = active_idx.index_select(0, failed_positions)
        if failed_global_idx.numel() > 0:
            last_info.index_copy_(
                0,
                failed_global_idx,
                info.index_select(0, failed_positions).to(torch.int32),
            )
            damp_percent.index_add_(
                0,
                failed_global_idx,
                torch.full(
                    (failed_global_idx.numel(),),
                    damp_auto_increment,
                    device=device,
                    dtype=torch.float32,
                ),
            )
            still_retry = (
                damp_percent.index_select(0, failed_global_idx) < 1
            )
            giveup_idx = failed_global_idx.index_select(
                0, (~still_retry).nonzero(as_tuple=False).flatten()
            )
            if giveup_idx.numel() > 0:
                hinv.index_copy_(
                    0,
                    giveup_idx,
                    eye.expand(giveup_idx.numel(), -1, -1),
                )
                pending.index_fill_(0, giveup_idx, False)
                for idx in giveup_idx.detach().cpu().tolist():
                    logging.warning(
                        "[realq.hessian] subgroup=%d reached damp %.5f; "
                        "using identity inverse (last info=%d).",
                        idx,
                        float(damp_percent[idx].item()),
                        int(last_info[idx].item()),
                    )
    return hinv


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
