"""fisher_mse loss used by block_gd refresh.

Fisher is precomputed in :mod:`realq.precompute.static_e2e` as the full
``(H, H)`` empirical Fisher at each transformer-block output, see
:func:`realq.precompute.hooks.FisherHookManager`. Here we reuse it as the
quadratic kernel of the block-boundary refresh loss:

    L = 0.5 * mean_b (Δy_b @ F @ Δy_b)
      = 0.5 * mean_b sum_t (Δy_{b,t} F Δy_{b,t})

with ``Δy = q_layer_output - fp_layer_output`` per (batch, token, hidden)
and ``F`` of shape ``(H, H)`` shared across (batch, token). Sub-task 5
only implements the full-matrix variant — the per-token-diag legacy form
(``legacy_fisher_diag_mse``) is explicitly out of scope per the rebuild
plan.
"""
from __future__ import annotations

import torch


# torch.quantile materialises a sorted view of the input. Past ~16M elements
# this OOMs on a 24 GB card for fp32 inputs and also issues a host sync.
# Mirrors old ``_TORCH_QUANTILE_SAFE_NUMEL`` (gptq_plus_utils.py:43).
_TORCH_QUANTILE_SAFE_NUMEL = 16 * 1024 * 1024


def _activation_clip_threshold(tensor: torch.Tensor, q: float) -> torch.Tensor | None:
    """Return the quantile cap used by :func:`_scale_delta_by_abs_quantile`.

    Small tensors keep ``torch.quantile``'s linear-interpolation semantics.
    Large tensors use the order-statistic fallback (topk on the smaller tail
    only) to dodge the quantile op's OOM / host sync. Mirrors old
    ``_activation_clip_threshold`` (gptq_plus_utils.py:69-89).
    """
    flat = tensor.reshape(-1)
    n = flat.numel()
    if n == 0:
        return None
    if n <= _TORCH_QUANTILE_SAFE_NUMEL:
        try:
            return torch.quantile(flat, q)
        except RuntimeError:
            pass
    k = max(1, min(n, int(n * q + 0.5)))
    upper_count = n - k + 1
    if k <= upper_count:
        return flat.topk(k, largest=False, sorted=False).values.max()
    return flat.topk(upper_count, largest=True, sorted=False).values.min()


def _scale_delta_by_abs_quantile(
    delta: torch.Tensor,
    ratio: float,
    threshold: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cap each element of ``delta`` at the ``ratio``-quantile of ``|delta|``.

    Returns ``delta * scale`` where ``scale`` is a DETACHED, per-element
    multiplier in (0, 1]. Elements with ``|delta| <= threshold`` get
    scale = 1; larger elements get scale = threshold / |delta| < 1, which
    pulls ``|delta * scale|`` back down to ``threshold`` while preserving
    sign. Because ``scale`` is detached, gradient still flows through
    ``delta`` itself — this is a soft mask on the loss magnitude, not on
    the autograd graph. Mirrors old ``_scale_delta_by_abs_quantile``
    (gptq_plus_utils.py:92-103).
    """
    if ratio >= 1.0:
        return delta
    abs_delta = delta.detach().float().abs()
    if threshold is None:
        threshold = _activation_clip_threshold(abs_delta, float(ratio))
    else:
        threshold = threshold.to(
            device=abs_delta.device, dtype=abs_delta.dtype
        )
    if threshold is None:
        return delta
    scale = abs_delta.clamp_min_(torch.finfo(abs_delta.dtype).tiny)
    scale.reciprocal_().mul_(threshold.detach()).clamp_(max=1.0)
    scale.nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)
    return delta * scale.to(delta.dtype)


def fisher_mse_loss(
    q_out: torch.Tensor,
    fp_out: torch.Tensor,
    fisher: torch.Tensor,
    a_loss_ratio: float = 1.0,
    a_loss_threshold: torch.Tensor | None = None,
) -> torch.Tensor:
    """``0.5 * mean_{b,t} (Δy_{b,t}^T F Δy_{b,t})``.

    Args:
        q_out: ``(B, T, H)`` — current quantised-layer output (requires_grad).
        fp_out: ``(B, T, H)`` — FP reference output (no grad).
        fisher: ``(H, H)`` — precomputed empirical Fisher at this layer's output.
        a_loss_ratio: kept-fraction quantile for outlier clipping on
            ``|delta|``. ``1.0`` (default) disables clipping. Matches old
            GPTQ+ ``--a_loss_ratio`` behaviour for fisher MSE losses
            (gptq_plus_utils.py:5660-5664).

    Notes:
        * ``q_out`` is replayed from the current quantized input stream, while
          ``fp_out`` is the matching full-precision block-output target for
          the same sample ids.  This deliberately includes accumulated
          upstream drift, matching REAL-Q's upstream-correction motivation.
        * Reduction matches old ``compute_refresh_loss(fisher_diag_mse)`` at
          line 5821-5826: per-token quadratic ``Δy^T F Δy``, then ``.mean()``
          over the FLATTENED ``(B*T,)`` axis. Earlier RealQ summed over T
          and meant only over B, which made loss ``T×`` larger and led
          ``grad_clip=1`` to clip every gradient to its limit ⇒ Adam saw
          uninformative grads and the resulting weights diverged from old.
    """
    if q_out.shape != fp_out.shape:
        raise ValueError(
            f"fisher_mse_loss: q_out shape {tuple(q_out.shape)} != fp_out shape {tuple(fp_out.shape)}"
        )
    if fisher.dim() != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(
            f"fisher_mse_loss: expected (H, H) fisher matrix, got shape {tuple(fisher.shape)}"
        )
    H = q_out.shape[-1]
    if fisher.shape[0] != H:
        raise ValueError(
            f"fisher_mse_loss: fisher hidden dim {fisher.shape[0]} != q_out hidden dim {H}"
        )
    # Preserve the legacy arithmetic order exactly.  The original path forms
    # Δ and, when enabled, applies the detached activation-loss clip in the
    # activation dtype (normally bf16), then promotes the clipped value for the
    # Fisher quadratic.  Promoting before clipping changes both the value and
    # bf16 gradient at the clipped tail and is amplified by subsequent Adam
    # refreshes.
    delta = q_out - fp_out                                        # (B, T, H)
    if a_loss_ratio < 1.0:
        delta = _scale_delta_by_abs_quantile(
            delta,
            float(a_loss_ratio),
            threshold=a_loss_threshold,
        )
    delta = delta.float()
    fisher = fisher.to(device=delta.device, dtype=torch.float32)  # (H, H)
    delta_flat = delta.reshape(-1, H)                             # (B*T, H)
    quad = (delta_flat @ fisher * delta_flat).sum(dim=-1)         # (B*T,)
    return 0.5 * quad.mean()
