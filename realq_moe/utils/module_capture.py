"""Small module-forward capture helpers used by MoE-only fast paths."""

from __future__ import annotations

import torch
import torch.nn as nn


class _InnerLinearInputCaptured(RuntimeError):
    """Private control flow used to stop before an unneeded Linear GEMM."""


@torch.no_grad()
def capture_inner_linear_input_without_gemm(
    module: nn.Module,
    linear: nn.Linear,
    inputs: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    """Return the exact input seen by ``linear`` without evaluating it.

    Calling the outer module is required when it transforms the activation
    before delegating to the underlying ``Linear`` (for example an
    ``ActQuantWrapper`` with ``online_full_had`` enabled).  A private exception
    from the inner pre-hook terminates the call before the Linear GEMM.
    """

    captured: list[torch.Tensor] = []

    def hook(_module, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError(f"{label} underlying Linear received no tensor.")
        captured.append(args[0].detach())
        raise _InnerLinearInputCaptured

    handle = linear.register_forward_pre_hook(hook)
    try:
        try:
            module(inputs)
        except _InnerLinearInputCaptured:
            pass
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"{label} underlying Linear must run exactly once, "
            f"observed {len(captured)} calls."
        )
    return captured[0]
