"""Research-only bounded temporary storage for solver finiteness checks.

No change to Cholesky, damping, retry or floating-point arithmetic. Boolean
AND is exact. Slice along the contiguous dimension to preserve dense reads
for both row-major Hessians and column-major CUDA factors. One host decision
per matrix, rather than one synchronization per chunk.
"""
import inspect
import textwrap

import torch


def all_finite_chunked(value, max_mask_elements=1024 * 1024):
    if max_mask_elements < 1:
        raise ValueError("positive mask element budget required")
    if value.ndim != 2 or value.layout != torch.strided:
        raise ValueError("solver finite screening expects a dense matrix")
    if not value.numel():
        return torch.ones((), dtype=torch.bool, device=value.device)
    # A full row of a row-major matrix, or full column of a column-major
    # matrix, is a view. Never flatten/contiguous-copy the whole matrix.
    axis = 0 if value.stride(1) == 1 else 1
    cross_section = value.shape[1 - axis]
    length = max(1, max_mask_elements // max(1, cross_section))
    result = torch.ones((), dtype=torch.bool, device=value.device)
    for begin in range(0, value.shape[axis], length):
        part = value.narrow(axis, begin, min(length, value.shape[axis] - begin))
        result.logical_and_(torch.isfinite(part).all())
    return result


def build_method(impl, *, workspace=False):
    if workspace:
        from tools.hessian_solver_workspace_prototype import build_method as build_workspace
        _, source = build_workspace(impl)
    else:
        source = textwrap.dedent(inspect.getsource(
            impl.GPTQPlus._compute_hessian_inverse_with_fallback))
    old = "if not torch.isfinite(Hinv_init).all() or not torch.isfinite(Hinv).all():"
    new = "if not _chunked_finite(Hinv_init) or not _chunked_finite(Hinv):"
    if source.count(old) != 1:
        raise RuntimeError("solver finiteness boundary changed")
    source = source.replace(old, new)
    namespace = dict(vars(impl), _chunked_finite=all_finite_chunked,
        _workspace_reference=impl.GPTQPlus._compute_hessian_inverse_with_fallback)
    exec(compile(source, "hessian_chunked_finite_prototype", "exec"), namespace)
    return namespace["_compute_hessian_inverse_with_fallback"], source
