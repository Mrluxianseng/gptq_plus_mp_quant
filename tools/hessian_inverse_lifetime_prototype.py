"""Research-only liveness change in the sequential GPTQ inverse helper.

Retains the original damping, retry, finite checks and all three CUDA solver
calls. No CPU transfer, arithmetic substitution, precision or stride change.
Not applied to batched/rank-group implementations without separate validation.
"""
import inspect
import textwrap


def build_method(impl):
    source = textwrap.dedent(inspect.getsource(
        impl.GPTQPlus._compute_hessian_inverse_with_fallback))
    before = """                chol = torch.linalg.cholesky(H_work)
                Hinv_init = torch.cholesky_inverse(chol)
                Hinv = torch.linalg.cholesky(Hinv_init, upper=True)"""
    after = """                chol = torch.linalg.cholesky(H_work)
                # Last consumer of the private damped working matrix completed.
                # PyTorch's allocator retains stream ordering for the same kernels.
                del H_work
                Hinv_init = torch.cholesky_inverse(chol)
                # Do not retain this factor while allocating the final factor.
                del chol
                Hinv = torch.linalg.cholesky(Hinv_init, upper=True)"""
    if source.count(before) != 1:
        raise RuntimeError("inverse helper changed; review the lifetime boundaries")
    source = source.replace(before, after)
    namespace = dict(vars(impl))
    exec(compile(source, "hessian_inverse_lifetime_prototype", "exec"), namespace)
    return namespace["_compute_hessian_inverse_with_fallback"], source


def install_down_proj(impl, *, enabled, records):
    original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
    proposed, source = build_method(impl)

    def inverse(self, H_sub, *args, **kwargs):
        name = str(getattr(self, "layer_name", ""))
        target = name.endswith("down_proj")
        method = proposed if enabled and target else original
        result = method(self, H_sub, *args, **kwargs)
        if target:
            init, factor, damping, fallback = result
            records.append(dict(layer=getattr(self, "layer_idx", None),
                                module=name, shape=list(H_sub.shape),
                                input_stride=list(H_sub.stride()),
                                factor_stride=list(factor.stride()),
                                dtype=str(H_sub.dtype), enabled=bool(enabled),
                                inverse_retained=init is not None,
                                damping=damping, identity_fallback=fallback))
        return result

    impl.GPTQPlus._compute_hessian_inverse_with_fallback = inverse
    return source


def build_reusing_fasterquant(impl):
    """Reuse each consumed H group for its immutable inverse-Cholesky factor.

    Keeps the original factor's exact strides, including identity fallback.
    Groups not yet consumed are untouched. Guards deliberately restrict the
    first experiment to alpha=0, sequential groups and no Hessian regularizer.
    """
    source = textwrap.dedent(inspect.getsource(impl.GPTQPlus.fasterquant))
    before = "    # Hinv is immutable for the subgroup. Under the guarded"
    # Locate by content then preserve the actual enclosing indentation.
    matching = [line for line in source.splitlines() if line.strip() == before.strip()]
    if len(matching) != 1:
        raise RuntimeError("inverse staging boundary changed")
    anchor = matching[0]
    indent = anchor[:len(anchor)-len(anchor.lstrip())]
    block = '''if (str(getattr(self, "layer_name", "")).endswith("down_proj")
        and self.alpha == 0 and not need_hessian_reg
        and not hessian_cpu_offload and not hinv_cpu_offload):
    # The current group's original H has no remaining consumers. Other
    # groups still hold their pristine H. Preserve solver layout exactly.
    _h_storage = self.H[sub_idx]
    if (not _h_storage.is_contiguous() or Hinv.shape != _h_storage.shape
            or Hinv.stride() not in ((_h_storage.shape[0], 1), (1, _h_storage.shape[0]))):
        raise RuntimeError("unsupported inverse/storage layout for reuse")
    _h_factor_destination = _h_storage.as_strided(Hinv.shape, Hinv.stride())
    _h_factor_destination.copy_(Hinv)
    Hinv = _h_factor_destination
    _reuse_records = getattr(self, "_hessian_factor_reuse_records", None)
    if _reuse_records is not None:
        _reuse_records.append(dict(layer=getattr(self, "layer_idx", None),
            module=str(getattr(self, "layer_name", "")), subgroup=sub_idx,
            shape=list(Hinv.shape), stride=list(Hinv.stride()),
            factor_shares_consumed_h_storage=True,
            identity_fallback=hessian_identity_fallback))
'''
    replacement = "\n".join(indent + line if line else "" for line in block.splitlines()) + "\n" + anchor
    source = source.replace(anchor, replacement)
    namespace = dict(vars(impl))
    exec(compile(source, "hessian_factor_reuse_prototype", "exec"), namespace)
    return namespace["fasterquant"], source
