"""Research-only two-buffer Cholesky solver storage, preserving API arithmetic.

Out= destinations never alias that operator's input. Restrict to non-inplace
damping; inplace callers keep the reference helper. Failed retries start from
the original H_sub as before. This is a workspace placement experiment.
"""
import inspect
import textwrap


def build_method(impl):
    original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
    source = textwrap.dedent(inspect.getsource(original))
    before = '''                chol = torch.linalg.cholesky(H_work)
                Hinv_init = torch.cholesky_inverse(chol)
                Hinv = torch.linalg.cholesky(Hinv_init, upper=True)'''
    after = '''                chol = torch.linalg.cholesky(H_work)
                # H_work's last input consumer has completed. Original inverse
                # and upper factors are column-major; preserve that layout.
                inverse_destination = H_work.as_strided(H_work.shape, chol.stride())
                Hinv_init = torch.cholesky_inverse(chol, out=inverse_destination)
                # chol is no longer an input consumer after the inverse call.
                # Reuse it as the distinct final factor destination.
                Hinv = torch.linalg.cholesky(Hinv_init, upper=True, out=chol)'''
    if source.count(before) != 1:
        raise RuntimeError("solver arithmetic boundary changed")
    # The original inplace input must remain pristine apart from its diagonal.
    # Do not overwrite it with an inverse through the out= destination.
    header = '    damp_percent = float(percdamp)'
    if source.count(header) != 1:
        raise RuntimeError("damping guard boundary changed")
    source = source.replace(before, after)
    source = source.replace(header, '''    if inplace_damping:
        return _workspace_reference(self, H_sub, percdamp=percdamp,
            damp_auto_increment=damp_auto_increment,
            profile_recorder=profile_recorder, profile_section=profile_section,
            log_context=log_context, need_hinv_init=need_hinv_init,
            inplace_damping=True)
''' + header)
    namespace = dict(vars(impl), _workspace_reference=original)
    exec(compile(source, "hessian_solver_workspace_prototype", "exec"), namespace)
    return namespace["_compute_hessian_inverse_with_fallback"], source


def install_down_proj(impl, *, enabled, records):
    original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
    proposed, source = build_method(impl)

    def inverse(self, h_sub, *args, **kwargs):
        target = str(getattr(self, "layer_name", "")).endswith("down_proj")
        result = (proposed if enabled and target else original)(self,h_sub,*args,**kwargs)
        if target:
            init, factor, damp, fallback = result
            records.append(dict(layer=getattr(self,"layer_idx",None),module=self.layer_name,
                                shape=list(h_sub.shape),input_stride=list(h_sub.stride()),
                                factor_stride=list(factor.stride()),dtype=str(h_sub.dtype),
                                enabled=bool(enabled),inverse_retained=init is not None,
                                damping=damp,identity_fallback=fallback,
                                workspace_reuse_enabled=bool(enabled and not kwargs.get("inplace_damping",False))))
        return result

    impl.GPTQPlus._compute_hessian_inverse_with_fallback = inverse
    return source
