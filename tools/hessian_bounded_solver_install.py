"""Opt-in real-trajectory installer for the bounded finite solver.

Prepare before CUDA screening, but do not interpret CPU integration checks as
real-Hessian or full-process memory acceptance. Only sequential down_proj is
covered; inplace damping retains the original helper through workspace guard.
"""
from tools.hessian_chunked_finite_prototype import build_method


def install_down_proj(impl, *, enabled, records):
    original = impl.GPTQPlus._compute_hessian_inverse_with_fallback
    proposed, source = build_method(impl, workspace=True)

    def inverse(self, h_sub, *args, **kwargs):
        target = str(getattr(self, "layer_name", "")).endswith("down_proj")
        active = bool(enabled and target)
        result = (proposed if active else original)(self, h_sub, *args, **kwargs)
        if target:
            init, factor, damp, fallback = result
            eligible = active and not kwargs.get("inplace_damping", False)
            records.append(dict(layer=getattr(self, "layer_idx", None),
                module=self.layer_name, shape=list(h_sub.shape),
                input_stride=list(h_sub.stride()), factor_stride=list(factor.stride()),
                dtype=str(h_sub.dtype), enabled=bool(enabled),
                inverse_retained=init is not None, damping=damp,
                identity_fallback=fallback, workspace_reuse_enabled=eligible,
                bounded_finite_enabled=eligible))
        return result

    impl.GPTQPlus._compute_hessian_inverse_with_fallback = inverse
    return source


def build_entry(raw_source, *, installer_sha, finite_sha, workspace_sha):
    """Replace bound installer in an unprofiled workspace entry, fail closed."""
    replacements = (
        ("from tools.hessian_solver_workspace_prototype import install_down_proj as _hinv_install",
         "from tools.hessian_bounded_solver_install import install_down_proj as _hinv_install"),
        ('_hinv_helper=_HinvPath.cwd()/"tools/hessian_solver_workspace_prototype.py"',
         '_hinv_helper=_HinvPath.cwd()/"tools/hessian_bounded_solver_install.py"'),
    )
    for old, new in replacements:
        if raw_source.count(old) != 1:
            raise RuntimeError("bound workspace entry installer boundary changed")
        raw_source = raw_source.replace(old, new)
    import re
    pattern = r'(?m)^_hinv_expected="[0-9a-f]{64}"$'
    if len(re.findall(pattern, raw_source)) != 1:
        raise RuntimeError("bound workspace entry SHA boundary changed")
    raw_source = re.sub(pattern, '_hinv_expected="' + installer_sha + '"', raw_source)
    anchor = "_hinv_source=_hinv_install(_hinv_impl,enabled=_hinv_release,records=_hinv_calls)"
    if raw_source.count(anchor) != 1:
        raise RuntimeError("bound workspace installer invocation changed")
    # These dependent modules are not covered by the installer's file hash.
    # Check them before installation, including when the control arm is used.
    guard = (
        '_finite_helper=_HinvPath.cwd()/"tools/hessian_chunked_finite_prototype.py"\n'
        f'if _hinv_hashlib.sha256(_finite_helper.read_bytes()).hexdigest()!={finite_sha!r}:\n'
        '    raise RuntimeError("bounded finite helper changed")\n'
        '_finite_workspace=_HinvPath.cwd()/"tools/hessian_solver_workspace_prototype.py"\n'
        f'if _hinv_hashlib.sha256(_finite_workspace.read_bytes()).hexdigest()!={workspace_sha!r}:\n'
        '    raise RuntimeError("bounded workspace dependency changed")\n'
    )
    # Insert at top level before the existing branch, preserving indentation.
    before = "if _hinv_pool_expected:\n    from tools.hessian_inverse_pool_prototype"
    if raw_source.count(before) != 1:
        raise RuntimeError("bound workspace installer branch changed")
    return raw_source.replace(before, guard + before)
