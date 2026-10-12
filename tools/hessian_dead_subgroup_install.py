"""Bound, down-proj-only lifetime followup; no all-linear or finite extension.

Retains the verified workspace/consumed-H/stable-down-write combination.
The added cleanup changes only the final dead private input's lifetime.
Real CUDA prefix and full acceptance remain required.
"""
import inspect

from tools.hessian_dead_subgroup_input_prototype import build_source


def install_builder(impl, stable_module):
    original = stable_module.build_stable_fasterquant

    def build(module):
        _, source = original(module)
        source = build_source(source)
        namespace = dict(vars(module))
        exec(compile(source, "hessian_dead_subgroup_install", "exec"), namespace)
        return namespace["fasterquant"], source

    stable_module.build_stable_fasterquant = build
    return original


def build_entry(source, *, installer_sha, prototype_sha):
    before = 'if _hinv_stable_expected:\n    from tools.hessian_stable_weight_prototype import install_fasterquant as _hinv_install_stable'
    if source.count(before) != 1:
        raise RuntimeError("stable down-only installer boundary changed")
    hook = (
        'from tools.hessian_dead_subgroup_install import install_builder as _dead_install_builder\n'
        'from tools import hessian_stable_weight_prototype as _dead_stable_module\n'
        '_dead_installer=_HinvPath.cwd()/"tools/hessian_dead_subgroup_install.py"\n'
        '_dead_prototype=_HinvPath.cwd()/"tools/hessian_dead_subgroup_input_prototype.py"\n'
        f'_dead_installer_sha={installer_sha!r}\n'
        f'_dead_prototype_sha={prototype_sha!r}\n'
        'if (_hinv_hashlib.sha256(_dead_installer.read_bytes()).hexdigest()!=_dead_installer_sha\n'
        '        or _hinv_hashlib.sha256(_dead_prototype.read_bytes()).hexdigest()!=_dead_prototype_sha):\n'
        '    raise RuntimeError("dead subgroup input helper changed")\n'
        '_dead_original_builder=_dead_install_builder(_hinv_impl,_dead_stable_module)\n'
        '_dead_records=[]\n'
    )
    source = source.replace(before, hook + before)
    init = '    self._hessian_factor_reuse_records=_hinv_reuse_calls'
    if source.count(init) != 1:
        raise RuntimeError("shared metadata initializer boundary changed")
    source = source.replace(init, init + '\n    self._dead_subgroup_input_records=_dead_records')
    anchor = '_hinv_atexit.register(_hinv_record)'
    if source.count(anchor) != 1:
        raise RuntimeError("shared factor observer finalizer changed")
    observer = '''
def _dead_save():
    _dead_ok=(_hinv_hashlib.sha256(_dead_installer.read_bytes()).hexdigest()==_dead_installer_sha
              and _hinv_hashlib.sha256(_dead_prototype.read_bytes()).hexdigest()==_dead_prototype_sha)
    _dead_destination=_HinvPath(_hinv_os.environ["RESEARCH_RUN_DIR"])
    (_dead_destination/f"dead_subgroup_input_{_hinv_stage}_rank{_hinv_os.environ.get('RANK','0')}.json").write_text(
        _hinv_json.dumps(dict(enabled=_hinv_stable,records=_dead_records,helpers_unchanged=_dead_ok,
            installer_sha256=_dead_installer_sha,prototype_sha256=_dead_prototype_sha,
            scope="down_proj final dead private input only; immutable factors retained"),indent=2))
_hinv_atexit.register(_dead_save)
'''
    source = source.replace(anchor, anchor + '\n' + observer)
    compile(source, "dead_subgroup_bound_entry", "exec")
    return source
