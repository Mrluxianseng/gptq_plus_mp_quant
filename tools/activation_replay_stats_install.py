"""Bound early-statistics cleanup, with shared CPU-only coverage observations."""
import inspect
import textwrap

from tools.activation_replay_stats_lifetime_prototype import build_source


def observed_source(source, *, enabled):
    result = build_source(source) if enabled else source
    marker = "del gptq" if enabled else "if memory_profiler is not None:"
    if enabled:
        matches = [line for line in result.splitlines() if line.strip() == marker]
        anchor = matches[0] + "\n" if len(matches) == 1 else ""
    else:
        replay = 'memory_profiler.mark(f"layer_{i}_activation_replay_and_cleanup", layer=i)'
        lines = result.splitlines()
        indexes = [i for i, line in enumerate(lines) if line.strip() == replay]
        if len(indexes) != 1:
            raise RuntimeError("replay observation boundary changed")
        matches = [lines[indexes[0]-1]]
        if matches[0].strip() != marker:
            raise RuntimeError("replay observation guard changed")
        anchor = matches[0] + "\n" + lines[indexes[0]] + "\n"
    if len(matches) != 1 or not anchor or result.count(anchor) != 1:
        raise RuntimeError("ambiguous replay observation anchor")
    line = matches[0]
    indent = line[:len(line)-len(line.lstrip())]
    observer = indent + "_replay_stats_record(i, gptq, saliency_dict, gradients_dict)\n"
    result = result.replace(anchor, observer + anchor, 1)
    result = result.replace("def gptq_fwrd(", "def _replay_stats_generated_fwrd(", 1)
    compile(result, "activation_replay_stats_install", "exec")
    return result


def storage_metadata(torch, gptq, saliency, gradients):
    """Count unique CUDA storage among statistics; retain only CPU integers."""
    storages = {}
    def visit(value):
        if torch.is_tensor(value):
            if value.is_cuda and value.numel():
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr())
                storages[key] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
    visit(saliency)
    visit(gradients)
    if isinstance(gptq, dict):
        for owner in gptq.values():
            for field in ("gradients", "saliency", "act_square", "H", "Losses"):
                visit(getattr(owner, field, None))
    return dict(unique_cuda_storage_count=len(storages), observed_cuda_storage_bytes=sum(storages.values()))


def install(impl, *, enabled, records):
    source = textwrap.dedent(inspect.getsource(impl.gptq_fwrd))
    proposed = observed_source(source, enabled=enabled)
    def record(layer, gptq, saliency, gradients):
        row = storage_metadata(impl.torch, gptq, saliency, gradients)
        row.update(layer=int(layer), enabled=bool(enabled), boundary="before_activation_replay",
                   observed_allocated_bytes=int(impl.torch.cuda.memory_allocated()))
        records.append(row)
    # Preserve live module globals, including collectors/hooks installed later.
    impl.__dict__["_replay_stats_record"] = record
    exec(compile(proposed, "activation_replay_stats_install", "exec"), impl.__dict__)
    impl.gptq_fwrd = impl.__dict__.pop("_replay_stats_generated_fwrd")
    return proposed


def build_entry(source, *, installer_sha, prototype_sha):
    anchor = "original_fwrd = impl.gptq_fwrd"
    if source.count(anchor) != 1 or "_hinv_stage=" not in source:
        raise RuntimeError("bound measured-forward entry changed")
    hook = f'''
_replay_stats_installer_path=_HinvPath.cwd()/"tools/activation_replay_stats_install.py"
_replay_stats_prototype_path=_HinvPath.cwd()/"tools/activation_replay_stats_lifetime_prototype.py"
_replay_stats_installer_sha={installer_sha!r}
_replay_stats_prototype_sha={prototype_sha!r}
def _replay_stats_sources_ok():
    return (_hinv_hashlib.sha256(_replay_stats_installer_path.read_bytes()).hexdigest()==_replay_stats_installer_sha
            and _hinv_hashlib.sha256(_replay_stats_prototype_path.read_bytes()).hexdigest()==_replay_stats_prototype_sha)
if not _replay_stats_sources_ok():
    raise RuntimeError("replay statistics helpers changed")
_replay_stats_enabled="--research_replay_early_stats" in sys.argv
if _replay_stats_enabled:
    sys.argv.remove("--research_replay_early_stats")
_replay_stats_records=[]
from tools.activation_replay_stats_install import install as _replay_stats_install
_replay_stats_source=_replay_stats_install(impl,enabled=_replay_stats_enabled,records=_replay_stats_records)
def _replay_stats_save():
    (_HinvPath(_hinv_os.environ["RESEARCH_RUN_DIR"])/f"replay_stats_lifetime_{{_hinv_stage}}_rank{{_hinv_os.environ.get('RANK','0')}}.json").write_text(
        _hinv_json.dumps(dict(enabled=_replay_stats_enabled,records=_replay_stats_records,
            helpers_unchanged=_replay_stats_sources_ok(),installer_sha256=_replay_stats_installer_sha,
            prototype_sha256=_replay_stats_prototype_sha,scope="existing private statistics deletes before replay; same model/input/math"),indent=2))
_hinv_atexit.register(_replay_stats_save)
'''
    result = source.replace(anchor, hook + "\n" + anchor)
    compile(result, "bound_replay_stats_entry", "exec")
    return result
