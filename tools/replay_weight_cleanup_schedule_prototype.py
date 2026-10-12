"""Move the existing final down-proj free cleanup after compact relocation.

H/Losses release stays at the original free call. Only that call's existing
allocator/GC cleanup tail moves after old parameter storages lose their last
references. No extra empty_cache calls, tensor retention or cleanup scans.
"""
import ast
import inspect
import textwrap


def install(impl, *, enabled):
    original=impl.GPTQPlus.free
    source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source)
    expected=ast.parse('''def free(self):
    self.H = None
    self.Losses = None
    torch.cuda.empty_cache()
    memory_utils.cleanup_memory(verbos=False)
''')
    if ast.dump(tree,include_attributes=False)!=ast.dump(expected,include_attributes=False):
        raise RuntimeError("existing free cleanup boundary changed")
    cleanup_source='def _compact_existing_cleanup():\n'+''.join(source.splitlines(keepends=True)[3:])
    namespace=dict(vars(impl))
    exec(compile(cleanup_source,'compact_existing_cleanup','exec'),namespace)
    cleanup=namespace['_compact_existing_cleanup']
    pending=[] # CPU layer indices only; never retain a GPTQ owner or tensor.
    old_prepare=impl.__dict__['_replay_compact_prepare']
    def free(self):
        if enabled and str(getattr(self,'layer_name','')).endswith('down_proj'):
            if pending:
                raise RuntimeError('unconsumed final-down cleanup')
            self.H=None
            self.Losses=None
            pending.append(int(self.layer_idx))
        else:
            original(self)
    def prepare(layer,index):
        if enabled and pending!=[int(index)]:
            raise RuntimeError('requires exactly one deferred original down cleanup per layer')
        ticket=old_prepare(layer,index)
        if enabled:
            cleanup()
            pending.clear()
            ticket[1]['rescheduled_existing_cleanup_calls']=1
        return ticket
    impl.GPTQPlus.free=free
    impl.__dict__['_replay_compact_prepare']=prepare
    return pending


def build_entry(source, *, helper_sha):
    anchor='_compact_original_builder=_compact_install(impl,_compact_stats_module,enabled=_replay_stats_enabled,records=_compact_records)'
    if source.count(anchor)!=1:
        raise RuntimeError('requires compact placement install before cleanup schedule')
    hook=f'''_compact_cleanup_path=_HinvPath.cwd()/"tools/replay_weight_cleanup_schedule_prototype.py"
_compact_cleanup_sha={helper_sha!r}
if _hinv_hashlib.sha256(_compact_cleanup_path.read_bytes()).hexdigest()!=_compact_cleanup_sha:
    raise RuntimeError("compact cleanup schedule helper changed")
from tools.replay_weight_cleanup_schedule_prototype import install as _compact_cleanup_install
_compact_cleanup_pending=_compact_cleanup_install(impl,enabled=_replay_stats_enabled)
def _compact_cleanup_save():
    (_HinvPath(_hinv_os.environ["RESEARCH_RUN_DIR"])/f"replay_weight_cleanup_{{_hinv_stage}}_rank{{_hinv_os.environ.get('RANK','0')}}.json").write_text(
        _hinv_json.dumps(dict(enabled=_replay_stats_enabled,pending_layers=_compact_cleanup_pending,
            helper_unchanged=_hinv_hashlib.sha256(_compact_cleanup_path.read_bytes()).hexdigest()==_compact_cleanup_sha),indent=2))
_hinv_atexit.register(_compact_cleanup_save)
'''
    result=source.replace(anchor,anchor+'\n'+hook)
    compile(result,'compact_cleanup_bound_entry','exec')
    return result
