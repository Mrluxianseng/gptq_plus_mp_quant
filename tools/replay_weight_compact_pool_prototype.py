"""Research: isolate current layer parameters during quantized replay.

One compact native-pool allocation replaces fragmented parameter storages.
Copies preserve values, parameter objects and contiguous shapes/strides.
The pool lives until the existing layer manager moves this layer to CPU.
No allocator flushing, host offload or arithmetic substitution is added.
"""
import inspect


def layout(layer):
    parameters = list(layer.named_parameters())
    if not parameters:
        raise RuntimeError("empty replay layer")
    dtype, device = parameters[0][1].dtype, parameters[0][1].device
    pointers = []
    for name, param in parameters:
        if (param.dtype != dtype or param.device != device or not param.is_contiguous()
                or param.storage_offset() != 0 or not param.numel()):
            raise RuntimeError("unsupported replay parameter layout: " + name)
        pointers.append(param.untyped_storage().data_ptr())
    if len(set(pointers)) != len(pointers):
        raise RuntimeError("aliased replay parameters require separate review")
    return parameters, dtype, device


def prepare(torch, layer, index, *, enabled, records):
    parameters, dtype, device = layout(layer)
    if device.type != "cuda":
        raise RuntimeError("replay layer must be on CUDA")
    specs = [dict(name=n, bytes=p.numel()*p.element_size(), shape=list(p.shape),
                  stride=list(p.stride()), dtype=str(p.dtype)) for n,p in parameters]
    row = dict(layer=int(index), enabled=bool(enabled), parameters=specs,
               parameter_count=len(specs), payload_bytes=sum(x["bytes"] for x in specs),
               boundary="after_statistics_delete_before_replay", released=False)
    records.append(row)
    if not enabled:
        return None
    pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(pool, device=device):
        arena = torch.empty(sum(p.numel() for _,p in parameters), dtype=dtype, device=device)
    offset = 0
    # Rebind only storage; module and Parameter identities remain unchanged.
    # No copies or Parameter references escape in CPU metadata.
    for _,param in parameters:
        view = arena[offset:offset+param.numel()].view(param.shape)
        view.copy_(param.data)
        param.data = view
        offset += param.numel()
    row.update(pool_segment_bytes=sum(s["total_size"] for s in pool.snapshot()),
               shared_storage_count=len({p.untyped_storage().data_ptr() for _,p in parameters}))
    return pool, row, device


def finish(torch, ticket):
    if ticket is None:
        return
    pool, row, device = ticket
    torch.cuda.synchronize(device)
    active = sum(b["size"] for s in pool.snapshot() for b in s["blocks"]
                 if b["state"].startswith("active"))
    row["active_pool_bytes_after_layer_release"] = active
    if active:
        raise RuntimeError("replay parameter storage escaped CPU layer release")
    row["released"] = True
    # Caller deletes its ticket immediately; no live tensor may escape.


def inject(source, *, enabled):
    start = 'memory_profiler.mark(f"layer_{i}_activation_replay_and_cleanup", layer=i)'
    lines = source.splitlines(keepends=True)
    positions = [i for i,x in enumerate(lines) if x.strip()==start]
    if len(positions)!=1 or lines[positions[0]-1].strip()!='if memory_profiler is not None:':
        raise RuntimeError("replay marker changed")
    pos=positions[0]-1
    indent=lines[pos][:len(lines[pos])-len(lines[pos].lstrip())]
    if enabled and lines[pos-1].strip()!='del saliency_dict, gradients_dict':
        raise RuntimeError("requires early statistics deletion before compact placement")
    lines.insert(pos, indent+'_replay_compact_ticket = _replay_compact_prepare(layer, i)\n')
    anchor='layer_manager.release_layer(i, layer, update_master=True, orig_device=orig_device)'
    positions=[i for i,x in enumerate(lines) if x.strip()==anchor]
    if len(positions)!=1:
        raise RuntimeError("CPU layer-release boundary changed")
    pos=positions[0]+1
    indent=lines[pos-1][:len(lines[pos-1])-len(lines[pos-1].lstrip())]
    lines[pos:pos]=[indent+'_replay_compact_finish(_replay_compact_ticket)\n',
                    indent+'del _replay_compact_ticket\n']
    result=''.join(lines)
    compile(result,'replay_weight_compact_pool_source','exec')
    return result


def install_source_builder(impl, stats_module, *, enabled, records):
    original=stats_module.observed_source
    def build(source, *, enabled):
        return inject(original(source,enabled=enabled),enabled=enabled)
    # stats enabled and compact enabled must match for this bounded experiment.
    def prepare_layer(layer,index):
        return prepare(impl.torch,layer,index,enabled=enabled,records=records)
    impl.__dict__['_replay_compact_prepare']=prepare_layer
    impl.__dict__['_replay_compact_finish']=lambda ticket:finish(impl.torch,ticket)
    stats_module.observed_source=build
    return original


def build_entry(source, *, helper_sha):
    anchor='_replay_stats_source=_replay_stats_install(impl,enabled=_replay_stats_enabled,records=_replay_stats_records)'
    if source.count(anchor)!=1:
        raise RuntimeError("requires bound stats installer before forward wrapper")
    hook=f'''_compact_path=_HinvPath.cwd()/"tools/replay_weight_compact_pool_prototype.py"
_compact_sha={helper_sha!r}
if _hinv_hashlib.sha256(_compact_path.read_bytes()).hexdigest()!=_compact_sha:
    raise RuntimeError("compact pool helper changed")
from tools import activation_replay_stats_install as _compact_stats_module
from tools.replay_weight_compact_pool_prototype import install_source_builder as _compact_install
_compact_records=[]
_compact_original_builder=_compact_install(impl,_compact_stats_module,enabled=_replay_stats_enabled,records=_compact_records)
def _compact_save():
    (_HinvPath(_hinv_os.environ["RESEARCH_RUN_DIR"])/f"replay_weight_compact_{{_hinv_stage}}_rank{{_hinv_os.environ.get('RANK','0')}}.json").write_text(
        _hinv_json.dumps(dict(enabled=_replay_stats_enabled,records=_compact_records,
            helper_unchanged=_hinv_hashlib.sha256(_compact_path.read_bytes()).hexdigest()==_compact_sha),indent=2))
_hinv_atexit.register(_compact_save)
'''
    result=source.replace(anchor,hook+'\n'+anchor)
    compile(result,'replay_compact_bound_entry','exec')
    return result
