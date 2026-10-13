"""Discover and validate matching REAL-Q caches without changing old artifacts.

No torch import during planning. Runtime validation loads tensor headers on CPU
with mmap and streams SHA256; it never holds all rank payloads in RAM at once.
"""
from __future__ import annotations

import hashlib
import ast
import json
import os
import errno
from pathlib import Path
import shutil

from utils.cache_identity import artifact_identity

MODULES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
           'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def production_fisher_dtype(source):
    """Bind admission to the collector's persisted dtype, not accumulation dtype."""
    collector = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'collect_static_end_to_end_saliency_and_fisher')
    stores = [node for node in ast.walk(collector)
              if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and node.value.func.value.id == 'static_fisher'
              and node.value.func.attr == 'append'
              and not (len(node.value.args) == 1 and isinstance(node.value.args[0], ast.Constant)
                       and node.value.args[0].value is None)]
    expected = ast.parse('static_fisher.append(fisher_mean.to(torch.bfloat16).cpu())').body[0]
    if len(stores) != 1 or ast.dump(stores[0]) != ast.dump(expected):
        raise ValueError('Production Fisher storage expression changed; review cache admission schema')
    return 'torch.bfloat16'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def contract(command):
    def value(flag, default=None):
        return command[command.index(flag)+1] if flag in command else default
    model = Path(value('--model')).resolve()
    world = next((int(x.split('=', 1)[1]) for x in command if x.startswith('--nproc_per_node=')), None)
    if world is None:
        world = int(value('--nproc_per_node', '1'))
    config = read(model/'config.json')
    def option(flag, default=None):
        return value(flag, default)
    # The filename has optional tags for features outside the supported formal
    # preset.  Refuse the cache path if one is enabled rather than accidentally
    # matching a legacy key whose payload has different contents.
    unsupported = {
        '--fisher_rademacher_k': (None, '0'),
        '--num_samples_for_grad': (None, '0'),
        '--enable_dynamic_saliency': (None, '0'),
        '--ffn_adaptive_fisher_shrinkage': (None,),
        '--ffn_adaptive_fisher_offdiag_shrinkage': (None,),
        '--attention_sink_size': (None, '0'),
        '--refined_rkl_num_A': (None, '1'),
    }
    for flag, allowed in unsupported.items():
        if option(flag) not in allowed:
            raise ValueError(f'Auto cache does not support {flag}={option(flag)!r}')
    result = dict(model=str(model), model_name=model.name, model_identity=artifact_identity(model),
                  hidden=int(config['hidden_size']), layers=int(config['num_hidden_layers']),
                  dataset=value('--dataset'), nsamples=int(value('--nsamples')), seq_len=int(value('--seq_len')),
                  seed=int(value('--seed', '0')), world=world, groups=int(value('--num_groups')),
                  global_bsz=int(value('--global_loss_bsz')), topk=int(value('--grad_hessian_topk', '-1')),
                  rotate='--rotate' in command, rotation_seed=int(value('--rotation_seed', '0')),
                  salclip=float(value('--saliency_clip_percentile', '.99')),
                  label_chunk=int(value('--static_fisher_label_chunk_tokens', '0')))
    # This entry is deliberately limited to the formal mainFisher preset.
    if value('--grad_refresh_loss') != 'fisher_diag_mse' or '--optimized_rotation_path' in command:
        raise ValueError('Auto cache currently supports the formal generated-rotation mainFisher preset only')
    if '--ignore_attention_sink' in command:
        raise ValueError('Auto cache does not support attention-sink filtering')
    # Bind the two content-schema constants to the production implementation.
    # This keeps a future schema/scale change from silently admitting old files.
    source = ast.parse((Path(__file__).resolve().parents[1]/'gptq_utils/gptq_plus_utils.py').read_text(encoding='utf-8'))
    constants = {}
    for node in source.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {'_STATIC_SALIENCY_SCHEMA_TAG', '_E2E_PRECOMPUTE_LOSS_GRAD_SCALE'}:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    if constants != {'_STATIC_SALIENCY_SCHEMA_TAG': 'salsumv2', '_E2E_PRECOMPUTE_LOSS_GRAD_SCALE': 1000.0}:
        raise ValueError(f'Production static-cache schema changed: {constants!r}')
    result['fisher_dtype'] = production_fisher_dtype(source)
    rotation = f"generated_hadamard:rotation_seed={result['rotation_seed']}" if result['rotate'] else 'disabled'
    result['rotation_identity'] = rotation
    hidden, intermediate = result['hidden'], int(config.get('intermediate_size', result['hidden']))
    heads = int(config.get('num_attention_heads', 1))
    head_dim = int(config.get('head_dim', hidden//heads))
    kv = int(config.get('num_key_value_heads', heads))*head_dim
    result['weight_shapes'] = {'self_attn.q_proj': [heads*head_dim, hidden], 'self_attn.k_proj': [kv, hidden],
                               'self_attn.v_proj': [kv, hidden], 'self_attn.o_proj': [hidden, heads*head_dim],
                               'mlp.gate_proj': [intermediate, hidden], 'mlp.up_proj': [intermediate, hidden],
                               'mlp.down_proj': [hidden, intermediate]}
    result['rotation_tag'] = hashlib.sha1(rotation.encode()).hexdigest()[:12]
    result['schema'] = 'salsumv2'
    salclip = f"{result['salclip']:.4f}".rstrip('0').rstrip('.')
    result['suffix'] = (f"_mid{result['model_identity'][:12]}_{result['dataset']}_s{result['nsamples']}"
                        f"_blk{result['seq_len']}_rot{int(result['rotate'])}_rotid{result['rotation_tag']}"
                        f"_g{result['groups']}_fisherfull_ghtk{result['topk']}_glbsz{result['global_bsz']}"
                        f"_cseed{result['seed']}_salclip{salclip}_salglobalv1_salsumv2_rklNA1_fpfinal0"
                        f"_e2els1000_mainFisher" + (f"_labelchunk{result['label_chunk']}" if result['label_chunk'] else '')
                        + f"_world{world}")
    return result


def checkpoint(directory, expected):
    """Require a complete compatible prepared model, not just Fisher shards."""
    directory = Path(directory)
    paths = list((directory/'_prepared_checkpoints').glob('*/gptqplus_fsdp_meta_checkpoint.json'))
    for path in sorted(paths):
        try:
            meta, config = read(path), read(path.parent/'config.json')
            if (meta.get('kind') != 'rotated' or not meta.get('rotate')
                    or meta.get('source_model_identity') != expected['model_identity']
                    or meta.get('rotation_identity') != expected['rotation_identity']
                    or meta.get('seq_len') != expected['seq_len']
                    or not (path.parent/'_SUCCESS').is_file()
                    or config.get('hidden_size') != expected['hidden']
                    or config.get('num_hidden_layers') != expected['layers']):
                continue
            index = path.parent/'model.safetensors.index.json'
            weight_map = read(index).get('weight_map', {}) if index.exists() else None
            names = set(weight_map.values()) if weight_map is not None else {'model.safetensors'}
            if not names or any(Path(n).name != n or not (path.parent/n).is_file() or (path.parent/n).stat().st_size == 0 for n in names):
                continue
            if not (path.parent/'tokenizer_config.json').is_file() or not any((path.parent/n).is_file() for n in ('tokenizer.json', 'tokenizer.model', 'vocab.json')):
                continue
            from safetensors import safe_open
            keys = set()
            shapes = {}
            for name in names:
                with safe_open(str(path.parent/name), framework='pt', device='cpu') as tensors:
                    for key in tensors.keys():
                        if key in keys:
                            raise ValueError('Duplicated prepared weight key')
                        keys.add(key)
                        view = tensors.get_slice(key)
                        shapes[key] = view.get_shape()
                        if not view.get_shape() or any(dim <= 0 for dim in view.get_shape()):
                            raise ValueError('Empty prepared weight')
            if weight_map is not None and set(weight_map) != keys:
                continue
            required = {f'model.layers.{layer}.{module}.weight' for layer in range(expected['layers']) for module in MODULES}
            if not required.issubset(keys):
                continue
            if any(shapes[f'model.layers.{layer}.{module}.weight'] != expected['weight_shapes'][module]
                   for layer in range(expected['layers']) for module in MODULES):
                continue
            return path.parent
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            continue
    return None


def cache_directories(roots):
    seen = set()
    for root in roots:
        root = Path(root).resolve()
        directories = [root, *sorted(p for p in root.glob('*') if p.is_dir()),
                       *sorted(root.glob('full_pipeline_profile_*/static_cache'))]
        for directory in directories:
            if directory in seen:
                continue
            seen.add(directory)
            yield directory


def prepared_admission(directory, expected):
    prepared = checkpoint(directory, expected)
    if prepared is None:
        raise ValueError('Prepared checkpoint missing/incomplete or incompatible')
    inventory = {str(path): sha(path) for path in sorted(prepared.rglob('*')) if path.is_file()}
    return dict(directory=str(Path(directory).resolve()), shards={}, prepared=str(prepared),
                sha256=inventory, contract=expected)


def find_prepared(roots, expected):
    """Prepared weights depend on model/rotation, not Fisher's rank layout."""
    for directory in cache_directories(roots):
        if checkpoint(directory, expected) is not None:
            return prepared_admission(directory, expected)
    return None


def candidates(roots, expected):
    for directory in cache_directories(roots):
        files = {}
        for rank in range(expected['world']):
            suffix = expected['suffix']+f'_rank{rank}.pt'
            matches = sorted(p for p in directory.glob('*.pt') if p.name.endswith(suffix))
            if not matches:
                break
            canonical = directory/(expected['model_name']+suffix)
            files[rank] = canonical if canonical in matches else matches[0]
        if len(files) == expected['world']:
            yield dict(directory=directory, shards=files, prepared=checkpoint(directory, expected))


def validate_payload(path, expected):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError('Cache payload must be a dict')
    saliency, fisher = payload.get('saliency'), payload.get('fisher')
    if not isinstance(saliency, (list, tuple)) or not isinstance(fisher, (list, tuple)) or len(saliency) != expected['layers'] or len(fisher) != expected['layers']:
        raise ValueError('Incomplete saliency/Fisher layer count')
    for index, (sal, matrix) in enumerate(zip(saliency, fisher)):
        if not isinstance(sal, dict) or set(sal) != set(MODULES):
            raise ValueError(f'Incomplete saliency modules at layer{index}')
        shape = (expected['nsamples']//expected['world'], expected['seq_len'], expected['groups'])
        if any(not isinstance(t, torch.Tensor) or t.is_meta or tuple(t.shape) != shape or not t.is_floating_point() for t in sal.values()):
            raise ValueError(f'Invalid saliency shape/dtype at layer{index}')
        fisher_shape = (expected['hidden'], expected['hidden'])
        fisher_dtype = expected['fisher_dtype']
        if (not isinstance(matrix, torch.Tensor) or matrix.is_meta
                or tuple(matrix.shape) != fisher_shape or str(matrix.dtype) != fisher_dtype):
            actual = (f'shape={tuple(matrix.shape)}, dtype={matrix.dtype}, meta={matrix.is_meta}'
                      if isinstance(matrix, torch.Tensor) else f'type={type(matrix).__name__}')
            raise ValueError(f'Invalid full Fisher at layer{index}: expected shape={fisher_shape}, '
                             f'dtype={fisher_dtype}; actual {actual}')
        for tensor in [*sal.values(), matrix]:
            flattened = tensor.reshape(-1)
            for start in range(0, flattened.numel(), 1024**2):
                if not torch.isfinite(flattened[start:start+1024**2]).all().item():
                    raise ValueError(f'Nonfinite cache tensor at layer{index}')
    del payload, saliency, fisher


def validate(candidate, expected, *, require_prepared=False):
    if require_prepared and not candidate['prepared']:
        raise ValueError('Prepared checkpoint missing/incomplete')
    inventory = {}
    for rank, path in candidate['shards'].items():
        print(f'[cache] CPU validating rank{rank}: {path}', flush=True)
        validate_payload(path, expected)
        inventory[str(path)] = sha(path)
    if candidate['prepared']:
        for path in sorted(candidate['prepared'].rglob('*')):
            if path.is_file():
                inventory[str(path)] = sha(path)
    return dict(directory=str(candidate['directory']), shards={str(k): str(v) for k,v in candidate['shards'].items()},
                prepared=str(candidate['prepared']) if candidate['prepared'] else None,
                sha256=inventory, contract=expected)


def verify_inventory(inventory):
    changed = [name for name, digest in inventory.items() if not Path(name).is_file() or sha(name) != digest]
    if changed:
        raise RuntimeError('Cache source changed: '+repr(changed))


def preflight_snapshots(admitted, destination_root, count, reserve_bytes=1024**3):
    """Check room for every remaining per-arm copy before launching Stage2."""
    if count <= 0:
        return dict(snapshot_count=0, bytes_per_snapshot=0, required_free_bytes=reserve_bytes,
                    free_bytes=shutil.disk_usage(destination_root).free, sufficient=True)
    bytes_per_snapshot = sum(Path(name).stat().st_size for name in admitted['sha256'])
    required = bytes_per_snapshot*count + reserve_bytes
    free = shutil.disk_usage(destination_root).free
    result = dict(snapshot_count=count, bytes_per_snapshot=bytes_per_snapshot,
                  required_free_bytes=required, free_bytes=free, sufficient=free >= required)
    if not result['sufficient']:
        raise RuntimeError('Insufficient disk for all remaining independent cache copies: '+repr(result))
    return result


def preflight_shared_views(admitted, destination_root, count, reserve_bytes=1024**3):
    """Reserve only small directory/metadata views, never another payload copy."""
    # File/directory entry overhead varies by filesystem. Budget 8KiB per
    # referenced file plus one private metadata record per arm, and headroom.
    per_view = (len(admitted['sha256']) + 4)*8192
    required = count*per_view + reserve_bytes
    free = shutil.disk_usage(destination_root).free
    result = dict(storage='shared', view_count=count,
                  shared_source_bytes=sum(Path(p).stat().st_size for p in admitted['sha256']),
                  additional_bulk_payload_bytes=0, estimated_bytes_per_view=per_view,
                  required_free_bytes=required, free_bytes=free, sufficient=free >= required)
    if not result['sufficient']:
        raise RuntimeError('Insufficient disk for small shared cache views: '+repr(result))
    return result


def _reference_file(source, target):
    """Keep the same stored bytes; unsupported sharing must never become a copy."""
    source = Path(source).resolve(strict=True)
    try:
        os.link(source, target)
        return 'hardlink'
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS,
                             errno.EOPNOTSUPP}:
            raise
    try:
        Path(target).symlink_to(source)
    except OSError as exc:
        raise RuntimeError('Shared cache requires hardlinks or symlinks; refusing a large copy') from exc
    return 'symlink'


def shared_view(admitted, destination):
    """Reference large immutable inputs; keep only alias metadata private.

    Readers load tensors into memory, never write through these links. The
    generated Stage2 entry forbids precompute/checkpoint writers and the runner
    checks source bytes before/after use. Hardlinks also keep bytes accessible
    when an obsolete source directory is later removed.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    verify_inventory(admitted['sha256'])
    expected = admitted['contract']
    records = []
    for rank, name in admitted['shards'].items():
        target = destination/(expected['model_name']+expected['suffix']+f'_rank{rank}.pt')
        records.append(dict(source=str(Path(name).resolve()), target=str(target),
                            kind=_reference_file(name, target), bytes=Path(name).stat().st_size))
    private_bytes = 0
    if admitted['prepared']:
        prepared = Path(admitted['prepared'])
        target_root = destination/'_prepared_checkpoints'/(expected['model_name']+'_rot_'+expected['rotation_tag'])
        for name in admitted['sha256']:
            path = Path(name)
            try:
                relative = path.relative_to(prepared)
            except ValueError:
                continue
            target = target_root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative.as_posix() == 'gptqplus_fsdp_meta_checkpoint.json':
                # The production readiness check requires the current model
                # path alias. Never mutate shared metadata to normalize it.
                meta = read(path)
                meta['source_model'] = expected['model']
                target.write_text(json.dumps(meta, indent=2), encoding='utf-8')
                private_bytes += target.stat().st_size
            else:
                records.append(dict(source=str(path.resolve()), target=str(target),
                                    kind=_reference_file(path, target), bytes=path.stat().st_size))
    verify_inventory(admitted['sha256'])
    report = dict(storage='shared', additional_bulk_payload_bytes=0,
                  private_metadata_bytes=private_bytes, references=records,
                  read_only_consumption=True, no_copy_fallback=True)
    with (destination.parent/'shared_cache_reference.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    return destination


def snapshot(admitted, destination):
    """Independent file copies; old cache/checkpoint bytes are never writable aliases."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    needed = sum(Path(name).stat().st_size for name in admitted['sha256'])
    if shutil.disk_usage(destination).free < needed + 1024**3:
        raise RuntimeError(f'Not enough disk for independent cache snapshot ({needed} bytes + 1GiB headroom)')
    verify_inventory(admitted['sha256'])
    expected = admitted['contract']
    for rank, name in admitted['shards'].items():
        target = destination/(expected['model_name']+expected['suffix']+f'_rank{rank}.pt')
        shutil.copy2(name, target)
        if sha(target) != admitted['sha256'][name]:
            raise RuntimeError('Copied shard digest mismatch')
    if admitted['prepared']:
        target = destination/'_prepared_checkpoints'/(expected['model_name']+'_rot_'+expected['rotation_tag'])
        shutil.copytree(admitted['prepared'], target)
        for name, digest in admitted['sha256'].items():
            path = Path(name)
            try:
                relative = path.relative_to(Path(admitted['prepared']))
            except ValueError:
                continue
            if sha(target/relative) != digest:
                raise RuntimeError('Copied prepared checkpoint digest mismatch')
        # Equivalent model-path aliases may differ from the current canonical
        # args.model. Adjust only this private metadata copy after identity match.
        meta_path = target/'gptqplus_fsdp_meta_checkpoint.json'
        meta = read(meta_path)
        meta['source_model'] = expected['model']
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    verify_inventory(admitted['sha256'])
    return destination
