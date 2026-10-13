"""Cache admission, immutable copies, and honest cached comparison scope."""
from copy import deepcopy
import ast
import errno
from functools import lru_cache
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import storage_method_cache as cache
from tools import run_storage_method_formal as formal
from tools.compare_storage_cached_profiles import compare as compare_cached
from tools.compare_full_pipeline_profiles import compare as compare_fresh, GLOBAL_FIELDS


def expected_fixture(root, world=2):
    model = root/'Qwen3-0.6B'
    model.mkdir()
    (model/'config.json').write_text(json.dumps(dict(hidden_size=2, num_hidden_layers=28)))
    args = SimpleNamespace(world_size=world, models={'qwen06b': model}, nsamples=4, seq_len=2, seed=1,
                           lr06='3e-4', lr17='5e-4', bsz=8, backward_bsz=4, qa_eval=False)
    cmd = formal.command(args, 'qwen06b', root/'entry.py', 'stage0', False)
    return cache.contract(cmd), cmd


@lru_cache(maxsize=1)
def production_serialization_code():
    """Execute the real collector conversion and the real cache payload mapping."""
    tree = ast.parse((formal.ROOT/'gptq_utils/gptq_plus_utils.py').read_text(encoding='utf-8'))
    collector = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'collect_static_end_to_end_saliency_and_fisher')
    store = next(n for n in ast.walk(collector) if isinstance(n, ast.Expr)
                 and ast.unparse(n).startswith('static_fisher.append(fisher_mean.'))
    mapping = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '_to_save' for t in n.targets)
                   and isinstance(n.value, ast.Dict)
                   and 'fisher' in [k.value for k in n.value.keys if isinstance(k, ast.Constant)])
    return tuple(compile(ast.Module(body=[node], type_ignores=[]), 'production_cache_writer', 'exec')
                 for node in (store, mapping))


def payload(expected):
    shape = (expected['nsamples']//expected['world'], expected['seq_len'], expected['groups'])
    convert, mapping = production_serialization_code()
    namespace = dict(torch=torch, static_fisher=[], fisher_mean=torch.eye(expected['hidden'], dtype=torch.float32))
    for _ in range(expected['layers']):
        exec(convert, namespace)
    namespace.update(static_saliency_by_layer=[{name: torch.ones(shape) for name in cache.MODULES} for _ in range(expected['layers'])],
                     static_fisher_by_layer=namespace['static_fisher'], static_fisher_shrinkage_by_layer=None,
                     static_legacy_fisher_diag_by_layer=[None]*expected['layers'], static_refined_A_by_layer=None)
    exec(mapping, namespace)
    return namespace['_to_save']


def write_cache(root, expected, prepared=True):
    root.mkdir(parents=True)
    for rank in range(expected['world']):
        torch.save(payload(expected), root/('old_model_alias'+expected['suffix']+f'_rank{rank}.pt'))
    if prepared:
        checkpoint = root/'_prepared_checkpoints/old_alias_rotated'
        checkpoint.mkdir(parents=True)
        (checkpoint/'_SUCCESS').write_text('ok')
        (checkpoint/'config.json').write_text(json.dumps(dict(hidden_size=expected['hidden'], num_hidden_layers=28)))
        (checkpoint/'gptqplus_fsdp_meta_checkpoint.json').write_text(json.dumps(dict(kind='rotated', rotate=True,
             source_model='/old/path/alias', source_model_identity=expected['model_identity'], rotation_identity=expected['rotation_identity'],
             seq_len=expected['seq_len'])))
        from safetensors.torch import save_file
        save_file({f'model.layers.{layer}.{name}.weight': torch.ones(2,2) for layer in range(28) for name in cache.MODULES}, checkpoint/'model.safetensors')
        (checkpoint/'tokenizer_config.json').write_text('{}')
        (checkpoint/'tokenizer.json').write_text('{}')
    return root


def test_complete_old_subdirectory_admitted_and_copied_without_source_writes(tmp_path):
    expected, cmd = expected_fixture(tmp_path)
    source = write_cache(tmp_path/'cache/formal_realq_qwen3_0_6b_s2048_l2048_seed1', expected)
    found = list(cache.candidates([tmp_path/'cache', source], expected))
    assert len(found) == 1 and found[0]['prepared']
    admitted = cache.validate(found[0], expected, require_prepared=True)
    before = dict(admitted['sha256'])
    target = cache.snapshot(admitted, tmp_path/'arm/static_cache')
    copied = list(cache.candidates([target], expected))
    assert len(copied) == 1
    cache.validate(copied[0], expected, require_prepared=True)
    for rank in range(expected['world']):
        path = target/(expected['model_name']+expected['suffix']+f'_rank{rank}.pt')
        assert path.is_file() and path.stat().st_ino != found[0]['shards'][rank].stat().st_ino
    meta = cache.read(copied[0]['prepared']/'gptqplus_fsdp_meta_checkpoint.json')
    assert meta['source_model'] == expected['model']
    cache.verify_inventory(before)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('flag,value', [('--seed','2'), ('--nsamples','8'), ('--seq_len','4'),
                                       ('--rotation_seed','7'), ('--global_loss_bsz','16')])
def test_changed_conditions_cannot_hit_old_files(tmp_path, flag, value):
    expected, cmd = expected_fixture(tmp_path)
    source = write_cache(tmp_path/'cache/old', expected)
    changed = list(cmd)
    changed[changed.index(flag)+1] = value
    assert not list(cache.candidates([source], cache.contract(changed)))


def test_world_change_and_missing_rank_do_not_hit(tmp_path):
    expected, cmd = expected_fixture(tmp_path, world=4)
    source = write_cache(tmp_path/'cache/old', expected)
    changed = [x.replace('--nproc_per_node=4', '--nproc_per_node=2') for x in cmd]
    assert not list(cache.candidates([source], cache.contract(changed)))
    next(source.glob('*rank3.pt')).unlink()
    assert not list(cache.candidates([source], expected))


@pytest.mark.parametrize('fault', ['corrupt', 'missing_fisher', 'short_layers', 'missing_module', 'wrong_shape',
                                 'nonfinite', 'fisher_fp32', 'fisher_shape', 'fisher_missing_layer'])
def test_invalid_payload_rejected_and_original_retained(tmp_path, fault):
    expected, _ = expected_fixture(tmp_path)
    source = write_cache(tmp_path/'old', expected)
    path = next(source.glob('*rank1.pt'))
    data = payload(expected)
    if fault == 'corrupt': path.write_bytes(b'truncated archive')
    elif fault == 'missing_fisher': data.pop('fisher')
    elif fault == 'short_layers': data['saliency'].pop()
    elif fault == 'missing_module': data['saliency'][0].pop(cache.MODULES[0])
    elif fault == 'wrong_shape': data['saliency'][0][cache.MODULES[0]] = torch.ones(1)
    elif fault == 'nonfinite': data['fisher'][0][0,0] = float('nan')
    elif fault == 'fisher_fp32': data['fisher'][0] = data['fisher'][0].to(torch.float32)
    elif fault == 'fisher_shape': data['fisher'][0] = torch.ones(1, dtype=torch.bfloat16)
    elif fault == 'fisher_missing_layer': data['fisher'][0] = None
    if fault != 'corrupt': torch.save(data, path)
    before = cache.sha(path)
    with pytest.raises(Exception): cache.validate(next(cache.candidates([source], expected)), expected)
    assert cache.sha(path) == before


def test_real_production_world4_cache_admitted_and_reused_without_conversion(tmp_path):
    expected, _ = expected_fixture(tmp_path, world=4)
    source = write_cache(tmp_path/'outputs/full_pipeline_profile_existing_precompute/static_cache', expected)
    found = next(cache.candidates([tmp_path/'outputs'], expected))
    assert found['directory'] == source.resolve()
    produced = torch.load(found['shards'][0], weights_only=True, mmap=True)
    assert len(produced['fisher']) == 28
    assert all(matrix.dtype == torch.bfloat16 for matrix in produced['fisher'])
    # This is exactly the old validator's erroneous dtype gate: it rejects
    # output from the actual production conversion/save expressions above.
    assert produced['fisher'][0].dtype != torch.float32
    assert expected['fisher_dtype'] == 'torch.bfloat16'
    admitted = cache.validate(found, expected, require_prepared=True)
    copied = cache.snapshot(admitted, tmp_path/'new_arm/static_cache')
    for rank, original in found['shards'].items():
        target = copied/(expected['model_name']+expected['suffix']+f'_rank{rank}.pt')
        assert cache.sha(target) == cache.sha(original)
    cache.validate(next(cache.candidates([copied], expected)), expected, require_prepared=True)
    assert not torch.cuda.is_initialized()


def test_future_production_fisher_dtype_change_requires_schema_review():
    source = (formal.ROOT/'gptq_utils/gptq_plus_utils.py').read_text(encoding='utf-8')
    source = source.replace('static_fisher.append(fisher_mean.to(torch.bfloat16).cpu())',
                            'static_fisher.append(fisher_mean.to(torch.float32).cpu())')
    with pytest.raises(ValueError, match='storage expression changed'):
        cache.production_fisher_dtype(ast.parse(source))


def test_shared_views_store_one_payload_and_private_alias_metadata(tmp_path, monkeypatch):
    expected, _ = expected_fixture(tmp_path, world=4)
    source = write_cache(tmp_path/'shared-source', expected)
    admitted = cache.validate(next(cache.candidates([source], expected)), expected, require_prepared=True)
    before = dict(admitted['sha256'])
    def forbid_copy(*args, **kwargs):
        pytest.fail('Shared storage must never copy a large file')
    monkeypatch.setattr(cache.shutil, 'copy2', forbid_copy)
    monkeypatch.setattr(cache.shutil, 'copytree', forbid_copy)
    stored_inodes = {(Path(name).stat().st_dev, Path(name).stat().st_ino) for name in before}
    view_inodes = set()
    for side in ('control', 'candidate'):
        view = cache.shared_view(admitted, tmp_path/side/'static_cache')
        found = next(cache.candidates([view], expected))
        cache.validate(found, expected, require_prepared=True)
        for rank, original in admitted['shards'].items():
            assert os.path.samefile(found['shards'][int(rank)], original)
        model = found['prepared']/'model.safetensors'
        assert os.path.samefile(model, Path(admitted['prepared'])/'model.safetensors')
        # Actual file-backed tensor readers must keep in-memory quantization
        # updates from writing through shared filesystem references.
        from safetensors.torch import load_file
        tensors = load_file(str(model), device='cpu')
        next(iter(tensors.values())).add_(1)
        for mmap in (False, True):
            loaded = torch.load(found['shards'][0], map_location='cpu', weights_only=True, mmap=mmap)
            loaded['fisher'][0].add_(1)
        cache.verify_inventory(before)
        metadata = found['prepared']/'gptqplus_fsdp_meta_checkpoint.json'
        assert not os.path.samefile(metadata, Path(admitted['prepared'])/metadata.name)
        assert cache.read(metadata)['source_model'] == expected['model']
        view_inodes.update((p.stat().st_dev, p.stat().st_ino) for p in view.rglob('*')
                           if p.is_file() and p.name != metadata.name)
        report = cache.read(view.parent/'shared_cache_reference.json')
        assert report['additional_bulk_payload_bytes'] == 0
        assert report['private_metadata_bytes'] == metadata.stat().st_size
        assert all(r['kind'] == 'hardlink' for r in report['references'])
    # Across two complete four-rank arm views, every large payload is still
    # one physical inode. Source metadata is untouched, including its alias.
    assert view_inodes.issubset(stored_inodes)
    cache.verify_inventory(before)
    assert cache.read(Path(admitted['prepared'])/'gptqplus_fsdp_meta_checkpoint.json')['source_model'] == '/old/path/alias'
    budget = cache.preflight_shared_views(admitted, tmp_path, 20)
    assert budget['additional_bulk_payload_bytes'] == 0 and budget['view_count'] == 20


def test_shared_fallback_uses_symlinks_and_never_silently_copies(tmp_path, monkeypatch):
    expected, _ = expected_fixture(tmp_path)
    source = write_cache(tmp_path/'shared-source', expected)
    admitted = cache.validate(next(cache.candidates([source], expected)), expected, require_prepared=True)
    def no_hardlink(*args, **kwargs):
        raise OSError(errno.EXDEV, 'cross-device fixture')
    monkeypatch.setattr(cache.os, 'link', no_hardlink)
    view = cache.shared_view(admitted, tmp_path/'linked-arm/static_cache')
    found = next(cache.candidates([view], expected))
    cache.validate(found, expected, require_prepared=True)
    assert all(path.is_symlink() for path in found['shards'].values())
    assert (found['prepared']/'model.safetensors').is_symlink()
    def no_symlink(*args, **kwargs):
        raise OSError(errno.EPERM, 'unsupported filesystem fixture')
    monkeypatch.setattr(Path, 'symlink_to', no_symlink)
    with pytest.raises(RuntimeError, match='refusing a large copy'):
        cache.shared_view(admitted, tmp_path/'unsupported-arm/static_cache')
    cache.verify_inventory(admitted['sha256'])


def test_shared_source_mutation_is_rejected(tmp_path):
    expected, _ = expected_fixture(tmp_path)
    source = write_cache(tmp_path/'shared-source', expected)
    admitted = cache.validate(next(cache.candidates([source], expected)), expected, require_prepared=True)
    cache.shared_view(admitted, tmp_path/'arm/static_cache')
    path = Path(next(iter(admitted['shards'].values())))
    path.write_bytes(b'mutated fixture cache')
    with pytest.raises(RuntimeError, match='Cache source changed'):
        cache.verify_inventory(admitted['sha256'])


def test_prepared_model_reused_across_world_change_without_reusing_fisher(tmp_path):
    expected, command = expected_fixture(tmp_path, world=2)
    source = write_cache(tmp_path/'cache/old-world2', expected)
    world4 = cache.contract([arg.replace('--nproc_per_node=2', '--nproc_per_node=4') for arg in command])
    assert not list(cache.candidates([tmp_path/'cache'], world4))
    prepared = cache.find_prepared([tmp_path/'cache'], world4)
    assert prepared is not None and prepared['shards'] == {}
    view = cache.shared_view(prepared, tmp_path/'new-world4/static_cache')
    assert not list(view.glob('*.pt'))
    model = cache.checkpoint(view, world4)
    assert model and os.path.samefile(model/'model.safetensors', Path(prepared['prepared'])/'model.safetensors')
    assert cache.read(model/'gptqplus_fsdp_meta_checkpoint.json')['source_model'] == world4['model']
    changed = dict(world4, rotation_identity='different rotation')
    assert cache.find_prepared([tmp_path/'cache'], changed) is None


def test_actual_precompute_pipeline_shares_prepared_and_writes_only_new_fisher(tmp_path, monkeypatch):
    from tools import run_storage_method_pipeline as pipeline
    import sys
    expected, stage0_command = expected_fixture(tmp_path)
    source = write_cache(tmp_path/'old-cache', expected)
    prepared = cache.prepared_admission(source, expected)
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    for name in pipeline.TRACKED:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('CPU fixture source')
    (tmp_path/'entry.py').write_text(formal.build_entry())
    s0, s2 = tmp_path/'stage0.json', tmp_path/'stage2.json'
    s0.write_text(json.dumps(dict(command=stage0_command)))
    stage2 = [v for v in stage0_command if v not in ('--fsdp_precompute', '--exit_after_precompute')]
    s2.write_text(json.dumps(dict(command=stage2)))
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--stage0-manifest', str(s0), '--stage2-manifest', str(s2),
                                     '--tag', 'reuse-prepared-fixture', '--stage0-only', '--reuse-prepared', str(source)])
    def monitor(*args, **kwargs):
        kwargs['stdout'].write('2026/10/13 00:00:00.000, 100, 8000, 1, 0, GPU-fixture\n')
        kwargs['stdout'].flush()
        return SimpleNamespace(terminate=lambda: None, wait=lambda **_: 0)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', monitor)
    monkeypatch.setattr(pipeline, 'profile_rows', lambda *args, **kwargs: [])
    def stage(name, command, root, env, log, *args):
        assert name == 'stage0' and env['RESEARCH_IMPORTED_STATIC_CACHE'] == '0'
        assert env['RESEARCH_SHARED_PREPARED_CHECKPOINT'] == '1'
        out = Path(env['RESEARCH_RUN_DIR'])
        view = Path(command[command.index('--static_cache_path')+1])
        assert not list(view.glob('*.pt'))
        model = cache.checkpoint(view, expected)
        assert model and os.path.samefile(model/'model.safetensors', Path(prepared['prepared'])/'model.safetensors')
        for rank in range(expected['world']):
            torch.save(payload(expected), view/(expected['model_name']+expected['suffix']+f'_rank{rank}.pt'))
            (out/f'process_allocator_peak_stage0_rank{rank}.json').write_text(json.dumps(
                dict(rank=rank, stage='stage0', peak_allocated_bytes=100, peak_reserved_bytes=200,
                     scope='complete_process_through_evaluation')))
        log.write_text('CPU fixture: new Fisher generated from shared prepared model')
        return dict(returncode=0)
    monkeypatch.setattr(pipeline, 'run_stage', stage)
    pipeline.main()
    out = tmp_path/'outputs/full_pipeline_profile_reuse-prepared-fixture'
    assert cache.read(out/'status.json')['valid']
    cache.validate(next(cache.candidates([out/'static_cache'], expected)), expected, require_prepared=True)
    cache.verify_inventory(prepared['sha256'])


def test_shards_without_prepared_model_require_preparation(tmp_path):
    expected, _ = expected_fixture(tmp_path)
    source = write_cache(tmp_path/'old', expected, prepared=False)
    found = next(cache.candidates([source], expected))
    assert cache.validate(found, expected)['prepared'] is None
    with pytest.raises(ValueError, match='Prepared checkpoint'):
        cache.validate(found, expected, require_prepared=True)


def test_model_mutation_changes_identity_and_misses(tmp_path):
    expected, cmd = expected_fixture(tmp_path)
    source = write_cache(tmp_path/'old', expected)
    (Path(expected['model'])/'extra_model_metadata.json').write_text('{"changed":true}')
    assert not list(cache.candidates([source], cache.contract(cmd)))


def test_cached_comparison_requires_two_cached_arms_and_complete_quality(tmp_path):
    model = tmp_path/'model'
    model.mkdir()
    (model/'config.json').write_text('{"num_hidden_layers":28}')
    sides = [tmp_path/'control', tmp_path/'candidate']
    for side in sides:
        side.mkdir()
        command = ['python', str(side/'entry.py'), '--model', str(model), '--exp', side.name, '--quant_stop_layer', '27']
        manifest = dict(stage0_command=command+['--fsdp_precompute'], stage2_command=command,
                        source_sha256_at_launch={str(side/'entry.py'):'same-source'})
        (side/'manifest.json').write_text(json.dumps(manifest))
        (side/'nvml_telemetry.csv').write_text('2026/10/13 00:00:00.000, 100, 8000, 1\n')
        status = dict(valid=True, scope='stage2_cached', returncode=0, sources_unchanged=True,
                      stage0=dict(returncode=None, skipped='validated_cache_reuse', measured=False), stage2=dict(returncode=0),
                      metrics=dict(state_sha256='same',state_tensors=507), kl=.125, ppl=20., memory_profile_files=[])
        status.update({key:100 for key in GLOBAL_FIELDS if not key.startswith('stage0_')})
        status.update({key:100 for key in ('stage2_process_peak_allocated_bytes','stage2_process_peak_reserved_bytes','full_process_peak_allocated_bytes','full_process_peak_reserved_bytes')})
        (side/'status.json').write_text(json.dumps(status))
        (side/'layer_output_fingerprints.json').write_text(json.dumps([dict(layer=i,sha256='a'*64,forward_count=1,shapes=[[1,2]]) for i in range(28)]))
        (side/'static_cache').mkdir()
        (side/'static_cache/fisher.pt').write_bytes(b'identical complete fixture inventory')
    result = compare_cached(*sides)
    assert result['full_model_quantization_validated'], result['exactness_errors']
    assert result['scope'] == 'stage2_cached' and result['precompute_measured'] is False
    assert not compare_fresh(*sides)['full_model_quantization_validated']
    status = cache.read(sides[1]/'status.json')
    status['scope'] = 'full_pipeline'
    (sides[1]/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError, match='both arms'):
        compare_cached(*sides)
