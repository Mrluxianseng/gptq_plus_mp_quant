"""Cache admission, immutable copies, and honest cached comparison scope."""
from copy import deepcopy
import json
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


def payload(expected):
    shape = (expected['nsamples']//expected['world'], expected['seq_len'], expected['groups'])
    return dict(saliency=[{name: torch.ones(shape) for name in cache.MODULES} for _ in range(expected['layers'])],
                fisher=[torch.eye(expected['hidden'], dtype=torch.float32) for _ in range(expected['layers'])])


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


@pytest.mark.parametrize('fault', ['corrupt', 'missing_fisher', 'short_layers', 'missing_module', 'wrong_shape', 'nonfinite', 'fisher_bf16'])
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
    elif fault == 'fisher_bf16': data['fisher'][0] = data['fisher'][0].to(torch.bfloat16)
    if fault != 'corrupt': torch.save(data, path)
    before = cache.sha(path)
    with pytest.raises(Exception): cache.validate(next(cache.candidates([source], expected)), expected)
    assert cache.sha(path) == before


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
