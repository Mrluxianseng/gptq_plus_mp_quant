"""CPU tests for formal workload wiring and fail-closed method coverage."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools import run_storage_method_formal as runner


def settings():
    return SimpleNamespace(world_size=4, models={'qwen06b': Path('/models/Qwen3-0.6B'),
                           'qwen17b': Path('/models/Qwen3-1.7B')}, nsamples=2048,
                           seq_len=2048, seed=1, lr06='3e-4', lr17='5e-4',
                           bsz=8, backward_bsz=4, qa_eval=True)


def test_formal_commands_preserve_math_and_separate_stages():
    from tools.run_replay_stats_memory_campaign import normalized_command
    for model, lr in runner.MODEL_LRS.items():
        for stage in ('stage0', 'stage2'):
            c = runner.command(settings(), model, Path('/run/entry.py'), stage, False)
            n = runner.command(settings(), model, Path('/run/entry.py'), stage, True)
            assert normalized_command(c) == normalized_command(n)
            for flag, value in {'--nsamples': '2048', '--seq_len': '2048', '--seed': '1',
                                '--grad_lr': lr, '--final_layer_grad_lr': '1e-5',
                                '--group_parallel_quant': 'none', '--eval_samples': '0',
                                '--static_fisher_label_chunk_tokens': '0'}.items():
                assert c[c.index(flag)+1] == value
            assert ('--fsdp_precompute' in c) == (stage == 'stage0')
            assert ('--stage2_cpu_master' in c) == (stage == 'stage2')
            assert '--lm_eval' in c if stage == 'stage2' else '--skip_eval' in c
            assert all(flag in n and flag not in c for flag in runner.METHOD_FLAGS)
            for flag, value in (('--nsamples','256'),('--grad_lr','1e-3'),('--seed','44')):
                altered = list(n)
                altered[altered.index(flag)+1] = value
                assert normalized_command(c) != normalized_command(altered)


def test_entry_real_source_builds_all_methods_without_output_dependencies():
    source = runner.build_entry()
    tree = ast.parse(source)
    assert source.count('original_fwrd = impl.gptq_fwrd') == 1
    assert '_dead_install_builder' in source and '_replay_stats_install(impl' in source
    assert 'outputs/replay-stats' not in source and '/mnt/d/' not in source
    assert "impl.collect_static_end_to_end_saliency_and_fisher=_forbid_precompute" in source
    # Compile the actual transformed fasterquant and forward against live source.
    from gptq_utils import gptq_plus_utils as impl
    from tools import hessian_stable_weight_prototype as stable
    from tools.hessian_dead_subgroup_input_prototype import build_source
    from tools.activation_replay_stats_install import observed_source
    import inspect
    import textwrap
    _, quant = stable.build_stable_fasterquant(impl)
    dead = build_source(quant)
    compile(dead, 'formal_dead_source', 'exec')
    compile(observed_source(textwrap.dedent(inspect.getsource(impl.gptq_fwrd)), enabled=True), 'formal_stats_source', 'exec')
    assert sum(isinstance(n, ast.FunctionDef) and n.name == '_hinv_init' for n in ast.walk(tree)) == 1


def metadata(enabled):
    inv = dict(solver_workspace=enabled, reuse=enabled, stable_weight=enabled, isolated_pool=False,
               helper_unchanged=True, stable_helper_unchanged=True,
               calls=[dict(layer=l, enabled=enabled, identity_fallback=False,
                           dtype='torch.float32',workspace_reuse_enabled=enabled) for l in range(28) for _ in range(4)],
               weight_write_calls=[dict(layer=l, enabled=enabled, pointer_preserved=True) for l in range(28)],
               reuse_calls=[dict(layer=l, factor_shares_consumed_h_storage=True,identity_fallback=False) for l in range(28) for _ in range(4)] if enabled else [])
    dead = dict(enabled=enabled, helpers_unchanged=True,records=[dict(layer=l,retained_factors=4,released_private_input_bytes=36*1024**2) for l in range(28)] if enabled else [])
    stats = dict(enabled=enabled,helpers_unchanged=True,records=[dict(layer=l,enabled=enabled,observed_cuda_storage_bytes=12*1024**2) for l in range(28)])
    return inv, dead, stats


def write_metadata(root, rank, records):
    for name, record in zip(('inverse_lifetime','dead_subgroup_input','replay_stats_lifetime'), records):
        (root/f'{name}_stage2_rank{rank}.json').write_text(json.dumps(record))


def test_all_ranks_coverage_required_and_disabled_workspace_rejected(tmp_path):
    for rank in range(4):
        write_metadata(tmp_path, rank, metadata(True))
    assert runner.method_coverage(tmp_path, 4, True)['valid']
    changed = deepcopy(metadata(True))
    changed[0]['calls'][0]['workspace_reuse_enabled'] = False
    write_metadata(tmp_path, 3, changed)
    assert not runner.method_coverage(tmp_path, 4, True)['valid']
    write_metadata(tmp_path, 3, metadata(True))
    (tmp_path/'dead_subgroup_input_stage2_rank3.json').unlink()
    with pytest.raises(FileNotFoundError):
        runner.method_coverage(tmp_path, 4, True)


def test_control_method_must_really_be_disabled(tmp_path):
    write_metadata(tmp_path, 0, metadata(False))
    assert runner.method_coverage(tmp_path, 1, False)['valid']
    write_metadata(tmp_path, 0, metadata(True))
    assert not runner.method_coverage(tmp_path, 1, False)['valid']


def test_commands_accepted_by_actual_production_parser(monkeypatch):
    import process_args
    for model in runner.MODEL_LRS:
        for stage in ('stage0', 'stage2'):
            cmd = runner.command(settings(), model, Path('/run/entry.py'), stage, True)
            cli = cmd[cmd.index('baseline-profile')+1:]
            cli = [value for value in cli if value not in (*runner.METHOD_FLAGS, '--research_stage0_checkpoint')]
            monkeypatch.setattr('sys.argv', ['ptq.py', *cli])
            monkeypatch.setattr(process_args, 'init_logging', lambda *args, **kwargs: None)
            args = process_args.parse_gen()
            assert args.nsamples == 2048 and args.seq_len == 2048
            assert args.grad_lr == float(runner.MODEL_LRS[model])
            assert args.grad_optimizer == 'adam' and args.group_parallel_quant == 'none'


def test_dry_run_defaults_to_auto_cache_and_labels_scope(tmp_path, monkeypatch):
    models = []
    for name, hidden, intermediate in (('Qwen3-0.6B', 1024, 3072), ('Qwen3-1.7B', 2048, 6144)):
        model = tmp_path/name
        model.mkdir()
        (model/'config.json').write_text(json.dumps(dict(hidden_size=hidden, intermediate_size=intermediate,
                                                         num_hidden_layers=28)))
        models.append(model)
    output = tmp_path/'plan'
    monkeypatch.setattr('sys.argv', ['run_storage_method_formal.py', '--dry-run',
        '--model06', str(models[0]), '--model17', str(models[1]), '--devices', '0,1,2,3',
        '--output-directory', str(output)])
    runner.main()
    plan = json.loads((output/'plan.json').read_text())
    assert plan['cache_mode'] == 'auto'
    assert plan['cache_storage'] == 'shared'
    assert all(job['launcher'][-2:] == ['--cache-storage', 'shared'] for job in plan['jobs'])
    assert plan['measurement_scope'] == 'stage2_cached'
    assert set(plan['optional_precompute_jobs']) == {'qwen06b', 'qwen17b'}
    assert Path(plan['cache_search_roots'][0]) == runner.ROOT/'cache'
    assert plan['gpu_work_started'] is False

    output_fresh = tmp_path/'fresh-plan'
    monkeypatch.setattr('sys.argv', ['run_storage_method_formal.py', '--dry-run',
        '--model06', str(models[0]), '--model17', str(models[1]), '--devices', '0,1,2,3',
        '--output-directory', str(output_fresh), '--cache-mode', 'fresh'])
    runner.main()
    fresh = json.loads((output_fresh/'plan.json').read_text())
    assert fresh['measurement_scope'] == 'full_pipeline'
    assert fresh['optional_precompute_jobs'] == {}


def test_formal_source_binding_excludes_unrelated_tools_and_includes_dependencies():
    paths = {path.relative_to(runner.ROOT).as_posix() for path in runner.bound_source_paths()}
    assert 'tools/find_large_files.py' not in paths
    assert 'tools/run_storage_method_pipeline.py' in paths
    assert 'tools/realq_checkpoint_research_entry.py' in paths
    assert 'tools/hessian_solver_workspace_prototype.py' in paths
    assert 'tools/activation_replay_stats_lifetime_prototype.py' in paths
    assert 'tools/storage_method_cache.py' in paths
    assert not any(name.startswith('tools/') and not (runner.ROOT/name).is_file() for name in paths)


@pytest.mark.parametrize('stage0', [False, True])
@pytest.mark.parametrize('prepared_shared', [False, True])
def test_shared_stage2_forbids_prepared_checkpoint_writer(monkeypatch, stage0, prepared_shared):
    from utils import model_utils
    writer = lambda *args, **kwargs: 'prepared-writing-fixture'
    monkeypatch.setattr(model_utils, '_save_prepared_checkpoint', writer)
    monkeypatch.setenv('RESEARCH_SHARED_STATIC_CACHE', '1')
    monkeypatch.setenv('RESEARCH_SHARED_PREPARED_CHECKPOINT', '1' if prepared_shared else '0')
    monkeypatch.setattr('sys.argv', ['entry.py', *(['--exit_after_precompute'] if stage0 else [])])
    tree = ast.parse(runner.build_entry())
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and 'RESEARCH_SHARED_STATIC_CACHE' in ast.unparse(node.test))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_forbid_precompute')
    import os
    import sys
    exec(compile(ast.Module(body=[definition, guard], type_ignores=[]), 'shared_checkpoint_guard', 'exec'),
         dict(os=os, sys=sys))
    if stage0 and not prepared_shared:
        assert model_utils._save_prepared_checkpoint() == 'prepared-writing-fixture'
    else:
        with pytest.raises(RuntimeError, match='consume its verified cache'):
            model_utils._save_prepared_checkpoint()


@pytest.mark.parametrize('hit', [False, True])
@pytest.mark.parametrize('storage', ['shared', 'copy'])
@pytest.mark.parametrize('qa_error', [False, True])
def test_actual_run_auto_cache_preflight_and_all_model_pairs(tmp_path, monkeypatch, hit, storage, qa_error):
    """Run the real outer --run loop with CPU fixture pipeline processes.

    Cache/schema, state/method and actual worker ownership have separate tests.
    This regression specifically executes controller scheduling after a cache
    hit or successful precompute, not just its --dry-run planning branch.
    """
    source = runner.build_entry()
    root = tmp_path/'repo'
    root.mkdir()
    marker = root/'controller-source.py'
    marker.write_text('unchanged CPU fixture source')
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner, 'build_entry', lambda: source)
    monkeypatch.setattr(runner, 'bound_source_paths', lambda: [marker])
    models = []
    directories, available = {}, {}
    for name, hidden, intermediate in (('qwen06b', 1024, 3072), ('qwen17b', 2048, 6144)):
        model = root/name
        model.mkdir()
        (model/'config.json').write_text(json.dumps(dict(hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=28)))
        models.append(model)
        directory = root/'cache'/name
        directory.mkdir(parents=True)
        (directory/'rank0.pt').write_bytes(b'CPU fixture cache')
        prepared = directory/'_prepared_checkpoints'/name
        prepared.mkdir(parents=True)
        (prepared/'model.safetensors').write_bytes(b'CPU fixture weights')
        directories[name], available[name] = directory, hit
    output = root/'outputs/actual-run'
    monkeypatch.setattr('sys.argv', ['formal', '--run', '--model06', str(models[0]), '--model17', str(models[1]),
                                     '--devices', '0,1,2,3', '--output-directory', str(output),
                                     '--repeats', '3', '--cache-storage', storage])
    monkeypatch.setattr(runner.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(runner.cache_manager, 'contract', lambda command: {'model_name': Path(command[command.index('--model')+1]).name})
    def candidates(roots, expected):
        name = expected['model_name']
        if not available[name]:
            return []
        directory = Path(roots[0]) if len(roots) == 1 else directories[name]
        return [dict(directory=directory.resolve(), prepared=directory/'_prepared_checkpoints'/name, shards={0: directory/'rank0.pt'})]
    monkeypatch.setattr(runner.cache_manager, 'candidates', candidates)
    def validate(found, expected, **kwargs):
        return dict(directory=str(found['directory']), prepared=str(found['prepared']), shards={str(k): str(v) for k,v in found['shards'].items()},
                    sha256={str(p): runner.sha(p) for p in found['directory'].rglob('*') if p.is_file()}, contract=expected)
    monkeypatch.setattr(runner.cache_manager, 'validate', validate)
    monkeypatch.setattr(runner.cache_manager, 'find_prepared', lambda *_: None)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: '')
    def qa_preflight(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['env']['HF_DATASETS_OFFLINE'] == '1'
        assert command[2].endswith('storage_method_qa_preflight.py')
        if qa_error:
            raise runner.subprocess.CalledProcessError(1, command)
        Path(command[command.index('--output')+1]).write_text(json.dumps({'valid': True, 'fixture': 'CPU scheduling only'}))
    monkeypatch.setattr(runner.subprocess, 'run', qa_preflight)
    launches = []
    class CPUProcess:
        returncode = 0
        def __init__(self, command, **kwargs):
            tag = command[command.index('--tag')+1]
            name = tag.split('-storage-formal-')[0]
            warm = '--stage0-only' in command
            launches.append((name, warm, list(command)))
            directory = root/'outputs'/('full_pipeline_profile_'+tag)
            directory.mkdir(parents=True)
            if warm:
                available[name] = True
                cache = directory/'static_cache'
                cache.mkdir()
                (cache/'rank0.pt').write_bytes(b'CPU newly precomputed cache')
                prepared = cache/'_prepared_checkpoints'/name
                prepared.mkdir(parents=True)
                (prepared/'model.safetensors').write_bytes(b'CPU newly prepared model')
            (directory/'status.json').write_text(json.dumps(dict(valid=True, stage0=dict(returncode=0))))
            kwargs['stdout'].write('CPU fixture pipeline completed\n')
            kwargs['stdout'].flush()
        def poll(self):
            return 0
    monkeypatch.setattr(runner.subprocess, 'Popen', CPUProcess)
    audited = []
    def audit(control, candidate, ranks, bindings):
        audited.append((control, candidate))
        return dict(valid=True, fixture='outer controller scheduling only')
    monkeypatch.setattr(runner, 'audit_pair', audit)
    if qa_error:
        with pytest.raises(runner.subprocess.CalledProcessError):
            runner.main()
        assert not launches and not audited
        assert not runner.read(output/'status.json')['valid']
        return
    runner.main()
    status = runner.read(output/'status.json')
    assert status['valid'] and len(status['completed_pairs']) == len(audited) == 6
    assert len([row for row in launches if not row[1]]) == 12
    assert len([row for row in launches if row[1]]) == (0 if hit else 2)
    assert len(status['precomputations']) == (0 if hit else 2)
    for name in directories:
        budget = runner.read(output/f'cache_disk_preflight_{name}.json')
        assert budget['view_count' if storage == 'shared' else 'snapshot_count'] == 6
        assert runner.read(output/f'cache_ready_{name}.json')['prepared']
        arms = [row[2] for row in launches if row[0] == name and not row[1]]
        assert all('--reuse-cache' in argv for argv in arms)
        assert ['candidate' if argv[argv.index('--tag')+1].endswith('candidate') else 'control' for argv in arms] == [
            'control', 'candidate', 'candidate', 'control', 'control', 'candidate']
