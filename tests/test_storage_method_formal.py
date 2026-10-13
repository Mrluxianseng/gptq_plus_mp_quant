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
