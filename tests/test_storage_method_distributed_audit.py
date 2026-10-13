"""Exercise actual entry functions with two Gloo ranks and real meta release."""
import ast
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tools import run_storage_method_formal as formal
from tools import run_storage_method_pipeline as pipeline
from tools.storage_method_distributed_audit import audit_directory, state_record, state_layout, validate_records


def functions(source, names, namespace):
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'actual_generated_functions', 'exec'), namespace)


def model_fixture():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(28)])
    for layer in model.layers:
        layer.weight.data.copy_(torch.eye(2))
    for index in range(479):
        model.register_buffer(f'fixed_{index}', torch.tensor([index], dtype=torch.int64))
    assert len(model.state_dict()) == 507
    model._gptqplus_stage2_cpu_master = True
    return model


def distributed_worker(rank, root, rendezvous, source):
    os.environ.update(RANK=str(rank), WORLD_SIZE='2')
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        from gptq_utils.gptq_plus_utils import Stage2CpuMasterLayerManager
        model = model_fixture()
        analyzer = SimpleNamespace(model=model)
        manager = Stage2CpuMasterLayerManager(analyzer, torch.device('cpu'), model.layers)
        # Use the production manager's ownership/death transition, not a fake
        # is_meta flag. Communication is Gloo/CPU; no CUDA context is created.
        def quantize(*_):
            for index in range(28):
                layer = model.layers[index]
                # Production materialization enforces CUDA/NCCL. Substitute
                # only its transport with CPU/Gloo, then call its real release.
                dist.broadcast(layer.weight.data, src=0)
                manager.materialized.add(index)
                assert not layer.weight.is_meta
                manager.release_layer(index, orig_device=torch.device('cpu'))
            return model
        fake_cuda = SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda: 0,
                                    max_memory_reserved=lambda: 0)
        torch_proxy = SimpleNamespace(cuda=fake_cuda, Tensor=torch.Tensor, uint8=torch.uint8)
        namespace = dict(torch=torch_proxy, hashlib=hashlib, json=json, logging=logging,
                         os=os, Path=Path, time=time, root=Path(root), mode='baseline-profile',
                         original_fwrd=quantize,
                         impl=SimpleNamespace(memory_utils=SimpleNamespace(get_process_cuda_memory_peaks=lambda: (0, 0))))
        def evaluate(args, analyzer):
            value = torch.ones(1, 2)
            for layer in analyzer.model.layers:
                value = layer(value)
            return (0.125, 20.0)
        namespace['_layer_trace_original_eval'] = evaluate
        functions(source, {'measured_fwrd', '_layer_trace_eval'}, namespace)
        args = SimpleNamespace(stage2_cpu_master=True)
        assert namespace['measured_fwrd'](args, analyzer, [], 'cpu') is model
        if rank == 1:
            assert all(layer.weight.is_meta for layer in model.layers)
            # Reproduce the failed server path on the same real tensors.
            old = (formal.ROOT/'tools/realq_checkpoint_research_entry.py').read_text()
            old_ns = dict(namespace, original_fwrd=lambda *_: model)
            functions(old, {'measured_fwrd'}, old_ns)
            with pytest.raises(NotImplementedError, match='meta tensor'):
                old_ns['measured_fwrd'](args, analyzer, [], 'cpu')
            (Path(root)/'rank1_old_meta_failure_reproduced.json').write_text(json.dumps({'reproduced': True}))
        else:
            assert namespace['_layer_trace_eval'](args, analyzer) == (0.125, 20.0)
            # Canonical owner trace is written by the actual production footer.
            # CPU footer skips CUDA peak/canonical writes; copy that owner-only
            # operation explicitly and leave non-owner trace absent.
            path = Path(root)/'layer_output_fingerprints_rank0.json'
            (Path(root)/'layer_output_fingerprints.json').write_bytes(path.read_bytes())
        dist.barrier()
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_two_real_ranks_actual_entry_meta_release_and_owner_evaluation(tmp_path):
    source = pipeline.prepare_entry_source(formal.build_entry())
    rendezvous = (tmp_path/'gloo_init').resolve().as_uri()
    mp.spawn(distributed_worker, args=(str(tmp_path), rendezvous, source), nprocs=2, join=True)
    report, owner = audit_directory(tmp_path, ranks=2, cpu_master=True)
    assert report['valid'], report
    assert report['evaluation_ranks'] == [0] and report['layout_record_ranks'] == [0, 1]
    assert owner['state_tensors'] == 507 and owner['state_meta_tensors'] == 0
    nonowner = json.loads((tmp_path/'baseline-profile_metrics_rank1.json').read_text())
    assert nonowner['state_meta_tensors'] == 28 and nonowner['state_tensors'] == 0
    assert nonowner['state_sha256'] is None
    assert not (tmp_path/'layer_output_fingerprints_rank1.json').exists()
    assert json.loads((tmp_path/'rank1_old_meta_failure_reproduced.json').read_text())['reproduced']
    # Numerical digest format stays byte-for-byte compatible with pilot auditing.
    digest = hashlib.sha256()
    for name, tensor in sorted(model_fixture().state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    assert owner['state_sha256'] == digest.hexdigest()
    (tmp_path/'baseline-profile_metrics_rank1.json').unlink()
    assert not audit_directory(tmp_path, ranks=2, cpu_master=True)[0]['valid']


def records_fixture(cpu_master=True):
    model = model_fixture()
    records = [state_record(model, cpu_master=cpu_master, rank=r, world_size=2) for r in range(2)]
    trace = [dict(layer=i, forward_count=1, shapes=[[1, 2]], sha256='a'*64) for i in range(28)]
    return records, {r: deepcopy(trace) for r in ([0] if cpu_master else [0, 1])}


def write_expected_layout(directory, ranks):
    expected = state_layout(model_fixture())
    for rank in range(ranks):
        (directory/f'state_expected_layout_rank{rank}.json').write_text(json.dumps(
            dict(rank=rank, scope='before_quantization', **expected)))


@pytest.mark.parametrize('fault', ['missing_rank', 'duplicate_rank', 'layout', 'owner_meta', 'owner_digest', 'nonowner_digest', 'ownership', 'world', 'count', 'missing_trace', 'empty_last', 'unexpected_trace'])
def test_incomplete_or_false_evidence_is_rejected(fault):
    records, traces = records_fixture()
    if fault == 'missing_rank': records.pop()
    elif fault == 'duplicate_rank': records[1]['rank'] = 0
    elif fault == 'layout': records[1]['state_layout_sha256'] = 'b'*64
    elif fault == 'owner_meta': records[0]['state_meta_tensors'] = 1
    elif fault == 'owner_digest': records[0]['state_sha256'] = None
    elif fault == 'nonowner_digest': records[1]['state_sha256'] = 'c'*64
    elif fault == 'ownership': records[1]['state_audit_authoritative'] = True
    elif fault == 'world': records[1]['world_size'] = 1
    elif fault == 'count': records[0]['state_tensors'] = 506
    elif fault == 'missing_trace': traces.clear()
    elif fault == 'empty_last': traces[0][-1]['forward_count'] = 0
    elif fault == 'unexpected_trace': traces[1] = deepcopy(traces[0])
    assert not validate_records(records, traces, ranks=2, cpu_master=True)['valid']


def test_replicated_mode_still_requires_every_rank_values_and_eval():
    records, traces = records_fixture(False)
    assert validate_records(records, traces, ranks=2, cpu_master=False)['valid']
    traces.pop(1)
    assert not validate_records(records, traces, ranks=2, cpu_master=False)['valid']
    records, traces = records_fixture(False)
    records[1]['state_sha256'] = 'c'*64
    assert not validate_records(records, traces, ranks=2, cpu_master=False)['valid']


def test_owner_meta_state_is_rejected_before_hashing():
    model = model_fixture().to_empty(device='meta')
    with pytest.raises(RuntimeError, match='Authoritative state'):
        state_record(model, cpu_master=True, rank=0, world_size=2)


@pytest.mark.parametrize('hidden,intermediate', [(1024, 3072), (2048, 6144)])
def test_real_qwen_meta_architecture_has_507_weights_and_28_rotation_buffers(hidden, intermediate):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from utils import model_utils, rotation_utils
    config = Qwen3Config(hidden_size=hidden, intermediate_size=intermediate,
                         num_hidden_layers=28, num_attention_heads=16,
                         num_key_value_heads=8, head_dim=128, vocab_size=64)
    with torch.device('meta'):
        model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
    analyzer = object.__new__(model_utils.ModelAnalyzer)
    analyzer.model, analyzer.model_arch = model, 'Qwen3ForCausalLM'
    rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
    state = model.state_dict()
    assert len(state) == 535
    assert sum(name.endswith('.weight') for name in state) == 507
    assert sum(name.endswith('.had_K') for name in state) == 28
    assert all(list(tensor.shape) == [12, 12] for name, tensor in state.items() if name.endswith('.had_K'))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault', [None, 'missing_buffer', 'shape', 'dtype', 'rename', 'missing_snapshot'])
def test_actual_cpu_qwen_rotation_full_state_and_pre_layout_are_required(tmp_path, fault):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from utils import model_utils, rotation_utils
    model = Qwen3ForCausalLM(Qwen3Config(hidden_size=16, intermediate_size=48,
        num_hidden_layers=28, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, vocab_size=32)).to(dtype=torch.bfloat16)
    analyzer = object.__new__(model_utils.ModelAnalyzer)
    analyzer.model, analyzer.model_arch = model, 'Qwen3ForCausalLM'
    rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
    expected = state_layout(model)
    assert expected['tensors'] == 535
    first = model.model.layers[0].mlp.down_proj
    if fault == 'missing_buffer': first.had_K = None
    elif fault == 'shape': first.had_K = first.had_K[:1]
    elif fault == 'dtype': first.had_K = first.had_K.to(dtype=torch.bfloat16)
    elif fault == 'rename': first.register_buffer('extra_had', first.had_K); first.had_K = None
    if fault not in (None, 'missing_snapshot'):
        with pytest.raises(RuntimeError, match='names/shapes/dtypes changed'):
            state_record(model, cpu_master=True, rank=0, world_size=1, expected_layout=expected)
        return
    row = state_record(model, cpu_master=True, rank=0, world_size=1, expected_layout=expected)
    trace = [dict(layer=i, forward_count=1, shapes=[[1, 2]], sha256='a'*64) for i in range(28)]
    if fault != 'missing_snapshot':
        (tmp_path/'state_expected_layout_rank0.json').write_text(json.dumps(dict(rank=0, scope='before_quantization', **expected)))
    for name, value in [('baseline-profile_metrics_rank0.json', row), ('baseline-profile_metrics.json', row),
                        ('layer_output_fingerprints_rank0.json', trace), ('layer_output_fingerprints.json', trace)]:
        (tmp_path/name).write_text(json.dumps(value))
    report, owner = audit_directory(tmp_path, ranks=1, cpu_master=True)
    assert report['valid'] == (fault is None), report
    assert owner['state_tensors'] == 535 and owner['state_meta_tensors'] == 0


@pytest.mark.parametrize('missing_peak', [False, True])
def test_actual_pipeline_main_selects_owner_and_requires_all_rank_peaks(tmp_path, monkeypatch, missing_peak):
    """Exercise status assembly with CPU fixture workers/telemetry, no GPU run."""
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    for name in pipeline.TRACKED:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture source')
    entry = tmp_path/'entry.py'
    entry.write_text(formal.build_entry())
    base = ['python', '--nproc_per_node=2', str(entry), 'baseline-profile', '--model', '/models/Qwen3-0.6B',
            '--dataset', 'wikitext2', '--nsamples', '2048', '--seq_len', '2048', '--seed', '1',
            '--num_groups', '4', '--exp', 'fixture', '--static_cache_path', 'fresh']
    stage0, stage2 = tmp_path/'s0.json', tmp_path/'s2.json'
    stage0.write_text(json.dumps({'command': [*base, '--fsdp_precompute', '--exit_after_precompute']}))
    stage2.write_text(json.dumps({'command': [*base, '--stage2_cpu_master']}))
    monkeypatch.setattr(sys, 'argv', ['profile', '--stage0-manifest', str(stage0), '--stage2-manifest', str(stage2), '--tag', 'owned-fixture'])
    def monitor(*args, **kwargs):
        kwargs['stdout'].write('2026/10/13 00:00:00.000, 100, 8000, 1, 0, GPU-fixture\n')
        kwargs['stdout'].flush()
        return SimpleNamespace(terminate=lambda: None, wait=lambda **_: 0)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', monitor)
    monkeypatch.setattr(pipeline, 'profile_rows', lambda *args, **kwargs: [])
    def run_stage(stage, command, root, env, log, *args):
        out = Path(env['RESEARCH_RUN_DIR'])
        if stage == 'stage0':
            cache = out/'static_cache'
            cache.mkdir()
            (cache/'fisher.pt').write_bytes(b'fixture')
            log.write_text('precompute complete')
        else:
            records, traces = records_fixture()
            write_expected_layout(out, 2)
            for record in records:
                (out/f"baseline-profile_metrics_rank{record['rank']}.json").write_text(json.dumps(record))
            (out/'baseline-profile_metrics.json').write_text(json.dumps(records[0]))
            (out/'layer_output_fingerprints_rank0.json').write_text(json.dumps(traces[0]))
            (out/'layer_output_fingerprints.json').write_text(json.dumps(traces[0]))
            # Non-owner logs first; status must not select this first row.
            log.write_text('OVERNIGHT_METRICS '+json.dumps(records[1])+'\nExact KL&PPL on wikitext2: 0.125, 20.0\n')
        for rank in range(2):
            if missing_peak and stage == 'stage2' and rank == 1:
                continue
            (out/f'process_allocator_peak_{stage}_rank{rank}.json').write_text(json.dumps(
                dict(rank=rank, stage=stage, peak_allocated_bytes=100, peak_reserved_bytes=200,
                     scope='complete_process_through_evaluation')))
        return {'returncode': 0}
    monkeypatch.setattr(pipeline, 'run_stage', run_stage)
    if missing_peak:
        with pytest.raises(RuntimeError, match='lacks complete metrics'):
            pipeline.main()
    else:
        pipeline.main()
    status = json.loads((tmp_path/'outputs/full_pipeline_profile_owned-fixture/status.json').read_text())
    assert status['valid'] == (not missing_peak)
    assert status['metrics']['rank'] == 0 and status['metrics']['state_tensors'] == 507
    assert status['expected_evaluation_ranks'] == [0]
    assert status['state_hashes_match_across_ranks'] is None
    assert status['layer_fingerprints_match_across_ranks'] is None
    assert len(status['state_metrics_by_rank']) == 2


@pytest.mark.parametrize('storage', ['shared', 'copy'])
def test_cached_pipeline_skips_stage0_without_fake_peak(tmp_path, monkeypatch, storage):
    """A cache hit launches only Stage2 and records no invented Stage0 peak."""
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    for name in pipeline.TRACKED:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture source')
    entry = tmp_path/'entry.py'
    entry.write_text(formal.build_entry())
    model = tmp_path/'Qwen3-0.6B'
    model.mkdir()
    (model/'config.json').write_text(json.dumps({'num_hidden_layers': 28}))
    base = ['python', '--nproc_per_node=2', str(entry), 'baseline-profile', '--model', str(model),
            '--dataset', 'wikitext2', '--nsamples', '2048', '--seq_len', '2048', '--seed', '1',
            '--num_groups', '4', '--exp', 'fixture', '--static_cache_path', 'fresh']
    stage0, stage2 = tmp_path/'s0.json', tmp_path/'s2.json'
    stage0.write_text(json.dumps({'command': [*base, '--fsdp_precompute', '--exit_after_precompute']}))
    stage2.write_text(json.dumps({'command': [*base, '--stage2_cpu_master']}))
    source_cache = tmp_path/'old-cache'
    source_cache.mkdir()
    monkeypatch.setattr(sys, 'argv', ['profile', '--stage0-manifest', str(stage0), '--stage2-manifest', str(stage2),
                                     '--tag', 'cached-fixture', '--reuse-cache', str(source_cache), '--cache-storage', storage])
    monkeypatch.setattr(pipeline, 'cache_contract', lambda _: {})
    monkeypatch.setattr(pipeline, 'cache_candidates', lambda *_args, **_kwargs: [
        {'directory': source_cache.resolve(), 'shards': {}, 'prepared': source_cache}])
    monkeypatch.setattr(pipeline, 'validate_cache', lambda *_args, **_kwargs: {'sha256': {}, 'contract': {}})
    def snapshot(_admitted, destination):
        destination.mkdir(parents=True)
        (destination/'rank.pt').write_bytes(b'validated fixture')
    monkeypatch.setattr(pipeline, 'share_cache' if storage == 'shared' else 'snapshot_cache', snapshot)
    monkeypatch.setattr(pipeline, 'verify_inventory', lambda _: None)
    def monitor(*args, **kwargs):
        kwargs['stdout'].write('2026/10/13 00:00:00.000, 100, 8000, 1, 0, GPU-fixture\n')
        kwargs['stdout'].flush()
        return SimpleNamespace(terminate=lambda: None, wait=lambda **_: 0)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', monitor)
    monkeypatch.setattr(pipeline, 'profile_rows', lambda *args, **kwargs: [])
    calls = []
    def run_stage(stage, command, root, env, log, *args):
        calls.append(stage)
        out = Path(env['RESEARCH_RUN_DIR'])
        records, traces = records_fixture()
        write_expected_layout(out, 2)
        for record in records:
            (out/f"baseline-profile_metrics_rank{record['rank']}.json").write_text(json.dumps(record))
            (out/f"process_allocator_peak_stage2_rank{record['rank']}.json").write_text(json.dumps(
                dict(rank=record['rank'], stage='stage2', peak_allocated_bytes=100, peak_reserved_bytes=200,
                     scope='complete_process_through_evaluation')))
        (out/'baseline-profile_metrics.json').write_text(json.dumps(records[0]))
        (out/'layer_output_fingerprints_rank0.json').write_text(json.dumps(traces[0]))
        (out/'layer_output_fingerprints.json').write_text(json.dumps(traces[0]))
        log.write_text('Exact KL&PPL on wikitext2: 0.125, 20.0\n')
        return {'returncode': 0}
    monkeypatch.setattr(pipeline, 'run_stage', run_stage)
    pipeline.main()
    status = json.loads((tmp_path/'outputs/full_pipeline_profile_cached-fixture/status.json').read_text())
    assert calls == ['stage2']
    assert status['valid'] and status['scope'] == 'stage2_cached'
    assert status['stage0'] == {'returncode': None, 'skipped': 'validated_cache_reuse', 'measured': False}
    assert status['stage0_process_peak_allocated_bytes'] is None
    assert status['full_process_peak_allocated_bytes'] == 100
    manifest = json.loads((tmp_path/'outputs/full_pipeline_profile_cached-fixture/manifest.json').read_text())
    assert manifest['cache_storage'] == storage
