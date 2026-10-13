"""Portable, serial formal W4A16 memory comparison; no historical outputs needed."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_hessian_solver_workspace_followup import prefix
from tools.hessian_dead_subgroup_install import build_entry as add_dead
from tools.activation_replay_stats_install import build_entry as add_stats
from tools.audit_memory_boundary_points import boundaries, compare as compare_points
from tools.audit_completed_memory_pair_capacity import intervals, cache_inventory
from tools import compare_full_pipeline_profiles as comparator
from tools.run_full_pipeline_memory_campaign import stop_process_tree
from tools.storage_method_distributed_audit import install_entry_audit, audit_directory
from tools.storage_method_log_stream import LiveFileRelay

METHOD_FLAGS = ('--research_hinv_solver_workspace', '--research_hinv_reuse_consumed_h',
                '--research_hinv_stable_weight', '--research_replay_early_stats')
MODEL_LRS = {'qwen06b': '3e-4', 'qwen17b': '5e-4'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def build_entry():
    """Bind the already tested helper chain to the portable checkpoint entry."""
    source = prefix(sha(ROOT/'tools/hessian_solver_workspace_prototype.py'),
                    stable_sha=sha(ROOT/'tools/hessian_stable_weight_prototype.py'))
    source += '\n' + (ROOT/'tools/realq_checkpoint_research_entry.py').read_text(encoding='utf-8')
    source = add_dead(source, installer_sha=sha(ROOT/'tools/hessian_dead_subgroup_install.py'),
                      prototype_sha=sha(ROOT/'tools/hessian_dead_subgroup_input_prototype.py'))
    source = add_stats(source, installer_sha=sha(ROOT/'tools/activation_replay_stats_install.py'),
                       prototype_sha=sha(ROOT/'tools/activation_replay_stats_lifetime_prototype.py'))
    source = install_entry_audit(source)
    anchor = "runpy.run_path('ptq.py', run_name='__main__')"
    assert source.count(anchor) == 1
    guard = "def _forbid_precompute(*args, **kwargs):\n    raise RuntimeError('formal Stage2 must consume its fresh verified cache')\nif '--exit_after_precompute' not in sys.argv:\n    impl.collect_static_end_to_end_saliency_and_fisher=_forbid_precompute\n\n"
    source = source.replace(anchor, guard + anchor)
    compile(source, 'formal_storage_entry', 'exec')
    return source


def command(args, model, entry, stage, candidate):
    # Formal workload/settings from formal_realq_compare.sh; model-specific
    # learning rates from the repository's paper PDF Table 6. No optimizer sweep.
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nnodes=1', f'--nproc_per_node={args.world_size}', str(entry), 'baseline-profile',
           '--model', str(args.models[model]), '--exp', 'formal_storage',
           '--dataset', 'wikitext2', '--nsamples', str(args.nsamples), '--seq_len', str(args.seq_len),
           '--seed', str(args.seed), '--rotation_seed', '0', '--refresh_seed', '0',
           '--w_method', 'gptq_plus', '--w_bits', '4', '--w_groupsize', '-1', '--w_clip',
           '--a_bits', '16', '--k_bits', '16', '--v_bits', '16', '--rotate', '--act_order',
           '--num_groups', '4', '--blocksize', '128', '--alpha', '0', '--enable_gptq_plus', '0',
           '--g_update_mode', 'block_gd', '--grad_optimizer', 'adam', '--final_layer_grad_optimizer', 'adam',
           '--grad_lr', args.lr06 if model == 'qwen06b' else args.lr17,
           '--final_layer_grad_lr', '1e-5', '--grad_lr_layer_schedule', 'cosine',
           '--grad_lr_layer_base_ratio', '0.01', '--grad_clip', '1.0', '--final_layer_grad_clip', '1.0',
           '--backward_samples', '32', '--backward_bsz', str(args.backward_bsz),
           '--final_layer_backward_bsz', str(args.backward_bsz), '--bsz', str(args.bsz),
           '--global_loss_bsz', str(args.bsz), '--hessian_accum_bsz', str(args.bsz),
           '--final_layer_stats_bsz', '4', '--global_loss', '--dp_global_shuffle', '--loss_slide_window',
           '--grad_refresh_loss', 'fisher_diag_mse', '--a_loss_ratio', '0.95',
           '--a_loss_clip_scope', 'local_backward_chunk', '--saliency_clip_percentile', '0.99',
           '--kl_topk', '-1', '--grad_hessian_topk', '-1', '--pre_gd_steps', '0',
           '--pre_grad_lr', '0', '--grad_reg_strategy', 'none',
           '--group_parallel_quant', 'none', '--gptq_inner_kernel', 'torch',
           '--offload_inps', '--kl_logit_chunk_tokens', '16', '--eval_logit_chunk_tokens', '16',
           '--eval_ce_chunk_tokens', '256', '--static_fisher_label_chunk_tokens', '0',
           '--eval_seq_len', str(args.seq_len), '--eval_samples', '0', '--eval_datasets', 'wikitext2',
           '--quant_stop_layer', '27', '--static_cache_path', 'SET_BY_FRESH_RUNNER']
    if stage == 'stage0':
        cmd += ['--fsdp_precompute', '--exit_after_precompute', '--skip_eval', '--research_stage0_checkpoint']
    else:
        cmd += ['--stage2_cpu_master']
        if args.qa_eval:
            cmd += ['--lm_eval', '--lm_eval_batch_size', '16']
    if candidate:
        cmd += list(METHOD_FLAGS)
    return cmd


def check_sources(bindings):
    changed = [name for name, expected in bindings.items() if sha(ROOT/name) != expected]
    if changed:
        raise RuntimeError('Bound source changed: ' + repr(changed))


def method_coverage(run, ranks, enabled, dimension=None):
    """Fail on skipped/batched paths even if their numerical result happens to match."""
    records = []
    groups, layers = Counter({i: 4 for i in range(28)}), Counter(range(28))
    for rank in range(ranks):
        inv = read(run/f'inverse_lifetime_stage2_rank{rank}.json')
        dead = read(run/f'dead_subgroup_input_stage2_rank{rank}.json')
        stats = read(run/f'replay_stats_lifetime_stage2_rank{rank}.json')
        ok = (inv['solver_workspace'] == enabled and inv['reuse'] == enabled
              and inv['stable_weight'] == enabled and not inv['isolated_pool']
              and inv['helper_unchanged'] and inv['stable_helper_unchanged']
              and dead['enabled'] == enabled and dead['helpers_unchanged']
              and stats['enabled'] == enabled and stats['helpers_unchanged'])
        ok = ok and len(inv['calls']) == 112 and Counter(r['layer'] for r in inv['calls']) == groups
        ok = ok and all(r['enabled'] == enabled and not r['identity_fallback'] and r['dtype'] == 'torch.float32'
                       and (not enabled or r['workspace_reuse_enabled']) for r in inv['calls'])
        if dimension is not None:
            ok = ok and all(r['module'] == 'mlp.down_proj' and r['shape'] == [dimension, dimension] for r in inv['calls'])
        ok = ok and len(inv['weight_write_calls']) == 28 and Counter(r['layer'] for r in inv['weight_write_calls']) == layers
        ok = ok and all(r['enabled'] == enabled and (not enabled or r['pointer_preserved']) for r in inv['weight_write_calls'])
        ok = ok and len(stats['records']) == 28 and Counter(r['layer'] for r in stats['records']) == layers
        ok = ok and all(r['enabled'] == enabled for r in stats['records'])
        if enabled:
            ok = ok and len(inv['reuse_calls']) == 112 and Counter(r['layer'] for r in inv['reuse_calls']) == groups
            ok = ok and all(r['factor_shares_consumed_h_storage'] and not r['identity_fallback'] for r in inv['reuse_calls'])
            ok = ok and len(dead['records']) == 28 and Counter(r['layer'] for r in dead['records']) == layers
            ok = ok and all(r['retained_factors'] == 4 and r['released_private_input_bytes'] > 0 for r in dead['records'])
            if dimension is not None:
                ok = ok and all(r['module'] == 'mlp.down_proj' and r['released_private_input_bytes'] == dimension*dimension*4 for r in dead['records'])
        else:
            ok = ok and not inv['reuse_calls'] and not dead['records']
        records.append(dict(rank=rank, valid=bool(ok), stats_bytes=[r['observed_cuda_storage_bytes'] for r in stats['records']]))
    return dict(valid=all(r['valid'] for r in records), ranks=records)


def nvml_peaks(run):
    result = {}
    with (run/'nvml_telemetry.csv').open(newline='') as stream:
        for row in csv.reader(stream):
            if len(row) >= 6:
                try:
                    key, value = row[5].strip(), int(row[1].strip())
                    result[key] = max(value, result.get(key, 0))
                except ValueError:
                    continue
    return result


def audit_pair(control, candidate, ranks, bindings):
    # Register the storage marker only in this process; original checker stays intact.
    original = comparator.SYSTEM_COMMAND_FLAGS
    try:
        comparator.SYSTEM_COMMAND_FLAGS = original | {'--research_replay_early_stats'}
        legacy = comparator.compare(control, candidate, compare_cache=True)
    finally:
        comparator.SYSTEM_COMMAND_FLAGS = original
    statuses = [read(run/'status.json') for run in (control, candidate)]
    paths = [[ROOT/f for f in s['memory_profile_files']] for s in statuses]
    points = compare_points(*(boundaries(files) for files in paths))
    rows = [intervals(files) for files in paths]
    missing = rows[0].keys() ^ rows[1].keys()
    interval_regressions = []
    for key in sorted(rows[0].keys() & rows[1].keys()):
        for metric in ('allocated', 'reserved'):
            a, b = [values[key][f'sampled_peak_{metric}_bytes'] for values in rows]
            if b > a:
                interval_regressions.append(dict(key=list(key), metric=metric, control=a, candidate=b))
    manifest = read(control/'manifest.json')['stage2_command']
    config = read(Path(manifest[manifest.index('--model')+1])/'config.json')
    dimension = config['intermediate_size']
    methods = [method_coverage(run, ranks, side == 1, dimension) for side, run in enumerate((control, candidate))]
    cpu_master = '--stage2_cpu_master' in manifest
    ownership = [audit_directory(run, ranks=ranks, cpu_master=cpu_master) for run in (control, candidate)]
    rank_states = all(report['valid'] and owner == status['metrics'] for (report, owner), status in zip(ownership, statuses))
    traces_ok = True
    for rank in ([0] if cpu_master else range(ranks)):
        a, b = [read(run/f'layer_output_fingerprints_rank{rank}.json') for run in (control, candidate)]
        traces_ok = traces_ok and a == b and len(a) == 28 and {r['layer'] for r in a} == set(range(28))
        traces_ok = traces_ok and all(r['forward_count'] > 0 and len(r['shapes']) == r['forward_count'] and len(r['sha256']) == 64 for r in a)
    caches = [cache_inventory(run/'static_cache') for run in (control, candidate)]
    gpu_peaks = [nvml_peaks(run) for run in (control, candidate)]
    global_peaks = {k: [s[k] for s in statuses] for k in ('full_process_peak_allocated_bytes', 'full_process_peak_reserved_bytes', 'max_nvml_memory_used_mib')}
    checks = dict(complete_exact_numerics_sources_commands_runtime_gpu=legacy['full_model_quantization_validated'],
                  complete_507_state_and_all_rank_ownership=rank_states, owner_all28_nonempty_exact=traces_ok,
                  complete_cache_inventory=bool(caches[0]) and caches[0] == caches[1],
                  method_coverage=all(m['valid'] for m in methods),
                  stats_holding_bytes_same=all(a['stats_bytes'] == b['stats_bytes'] for a, b in zip(methods[0]['ranks'], methods[1]['ranks'])),
                  complete_exact_intervals=bool(rows[0]) and not missing and all(r.get('peak_measurement_mode') == 'exact' for arm in rows for r in arm.values()),
                  separate_points_complete=not points['missing'] and points['matched_values'] == 4*len(rows[0]),
                  every_stage_rank_has_intervals=all(any(k[0] == stage and k[2] == rank for k in rows[0]) for stage in ('stage0', 'stage2') for rank in range(ranks)),
                  all_full_global_peaks_nonincreasing=all(b <= a for a, b in global_peaks.values()),
                  sampled_each_physical_gpu_peak_nonincreasing=bool(gpu_peaks[0]) and gpu_peaks[0].keys() == gpu_peaks[1].keys() and all(gpu_peaks[1][gpu] <= value for gpu, value in gpu_peaks[0].items()))
    check_sources(bindings)
    checks['bound_sources_unchanged'] = True
    return dict(valid=all(checks.values()), capacity_valid=all(checks.values()), checks=checks,
                full_global_peaks=global_peaks, sampled_nvml_peaks_by_uuid=gpu_peaks,
                state_ownership=[report for report, _ in ownership],
                cache_files=caches, method_coverage=methods, interval_count=len(rows[0]),
                interval_regressions=interval_regressions, missing_intervals=[list(k) for k in sorted(missing)],
                points=points, strict_interval_point_valid=all(checks.values()) and not interval_regressions and points['valid'],
                original_comparison=legacy, status_sha256=[sha(run/'status.json') for run in (control, candidate)],
                criterion='Exact quality/source/cache/method/full telemetry and global capacity; local regressions retained as diagnostics. NVML sampled; profiling time is not speed evidence.')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true', help='Write fresh plan/entry/manifests without importing torch or launching GPU work.')
    modes.add_argument('--run', action='store_true')
    p.add_argument('--model06', type=Path, required=True)
    p.add_argument('--model17', type=Path, required=True)
    p.add_argument('--only', choices=('both', 'qwen06b', 'qwen17b'), default='both')
    p.add_argument('--devices', default='0,1,2,3')
    p.add_argument('--output-directory', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--nsamples', type=int, default=2048)
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--bsz', type=int, default=8)
    p.add_argument('--backward-bsz', type=int, default=4)
    p.add_argument('--lr06', default=MODEL_LRS['qwen06b'])
    p.add_argument('--lr17', default=MODEL_LRS['qwen17b'])
    p.add_argument('--skip-qa-eval', action='store_false', dest='qa_eval', default=True,
                   help='Omit QA tasks for an initial memory-only comparison; default matches the previous formal script QA setting.')
    p.add_argument('--timeout-seconds', type=int, default=86400)
    return p


def main():
    args = parser().parse_args()
    devices = args.devices.split(',')
    if len(devices) != len(set(devices)) or any(not d.isdigit() for d in devices):
        raise ValueError('Use distinct physical GPU indices, e.g. 0,1,2,3')
    args.world_size = len(devices)
    if any(x <= 0 for x in (args.repeats, args.nsamples, args.seq_len, args.bsz, args.backward_bsz, args.timeout_seconds)):
        raise ValueError('Counts/batches/timeout must be positive')
    if 32 % args.world_size or args.nsamples % args.world_size or args.bsz % args.world_size:
        raise ValueError('GPU count must divide backward_samples=32, calibration samples and global_loss_bsz')
    args.models = {'qwen06b': args.model06.resolve(), 'qwen17b': args.model17.resolve()}
    selected = list(args.models) if args.only == 'both' else [args.only]
    for name in selected:
        config = read(args.models[name]/'config.json')
        expected = (1024, 3072) if name == 'qwen06b' else (2048, 6144)
        if (config.get('hidden_size'), config.get('intermediate_size')) != expected or config.get('num_hidden_layers') != 28:
            raise ValueError('Model architecture does not match ' + name)
    out = args.output_directory.resolve()
    out.mkdir(parents=True, exist_ok=False)
    entry = out/'entry.py'
    entry.write_text(build_entry(), encoding='utf-8')
    files = {ROOT/'ptq.py', ROOT/'process_args.py', Path(__file__), ROOT/'scripts/run_storage_method_formal.sh', ROOT/'scripts/formal_realq_compare.sh'}
    files.update(p for folder in ('tools', 'gptq_utils', 'utils') for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts)
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    save(out/'source_bindings.json', bindings)
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    jobs = []
    for model in selected:
        for pair in range(1, args.repeats + 1):
            order = ('control', 'candidate') if pair % 2 else ('candidate', 'control')
            for side in order:
                tag = f'{model}-storage-formal-{run_id}-p{pair}-{side}'
                manifests = []
                for stage in ('stage0', 'stage2'):
                    path = out/f'{tag}-{stage}.json'
                    cmd = command(args, model, entry, stage, side == 'candidate')
                    save(path, dict(command=cmd, scope='Formal server workload; unvalidated before execution', model=model, pair=pair, side=side))
                    manifests.append(path)
                launcher = [sys.executable, '-u', str(ROOT/'tools/run_storage_method_pipeline.py'),
                            '--stage0-manifest', str(manifests[0]), '--stage2-manifest', str(manifests[1]),
                            '--tag', tag, '--timeout-seconds', str(args.timeout_seconds),
                            '--exact-phase-peaks', '--gpu-indices', args.devices]
                jobs.append(dict(model=model, pair=pair, side=side, tag=tag, launcher=launcher,
                                 result=str(ROOT/'outputs'/('full_pipeline_profile_'+tag))))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices,
               PYTORCH_ALLOC_CONF='backend:native', PYTORCH_CUDA_ALLOC_CONF='backend:native',
               OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '1'),
               NCCL_SHM_DISABLE=os.environ.get('NCCL_SHM_DISABLE', '1'), PYTHONUNBUFFERED='1')
    save(out/'plan.json', dict(created=datetime.now().isoformat(), gpu_work_started=False, aggregate_worker_stdout=True,
         formal_workload=args.nsamples == 2048 and args.seq_len == 2048,
         config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k != 'models'},
         model_config_sha256={m: sha(args.models[m]/'config.json') for m in selected},
         entry_sha256=sha(entry), environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'PYTORCH_ALLOC_CONF', 'NCCL_SHM_DISABLE', 'OMP_NUM_THREADS', 'PYTHONUNBUFFERED')},
         jobs=jobs, note='Sequential groups, DP/FSDP across selected GPUs; fresh Stage0 per arm. No pilot canonical hashes reused as formal references.'))
    print(f'Plan: {out / "plan.json"}\n2048x2048 defaults; group_parallel_quant=none; GPUs={args.devices}', flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"{job['model']} pair{job['pair']} {job['side']}: {job['result']}")
        return
    active = None
    completed = []
    results = {}
    status = dict(valid=False, completed_pairs=completed, workload='formal server comparison')
    def interrupted(*_):
        raise KeyboardInterrupt('Interrupted; original evidence retained')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for job in jobs:
            check_sources(bindings)
            if sha(entry) != read(out/'plan.json')['entry_sha256']:
                raise RuntimeError('Generated entry changed')
            for model in selected:
                if sha(args.models[model]/'config.json') != read(out/'plan.json')['model_config_sha256'][model]:
                    raise RuntimeError('Model config changed')
            query = subprocess.check_output(['nvidia-smi', f'--id={args.devices}', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
            if any(line.strip().isdigit() for line in query.splitlines()):
                raise RuntimeError('Selected GPU has another compute process; no concurrent launch')
            print(f"Starting {job['tag']}; log {out/(job['tag']+'.log')}", flush=True)
            with (out/(job['tag']+'.log')).open('x') as log:
                active = subprocess.Popen(job['launcher'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                relay = LiveFileRelay(out/(job['tag']+'.log'))
                while active.poll() is None:
                    relay.forward()
                    time.sleep(0.25)
                relay.forward()
                code = active.returncode
            active = None
            if code:
                raise RuntimeError('Full pipeline failed: ' + job['tag'])
            check_sources(bindings)
            if sha(entry) != read(out/'plan.json')['entry_sha256']:
                raise RuntimeError('Generated entry changed during run')
            key = (job['model'], job['pair'])
            results.setdefault(key, {})[job['side']] = Path(job['result'])
            if len(results[key]) == 2:
                pair = audit_pair(results[key]['control'], results[key]['candidate'], args.world_size, bindings)
                path = out/f"audit_{job['model']}_p{job['pair']}.json"
                save(path, pair)
                if not pair['valid']:
                    raise RuntimeError('Formal acceptance failed; preserved audit: ' + str(path))
                completed.append(dict(model=key[0], pair=key[1], audit=str(path), audit_sha256=sha(path), capacity_valid=True))
                print('Accepted complete formal pair: ' + str(path), flush=True)
        status['valid'] = True
    except BaseException as exc:
        status['error'] = repr(exc)
        if active is not None and active.poll() is None:
            stop_process_tree(active.pid)
        raise
    finally:
        status['finished'] = datetime.now().isoformat()
        save(out/'status.json', status)


if __name__ == '__main__':
    main()
