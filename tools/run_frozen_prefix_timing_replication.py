"""Repeat full-model REAL-Q/cache equivalence and timing in C-B-B-C order.

The prefix cache only changes execution of eligible Qwen3 down_proj refreshes.
All arms use identical PTQ arguments and the current checked-out source.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

os.chdir('/mnt/d/gptq_plus_realq')
root = Path('outputs/frozen_prefix_timing_replication_20260929').resolve()
root.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
common = [
    '--model', '/mnt/d/llamaModels/Qwen3-0.6B', '--dataset', 'wikitext2',
    '--nsamples', '32', '--seq_len', '256', '--w_method', 'gptq_plus',
    '--w_bits', '4', '--w_clip', '--num_groups', '4', '--blocksize', '128',
    '--act_order', '--kl_topk', '-1', '--bsz', '1', '--final_layer_stats_bsz', '4',
    '--alpha', '0.0', '--enable_gptq_plus', '0', '--backward_samples', '32',
    '--backward_bsz', '4', '--final_layer_backward_bsz', '4', '--g_update_mode', 'block_gd',
    '--grad_lr', '5e-5', '--grad_optimizer', 'adam', '--grad_refresh_loss', 'fisher_diag_mse',
    '--global_loss', '--global_loss_bsz', '1', '--grad_clip', '5e-5',
    '--final_layer_grad_clip', '5e-4', '--final_layer_grad_lr', '1e-6',
    '--group_parallel_quant', 'none', '--eval_seq_len', '1024', '--eval_datasets', 'wikitext2',
    '--seed', '42',
]
schedule = [
    ('prefix-fast', 'pair1_candidate_first'),
    ('baseline-profile', 'pair1_control_second'),
    ('baseline-profile', 'pair2_control_first'),
    ('prefix-fast', 'pair2_candidate_second'),
]
source_paths = ['gptq_utils/gptq_plus_utils.py', 'process_args.py',
                'tools/overnight_ptq_probe.py', 'gptq_utils/frozen_prefix_replay.py']
source_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}
rows = []

for mode, label in schedule:
    folder = root / label
    folder.mkdir()
    entry = folder / 'entry.py'
    replay = folder / 'frozen_prefix_replay.py'
    shutil.copyfile('tools/overnight_ptq_probe.py', entry)
    shutil.copyfile('gptq_utils/frozen_prefix_replay.py', replay)
    command = [sys.executable, '-u', '-m', 'torch.distributed.run',
               '--nnodes=1', '--nproc_per_node=1', '--rdzv_endpoint=localhost:29539',
               str(entry), mode, '--exp', 'prefix_timing_' + label, *common]
    hashes = dict(source_hashes)
    hashes[str(entry)] = hashlib.sha256(entry.read_bytes()).hexdigest()
    hashes[str(replay)] = hashlib.sha256(replay.read_bytes()).hexdigest()
    manifest = {'mode': mode, 'label': label, 'command': command, 'hashes': hashes,
                'schedule': 'candidate, control, control, candidate'}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    run_env = env.copy()
    run_env.update(RESEARCH_RUN_DIR=str(folder), REPLAY_IMPL_PATH=str(replay))
    print('START', label, time.strftime('%F %T'), flush=True)
    started = time.monotonic()
    with (folder / 'run.log').open('w') as log:
        result = subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s', '3600', *command],
                                env=run_env, stdout=log, stderr=subprocess.STDOUT)
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                    for p, h in source_hashes.items())
    log_text = (folder / 'run.log').read_text(errors='replace')
    kl_ppl = re.findall(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log_text)
    metric_path = folder / (mode + '_metrics.json')
    status = {'mode': mode, 'label': label, 'returncode': result.returncode,
              'wall_seconds': time.monotonic() - started, 'sources_unchanged': unchanged,
              'preview_events': log_text.count('QUANT_PREVIEW_NEXT_BLOCK'),
              'cache_events': log_text.count('OVERNIGHT_PREFIX hits=')}
    if metric_path.exists():
        status.update(json.loads(metric_path.read_text()))
    if kl_ppl:
        status.update(kl=float(kl_ppl[-1][0]), ppl=float(kl_ppl[-1][1]))
    (folder / 'status.json').write_text(json.dumps(status, indent=2))
    rows.append(status)
    print('END', json.dumps(status), flush=True)
    if result.returncode or not unchanged or not kl_ppl or not metric_path.exists():
        raise RuntimeError('run failed or incomplete; preserve artifacts and stop')

pair1 = (rows[0], rows[1])
pair2 = (rows[2], rows[3])
for pair in (pair1, pair2):
    assert pair[0]['state_sha256'] == pair[1]['state_sha256'], 'final weights differ'
    assert pair[0]['kl'] == pair[1]['kl'] and pair[0]['ppl'] == pair[1]['ppl'], 'quality differs'
    assert all(r['preview_events'] == 0 for r in pair), 'unrelated next-block preview was enabled'
    candidate = next(r for r in pair if r['mode'] == 'prefix-fast')
    control = next(r for r in pair if r['mode'] == 'baseline-profile')
    assert candidate['cache_events'] > 0 and control['cache_events'] == 0, 'cache activation mismatch'
summary = {'schedule': [r['label'] for r in rows], 'pairs': []}
for i, pair in enumerate((pair1, pair2), start=1):
    candidate = next(r for r in pair if r['mode'] == 'prefix-fast')
    control = next(r for r in pair if r['mode'] == 'baseline-profile')
    summary['pairs'].append({
        'pair': i, 'candidate_first': pair[0]['mode'] == 'prefix-fast',
        'final_state_equal': candidate['state_sha256'] == control['state_sha256'],
        'exact_quality_equal': (candidate['kl'], candidate['ppl']) == (control['kl'], control['ppl']),
        'control_quantization_seconds': control['quantization_seconds'],
        'candidate_quantization_seconds': candidate['quantization_seconds'],
        'quantization_time_reduction_percent': 100 * (1 - candidate['quantization_seconds'] / control['quantization_seconds']),
        'control_wall_seconds': control['wall_seconds'], 'candidate_wall_seconds': candidate['wall_seconds'],
        'process_time_reduction_percent': 100 * (1 - candidate['wall_seconds'] / control['wall_seconds']),
        'peak_allocated_delta_mib': (candidate['peak_allocated'] - control['peak_allocated']) / 2**20,
        'kl': candidate['kl'], 'ppl': candidate['ppl'],
    })
(root / 'paired_summary.json').write_text(json.dumps(summary, indent=2))
print('SUMMARY', json.dumps(summary, indent=2), flush=True)
