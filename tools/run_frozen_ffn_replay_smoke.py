"""Paired 5-layer correctness smoke for generic FFN-prefix replay."""
import hashlib
import json
import os
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'gptq_utils/frozen_ffn_replay.py', 'gptq_utils/frozen_prefix_replay.py',
    'gptq_utils/gptq_plus_utils.py', 'process_args.py', 'ptq.py',
    'utils/data_utils.py', 'utils/eval_utils.py', 'tools/frozen_ffn_replay_probe.py',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['smoke', 'full'], default='smoke')
    scope = parser.parse_args().scope
    run_name = 'full_replication' if scope == 'full' else 'smoke'
    root = ROOT / f'outputs/frozen_ffn_replay_{run_name}_20260929'
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / 'outputs/ffn_hybrid_factorial_audit_20260928/seed44_windows0_control/manifest.json'
    template = json.loads(manifest_path.read_text())['command']
    entry_index = next(i for i, v in enumerate(template) if v.endswith('/entry.py'))
    del template[entry_index + 1]  # source wrapper has no legacy positional mode
    source_hashes = {s: digest(ROOT / s) for s in SOURCES}
    snapshots = root / 'source'; snapshots.mkdir()
    for source in SOURCES:
        dest = snapshots / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, dest)

    results = []
    cases = (('candidate_verify', True, True), ('control', False, False)) if scope == 'smoke' else (
        ('control', False, False), ('candidate', True, False))
    for arm, enabled, verify in cases:
        folder = root / arm; folder.mkdir()
        command = list(template)
        command[entry_index] = str(ROOT / 'tools/frozen_ffn_replay_probe.py')
        command[6] = '--rdzv_endpoint=localhost:29540'
        for flag, value in {
            '--exp': 'frozen_ffn_' + arm, '--seed': 42,
            '--nsamples': 32, '--backward_samples': 32,
            '--eval_samples': 8 if scope == 'smoke' else 0, '--eval_start_sample': 0,
            '--sequence_fisher_mode': 'off',
        }.items():
            set_value(command, flag, value)
        if scope == 'smoke':
            set_value(command, '--quant_stop_layer', 4)
        manifest = {'arm': arm, 'command': command, 'source_sha256': source_hashes,
                    'cache_enabled': enabled, 'verify_every_hit': verify,
                    'scope': 'REAL-Q W4; ' + ('layers 0-4' if scope == 'smoke' else 'all layers')
                             + '; cache changes execution only'}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', RESEARCH_RUN_DIR=str(folder),
                   FROZEN_FFN_REPLAY='1' if enabled else '0',
                   FROZEN_FFN_VERIFY='1' if verify else '0')
        print('START', arm, time.strftime('%F %T'), flush=True)
        started = time.monotonic()
        with (folder / 'run.log').open('w') as stream:
            proc = subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s', '1800', *command],
                                  env=env, stdout=stream, stderr=subprocess.STDOUT)
        log = (folder / 'run.log').read_text(errors='replace')
        match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log)
        metrics_path = folder / 'metrics.json'
        unchanged = all(digest(ROOT / s) == h for s, h in source_hashes.items())
        row = {'arm': arm, 'returncode': proc.returncode, 'wall_seconds': time.monotonic() - started,
               'unchanged': unchanged, 'kl': float(match[1]) if match else None,
               'ppl': float(match[2]) if match else None,
               'metrics': json.loads(metrics_path.read_text()) if metrics_path.exists() else None}
        (folder / 'status.json').write_text(json.dumps(row, indent=2)); results.append(row)
        (root / 'results.json').write_text(json.dumps(results, indent=2))
        print('END', json.dumps(row), flush=True)
        if proc.returncode or not match or not unchanged or not metrics_path.exists():
            raise RuntimeError(f'incomplete arm: {arm}')

    if scope == 'smoke':
        candidate = next(r for r in results if r['arm'] == 'candidate_verify')
        control = next(r for r in results if r['arm'] == 'control')
    else:
        candidate = next(r for r in results if r['arm'] == 'candidate')
        control = next(r for r in results if r['arm'] == 'control')
    cstatic = json.loads((root / candidate['arm'] / 'static_stats.json').read_text())
    bstatic = json.loads((root / 'control' / 'static_stats.json').read_text())
    checks = {
        'cache_hits_and_misses_present': (candidate['metrics']['cache_hits'] > 0
                                          and candidate['metrics']['cache_misses'] > 0),
        'cache_within_128_mib': candidate['metrics']['cache_peak_bytes'] <= 128 * 1024 * 1024,
        'final_state_equal': candidate['metrics']['state_sha256'] == control['metrics']['state_sha256'],
        'quality_equal': candidate['kl'] == control['kl'] and candidate['ppl'] == control['ppl'],
        'fisher_and_saliency_equal': cstatic == bstatic,
    }
    if scope == 'smoke':
        checks['verified_cache_hits_present'] = candidate['metrics']['cache_verified_hits'] > 0
    summary = {'scope': scope,
               'candidate_full_forward_verified_hits': candidate['metrics']['cache_verified_hits'],
               'candidate_cache_hits': candidate['metrics']['cache_hits'],
               'candidate_cache_misses': candidate['metrics']['cache_misses'],
               'candidate_cache_peak_bytes': candidate['metrics']['cache_peak_bytes'],
               'checks': checks, 'passed': all(checks.values()),
               'final_state_equal': checks['final_state_equal'],
               'quality_equal': checks['quality_equal'],
               'kl': candidate['kl'], 'ppl': candidate['ppl'],
               'fisher_and_saliency_equal': checks['fisher_and_saliency_equal'],
               'quantization_time_reduction_percent': 100 * (1 - candidate['metrics']['quantization_seconds'] /
                                                               control['metrics']['quantization_seconds'])}
    (root / 'audited_summary.json').write_text(json.dumps(summary, indent=2))
    print('AUDIT PASS' if summary['passed'] else 'AUDIT FAIL', json.dumps(summary), flush=True)
    if not summary['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
