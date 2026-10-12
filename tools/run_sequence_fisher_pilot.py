"""Frozen three-arm pilot; no parameter search. Run inside the existing WSL venv."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('gptq_utils/gptq_plus_utils.py', 'gptq_utils/sequence_fisher.py',
           'process_args.py', 'ptq.py', 'utils/data_utils.py', 'utils/eval_utils.py',
           'tools/sequence_fisher_probe.py', 'tools/run_sequence_fisher_pilot.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['noop', 'pilot', 'replication', 'cross_test'], required=True)
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    os.chdir(ROOT)
    root = Path(options.output).resolve(); root.mkdir(parents=True, exist_ok=False)
    template = json.loads((ROOT / 'outputs/ffn_hybrid_factorial_audit_20260928/seed44_windows0_control/manifest.json').read_text())['command']
    entry_index = next(i for i, v in enumerate(template) if v.endswith('/entry.py'))
    del template[entry_index + 1]  # legacy probe's positional mode
    source_hashes = {s: digest(ROOT / s) for s in SOURCES}
    snapshots = root / 'source'; snapshots.mkdir()
    for s in SOURCES:
        destination = snapshots / s; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / s, destination)
    cases = [(42, 'off', 0), (42, 'sequence', 0)] if options.stage == 'noop' else (
        [(42, 'off', .5), (42, 'token', .5), (42, 'sequence', .5),
         (44, 'sequence', .5), (44, 'token', .5), (44, 'off', .5)]
        if options.stage == 'pilot' else
        ([(46, 'off', 0), (46, 'cross', 1),
          (48, 'cross', 1), (48, 'off', 0),
          (50, 'off', 0), (50, 'cross', 1)] if options.stage == 'cross_test' else
         [(46, 'off', .5), (46, 'token', .5), (46, 'sequence', .5),
          (48, 'sequence', .5), (48, 'off', .5), (48, 'token', .5),
          (50, 'token', .5), (50, 'sequence', .5), (50, 'off', .5)]))
    results = []
    for seed, mode, mix in cases:
        folder = root / f'seed{seed}_{mode}'; folder.mkdir()
        command = list(template); command[entry_index] = str(ROOT / 'tools/sequence_fisher_probe.py')
        command[6] = '--rdzv_endpoint=localhost:29538'
        for flag, value in {
            '--exp': f'seqf_{root.name}_{seed}_{mode}', '--seed': seed,
            '--nsamples': 4 if options.stage == 'noop' else 32,
            '--backward_samples': 4 if options.stage == 'noop' else 32,
            '--quant_stop_layer': 0 if options.stage == 'noop' else 4,
            '--eval_samples': 2 if options.stage == 'noop' else 8,
            '--sequence_fisher_mode': mode, '--sequence_fisher_mix': mix,
            '--sequence_fisher_probes': 4,
        }.items():
            set_value(command, flag, value)
        manifest = dict(command=command, source_sha256=source_hashes, seed=seed, mode=mode, mix=mix,
                        scope='prefix-only W4; remaining layers FP; FFN loss only')
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        assert all(digest(ROOT / s) == h for s, h in source_hashes.items())
        env = os.environ.copy(); env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            RESEARCH_RUN_DIR=str(folder), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        print('START', seed, mode, time.strftime('%F %T'), flush=True)
        started = time.monotonic()
        with (folder / 'run.log').open('w') as stream:
            proc = subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s', '1800', *command],
                                  env=env, stdout=stream, stderr=subprocess.STDOUT)
        log = (folder / 'run.log').read_text(errors='replace')
        match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log)
        metric_file = folder / 'metrics.json'
        row = dict(seed=seed, mode=mode, mix=mix, returncode=proc.returncode,
                   wall_seconds=time.monotonic() - started,
                   unchanged=all(digest(ROOT / s) == h for s, h in source_hashes.items()),
                   kl=float(match[1]) if match else None, ppl=float(match[2]) if match else None,
                   refresh_events=log.count('SEQUENCE_FISHER_REFRESH'),
                   metrics=json.loads(metric_file.read_text()) if metric_file.exists() else {})
        (folder / 'status.json').write_text(json.dumps(row, indent=2)); results.append(row)
        (root / 'results.json').write_text(json.dumps(results, indent=2))
        print('END', json.dumps(row), flush=True)
        if proc.returncode or not match or not row['unchanged']:
            raise RuntimeError(f'Invalid run: {folder}')
    for seed in set(row['seed'] for row in results):
        paths = [root / f"seed{seed}_{r['mode']}" for r in results if r['seed'] == seed]
        stats = [json.loads((p / 'static_stats.json').read_text()) for p in paths]
        assert len({s['saliency'] for s in stats}) == 1, 'Original saliency changed!'
        assert len({s['fisher'] for s in stats}) == 1, 'Original Fisher changed!'
        for path in paths:
            windows = json.loads((path / 'eval_window_kl.json').read_text())
            assert len(windows) == (2 if options.stage == 'noop' else 8), 'Missing per-window KL'
    if options.stage == 'noop':
        assert results[0]['metrics']['state_sha256'] == results[1]['metrics']['state_sha256'], 'No-op changed model!'
        assert results[0]['kl'] == results[1]['kl'] and results[0]['ppl'] == results[1]['ppl']
    print('AUDIT PASS: original statistics match across arms' + ('; no-op is bitwise identical' if options.stage == 'noop' else ''), flush=True)


if __name__ == '__main__':
    main()
