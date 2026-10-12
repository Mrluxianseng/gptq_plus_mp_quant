"""Matched 8-window quality check for REAL-Q's first-order refined surrogate.

The candidate is the existing refined_mse path: Fisher second-order term plus
the collected end-to-end first-order KL term. This is one mechanism test, not
a hyperparameter/configuration sweep. A short smoke can be selected separately.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path('/mnt/d/gptq_plus_realq')
BASE_MANIFEST = ROOT / 'outputs/resume_20260927_1216/baseline-profile/manifest.json'
PROBE = ROOT / 'tools/overnight_ptq_probe.py'
SOURCES = (
    'gptq_utils/gptq_plus_utils.py',
    'process_args.py',
    'ptq.py',
    'utils/data_utils.py',
    'tools/overnight_ptq_probe.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def set_switch(command, flag, enabled):
    while flag in command:
        command.remove(flag)
    if enabled:
        command.append(flag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/resume_20260927_refined_mse_quality8')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    root = (ROOT / args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.smoke_only and any((root / name).exists() for name in ('control-fisher', 'candidate-refined-mse')):
        raise FileExistsError(f'quality-arm directory already exists under {root}; choose a fresh --out')
    base = json.loads(BASE_MANIFEST.read_text())['command']
    entry_index = next(i for i, item in enumerate(base) if item.endswith('/entry.py'))

    def run(label, treatment, smoke=False):
        folder = root / label
        folder.mkdir()
        entry = folder / 'entry.py'
        shutil.copyfile(PROBE, entry)
        command = list(base)
        command[entry_index] = str(entry)
        command[entry_index + 1] = 'baseline-profile'
        set_value(command, '--exp', f'resume_20260927_refined_mse_{label}')
        set_value(command, '--num_groups', 4)
        set_value(command, '--group_parallel_quant', 'none')
        set_value(command, '--grad_refresh_loss', 'refined_mse' if treatment else 'fisher_diag_mse')
        set_value(command, '--num_samples_for_refined_mse', 4 if smoke else 32)
        set_value(command, '--quant_stop_layer', 3 if smoke else 26)
        set_value(command, '--eval_samples', 0 if smoke else 8)
        if smoke:
            set_value(command, '--nsamples', 4)
            set_value(command, '--backward_samples', 4)
            set_value(command, '--backward_bsz', 2)
            set_value(command, '--global_loss_bsz', 1)
            set_switch(command, '--skip_eval', True)
        hashes = {path: digest(path) for path in SOURCES}
        env = os.environ.copy()
        env.update(
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            RESEARCH_RUN_DIR=str(folder),
        )
        manifest = {'command': command, 'hashes': hashes, 'treatment': treatment, 'smoke': smoke}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        started = time.monotonic()
        print('START', label, time.strftime('%F %T'), flush=True)
        with (folder / 'run.log').open('w') as log:
            result = subprocess.run(
                ['timeout', '--signal=TERM', '--kill-after=30s', '3600', *command],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        unchanged = all(digest(path) == value for path, value in hashes.items())
        status = {
            'returncode': result.returncode,
            'wall_seconds': time.monotonic() - started,
            'sources_unchanged': unchanged,
            'finished': time.strftime('%F %T'),
        }
        (folder / 'status.json').write_text(json.dumps(status, indent=2))
        print('END', label, json.dumps(status), flush=True)
        if result.returncode or not unchanged:
            raise RuntimeError(f'{label} failed or source changed; inspect log before any quality claim')
        return folder

    if args.smoke_only:
        run('smoke-refined-mse', True, smoke=True)
        print('SMOKE_OK; no quality claim made', flush=True)
        return

    run('control-fisher', False)
    run('candidate-refined-mse', True)

    import re
    results = {}
    for label in ('control-fisher', 'candidate-refined-mse'):
        folder = root / label
        log = (folder / 'run.log').read_text(errors='replace')
        status = json.loads((folder / 'status.json').read_text())
        match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log)
        if not match:
            raise RuntimeError(f'{label}: successful exit but missing exact quality metrics')
        results[label] = {
            **status,
            'kl': float(match.group(1)),
            'ppl': float(match.group(2)),
            'state_sha256': json.loads((folder / 'baseline-profile_metrics.json').read_text()).get('state_sha256'),
        }
    results['scope'] = (
        'Matched Qwen3-0.6B; W4 layers 0-26, layer 27 FP; same baseline command, '
        '32 calibration samples, seed and 8 fixed 1024-token WikiText2 windows. '
        'Only grad_refresh_loss changes from fisher_diag_mse to refined_mse. '
        'Directional partial-W4 screen, not a full-test/full-W4 result.'
    )
    (root / 'quality_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
