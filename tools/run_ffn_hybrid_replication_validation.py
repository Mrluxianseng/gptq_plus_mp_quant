"""Two matched full-W4 replications: new calibration seed and held-out test windows."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path('/mnt/d/gptq_plus_realq')
BASE_MANIFEST = ROOT / 'outputs/resume_20260927_1216/baseline-profile/manifest.json'
PROBE = ROOT / 'tools/overnight_ptq_probe.py'
SOURCES = (
    'gptq_utils/gptq_plus_utils.py', 'process_args.py', 'ptq.py',
    'utils/data_utils.py', 'tools/overnight_ptq_probe.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    root = ROOT / 'outputs/resume_20260927_ffn_hybrid_validation'
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    template = json.loads(BASE_MANIFEST.read_text())['command']
    entry_index = next(i for i, item in enumerate(template) if item.endswith('/entry.py'))
    definitions = (
        ('seed43', 43, 0),       # independent calibration samples, original test windows
        ('heldout_windows', 42, 8),  # original calibration samples, next eight test windows
    )
    all_hashes = {path: digest(path) for path in SOURCES}
    runs = {}

    for pair_name, calibration_seed, eval_start in definitions:
        for treatment in (False, True):
            label = f'{pair_name}_' + ('candidate' if treatment else 'control')
            folder = root / label
            folder.mkdir()
            entry = folder / 'entry.py'
            shutil.copyfile(PROBE, entry)
            command = list(template)
            command[entry_index] = str(entry)
            command[entry_index + 1] = 'baseline-profile'
            set_value(command, '--exp', f'resume_20260927_ffn_verify_{label}')
            set_value(command, '--seed', calibration_seed)
            set_value(command, '--eval_samples', 8)
            set_value(command, '--eval_start_sample', eval_start)
            if treatment:
                command.append('--ffn_nonlinear_hybrid_refresh')

            hashes = dict(all_hashes)
            hashes[str(entry)] = digest(entry)
            env = os.environ.copy()
            env.update(
                HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                RESEARCH_RUN_DIR=str(folder),
            )
            (folder / 'manifest.json').write_text(json.dumps({
                'command': command,
                'source_sha256': hashes,
                'treatment': treatment,
                'calibration_seed': calibration_seed,
                'eval_start_sample': eval_start,
                'eval_samples': 8,
                'full_w4': True,
            }, indent=2))
            started = time.monotonic()
            print('START', label, time.strftime('%F %T'), flush=True)
            with (folder / 'run.log').open('w') as log:
                proc = subprocess.run(
                    ['timeout', '--signal=TERM', '--kill-after=30s', '3600', *command],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
            log_text = (folder / 'run.log').read_text(errors='replace')
            unchanged = all(digest(path) == value for path, value in hashes.items())
            metrics = json.loads((folder / 'baseline-profile_metrics.json').read_text()) if proc.returncode == 0 else {}
            match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log_text)
            status = {
                'returncode': proc.returncode,
                'wall_seconds': time.monotonic() - started,
                'sources_unchanged': unchanged,
                'ffn_refresh_events': log_text.count('FFN_NONLINEAR_REFRESH'),
                'kl': float(match.group(1)) if match else None,
                'ppl': float(match.group(2)) if match else None,
                'quantization_seconds': metrics.get('quantization_seconds'),
                'state_sha256': metrics.get('state_sha256'),
                'peak_allocated': metrics.get('peak_allocated'),
                'finished': time.strftime('%F %T'),
            }
            (folder / 'status.json').write_text(json.dumps(status, indent=2))
            runs[label] = status
            print('END', label, json.dumps(status), flush=True)
            if proc.returncode or not unchanged or not match:
                (root / 'partial_results.json').write_text(json.dumps(runs, indent=2))
                raise RuntimeError(f'{label} failed, mutated code, or lacks exact quality metrics')

    pairs = {}
    for pair_name, _, eval_start in definitions:
        control = runs[f'{pair_name}_control']
        candidate = runs[f'{pair_name}_candidate']
        pairs[pair_name] = {
            'control': control,
            'candidate': candidate,
            'delta_percent': {
                'kl': (candidate['kl'] / control['kl'] - 1) * 100,
                'ppl': (candidate['ppl'] / control['ppl'] - 1) * 100,
                'quantization_time': (
                    candidate['quantization_seconds'] / control['quantization_seconds'] - 1
                ) * 100,
            },
            'eval_window_indices': list(range(eval_start, eval_start + 8)),
        }
    result = {
        'pairs': pairs,
        'scope': (
            'Each pair is matched on all 28 W4 layers, 32x256 calibration examples, '
            'rotation seed, refresh seed, and the same eight 1024-token windows within '
            'that pair. seed43 changes only the calibration seed. heldout_windows '
            'keeps calibration seed 42 and evaluates windows 8-15 instead of 0-7.'
        ),
    }
    (root / 'quality_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
