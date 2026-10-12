"""Matched full-W4 control/treatment pair for the fixed Fisher+nonlinear hybrid."""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/resume_20260927_ffn_hybrid_full8')
    parser.add_argument('--summarize-existing', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    root = (ROOT / args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    template = json.loads(BASE_MANIFEST.read_text())['command']
    entry_index = next(i for i, item in enumerate(template) if item.endswith('/entry.py'))

    def run(label, treatment):
        folder = root / label
        if folder.exists():
            raise FileExistsError(f'run directory already exists: {folder}')
        folder.mkdir()
        entry = folder / 'entry.py'
        shutil.copyfile(PROBE, entry)
        command = list(template)
        command[entry_index] = str(entry)
        command[entry_index + 1] = 'baseline-profile'
        set_value(command, '--exp', f'resume_20260927_ffn_hybrid_{label}')
        set_value(command, '--eval_samples', 8)
        if treatment:
            command.append('--ffn_nonlinear_hybrid_refresh')
        hashes = {path: digest(path) for path in SOURCES}
        hashes[str(entry)] = digest(entry)
        env = os.environ.copy()
        env.update(
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            RESEARCH_RUN_DIR=str(folder),
        )
        (folder / 'manifest.json').write_text(json.dumps({
            'command': command, 'source_sha256': hashes,
            'treatment': treatment, 'full_w4': True,
            'candidate': 'equal average of Fisher quadratic and trace-scaled exact layer-output MSE on FFN projections',
        }, indent=2))
        started = time.monotonic()
        print('START', label, time.strftime('%F %T'), flush=True)
        with (folder / 'run.log').open('w') as log:
            result = subprocess.run(
                ['timeout', '--signal=TERM', '--kill-after=30s', '3600', *command],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        log_text = (folder / 'run.log').read_text(errors='replace')
        unchanged = all(digest(path) == value for path, value in hashes.items())
        metrics = json.loads((folder / 'baseline-profile_metrics.json').read_text()) if result.returncode == 0 else {}
        match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log_text)
        status = {
            'returncode': result.returncode,
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
        print('END', label, json.dumps(status), flush=True)
        if result.returncode or not unchanged or not match:
            raise RuntimeError(f'{label} failed, mutated code, or lacks exact quality metrics')

    if not args.summarize_existing:
        run('control', False)
        run('candidate', True)
    control = json.loads((root / 'control/status.json').read_text())
    candidate = json.loads((root / 'candidate/status.json').read_text())
    assert control['returncode'] == candidate['returncode'] == 0
    result = {
        'control': control, 'candidate': candidate,
        'delta_percent': {
            'kl': (candidate['kl'] / control['kl'] - 1.0) * 100.0,
            'ppl': (candidate['ppl'] / control['ppl'] - 1.0) * 100.0,
            'quantization_time': (candidate['quantization_seconds'] / control['quantization_seconds'] - 1.0) * 100.0,
        },
        'scope': (
            'Matched full W4 Qwen3-0.6B (all 28 transformer layers), same 32x256 '
            'calibration set, seed, and first 8 fixed 1024-token WikiText2 test '
            'windows. Only --ffn_nonlinear_hybrid_refresh differs.'
        ),
    }
    (root / 'quality_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
