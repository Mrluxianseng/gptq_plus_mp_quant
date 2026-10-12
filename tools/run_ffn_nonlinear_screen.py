"""Smoke and matched 8-window screen for selective nonlinear FFN refresh.

The treatment switches only the Block-GD refresh objective for Qwen/Llama-style
mlp.gate_proj, mlp.up_proj and mlp.down_proj. Attention projections retain the
existing Fisher objective. The reference arm is the already-completed matched
plain-g1 arm in outputs/resume_20260927_ts_quality-screen8.
"""
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
REFERENCE = ROOT / 'outputs/resume_20260927_ts_quality-screen8/plain-g1'
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/resume_20260927_ffn_nonlinear8')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--summarize-existing', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    root = (ROOT / args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    command_template = json.loads((REFERENCE / 'manifest.json').read_text())['command']
    entry_index = next(i for i, item in enumerate(command_template) if item.endswith('/entry.py'))

    def run(label, smoke=False):
        folder = root / label
        if folder.exists():
            raise FileExistsError(f'run directory already exists: {folder}')
        folder.mkdir()
        entry = folder / 'entry.py'
        shutil.copyfile(PROBE, entry)
        command = list(command_template)
        command[entry_index] = str(entry)
        command[entry_index + 1] = 'baseline-profile'
        set_value(command, '--exp', f'resume_20260927_ffn_nonlinear_{label}')
        set_value(command, '--quant_stop_layer', 1 if smoke else 26)
        set_value(command, '--eval_samples', 0 if smoke else 8)
        command.append('--ffn_nonlinear_refresh')
        if smoke:
            set_value(command, '--nsamples', 4)
            set_value(command, '--backward_samples', 4)
            command.append('--skip_eval')
        hashes = {path: digest(path) for path in SOURCES}
        hashes[str(entry)] = digest(entry)
        env = os.environ.copy()
        env.update(
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            RESEARCH_RUN_DIR=str(folder),
        )
        manifest = {
            'command': command,
            'source_sha256': hashes,
            'reference_manifest': str(REFERENCE / 'manifest.json'),
            'smoke': smoke,
        }
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
            'ffn_refresh_events': (folder / 'run.log').read_text(errors='replace').count('FFN_NONLINEAR_REFRESH'),
            'finished': time.strftime('%F %T'),
        }
        (folder / 'status.json').write_text(json.dumps(status, indent=2))
        print('END', label, json.dumps(status), flush=True)
        if result.returncode or not unchanged:
            raise RuntimeError(f'{label} failed or source changed; inspect log before any quality claim')
        return folder

    if args.smoke_only:
        run('smoke', smoke=True)
        print('SMOKE_OK; no quality claim made', flush=True)
        return

    folder = root / 'candidate' if args.summarize_existing else run('candidate')
    log = (folder / 'run.log').read_text(errors='replace')
    match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log)
    if not match:
        raise RuntimeError('candidate exited successfully but has no exact WikiText2 KL/PPL metrics')
    control = json.loads((REFERENCE.parent / 'quality_results.json').read_text())['arms']['plain-g1']
    status = json.loads((folder / 'status.json').read_text())
    candidate = {
        **status,
        'kl': float(match.group(1)),
        'ppl': float(match.group(2)),
        'quantization_seconds': json.loads((folder / 'baseline-profile_metrics.json').read_text())['quantization_seconds'],
        'state_sha256': json.loads((folder / 'baseline-profile_metrics.json').read_text())['state_sha256'],
    }
    results = {
        'control_plain_g1': control,
        'candidate_ffn_nonlinear': candidate,
        'delta_percent': {
            'kl': (candidate['kl'] / control['kl'] - 1.0) * 100.0,
            'ppl': (candidate['ppl'] / control['ppl'] - 1.0) * 100.0,
            'quantization_time': (
                candidate['quantization_seconds'] / control['quantization_seconds'] - 1.0
            ) * 100.0,
        },
        'scope': (
            'Matched Qwen3-0.6B, W4 layers 0-26, layer 27 FP; 32 calibration '
            'samples, seed and first 8 fixed 1024-token WikiText2 test windows. '
            'The only treatment is --ffn_nonlinear_refresh. Partial-W4 directional '
            'screen, not a full-test/full-W4 claim.'
        ),
    }
    (root / 'quality_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
