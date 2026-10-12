"""Third matched full-W4 FFN hybrid replication on new calibration/eval data."""
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
    'tools/run_ffn_hybrid_third_replication.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    root = ROOT / 'outputs/ffn_hybrid_third_replication_20260928'
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    template = json.loads(BASE_MANIFEST.read_text())['command']
    entry_index = next(i for i, item in enumerate(template) if item.endswith('/entry.py'))
    sources = {source: digest(ROOT / source) for source in SOURCES}
    runs = {}

    for candidate in (False, True):
        label = 'candidate' if candidate else 'control'
        folder = root / label
        folder.mkdir()
        entry = folder / 'entry.py'
        shutil.copyfile(PROBE, entry)
        command = list(template)
        command[entry_index] = str(entry)
        command[entry_index + 1] = 'baseline-profile'
        set_value(command, '--exp', f'ffn_hybrid_third_{label}')
        set_value(command, '--seed', 44)
        set_value(command, '--nsamples', 32)
        set_value(command, '--backward_samples', 32)
        set_value(command, '--eval_samples', 8)
        set_value(command, '--eval_start_sample', 16)
        if candidate:
            command.append('--ffn_nonlinear_hybrid_refresh')
        hashes = dict(sources)
        hashes[str(entry)] = digest(entry)
        (folder / 'manifest.json').write_text(json.dumps({
            'command': command,
            'source_sha256': hashes,
            'treatment': 'ffn_nonlinear_hybrid_refresh' if candidate else 'REALQ full-Fisher refresh',
            'calibration_seed': 44,
            'calibration_samples': 32,
            'eval_start_sample': 16,
            'eval_samples': 8,
            'full_w4': True,
        }, indent=2))
        env = os.environ.copy()
        env.update(
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            RESEARCH_RUN_DIR=str(folder),
        )
        started = time.monotonic()
        print('START', label, time.strftime('%F %T'), flush=True)
        with (folder / 'run.log').open('w') as log:
            proc = subprocess.run(
                ['timeout', '--signal=TERM', '--kill-after=30s', '3600', *command],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        log_text = (folder / 'run.log').read_text(errors='replace')
        unchanged = all(digest(ROOT / source) == value for source, value in hashes.items())
        metric_path = folder / 'baseline-profile_metrics.json'
        metrics = json.loads(metric_path.read_text()) if metric_path.exists() else {}
        match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log_text)
        status = {
            'returncode': proc.returncode,
            'wall_seconds': time.monotonic() - started,
            'sources_unchanged': unchanged,
            'hybrid_refresh_events': log_text.count('FFN_NONLINEAR_REFRESH'),
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

    control, candidate = runs['control'], runs['candidate']
    result = {
        'control': control,
        'candidate': candidate,
        'delta_percent': {
            'kl': (candidate['kl'] / control['kl'] - 1) * 100,
            'ppl': (candidate['ppl'] / control['ppl'] - 1) * 100,
            'quantization_time': (
                candidate['quantization_seconds'] / control['quantization_seconds'] - 1
            ) * 100 if candidate['quantization_seconds'] and control['quantization_seconds'] else None,
        },
        'scope': 'Same full-W4 28-layer pair; seed 44, 32 calibration samples, WikiText2 windows 16-23.',
    }
    (root / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
