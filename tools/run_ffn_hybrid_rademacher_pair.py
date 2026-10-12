"""Matched full-W4 test of tokenwise Rademacher Fisher vs summed-token Fisher."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path('/mnt/d/gptq_plus_realq')
TEMPLATE_MANIFEST = ROOT / 'outputs/ffn_hybrid_factorial_audit_20260928/seed44_windows0_control/manifest.json'
PROBE = ROOT / 'tools/overnight_ptq_probe.py'
SOURCES = (
    'gptq_utils/gptq_plus_utils.py', 'process_args.py', 'ptq.py',
    'utils/data_utils.py', 'tools/overnight_ptq_probe.py',
    'tools/run_ffn_hybrid_rademacher_pair.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    root = ROOT / 'outputs/ffn_hybrid_rademacher_pair_20260928'
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    template = json.loads(TEMPLATE_MANIFEST.read_text())['command']
    entry_index = next(i for i, item in enumerate(template) if item.endswith('/entry.py'))
    sources = {source: digest(ROOT / source) for source in SOURCES}
    results = {}

    # Same calibration seed and held-out windows as the completed seed44/windows0 pair.
    # Only the Fisher estimator changes from signed-sum k=0 to token-signed k=4;
    # within each pair, the hybrid switch is the only treatment difference.
    for candidate in (False, True):
        label = 'hybrid' if candidate else 'fisher_control'
        folder = root / label
        folder.mkdir()
        entry = folder / 'entry.py'
        shutil.copyfile(PROBE, entry)
        command = list(template)
        command[entry_index] = str(entry)
        command[entry_index + 1] = 'baseline-profile'
        set_value(command, '--exp', f'ffn_hybrid_rademacher4_seed44_windows0_{label}')
        set_value(command, '--seed', 44)
        set_value(command, '--nsamples', 32)
        set_value(command, '--backward_samples', 32)
        set_value(command, '--eval_samples', 8)
        set_value(command, '--eval_start_sample', 0)
        set_value(command, '--fisher_rademacher_k', 4)
        if candidate:
            command.append('--ffn_nonlinear_hybrid_refresh')
        hashes = dict(sources)
        hashes[str(entry)] = digest(entry)
        (folder / 'manifest.json').write_text(json.dumps({
            'command': command,
            'source_sha256': hashes,
            'candidate': candidate,
            'fisher_rademacher_k': 4,
            'calibration_seed': 44,
            'calibration_samples': 32,
            'eval_start_sample': 0,
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
            'rademacher_log_confirmed': 'Rademacher token signs: k=4' in log_text,
            'hybrid_refresh_events': log_text.count('FFN_NONLINEAR_REFRESH'),
            'kl': float(match.group(1)) if match else None,
            'ppl': float(match.group(2)) if match else None,
            'quantization_seconds': metrics.get('quantization_seconds'),
            'state_sha256': metrics.get('state_sha256'),
            'peak_allocated': metrics.get('peak_allocated'),
            'finished': time.strftime('%F %T'),
        }
        (folder / 'status.json').write_text(json.dumps(status, indent=2))
        results[label] = status
        print('END', label, json.dumps(status), flush=True)
        if proc.returncode or not unchanged or not match or not status['rademacher_log_confirmed']:
            (root / 'partial_results.json').write_text(json.dumps(results, indent=2))
            raise RuntimeError(f'{label} failed validation or lacks exact metrics')

    base, hybrid = results['fisher_control'], results['hybrid']
    results['delta_percent'] = {
        'kl': (hybrid['kl'] / base['kl'] - 1) * 100,
        'ppl': (hybrid['ppl'] / base['ppl'] - 1) * 100,
        'quantization_time': (
            hybrid['quantization_seconds'] / base['quantization_seconds'] - 1
        ) * 100 if hybrid['quantization_seconds'] and base['quantization_seconds'] else None,
    }
    results['scope'] = (
        'Matched full-W4 pair on calibration seed 44 and WikiText2 eval windows 0-7. '
        'Both arms use fisher_rademacher_k=4; only hybrid enables the FFN nonlinear loss.'
    )
    (root / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
