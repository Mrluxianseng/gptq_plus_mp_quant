"""Full 28-layer paired test of off-diagonal-only down_proj refresh Fisher."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path('/mnt/d/gptq_plus_realq')
TEMPLATE = ROOT / 'outputs/resume_20260927_1216/cross-control/manifest.json'
ENTRY_SOURCE = ROOT / 'tools/overnight_ptq_probe.py'
SOURCES = (
    'gptq_utils/gptq_plus_utils.py',
    'process_args.py',
    'ptq.py',
    'utils/data_utils.py',
    'tools/overnight_ptq_probe.py',
    'tools/run_offdiag_refresh_full_pair.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    root = ROOT / 'outputs/offdiag_refresh_full_pair_20260928'
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    template = json.loads(TEMPLATE.read_text())['command']
    entry_index = next(i for i, item in enumerate(template) if item.endswith('/entry.py'))
    # The template was a four-layer smoke. Remove that cap for a complete run.
    if '--quant_stop_layer' in template:
        index = template.index('--quant_stop_layer')
        del template[index:index + 2]
        if index < entry_index:
            entry_index -= 2
    source_hashes = {source: digest(ROOT / source) for source in SOURCES}
    results = {}

    for offdiag_only in (False, True):
        label = 'offdiag_only' if offdiag_only else 'full_fisher'
        folder = root / label
        folder.mkdir()
        entry = folder / 'entry.py'
        shutil.copyfile(ENTRY_SOURCE, entry)
        command = list(template)
        command[entry_index] = str(entry)
        command[entry_index + 1] = 'baseline-profile'
        set_value(command, '--exp', f'offdiag_refresh_full_{label}')
        set_value(command, '--seed', 42)
        set_value(command, '--nsamples', 32)
        set_value(command, '--backward_samples', 32)
        set_value(command, '--eval_samples', 8)
        set_value(command, '--eval_start_sample', 0)
        command.append('--cross_module_compensate')
        if offdiag_only:
            command.append('--refresh_fisher_offdiag_only')
        hashes = dict(source_hashes)
        hashes[str(entry)] = digest(entry)
        (folder / 'manifest.json').write_text(json.dumps({
            'command': command,
            'source_sha256': hashes,
            'cross_module_compensate': True,
            'refresh_fisher_offdiag_only': offdiag_only,
            'quantized_layers': 'all 28',
            'calibration_samples': 32,
            'calibration_seed': 42,
            'eval_samples': 8,
            'eval_start_sample': 0,
        }, indent=2))
        env = os.environ.copy()
        env.update(
            HF_HUB_OFFLINE='1',
            HF_DATASETS_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',
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
        unchanged = all(digest(ROOT / source) == value for source, value in source_hashes.items())
        match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log_text)
        metric_path = folder / 'baseline-profile_metrics.json'
        metrics = json.loads(metric_path.read_text()) if metric_path.exists() else {}
        status = {
            'returncode': proc.returncode,
            'wall_seconds': time.monotonic() - started,
            'sources_unchanged': unchanged,
            'offdiag_refresh_events': log_text.count('FISHER_OFFDIAG_ONLY'),
            'cross_compensation_events': log_text.count('cross_module_compensate:'),
            'kl': float(match.group(1)) if match else None,
            'ppl': float(match.group(2)) if match else None,
            'quantization_seconds': metrics.get('quantization_seconds'),
            'peak_allocated': metrics.get('peak_allocated'),
            'finished': time.strftime('%F %T'),
        }
        (folder / 'status.json').write_text(json.dumps(status, indent=2))
        results[label] = status
        print('END', label, json.dumps(status), flush=True)
        if proc.returncode or not unchanged or not match:
            (root / 'partial_results.json').write_text(json.dumps(results, indent=2))
            raise RuntimeError(f'{label} failed, mutated sources, or lacks quality metrics')

    control, candidate = results['full_fisher'], results['offdiag_only']
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
        'scope': 'Same full-W4 28-layer run, seed 42, 32 calibration samples, and WikiText2 windows 0-7.',
    }
    (root / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
