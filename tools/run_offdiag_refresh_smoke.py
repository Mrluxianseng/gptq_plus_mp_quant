"""Matched four-layer smoke test for zeroing the Adam-refresh Fisher diagonal."""
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
    'tools/run_offdiag_refresh_smoke.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    root = ROOT / 'outputs/offdiag_refresh_smoke_v2_20260928'
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    template = json.loads(TEMPLATE.read_text())['command']
    entry_index = next(i for i, item in enumerate(template) if item.endswith('/entry.py'))
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
        set_value(command, '--exp', f'offdiag_refresh_smoke_{label}')
        set_value(command, '--seed', 42)
        set_value(command, '--nsamples', 8)
        set_value(command, '--backward_samples', 8)
        set_value(command, '--quant_stop_layer', 4)
        set_value(command, '--eval_samples', 8)
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
            'quant_stop_layer': 4,
            'calibration_samples': 8,
            'eval_samples': 8,
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
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        log_text = (folder / 'run.log').read_text(errors='replace')
        unchanged = all(digest(ROOT / source) == value for source, value in source_hashes.items())
        match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log_text)
        status = {
            'returncode': proc.returncode,
            'wall_seconds': time.monotonic() - started,
            'sources_unchanged': unchanged,
            'offdiag_refresh_events': log_text.count('FISHER_OFFDIAG_ONLY'),
            'kl': float(match.group(1)) if match else None,
            'ppl': float(match.group(2)) if match else None,
            'finished': time.strftime('%F %T'),
        }
        (folder / 'status.json').write_text(json.dumps(status, indent=2))
        results[label] = status
        print('END', label, json.dumps(status), flush=True)
        if proc.returncode or not unchanged or not match:
            (root / 'partial_results.json').write_text(json.dumps(results, indent=2))
            raise RuntimeError(f'{label} failed, mutated sources, or lacks quality metrics')

    (root / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
