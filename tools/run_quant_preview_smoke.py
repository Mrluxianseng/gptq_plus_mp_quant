"""Matched four-layer smoke test for next-column-block quantization preview."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path('/mnt/d/gptq_plus_realq')
TEMPLATE = ROOT / 'outputs/ffn_hybrid_factorial_audit_20260928/seed44_windows0_control/manifest.json'
PROBE = ROOT / 'tools/overnight_ptq_probe.py'
SOURCES = (
    'gptq_utils/gptq_plus_utils.py', 'process_args.py', 'ptq.py',
    'utils/data_utils.py', 'tools/overnight_ptq_probe.py',
    'tools/run_quant_preview_smoke.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    calibration_samples = 32
    root = ROOT / 'outputs/quant_preview_smoke32_20260928'
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    template = json.loads(TEMPLATE.read_text())['command']
    entry_index = next(i for i, item in enumerate(template) if item.endswith('/entry.py'))
    sources = {source: digest(ROOT / source) for source in SOURCES}
    results = {}

    for candidate in (False, True):
        label = 'candidate' if candidate else 'control'
        folder = root / label
        folder.mkdir()
        entry = folder / 'entry.py'
        shutil.copyfile(PROBE, entry)
        command = list(template)
        command[entry_index] = str(entry)
        command[entry_index + 1] = 'baseline-profile'
        set_value(command, '--exp', f'quant_preview_smoke_{label}')
        set_value(command, '--seed', 44)
        set_value(command, '--nsamples', calibration_samples)
        set_value(command, '--backward_samples', calibration_samples)
        set_value(command, '--quant_stop_layer', 4)
        set_value(command, '--eval_samples', 8)
        if candidate:
            command.append('--preview_next_quant_block')
        hashes = dict(sources)
        hashes[str(entry)] = digest(entry)
        (folder / 'manifest.json').write_text(json.dumps({
            'command': command,
            'source_sha256': hashes,
            'candidate': candidate,
            'seed': 44,
            'quant_stop_layer': 4,
            'calibration_samples': calibration_samples,
            'eval_samples': 8,
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
                ['timeout', '--signal=TERM', '--kill-after=30s', '1800', *command],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        log_text = (folder / 'run.log').read_text(errors='replace')
        unchanged = all(digest(ROOT / source) == value for source, value in sources.items())
        metric_path = folder / 'baseline-profile_metrics.json'
        metrics = json.loads(metric_path.read_text()) if metric_path.exists() else {}
        match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log_text)
        status = {
            'returncode': proc.returncode,
            'wall_seconds': time.monotonic() - started,
            'sources_unchanged': unchanged,
            'preview_events': log_text.count('QUANT_PREVIEW_NEXT_BLOCK'),
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
        if proc.returncode or not unchanged or not match:
            (root / 'partial_results.json').write_text(json.dumps(results, indent=2))
            raise RuntimeError(f'{label} failed, mutated source, or lacks exact metrics')

    (root / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
