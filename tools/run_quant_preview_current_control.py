"""Run the no-preview full-model control with the current source tree."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/mnt/d/gptq_plus_realq')
CONTROL = ROOT / 'outputs/ffn_hybrid_factorial_audit_20260928/seed44_windows0_control'
PROBE = ROOT / 'tools/overnight_ptq_probe.py'
SOURCES = (
    'gptq_utils/gptq_plus_utils.py', 'process_args.py', 'ptq.py',
    'utils/data_utils.py', 'tools/overnight_ptq_probe.py',
    'tools/run_quant_preview_current_control.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder = ROOT / 'outputs/quant_preview_current_control_20260928'
    folder.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    command = list(json.loads((CONTROL / 'manifest.json').read_text())['command'])
    entry_index = next(i for i, value in enumerate(command) if value.endswith('/entry.py'))
    entry = folder / 'entry.py'
    shutil.copyfile(PROBE, entry)
    command[entry_index] = str(entry)
    command[entry_index + 1] = 'baseline-profile'
    command[command.index('--exp') + 1] = 'quant_preview_current_control_seed44_windows0'
    command = [value for value in command if value != '--preview_next_quant_block']
    hashes = {source: digest(ROOT / source) for source in SOURCES}
    hashes[str(entry)] = digest(entry)
    (folder / 'manifest.json').write_text(json.dumps({
        'command': command,
        'source_sha256': hashes,
        'candidate': False,
        'control': 'current source tree; preview flag disabled',
        'calibration_seed': 44,
        'calibration_samples': 32,
        'eval_start_sample': 0,
        'eval_samples': 8,
        'memory_mode': 'no input offload',
        'reference_control_folder': str(CONTROL),
        'reference_control_source_sha256': json.loads((CONTROL / 'manifest.json').read_text())['source_sha256'],
    }, indent=2))
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', RESEARCH_RUN_DIR=str(folder),
    )
    started = time.monotonic()
    print('START current_source_control', time.strftime('%F %T'), flush=True)
    with (folder / 'run.log').open('w') as log:
        proc = subprocess.run(
            ['timeout', '--signal=TERM', '--kill-after=30s', '3600', *command],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    text = (folder / 'run.log').read_text(errors='replace')
    match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', text)
    metrics_path = folder / 'baseline-profile_metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    unchanged = all(digest(ROOT / source) == hashes[source] for source in SOURCES)
    result = {
        'returncode': proc.returncode,
        'wall_seconds': time.monotonic() - started,
        'sources_unchanged': unchanged,
        'kl': float(match.group(1)) if match else None,
        'ppl': float(match.group(2)) if match else None,
        'quantization_seconds': metrics.get('quantization_seconds'),
        'state_sha256': metrics.get('state_sha256'),
        'peak_allocated': metrics.get('peak_allocated'),
        'peak_reserved': metrics.get('peak_reserved'),
        'finished': time.strftime('%F %T'),
    }
    (folder / 'status.json').write_text(json.dumps(result, indent=2))
    print('END current_source_control', json.dumps(result), flush=True)
    if proc.returncode or not unchanged or not match:
        raise RuntimeError('Current-source control failed, mutated source, or lacks exact metrics')


if __name__ == '__main__':
    main()
