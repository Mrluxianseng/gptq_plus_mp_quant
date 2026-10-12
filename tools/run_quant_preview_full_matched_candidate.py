"""Full-model candidate with the exact control memory path (no input offload)."""
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
TEMPLATE = CONTROL / 'manifest.json'
PROBE = ROOT / 'tools/overnight_ptq_probe.py'
SOURCES = (
    'gptq_utils/gptq_plus_utils.py', 'process_args.py', 'ptq.py',
    'utils/data_utils.py', 'tools/overnight_ptq_probe.py',
    'tools/run_quant_preview_full_matched_candidate.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder = ROOT / 'outputs/quant_preview_full_matched_candidate_20260928'
    folder.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    control_manifest = json.loads(TEMPLATE.read_text())
    command = list(control_manifest['command'])
    entry_index = next(i for i, item in enumerate(command) if item.endswith('/entry.py'))
    entry = folder / 'entry.py'
    shutil.copyfile(PROBE, entry)
    command[entry_index] = str(entry)
    command[entry_index + 1] = 'baseline-profile'
    command[command.index('--exp') + 1] = 'quant_preview_full_matched_candidate_seed44_windows0'
    command.append('--preview_next_quant_block')
    hashes = {source: digest(ROOT / source) for source in SOURCES}
    hashes[str(entry)] = digest(entry)
    (folder / 'manifest.json').write_text(json.dumps({
        'command': command,
        'source_sha256': hashes,
        'candidate': True,
        'preview': 'next 128-column block; standard per-row symmetric W4 path',
        'calibration_seed': 44,
        'calibration_samples': 32,
        'eval_start_sample': 0,
        'eval_samples': 8,
        'memory_mode': 'matched to control; no input offload',
        'matched_control_folder': str(CONTROL),
        'matched_control_status': json.loads((CONTROL / 'status.json').read_text()),
    }, indent=2))
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        RESEARCH_RUN_DIR=str(folder),
    )
    started = time.monotonic()
    print('START full_matched_candidate', time.strftime('%F %T'), flush=True)
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
    control = json.loads((CONTROL / 'status.json').read_text())
    result = {
        'returncode': proc.returncode,
        'wall_seconds': time.monotonic() - started,
        'sources_unchanged': unchanged,
        'preview_events': log_text.count('QUANT_PREVIEW_NEXT_BLOCK'),
        'kl': float(match.group(1)) if match else None,
        'ppl': float(match.group(2)) if match else None,
        'quantization_seconds': metrics.get('quantization_seconds'),
        'state_sha256': metrics.get('state_sha256'),
        'peak_allocated': metrics.get('peak_allocated'),
        'peak_reserved': metrics.get('peak_reserved'),
        'control_kl': control['kl'],
        'control_ppl': control['ppl'],
        'delta_percent': {
            'kl': None if not match else (float(match.group(1)) / control['kl'] - 1) * 100,
            'ppl': None if not match else (float(match.group(2)) / control['ppl'] - 1) * 100,
        },
        'finished': time.strftime('%F %T'),
    }
    (folder / 'status.json').write_text(json.dumps(result, indent=2))
    print('END full_matched_candidate', json.dumps(result), flush=True)
    if proc.returncode or not unchanged or not match:
        raise RuntimeError('Full matched candidate failed, mutated source, or lacks exact metrics')


if __name__ == '__main__':
    main()
