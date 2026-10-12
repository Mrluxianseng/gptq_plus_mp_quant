"""Paired current-source comparison through layers 0-26 (final layer stays FP)."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/mnt/d/gptq_plus_realq')
REFERENCE = ROOT / 'outputs/ffn_hybrid_factorial_audit_20260928/seed44_windows0_control'
PROBE = ROOT / 'tools/overnight_ptq_probe.py'
PAIR = ROOT / 'outputs/quant_preview_prefix_pair_20260928'
SOURCES = (
    'gptq_utils/gptq_plus_utils.py', 'process_args.py', 'ptq.py',
    'utils/data_utils.py', 'tools/overnight_ptq_probe.py',
    'tools/run_quant_preview_prefix_pair.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_arm(name, preview):
    folder = PAIR / name
    folder.mkdir(parents=True, exist_ok=False)
    ref_manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    command = list(ref_manifest['command'])
    entry_index = next(i for i, item in enumerate(command) if item.endswith('/entry.py'))
    entry = folder / 'entry.py'
    shutil.copyfile(PROBE, entry)
    command[entry_index] = str(entry)
    command[entry_index + 1] = 'baseline-profile'
    command[command.index('--exp') + 1] = f'quant_preview_prefix_{name}_seed44_stop26'
    command += ['--quant_stop_layer', '26']
    if preview:
        command.append('--preview_next_quant_block')
    hashes = {source: digest(ROOT / source) for source in SOURCES}
    hashes[str(entry)] = digest(entry)
    (folder / 'manifest.json').write_text(json.dumps({
        'command': command,
        'source_sha256': hashes,
        'candidate': preview,
        'preview': 'layers 0-26, next 128-column block' if preview else None,
        'quant_stop_layer': 26,
        'final_layer_27': 'left floating point; common to both arms',
        'calibration_seed': 44,
        'calibration_samples': 32,
        'eval_start_sample': 0,
        'eval_samples': 8,
        'matched_control_source': 'same current source tree and CLI except preview flag',
        'reference_old_full_control': str(REFERENCE),
    }, indent=2))
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        RESEARCH_RUN_DIR=str(folder),
    )
    start = time.monotonic()
    with (folder / 'run.log').open('w') as log:
        proc = subprocess.run(
            ['timeout', '--signal=TERM', '--kill-after=30s', '3600', *command],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    log_text = (folder / 'run.log').read_text(errors='replace')
    match = re.search(r'Exact KL&PPL on wikitext2: ([0-9.eE+-]+), ([0-9.eE+-]+)', log_text)
    metrics_path = folder / 'baseline-profile_metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    unchanged = all(digest(ROOT / source) == hashes[source] for source in SOURCES)
    status = {
        'returncode': proc.returncode,
        'wall_seconds': time.monotonic() - start,
        'sources_unchanged': unchanged,
        'preview_events': log_text.count('QUANT_PREVIEW_NEXT_BLOCK'),
        'kl': float(match.group(1)) if match else None,
        'ppl': float(match.group(2)) if match else None,
        'quantization_seconds': metrics.get('quantization_seconds'),
        'state_sha256': metrics.get('state_sha256'),
        'peak_allocated': metrics.get('peak_allocated'),
        'peak_reserved': metrics.get('peak_reserved'),
        'finished': time.strftime('%F %T'),
    }
    (folder / 'status.json').write_text(json.dumps(status, indent=2))
    return status


def main():
    PAIR.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    print('START paired_prefix_comparison', time.strftime('%F %T'), flush=True)
    control = run_arm('control', preview=False)
    candidate = run_arm('preview', preview=True)
    result = {
        'control': control,
        'candidate': candidate,
        'delta_percent': {
            'kl': None if control['kl'] is None or candidate['kl'] is None else (candidate['kl'] / control['kl'] - 1) * 100,
            'ppl': None if control['ppl'] is None or candidate['ppl'] is None else (candidate['ppl'] / control['ppl'] - 1) * 100,
            'quantization_time': None if control['quantization_seconds'] is None or candidate['quantization_seconds'] is None else (candidate['quantization_seconds'] / control['quantization_seconds'] - 1) * 100,
        },
        'total_wall_seconds': time.monotonic() - start,
    }
    (PAIR / 'paired_summary.json').write_text(json.dumps(result, indent=2))
    print('PAIRED_PREFIX_RESULT', json.dumps(result), flush=True)
    if control['returncode'] or candidate['returncode'] or not control['sources_unchanged'] or not candidate['sources_unchanged'] or candidate['preview_events'] <= 0:
        raise RuntimeError('Prefix pair failed; inspect the per-arm status/logs')


if __name__ == '__main__':
    main()
