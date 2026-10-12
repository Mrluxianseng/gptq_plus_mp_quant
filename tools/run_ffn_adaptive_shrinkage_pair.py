"""Matched full-W4 audit of sequence-adaptive FFN Fisher shrinkage."""
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
    'tools/run_ffn_adaptive_shrinkage_pair.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    seed = int(os.environ.get('FFN_SHRINK_SEED', '44'))
    eval_start = int(os.environ.get('FFN_SHRINK_EVAL_START', '0'))
    run_tag = os.environ.get('FFN_SHRINK_RUN_TAG', 'pair')
    target = os.environ.get('FFN_SHRINK_TARGET', 'isotropic')
    if target not in ('isotropic', 'diagonal'):
        raise ValueError('FFN_SHRINK_TARGET must be isotropic or diagonal')
    root = ROOT / f'outputs/ffn_adaptive_shrinkage_{run_tag}_20260928'
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    template = json.loads(TEMPLATE_MANIFEST.read_text())['command']
    entry_index = next(i for i, item in enumerate(template) if item.endswith('/entry.py'))
    sources = {source: digest(ROOT / source) for source in SOURCES}
    results = {}

    # Reuse the matched full-W4 protocol. The only
    # treatment difference is whether FFN Fisher curvature is data-adaptively
    # shrunk toward its trace-matched isotropic target.
    for candidate in (False, True):
        label = 'adaptive_shrinkage' if candidate else 'fisher_control'
        folder = root / label
        folder.mkdir()
        entry = folder / 'entry.py'
        shutil.copyfile(PROBE, entry)
        command = list(template)
        command[entry_index] = str(entry)
        command[entry_index + 1] = 'baseline-profile'
        set_value(command, '--exp', f'ffn_adaptive_shrink_seed{seed}_windows{eval_start}_{label}')
        set_value(command, '--seed', seed)
        set_value(command, '--nsamples', 32)
        set_value(command, '--backward_samples', 32)
        set_value(command, '--eval_samples', 8)
        set_value(command, '--eval_start_sample', eval_start)
        if candidate:
            command.append(
                '--ffn_adaptive_fisher_offdiag_shrinkage'
                if target == 'diagonal' else '--ffn_adaptive_fisher_shrinkage'
            )
        hashes = dict(sources)
        hashes[str(entry)] = digest(entry)
        (folder / 'manifest.json').write_text(json.dumps({
            'command': command,
            'source_sha256': hashes,
            'candidate': candidate,
            'calibration_seed': seed,
            'calibration_samples': 32,
            'eval_start_sample': eval_start,
            'eval_samples': 8,
            'full_w4': True,
            'method': (
                'sequence-level data-adaptive linear shrinkage of FFN layer-output Fisher '
                + ('toward diag(F), preserving channel curvatures' if target == 'diagonal'
                   else 'toward trace(F)/H * I')
            ),
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
        shrink_stats = [
            {
                'layer': int(layer), 'lambda': float(value),
                'samples': int(samples), 'beta': float(beta),
                'target_distance': float(distance),
                'target': target_name,
            }
            for layer, value, samples, beta, distance, target_name in re.findall(
                r'FFN_FISHER_SHRINKAGE_STATS layer=(\d+) lambda=([0-9.eE+-]+) '
                r'samples=(\d+) beta=([0-9.eE+-]+) target_distance=([0-9.eE+-]+) target=([^\s]+)',
                log_text,
            )
        ]
        status = {
            'returncode': proc.returncode,
            'wall_seconds': time.monotonic() - started,
            'sources_unchanged': unchanged,
            'shrinkage_stats_layers': len(shrink_stats),
            'shrinkage_lambda_min': min((x['lambda'] for x in shrink_stats), default=None),
            'shrinkage_lambda_median': sorted(x['lambda'] for x in shrink_stats)[len(shrink_stats)//2] if shrink_stats else None,
            'shrinkage_lambda_max': max((x['lambda'] for x in shrink_stats), default=None),
            'shrinkage_stats': shrink_stats,
            'apply_events': log_text.count('FFN_FISHER_SHRINKAGE_APPLY'),
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
            raise RuntimeError(f'{label} failed or lacks exact metrics')
        if candidate and len(shrink_stats) != 28:
            (root / 'partial_results.json').write_text(json.dumps(results, indent=2))
            raise RuntimeError(f'Expected 28 layer shrinkage estimates; got {len(shrink_stats)}')

    base, adaptive = results['fisher_control'], results['adaptive_shrinkage']
    results['delta_percent'] = {
        'kl': (adaptive['kl'] / base['kl'] - 1) * 100,
        'ppl': (adaptive['ppl'] / base['ppl'] - 1) * 100,
        'quantization_time': (
            adaptive['quantization_seconds'] / base['quantization_seconds'] - 1
        ) * 100 if adaptive['quantization_seconds'] and base['quantization_seconds'] else None,
    }
    results['scope'] = (
        f'Matched full-W4 pair on calibration seed {seed} and WikiText2 eval windows '
        f'{eval_start}-{eval_start + 7}. '
        'Both arms use the same calibration examples, model, and configuration; '
        'only the adaptive FFN Fisher shrinkage objective differs.'
    )
    (root / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
