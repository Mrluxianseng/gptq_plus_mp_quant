"""Compare layer-0 down_proj Fisher components around cross closed-form compensation."""
import hashlib
import json
import os
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
    'tools/run_fisher_split_diag_probe.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_value(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    root = ROOT / 'outputs/fisher_split_diag_probe_v2_20260928'
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    template = json.loads(TEMPLATE.read_text())['command']
    entry_index = next(i for i, item in enumerate(template) if item.endswith('/entry.py'))
    all_hashes = {source: digest(ROOT / source) for source in SOURCES}
    summaries = {}

    for with_cross in (False, True):
        label = 'cross' if with_cross else 'control'
        folder = root / label
        folder.mkdir()
        entry = folder / 'entry.py'
        shutil.copyfile(ENTRY_SOURCE, entry)
        command = list(template)
        command[entry_index] = str(entry)
        command[entry_index + 1] = 'baseline-profile'
        set_value(command, '--exp', f'fisher_split_probe_{label}')
        set_value(command, '--seed', 42)
        set_value(command, '--nsamples', 8)
        set_value(command, '--backward_samples', 8)
        set_value(command, '--quant_stop_layer', 1)
        set_value(command, '--eval_samples', 2)
        command.append('--diagnose_fisher_split')
        if with_cross:
            command.append('--cross_module_compensate')

        hashes = dict(all_hashes)
        hashes[str(entry)] = digest(entry)
        (folder / 'manifest.json').write_text(json.dumps({
            'command': command,
            'source_sha256': hashes,
            'cross_module_compensate': with_cross,
            'quant_stop_layer': 1,
            'calibration_samples': 8,
            'fisher_probe_samples': 8,
            'probe_scope': 'layer 0 mlp.down_proj only',
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
        unchanged = all(digest(ROOT / source) == value for source, value in all_hashes.items())
        probe_lines = [
            line for line in log_text.splitlines()
            if 'FISHER_SPLIT_PROBE ' in line or 'CROSS_RESIDUAL_PROBE ' in line
        ]
        correction_lines = [line for line in log_text.splitlines() if 'cross_module_compensate:' in line]
        status = {
            'returncode': proc.returncode,
            'wall_seconds': time.monotonic() - started,
            'sources_unchanged': unchanged,
            'probe_lines': probe_lines,
            'cross_correction_lines': correction_lines,
            'finished': time.strftime('%F %T'),
        }
        (folder / 'status.json').write_text(json.dumps(status, indent=2))
        summaries[label] = status
        print('END', label, json.dumps(status), flush=True)
        if proc.returncode or not unchanged or not probe_lines:
            (root / 'partial_results.json').write_text(json.dumps(summaries, indent=2))
            raise RuntimeError(f'{label} failed, mutated sources, or produced no Fisher split probes')

    (root / 'results.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
