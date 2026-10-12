"""Two-layer GPU smoke for the opt-in Fisher+nonlinear FFN hybrid."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/mnt/d/gptq_plus_realq')
REFERENCE = ROOT / 'outputs/resume_20260927_ffn_nonlinear8/smoke/manifest.json'
PROBE = ROOT / 'tools/overnight_ptq_probe.py'
OUT = ROOT / 'outputs/resume_20260927_ffn_hybrid_smoke'


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(REFERENCE.read_text())
    command = list(manifest['command'])
    command = [arg for arg in command if arg != '--ffn_nonlinear_refresh']
    entry_index = next(i for i, arg in enumerate(command) if arg.endswith('/entry.py'))
    entry = OUT / 'entry.py'
    shutil.copyfile(PROBE, entry)
    command[entry_index] = str(entry)
    command.append('--ffn_nonlinear_hybrid_refresh')
    hashes = {
        'gptq_utils/gptq_plus_utils.py': hashlib.sha256(Path('gptq_utils/gptq_plus_utils.py').read_bytes()).hexdigest(),
        'process_args.py': hashlib.sha256(Path('process_args.py').read_bytes()).hexdigest(),
        str(entry): hashlib.sha256(entry.read_bytes()).hexdigest(),
    }
    (OUT / 'manifest.json').write_text(json.dumps({'command': command, 'source_sha256': hashes}, indent=2))
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', RESEARCH_RUN_DIR=str(OUT),
    )
    started = time.monotonic()
    with (OUT / 'run.log').open('w') as log:
        result = subprocess.run(
            ['timeout', '--signal=TERM', '--kill-after=30s', '600', *command],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    log_text = (OUT / 'run.log').read_text(errors='replace')
    unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in hashes.items())
    status = {
        'returncode': result.returncode,
        'wall_seconds': time.monotonic() - started,
        'sources_unchanged': unchanged,
        'ffn_refresh_events': log_text.count('FFN_NONLINEAR_REFRESH'),
        'hybrid_objective_logged': 'loss=ffn_nonlinear_hybrid' in log_text,
    }
    (OUT / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2), flush=True)
    if result.returncode or not unchanged or not status['hybrid_objective_logged']:
        raise RuntimeError(f'Hybrid smoke failed; inspect {OUT / "run.log"}')


if __name__ == '__main__':
    main()
