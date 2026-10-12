"""Audit the selective FFN nonlinear screen against its matched control."""
import hashlib
import json
import re
from pathlib import Path


ROOT = Path('/mnt/d/gptq_plus_realq')
BASE = ROOT / 'outputs/resume_20260927_ts_quality-screen8/plain-g1'
RUN = ROOT / 'outputs/resume_20260927_ffn_nonlinear8/candidate'
RESULT = ROOT / 'outputs/resume_20260927_ffn_nonlinear8/quality_results.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_command(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    command = list(manifest['command'])
    entry_pos = next(i for i, item in enumerate(command) if item.endswith('/entry.py'))
    command[entry_pos] = '<entry.py>'
    exp_pos = command.index('--exp') + 1
    command[exp_pos] = '<exp>'
    treatments = [item for item in command if item == '--ffn_nonlinear_refresh']
    command = [item for item in command if item != '--ffn_nonlinear_refresh']
    return command, len(treatments), manifest


def main():
    base_command, base_treatment_count, base_manifest = normalized_command(BASE)
    candidate_command, candidate_treatment_count, candidate_manifest = normalized_command(RUN)
    assert base_command == candidate_command, 'commands differ beyond entry/exp/treatment normalization'
    assert base_treatment_count == 0 and candidate_treatment_count == 1
    status = json.loads((RUN / 'status.json').read_text())
    assert status['returncode'] == 0 and status['sources_unchanged']
    log = (RUN / 'run.log').read_text(errors='replace')
    events = re.findall(r'FFN_NONLINEAR_REFRESH layer=(\d+) module=(mlp\.(?:gate_proj|up_proj|down_proj))', log)
    assert len(events) == 81 and len(set(events)) == 81, f'expected 81 unique objective switches, got {len(events)}'
    assert {int(layer) for layer, _ in events} == set(range(27))
    assert {module for _, module in events} == {'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'}
    assert not re.search(r'Traceback|OutOfMemory|CUDA out of memory|\bNaN\b|\bInf\b', log, re.I)

    for name, expected in candidate_manifest['source_sha256'].items():
        assert sha256(name) == expected, f'source changed after run: {name}'
    assert candidate_manifest['command'][candidate_manifest['command'].index('--eval_samples') + 1] == '8'
    assert candidate_manifest['command'][candidate_manifest['command'].index('--quant_stop_layer') + 1] == '26'

    results = json.loads(RESULT.read_text())
    control = results['control_plain_g1']
    candidate = results['candidate_ffn_nonlinear']
    assert candidate['state_sha256'] != control['state_sha256']
    assert candidate['kl'] < control['kl'] and candidate['ppl'] < control['ppl']
    print(json.dumps({
        'matched_command_except_treatment': True,
        'candidate_events': len(events),
        'candidate_kl': candidate['kl'],
        'control_kl': control['kl'],
        'candidate_ppl': candidate['ppl'],
        'control_ppl': control['ppl'],
        'quality_deltas_percent': results['delta_percent'],
        'no_runtime_or_numeric_errors': True,
        'source_hashes_unchanged': True,
    }, indent=2))


if __name__ == '__main__':
    main()
