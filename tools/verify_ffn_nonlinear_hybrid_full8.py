"""Audit the matched full-W4 hybrid screen for treatment and log integrity."""
import json
import re
from pathlib import Path

ROOT = Path('/mnt/d/gptq_plus_realq/outputs/resume_20260927_ffn_hybrid_full8')
TREATMENT = '--ffn_nonlinear_hybrid_refresh'
EXPECTED_MODULES = {'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'}


def main():
    control_manifest = json.loads((ROOT / 'control/manifest.json').read_text())
    candidate_manifest = json.loads((ROOT / 'candidate/manifest.json').read_text())
    control_command = control_manifest['command']
    candidate_command = candidate_manifest['command']
    assert TREATMENT not in control_command
    assert candidate_command.count(TREATMENT) == 1
    def normalized(command):
        result = [x for x in command if x != TREATMENT]
        index = result.index('--exp')
        result[index + 1] = '<EXP>'
        entry_index = next(i for i, value in enumerate(result) if value.endswith('/entry.py'))
        result[entry_index] = '<ENTRY>'
        return result

    assert normalized(candidate_command) == normalized(control_command)
    control_hashes = {Path(path).name: digest for path, digest in control_manifest['source_sha256'].items()}
    candidate_hashes = {Path(path).name: digest for path, digest in candidate_manifest['source_sha256'].items()}
    assert control_hashes == candidate_hashes

    result = json.loads((ROOT / 'quality_results.json').read_text())
    assert result['control']['returncode'] == result['candidate']['returncode'] == 0
    assert result['control']['sources_unchanged'] and result['candidate']['sources_unchanged']
    assert result['control']['state_sha256'] == '7ad259c0465d1df4483996ce692a91bc14a2bb9f9f6b589ef03588660ac27046'

    log = (ROOT / 'candidate/run.log').read_text(errors='replace')
    routed = re.findall(
        r'FFN_NONLINEAR_REFRESH layer=(\d+) module=(\S+) loss=ffn_nonlinear_hybrid',
        log,
    )
    assert len(routed) == 81, f'Expected 81 treatment routes; found {len(routed)}'
    pairs = {(int(layer), module) for layer, module in routed}
    expected = {(layer, module) for layer in range(27) for module in EXPECTED_MODULES}
    assert pairs == expected, f'Route mismatch: missing={expected-pairs}, extra={pairs-expected}'
    assert 'Exact KL&PPL on wikitext2:' in log
    error_tokens = ('Traceback (most recent call last)', 'CUDA out of memory', 'nan detected')
    assert not any(token.lower() in log.lower() for token in error_tokens)

    print(json.dumps({
        'command_difference': 'run-specific --exp plus one opt-in treatment flag only',
        'control_state_hash_matches_prior_baseline': True,
        'candidate_route_count': len(routed),
        'candidate_layers': 27,
        'candidate_modules_per_layer': sorted(EXPECTED_MODULES),
        'both_runs_successful_and_sources_unchanged': True,
        'full_quality_metrics_present': True,
        'delta_percent': result['delta_percent'],
    }, indent=2))


if __name__ == '__main__':
    main()
