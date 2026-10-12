"""Audit saved paired runs; never select hyperparameters or discard a seed."""
import argparse
import json
from pathlib import Path
import re
import torch


def load(path):
    return json.loads(path.read_text())


def compare_outputs(a, b):
    torch.testing.assert_close(a['target'], b['target'], rtol=0, atol=0)
    delta = a['output'].float() - b['output'].float()
    return dict(rmse=delta.square().mean().sqrt().item(), max_abs=delta.abs().max().item(),
                changed_fraction=(delta != 0).float().mean().item())


def token_error_profile(tensors):
    error = (tensors['output'].float() - tensors['target'].float()).square().mean(-1)
    return dict(first_token_mse=error[:, 0].mean().item(),
                remaining_tokens_mse=error[:, 1:].mean().item(),
                maximum_error_positions=error.argmax(-1).tolist())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('root')
    root = Path(parser.parse_args().root)
    rows = load(root / 'results.json')
    assert all(r['returncode'] == 0 and r['unchanged'] for r in rows)
    summary = dict(scope='Only blocks 0-4 quantized; two calibration seeds; eight shared evaluation windows.', pairs={})
    source_hashes = None
    for seed in sorted({r['seed'] for r in rows}):
        runs = {r['mode']: r for r in rows if r['seed'] == seed}
        paths = {mode: root / f'seed{seed}_{mode}' for mode in runs}
        stat_rows = {mode: load(p / 'static_stats.json') for mode, p in paths.items()}
        assert len({r['fisher'] for r in stat_rows.values()}) == 1
        assert len({r['saliency'] for r in stat_rows.values()}) == 1
        sketches = {}
        for mode, path in paths.items():
            manifest = load(path / 'manifest.json')
            if source_hashes is None:
                source_hashes = manifest['source_sha256']
            assert source_hashes == manifest['source_sha256']
            log = (path / 'run.log').read_text(errors='replace')
            assert not re.search(r'Traceback|CUDA out of memory|Non-finite Fisher', log)
            assert runs[mode]['refresh_events'] == (0 if mode == 'off' else 15)
            if mode != 'off':
                sketches[mode] = dict(re.findall(r'SEQUENCE_FISHER_CACHE layer=(\d+).*?sha256=([0-9a-f]+)', log))
                assert len(sketches[mode]) == 5
        if 'token' in sketches and 'sequence' in sketches:
            assert sketches['token'] == sketches['sequence']
        elif 'cross' in sketches:
            assert len(sketches['cross']) == 5
        pairs = dict(original_statistics_identical=True, sketches={},
                     runs=runs, static_precompute=stat_rows, comparisons={}, layers=[], evaluation_windows={})
        pairs['sketches'] = sketches
        for mode, path in paths.items():
            pairs['evaluation_windows'][mode] = load(path / 'eval_window_kl.json')
        for mode in (m for m in runs if m != 'off'):
            a, b = runs[mode], runs['off']
            pairs['comparisons'][mode + '_vs_off'] = dict(
                kl_delta_percent=100 * (a['kl'] / b['kl'] - 1),
                ppl_delta_percent=100 * (a['ppl'] / b['ppl'] - 1),
                time_delta_percent=100 * (a['metrics']['quantization_seconds'] / b['metrics']['quantization_seconds'] - 1),
                peak_allocated_delta_MiB=(a['metrics']['peak_allocated'] - b['metrics']['peak_allocated']) / 2**20)
        if 'sequence' in runs and 'token' in runs:
            a, b = runs['sequence'], runs['token']
            pairs['comparisons']['sequence_vs_token'] = dict(
                kl_delta_percent=100 * (a['kl'] / b['kl'] - 1),
                ppl_delta_percent=100 * (a['ppl'] / b['ppl'] - 1),
                time_delta_percent=100 * (a['metrics']['quantization_seconds'] / b['metrics']['quantization_seconds'] - 1),
                peak_allocated_delta_MiB=(a['metrics']['peak_allocated'] - b['metrics']['peak_allocated']) / 2**20)
        local_metrics = {mode: [json.loads(line) for line in (path / 'layer_metrics.jsonl').read_text().splitlines()]
                         for mode, path in paths.items()}
        for idx in range(5):
            tensors = {mode: torch.load(path / f'layer_{idx}_outputs.pt', map_location='cpu', weights_only=True)
                       for mode, path in paths.items()}
            layer = dict(layer=idx, metrics={m: v[idx] for m, v in local_metrics.items()},
                         token_error_profiles={m: token_error_profile(t) for m, t in tensors.items()},
                         output_vs_off={})
            for mode in (m for m in runs if m != 'off'):
                layer['output_vs_off'][mode] = compare_outputs(tensors[mode], tensors['off'])
            pairs['layers'].append(layer)
        summary['pairs'][str(seed)] = pairs
    summary['sequence_hybrid_repeatable_vs_control'] = bool(summary['pairs']) and all(
        'sequence_vs_off' in p['comparisons'] and p['comparisons']['sequence_vs_off']['kl_delta_percent'] < 0
        for p in summary['pairs'].values())
    summary['sequence_hybrid_repeatable_vs_token'] = bool(summary['pairs']) and all(
        'sequence_vs_token' in p['comparisons'] and p['comparisons']['sequence_vs_token']['kl_delta_percent'] < 0
        for p in summary['pairs'].values())
    summary['cross_addition_repeatable_vs_control'] = bool(summary['pairs']) and all(
        'cross_vs_off' in p['comparisons'] and p['comparisons']['cross_vs_off']['kl_delta_percent'] < 0
        for p in summary['pairs'].values())
    (root / 'audited_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({seed: p['comparisons'] for seed, p in summary['pairs'].items()}, indent=2))
    print('All integrity checks passed; flags:', summary['sequence_hybrid_repeatable_vs_control'],
          summary['sequence_hybrid_repeatable_vs_token'], summary['cross_addition_repeatable_vs_control'])


if __name__ == '__main__':
    main()
