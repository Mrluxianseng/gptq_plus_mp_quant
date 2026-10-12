"""Serial cost reproduction of the exact, globally accepted two-model system.

Three alternating-order pairs per model; no dense memory profiler. Fresh
Stage0 caches, all507 states/all28 heldout outputs and real method coverage
remain mandatory. Collector plus quantization is not end-to-end latency.
This controller must not start until the hardware followup has completed.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_low_overhead_speed_campaign as timing
from tools.audit_completed_memory_pair_capacity import audit, cache_inventory, digest, host_path, read
from tools.prepare_replay_stats_low_overhead_entry import prepare_entry
from tools.run_replay_stats_full_followup_v2 import save_new
from tools.run_replay_stats_memory_campaign import normalized_command
from tools.timing_completed_prepared_disk import link_completed


def check_bound(bound):
    changed = [p for p, h in bound.items() if not host_path(p).is_file() or digest(host_path(p)) != h]
    if changed:
        raise RuntimeError('bound timing source changed: ' + repr(changed))


def adapted_arm(entry, bound):
    """Reuse the existing serial timer, with exact entry and complete provenance.

    The original timer tool stays unchanged. Fail closed if its entry-copy
    boundary changes. Cache inventory uses streaming and extended host paths.
    """
    source = ast.parse(inspect.getsource(timing.run_arm))
    constants = [n for n in ast.walk(source) if isinstance(n, ast.Constant)
                 and n.value == 'tools/realq_low_overhead_speed_entry.py']
    if len(constants) != 1:
        raise RuntimeError('existing timing entry-copy boundary changed')
    constants[0].value = str(entry)
    namespace = dict(vars(timing))
    namespace['check_sources'] = lambda _expected: check_bound(bound)
    namespace['cache_hashes'] = lambda directory, **_: cache_inventory(Path(directory)/'static_cache')
    exec(compile(source, 'bound_replay_stats_timing_arm', 'exec'), namespace)
    return namespace['run_arm']


def verify_arm(destination, reference, record):
    """Check all heldout layers and every observed method record after timing."""
    destination, reference = Path(destination), Path(reference)
    fingerprints = read(destination/'layer_output_fingerprints.json')
    if (fingerprints != read(reference/'layer_output_fingerprints.json') or len(fingerprints) != 28
            or {r['layer'] for r in fingerprints} != set(range(28))
            or any(len(r['sha256']) != 64 or r['forward_count'] != 2 or not r['shapes'] for r in fingerprints)):
        raise RuntimeError('timing heldout all28 fingerprints differ or are incomplete')
    # Exact metadata comparison preserves every group/layer/shape/flag/pointer
    # condition already independently established by the full memory audit.
    # Allocation addresses and observational byte counters are not numerical
    # method identities; pointer preservation itself remains mandatory.
    def identities(value):
        if isinstance(value, list):
            return [identities(x) for x in value]
        if isinstance(value, dict):
            return {k: identities(v) for k, v in value.items()
                    if k not in {'observed_allocated_bytes', 'observed_reserved_bytes',
                                 'pointer', 'segment_bytes', 'inactive_bytes'}
                    and not k.endswith('_ptr') and not k.endswith('_pointer')}
        return value
    files = ['inverse_lifetime_stage2_rank0.json', 'dead_subgroup_input_stage2_rank0.json',
             'replay_stats_lifetime_stage2_rank0.json']
    for name in files:
        actual, expected = read(destination/name), read(reference/name)
        if identities(actual) != identities(expected):
            raise RuntimeError('timing method coverage differs: ' + name)
    metric = record['metrics_by_rank'][0]
    if (metric.get('exact_phase_memory_profile') is not False
            or metric.get('component_event_profile') is not False
            or not metric.get('heldout_layer_output_capture_after_timer')
            or not metric.get('shared_cpu_method_storage_observers_retained')):
        raise RuntimeError('missing reduced-instrumentation scope evidence')
    if cache_inventory(destination/'static_cache') != cache_inventory(reference/'static_cache'):
        raise RuntimeError('post-quantization cache differs')
    manifest = read(reference/'manifest.json')
    if any(normalized_command(record['commands'][stage]) != normalized_command(
            timing.without_profile(manifest[stage+'_command'])) for stage in ('stage0', 'stage2')):
        raise RuntimeError('algorithm/data timing command differs')
    return dict(valid=True, all28_outputs_exact=True, method_records_exact=True,
                post_quantization_cache_exact=True, outside_speed_timers=True)


def admit(followup, out):
    status = read(followup/'status.json')
    if (not status['valid'] or len(status['completed_pairs']) != 3
            or [x['repeat'] for x in status['completed_pairs']] != [1, 2, 3]
            or not all(x['capacity_valid'] for x in status['completed_pairs'])):
        raise RuntimeError('three complete hardware repeats required before timing')
    old = ROOT/'outputs/replay-stats-full-followup-20261012-v2'
    rows = []
    references = {}
    for model in ('qwen06b', 'qwen17b'):
        for pair in (1, 2, 3):
            original = model == 'qwen17b' or pair == 1
            tag = (f'{model}-replay-stats-full-20261012-v2-p{pair}' if original
                   else f'qwen06b-replay-stats-hardware-20261012-v1-p{pair}')
            bindings = (old if original else followup)/'source_bindings.json'
            value = audit(tag, bindings, method='stable')
            save_new(out/f'independent_capacity_admission_{model}_p{pair}.json', value)
            if not value['capacity_valid']:
                raise RuntimeError('fresh six-pair hardware readmission failed: ' + tag)
            rows.append(value)
            if pair == 1:
                references[model] = {side: ROOT/f'outputs/full_pipeline_profile_{tag}-r1-{side}'
                                     for side in ('control', 'candidate')}
    return references, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capacity-followup', type=Path, required=True)
    p.add_argument('--output-directory', type=Path, required=True)
    p.add_argument('--timeout-seconds', type=int, default=14400)
    args = p.parse_args()
    out = args.output_directory.resolve()
    out.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    report = dict(valid=False, completed_pairs=[], requested_pairs_per_model=3)
    log = timing.CombinedLog(out/'combined.log')
    try:
        refs, admissions = admit(args.capacity_followup.resolve(), out)
        entry = out/'entry.py'
        with entry.open('x', encoding='utf-8') as stream:
            stream.write(prepare_entry())
        bound = dict(read(ROOT/'outputs/replay-stats-full-followup-20261012-v2/source_bindings.json'))
        for path in (entry, Path(__file__), ROOT/'tools/run_low_overhead_speed_campaign.py',
                     ROOT/'tools/realq_low_overhead_speed_entry.py', ROOT/'tools/prepare_replay_stats_low_overhead_entry.py',
                     ROOT/'tools/audit_completed_memory_pair_capacity.py',
                     ROOT/'tools/timing_completed_prepared_disk.py',
                     ROOT/'tools/deduplicate_completed_prepared_checkpoint.py'):
            bound[str(path.resolve())] = digest(path)
        save_new(out/'source_bindings.json', bound)
        check_bound(bound)
        query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        if any(x.strip().isdigit() for x in query.splitlines()):
            raise RuntimeError('another CUDA process is active')
        arm = adapted_arm(entry, bound)
        for model, references in refs.items():
            launches = {s: timing.reference(d) for s, d in references.items()}
            caches = {s: cache_inventory(d/'static_cache') for s, d in references.items()}
            if len(caches['control']) != 13 or caches['control'] != caches['candidate']:
                raise RuntimeError('reference fresh cache provenance differs')
            for pair in (1, 2, 3):
                order = ('control', 'candidate') if pair % 2 else ('candidate', 'control')
                arms = {}
                for side in order:
                    if shutil.disk_usage(out).free < sum(x['bytes'] for x in caches[side].values()) + 1024**3:
                        raise RuntimeError('insufficient disk headroom for a complete fresh timing arm; no CUDA launched')
                    tag = f'{model}-replay-stats-cost-p{pair}-{side}'
                    state, launch = launches[side]
                    arms[side] = arm(out, tag, state, launch, bound, args.timeout_seconds, log, caches[side])
                    extra = verify_arm(out/tag, references[side], arms[side])
                    check_bound(bound)
                    extra['source_count'] = len(bound)
                    extra['entry_sha256'] = digest(out/tag/'entry.py')
                    if extra['entry_sha256'] != digest(entry):
                        raise RuntimeError('actual worker entry differs')
                    save_new(out/tag/'independent_complete_method_and_output_audit.json', extra)
                    disk = link_completed(out/tag, references[side])
                    save_new(out/tag/'completed_prepared_disk_link.json', disk)
                c, n = [arms[s]['quantization_seconds_slowest_rank'] for s in ('control', 'candidate')]
                sc, sn = [arms[s]['static_collection_seconds_slowest_rank'] for s in ('control', 'candidate')]
                row = dict(model=model, pair=pair, order=order, control_seconds=c, candidate_seconds=n,
                           control_collector_seconds=sc, candidate_collector_seconds=sn,
                           control_collector_plus_quantization=sc+c, candidate_collector_plus_quantization=sn+n,
                           paired_speedup=c/n, paired_collector_plus_quantization_speedup=(sc+c)/(sn+n))
                report['completed_pairs'].append(row)
                save_new(out/f'completed_{model}_p{pair}.json', row)
                log.write('[cost] complete independently audited pair: ' + json.dumps(row)+'\n')
        report['summaries'] = {model: {
            key: dict(mean=statistics.mean(values), stdev=statistics.stdev(values), min=min(values), max=max(values))
            for key in ('control_seconds', 'candidate_seconds', 'paired_speedup',
                        'control_collector_plus_quantization', 'candidate_collector_plus_quantization')
            for values in [[r[key] for r in report['completed_pairs'] if r['model'] == model]]}
            for model in refs}
        report['valid'] = True
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report.update(finished=time.strftime('%F %T'), scope='CUDA synchronized collector and full quantization; excludes preparation/evaluation/hash; shared CPU observers retained; no outlier removal. Capacity is established separately by dense-profile six-pair admission, not this timing run.')
        save_new(out/'status.json', report)
        log.close()


if __name__ == '__main__':
    main()
