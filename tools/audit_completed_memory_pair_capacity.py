"""Independent complete-pair audit under the clarified hardware peak criterion.

Original interval and separate-point regressions stay in the output. Only
complete numerical/source/cache/method coverage and whole-flow global peaks
determine capacity acceptance. Does not mutate any original status or gate.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_memory_boundary_points import boundaries,compare
from tools.run_replay_stats_full_followup_v2 import normalized_command,replay_coverage


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def host_path(path):
    value=str(path)
    if os.name=='nt' and value.startswith('/mnt/d/'):value='D:/'+value[7:]
    if os.name!='nt' and value.lower().startswith('d:/'):value='/mnt/d/'+value[3:]
    return Path(value)


def cache_inventory(folder):
    folder=Path(folder).resolve()
    if os.name=='nt':folder=Path('\\\\?\\'+str(folder))
    return {p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
            for p in folder.rglob('*') if p.is_file()}


def intervals(paths):
    result={};counts=Counter()
    for path in paths:
        stage='stage0' if '_stage0/' in path.as_posix() else 'stage2'
        kind='evaluation' if 'evaluation' in path.name else 'quantization'
        match=re.search(r'rank(\d+)',path.name)
        if match is None:raise RuntimeError('rank identity missing')
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if 'phase' not in row:continue
            identity=(stage,kind,int(match[1]),row['phase'])
            key=(*identity,counts[identity]);counts[identity]+=1;result[key]=row
    return result


def capacity_gate(checks,peaks):
    return all(checks.values()) and bool(peaks) and all(b<=a for a,b in peaks.values())


def audit(tag,bindings,*,method='stable',component=None):
    runs=[ROOT/('outputs/full_pipeline_profile_'+tag+'-r1-'+side) for side in ('control','candidate')]
    statuses=[read(run/'status.json') for run in runs];manifests=[read(run/'manifest.json') for run in runs]
    model='qwen06b' if tag.startswith('qwen06b') else 'qwen17b'
    accepted=('qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457' if model=='qwen06b' else 'qwen17b-checkpoint-ce-matched-full-threepairs-20261010-v6-2125')
    reference=ROOT/('outputs/full_pipeline_profile_'+accepted+'-r1-candidate');canonical=read(reference/'status.json')
    conflicts=[];bound=dict(read(bindings))
    for manifest in manifests:
        for filename,h in manifest['source_sha256_at_launch'].items():
            if filename in bound and bound[filename]!=h:conflicts.append(filename)
            bound[filename]=h
    changed=[p for p,h in bound.items() if not host_path(p).is_file() or digest(host_path(p))!=h]
    fingerprints=[read(run/'layer_output_fingerprints.json') for run in runs]
    paths=[[ROOT/p for p in s['memory_profile_files']] for s in statuses]
    points=compare(*(boundaries(p) for p in paths));rows=[intervals(p) for p in paths]
    missing=[list(k) for k in sorted(rows[0].keys()^rows[1].keys())];regressions=[]
    for key in sorted(rows[0].keys()&rows[1].keys()):
        for metric in ('allocated','reserved'):
            a,b=[r[key]['sampled_peak_'+metric+'_bytes'] for r in rows]
            if b>a:regressions.append(dict(key=list(key),metric=metric,control_bytes=a,candidate_bytes=b,delta_bytes=b-a))
    caches=[cache_inventory(run/'static_cache') for run in runs]
    metadata=[read(run/'inverse_lifetime_stage2_rank0.json') for run in runs]
    deads=[read(run/'dead_subgroup_input_stage2_rank0.json') for run in runs]
    group_counts=Counter({i:4 for i in range(28)});layer_counts=Counter(range(28));dim=3072 if model=='qwen06b' else 6144
    method_ok=[]
    for i,(inv,dead) in enumerate(zip(metadata,deads)):
        enabled=i==1
        ok=(inv['solver_workspace']==enabled and inv['reuse']==enabled and inv['helper_unchanged']
            and inv['stable_helper_unchanged'] and dead['enabled']==enabled and dead['helpers_unchanged']
            and len(inv['calls'])==112 and Counter(x['layer'] for x in inv['calls'])==group_counts
            and all(x['module']=='mlp.down_proj' and x['shape']==[dim,dim] and x['dtype']=='torch.float32'
                    and x['enabled']==enabled and not x['identity_fallback'] for x in inv['calls'])
            and len(inv['weight_write_calls'])==28 and Counter(x['layer'] for x in inv['weight_write_calls'])==layer_counts)
        if method=='stable':
            ok=ok and all(x['enabled']==enabled and (not enabled or x['pointer_preserved']) for x in inv['weight_write_calls'])
        else:
            from tools.run_weight_size_policy_screen import policy_coverage
            ok=ok and policy_coverage(inv,dead,enabled=enabled)
        if enabled:
            ok=ok and len(inv['reuse_calls'])==112 and Counter(x['layer'] for x in inv['reuse_calls'])==group_counts and all(x['factor_shares_consumed_h_storage'] and not x['identity_fallback'] for x in inv['reuse_calls'])
            ok=ok and len(dead['records'])==28 and Counter(x['layer'] for x in dead['records'])==layer_counts and all(x['released_private_input_bytes']==dim*dim*4 and x['retained_factors']==4 for x in dead['records'])
        else:ok=ok and not inv['reuse_calls'] and not dead['records']
        method_ok.append(ok)
    replay=replay_coverage(*[read(run/'replay_stats_lifetime_stage2_rank0.json') for run in runs])
    compact=True
    if method=='compact':
        from tools.run_replay_weight_compact_screen import compact_coverage
        comp=read(component);expected=next(m['records'][0] for m in comp['models'] if m['model']=='Qwen3-0.6B')
        compact=compact_coverage(*[read(run/'replay_weight_compact_stage2_rank0.json') for run in runs],
            [read(run/'replay_weight_cleanup_stage2_rank0.json') for run in runs],expected)
    fullpeaks={k:[s[k] for s in statuses] for k in ('full_process_peak_allocated_bytes','full_process_peak_reserved_bytes','max_nvml_memory_used_mib')}
    legacy=read(ROOT/('outputs/memory_campaign_'+tag+'/pair1_audit/comparison.json'))
    diagnostics={k:[s[k] for s in statuses] for k in statuses[0] if ('peak_allocated_bytes' in k or 'peak_reserved_bytes' in k) and isinstance(statuses[0][k],(int,float))}
    checks=dict(all_stages_exit_zero=all(s['valid'] and s['returncode']==0 and s['stage0']['returncode']==0 and s['stage2']['returncode']==0 for s in statuses),
        canonical_507_state_kl_ppl_exact=all(s['metrics']['state_tensors']==507 and s['metrics']['state_sha256']==canonical['metrics']['state_sha256'] and (s['kl'],s['ppl'])==(canonical['kl'],canonical['ppl']) for s in statuses),
        canonical_all28_nonempty_outputs_exact=all(f==read(reference/'layer_output_fingerprints.json') and len(f)==28 and {x['layer'] for x in f}==set(range(28)) and all(len(x['sha256'])==64 and x['forward_count']==2 and x['shapes'] for x in f) for f in fingerprints),
        sources_unchanged=not changed and not conflicts,
        original_pair_infrastructure_identity_complete=all(legacy.get(k) for k in ('exactness_passed','full_model_quantization_validated','tracked_source_snapshots_match','algorithm_commands_match','recorded_runtime_environments_match','monitored_gpu_identities_match')),
        both_stage_commands_equal=all(normalized_command(manifests[0][st+'_command'])==normalized_command(manifests[1][st+'_command']) for st in ('stage0','stage2')),
        single_rank_full_flow=all(m['expected_ranks']==dict(stage0=1,stage2=1) for m in manifests),
        all13_cache_bytes_equal=len(caches[0])==len(caches[1])==13 and caches[0]==caches[1],
        inverse_reuse_write_dead_stats_compact_coverage=all(method_ok) and replay and compact,
        all2961_interval_telemetry_complete=len(rows[0])==len(rows[1])==2961 and not missing and all(x.get('peak_measurement_mode')=='exact' for arm in rows for x in arm.values()),
        all11844_point_telemetry_complete=points['matched_values']==11844 and not points['missing'])
    capacity=capacity_gate(checks,fullpeaks)
    return dict(time=time.strftime('%F %T'),tag=tag,method=method,valid=capacity,capacity_valid=capacity,
        strict_interval_point_valid=capacity and not regressions and points['valid'],checks=checks,
        full_global_peaks=fullpeaks,stage_global_diagnostics=diagnostics,
        global_peaks_no_regression=all(b<=a for a,b in fullpeaks.values()),
        source_count=len(bound),changed_sources=changed,source_conflicts=conflicts,
        status_sha256=[digest(run/'status.json') for run in runs],bindings_sha256=digest(bindings),
        audit_source_sha256=digest(Path(__file__)),cache_files=caches,interval_count=[len(x) for x in rows],
        missing_intervals=missing,interval_regressions=regressions,points=points,
        kl=[s['kl'] for s in statuses],ppl=[s['ppl'] for s in statuses],
        diagnostic_quantization_seconds=[s['metrics']['quantization_seconds'] for s in statuses],
        criterion='Complete exact quality/source/cache/method/telemetry; full-process allocated/reserved and sampled whole-device NVML peaks <= control. Local changes are diagnostics.',
        measurement_limit='NVML sampled, not continuous every-instant proof; cal2x1024 pilot, no formal2048 or actual smaller-GPU/multi-GPU proof.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True);parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--method',choices=('stable','size','compact'),default='stable')
    parser.add_argument('--component',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.tag,args.bindings,method=args.method,component=args.component)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(dict(valid=report['valid'],strict=report['strict_interval_point_valid'],source_count=report['source_count'],interval_regressions=len(report['interval_regressions']),point_regressions=len(report['points']['regressions']),peaks=report['full_global_peaks'])))


if __name__=='__main__':main()
