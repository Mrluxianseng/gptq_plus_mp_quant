"""One fresh full 0.6B placement screen after confirmed size-policy failure."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_weight_size_policy_screen as policy
from tools import run_replay_stats_full_followup_v2 as base
from tools.replay_weight_compact_pool_prototype import build_entry as compact_entry
from tools.replay_weight_cleanup_schedule_prototype import build_entry as cleanup_entry
from tools.run_full_pipeline_memory_profile import set_option


def prepare_entry(queue):
    source=compact_entry(policy.prepare_entry(queue),helper_sha=base.digest(ROOT/'tools/replay_weight_compact_pool_prototype.py'))
    return cleanup_entry(source,helper_sha=base.digest(ROOT/'tools/replay_weight_cleanup_schedule_prototype.py'))


def compact_coverage(control,candidate,cleanups,expected):
    def spec(rows):
        return [dict(x,name=x['name'].replace('.module.','.')) for x in rows]
    for arm,enabled in ((control,False),(candidate,True)):
        rows=arm['records']
        if arm['enabled']!=enabled or not arm['helper_unchanged'] or len(rows)!=28:
            return False
        if [r['layer'] for r in rows]!=list(range(28)):
            return False
        for row in rows:
            if (row['enabled']!=enabled or row['boundary']!='after_statistics_delete_before_replay'
                    or row['payload_bytes']!=expected['payload_bytes']
                    or row['parameter_count']!=expected['parameter_count']
                    or spec(row['parameters'])!=spec(expected['parameters'])):
                return False
            if enabled and (not row['released'] or row['active_pool_bytes_after_layer_release']!=0
                    or row['shared_storage_count']!=1 or row['pool_segment_bytes']<row['payload_bytes']
                    or row.get('rescheduled_existing_cleanup_calls')!=1):
                return False
    return all(c['enabled']==(i==1) and not c['pending_layers'] and c['helper_unchanged']
               for i,c in enumerate(cleanups))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-independent-audit',type=Path,required=True)
    parser.add_argument('--component',type=Path,required=True)
    parser.add_argument('--workspace-queue',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args();out=args.output_directory.resolve();out.mkdir(exist_ok=False)
    proc=reader=None
    report=dict(valid=False,scope='One fresh 0.6B compact replay-weight native pool + existing cleanup reschedule; no repetitions/speed claim')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        failed=base.read(args.failed_independent_audit)
        keys=('all_stages_exit_zero','canonical_507_state_kl_ppl_exact','canonical_all28_nonempty_outputs_exact',
              'sources_unchanged','both_stage_commands_equal','all13_cache_bytes_equal','global_peaks_no_regression',
              'method_112_inverse_28_requested_selected_write_dead_stats_covered')
        if failed['valid'] or not all(failed['checks'].get(k) for k in keys) or not failed['interval_regressions'] or failed['points']['missing']:
            raise RuntimeError('requires complete exact/source/cache/method confirmed resource failure')
        component=base.read(args.component)
        if not component['valid'] or len(component['models'])!=2 or component['helper_sha256']!=base.digest(ROOT/'tools/replay_weight_compact_pool_prototype.py'):
            raise RuntimeError('two-model real decoder component required')
        for model in component['models']:
            if not all(model.get(k) for k in ('values_exact','parameter_ids_preserved','forward_exact','cpu_release_values_exact','pool_released')):
                raise RuntimeError('component incomplete')
        expected=next(m['records'][0] for m in component['models'] if m['model']=='Qwen3-0.6B')
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
        if any(s.strip().isdigit() for s in gpu.stdout.splitlines()):raise RuntimeError('another CUDA process active')
        bound=base.read(ROOT/'outputs/weight-size-policy-screen-20261012-v1/source_bindings.json')
        base.check_bound_sources(bound)
        entry=out/'entry.py';entry.write_text(prepare_entry(args.workspace_queue))
        for path in (entry,Path(__file__).resolve(),ROOT/'tools/replay_weight_compact_pool_prototype.py',ROOT/'tools/replay_weight_cleanup_schedule_prototype.py'):
            bound[str(path)]=base.digest(path)
        base.save_new(out/'source_bindings.json',bound);base.check_bound_sources(bound)
        reference=ROOT/'outputs/full_pipeline_profile_qwen06b-checkpoint-ce-original-input-full-threepairs-20261010-1457-r1-candidate'
        canonical=base.read(reference/'status.json');ref=base.read(reference/'manifest.json');folder=out/'manifests';folder.mkdir()
        for side in ('control','candidate'):
            for stage in ('stage0','stage2'):
                data=base.read(ref[stage+'_source_manifest']);command=data['command'];command[command.index('baseline-profile')-1]=str(entry)
                if stage=='stage2':
                    set_option(command,'--quant_stop_layer','27')
                    if side=='candidate':command.extend(['--research_hinv_solver_workspace','--research_hinv_reuse_consumed_h','--research_hinv_stable_weight','--research_replay_early_stats'])
                data['scope_note']='After last quantization/statistics consumers, compact current layer parameters in a native pool; move existing final down cleanup tail after placement. Same math/data, fresh Stage0.'
                base.save_new(folder/(side+'-'+stage+'.json'),data)
        for stage in ('stage0','stage2'):
            commands=[base.read(folder/(side+'-'+stage+'.json'))['command'] for side in ('control','candidate')]
            if base.normalized_command(commands[0])!=base.normalized_command(commands[1]):raise RuntimeError('algorithm/data changed')
        reader=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/stream_memory_research_progress.py'),'--queues',str(out),'--output',str(ROOT/'logs/realq-system-restart-20261010-1408.log'),'--append'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        cmd=[sys.executable,'-u',str(ROOT/'tools/run_replay_stats_memory_campaign.py'),'--tag',args.tag,'--pairs','1','--gpu-indices','0','--timeout-seconds','7200']
        for side in ('control','candidate'):
            for stage in ('stage0','stage2'):cmd.extend([f'--{side}-{stage}-manifest',str(folder/(side+'-'+stage+'.json'))])
        base.save_new(out/'active.json',dict(label=args.tag,command=cmd))
        with (out/(args.tag+'.log')).open('x') as stream:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);code=proc.wait()
        runs=[ROOT/('outputs/full_pipeline_profile_'+args.tag+'-r1-'+side) for side in ('control','candidate')]
        statuses=[base.read(run/'status.json') for run in runs];points=base.audit_points(*statuses);base.save_new(out/'separate_start_end.json',points)
        coverage=[policy.policy_coverage(base.read(run/'inverse_lifetime_stage2_rank0.json'),base.read(run/'dead_subgroup_input_stage2_rank0.json'),enabled=i==1) for i,run in enumerate(runs)]
        replay=base.replay_coverage(*[base.read(run/'replay_stats_lifetime_stage2_rank0.json') for run in runs])
        compact=compact_coverage(*[base.read(run/'replay_weight_compact_stage2_rank0.json') for run in runs],
            [base.read(run/'replay_weight_cleanup_stage2_rank0.json') for run in runs],expected)
        quality=all(s['metrics']['state_tensors']==507 and s['metrics']['state_sha256']==canonical['metrics']['state_sha256'] and (s['kl'],s['ppl'])==(canonical['kl'],canonical['ppl']) and base.read(run/'layer_output_fingerprints.json')==base.read(reference/'layer_output_fingerprints.json') for s,run in zip(statuses,runs))
        base.check_bound_sources(bound)
        audit=base.read(ROOT/('outputs/memory_campaign_'+args.tag+'/pair1_audit/comparison.json'))
        report.update(campaign_returncode=code,canonical_quality_passed=quality,size_policy_coverage=coverage,replay_coverage=replay,
            compact_pool_coverage=compact,separate_points_passed=points['valid'],strict_memory_gate=audit['strict_measured_memory_gate_passed'],cache_files_match=audit['cache_files_match'],sources_unchanged=True,
            trigger_sha256=base.digest(args.failed_independent_audit),component_sha256=base.digest(args.component))
        report['valid']=code==0 and quality and all(coverage) and replay and compact and points['valid'] and audit['strict_measured_memory_gate_passed'] and audit['cache_files_match']
    except BaseException as exc:
        report['error']=repr(exc)
        if proc and proc.poll() is None:base.stop_process_tree(proc.pid);proc.wait()
        raise
    finally:
        report['finished']=time.strftime('%F %T');base.save_new(out/'status.json',report)
        if reader:reader.wait(timeout=15)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
