"""Additional independent task/phase/device/payload coverage checks."""
import json
from pathlib import Path
from tools.audit_qa_server_mapped import audit as base_audit
from tools.qa_server_mapped_validation import sha


def phases(calls,tasks):
    result={'qa_setup','qa_mapped_storage_cleanup','qa_bridge_cleanup_complete'}
    for i in range(1,len(tasks)+1):result.update((f'qa_task{i}_start',f'qa_task{i}_end'))
    for i in range(1,calls+1):
        result.update(f'qa_forward{i}_{s}' for s in ('model_start','logits_ready','scoring'))
        result.update(f'qa_forward{i}_layer{k}_observed' for k in range(28))
        result.update(f'forward{i}_layer{k}_{s}' for k in range(28) for s in ('load_start','compute_start','offload_start','end'))
        result.update(f'head{i}_storage_{s}' for s in ('start','end','release_start','release_end'))
    return result


def audit(control,candidate,expected_tasks,bindings,*,expected_devices=4):
    from tools.audit_completed_memory_pair_capacity import intervals
    from tools.audit_memory_boundary_points import boundaries
    result=base_audit(control,candidate,expected_tasks,bindings,expected_devices=expected_devices)
    checks={}
    for side,root in zip(('control','candidate'),map(Path,(control,candidate))):
        summary=json.loads((root/'qa_complete/qa_complete_summary.json').read_text())
        head=json.loads((root/'qa_mapped_head_rank0.json').read_text())['head']
        owner=json.loads((root/'qa_counter_owner_rank0.json').read_text())
        checks[side+'_exclusive_counter_handoff']=owner['completed'] and not owner['entry_active_profiles'] and not owner['exit_active_profiles']
        buffers={k:v for k,v in summary['before']['layout'].items() if k.endswith('.had_K')}
        checks[side+'_all_Had_shapes']=len(buffers)==28 and all(v==[[12,12],'torch.float32'] for v in buffers.values())
        payloads=[];inputs=[]
        with (root/'qa_complete/qa_complete_events.jsonl').open() as f:
            for line in f:
                row=json.loads(line)
                if row['kind']=='logits':payloads.append(row)
                if row['kind']=='model_input':inputs.append(row)
        checks[side+'_head_logits_payload']=len(payloads)==len(inputs)==len(head['records']) and all(
            h['output_shape']==o['shape']==[inp['shape'][0],inp['shape'][1],151936]
            and h['input_shape'][0]==inp['shape'][0] and 1<=inp['shape'][0]<=16
            and o['dtype']==h['dtype']=='torch.bfloat16'
            and h['output_payload_bytes']==2*o['shape'][0]*o['shape'][1]*o['shape'][2]
            for h,o,inp in zip(head['records'],payloads,inputs))
        expected=phases(summary['forward_count'],expected_tasks)
        for device in range(expected_devices):
            paths=sorted((root/f'qa_stage2/memory_profile/device{device}').glob('*.jsonl'))
            table=intervals(paths);points=boundaries(paths)
            checks[f'{side}_device{device}_every_phase']=len(paths)==1 and len(table)==len(expected) and {k[3] for k in table}=={f'qa_device{device}__{phase}' for phase in expected}
            checks[f'{side}_device{device}_every_point']=len(points)==4*len(expected) and all(k[:3]==('stage2','evaluation',0) and k[4]==0 for k in points)
            checks[f'{side}_device{device}_actual_peak_bounds']=all(row['device_index']==device and row['rank']==0 and
                all(row['sampled_peak_'+m+'_bytes']>=max(row[p+'_'+m+'_bytes'] for p in ('start','end')) for m in ('allocated','reserved')) for row in table.values())
        checks[side+'_original_Linear_source']=head['original_forward_sha256']==__import__('hashlib').sha256(
            __import__('inspect').getsource(__import__('torch').nn.Linear.forward).encode()).hexdigest()
    result['independent_server_coverage_checks']=checks
    result['valid']=result['valid'] and all(checks.values())
    result['auditor_sha256']=sha(__file__)
    return result
