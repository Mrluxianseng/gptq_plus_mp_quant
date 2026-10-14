"""Small real CUDA counter probe; no QA capacity or numerical-method claim."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import torch
from tools.qa_visible_device_phase_profile import VisibleDevicePhases
from tools.audit_memory_boundary_points import boundaries
from tools.audit_completed_memory_pair_capacity import intervals


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(inspect.getfile(VisibleDevicePhases)),
             Path(inspect.getfile(torch.cuda.memory_stats)),
             Path('tools/audit_memory_boundary_points.py'),Path('tools/audit_completed_memory_pair_capacity.py')]
    bindings={str(p.resolve()):sha(p) for p in sources}
    (root/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    if torch.cuda.is_initialized(): raise RuntimeError('Probe requires a fresh CUDA process')
    profile=VisibleDevicePhases(torch.cuda,root/'qa_stage2/memory_profile',exclusive_peak_owner=True)
    idle_did_not_initialize=not torch.cuda.is_initialized()
    try:
        block=torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda:0')
        block.fill_(7);torch.cuda.synchronize(0)
        time.sleep(0.3)
        del block
        profile.mark('after_transient')
        report=profile.finish('completed')
    except BaseException:
        if not profile.closed: profile.finish('failed')
        raise
    paths=[Path(p) for p in report['memory_profile_files']]
    rows=intervals(paths);points=boundaries(paths)
    first=[r for r in rows.values() if r['logical_phase']=='qa_setup' and r['device_index']==0]
    transient=first[0]
    checks=dict(no_idle_context_creation=idle_did_not_initialize,
        transient_peak_actual=transient['sampled_peak_allocated_bytes']>=64*1024*1024,
        transient_not_boundary_max=transient['sampled_peak_allocated_bytes']>
            max(transient['start_allocated_bytes'],transient['end_allocated_bytes']),
        every_visible_device_interval=len(rows)==2*torch.cuda.device_count(),
        complete_individual_points=len(points)==8*torch.cuda.device_count(),
        sources_unchanged=all(sha(p)==h for p,h in bindings.items()))
    physical=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    result=dict(valid=all(checks.values()),checks=checks,memory=report,
        interval_count=len(rows),point_count=len(points),physical_gpu_inventory=physical,
        scope='tiny CUDA allocator-counter probe only; no model QA, multi-GPU or NVML peak validation')
    with (root/'status.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2),flush=True)
    if not result['valid']: raise RuntimeError('CUDA counter probe rejected')


if __name__=='__main__': main()
