"""Fresh complete native-dispatch/mapped interface audit, no GPU experiment."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_qa_mapped_observed_bridge_completed import audit
from tools.audit_qa_memory_rows_completed import sha


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    reports={};artifacts={}
    for name in ('qwen06','qwen17'):
        root=ROOT/('outputs/qa-mapped-observed-native-'+name+'-20261014-v1')
        independent=root/'independent_completed_audit-20261014-v1.json'
        old=json.loads(independent.read_text());fresh=audit(root)
        if old['valid'] is not True or fresh['valid'] is not True or fresh['interval_count']!=593 or fresh['points']['matched_values']!=2372 or fresh['points']['missing']:
            raise RuntimeError('Complete composed numerical/source/method/telemetry audit required: '+name)
        reports[name]=fresh
        for path in (root/'status.json',independent,root/'source_bindings.json',root/'worker_source.py',
            root/'control.json',root/'candidate.json',root/'control/qa_mapped_head_rank0.json',root/'candidate/qa_mapped_head_rank0.json'):
            artifacts[str(path)]=sha(path)
        print(name,'fresh full composed audit True',flush=True)
    result=dict(valid=True,reports=reports,interval_count=1186,independently_matched_point_values=4744,
        source_artifact_sha256=artifacts,summary_source_sha256=sha(__file__),formal_enabled=False,
        scope='two local single-device native-dispatch compatibility pairs, unrotatedbase535/torch_fallback/65synthetic/natural configured16; NOT formal QA/REAL-Q/multidevice capacity',
        nvml_is_sampled=True,native_cuda_counters_include_mapped_host=True)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(valid=True,nvml={name:r['nvml_peak_mib'] for name,r in reports.items()},intervals=1186,points=4744),indent=2))
