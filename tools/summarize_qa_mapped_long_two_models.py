"""Fresh read-only two-model audit; no model copies or GPU experiment."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_qa_mapped_long_batch_completed import audit
from tools.audit_qa_memory_rows_completed import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    reports={};artifacts={}
    for name in ('qwen06','qwen17'):
        root=ROOT/('outputs/qa-mapped-long-batch-'+name+'-ctx1270-20261014-v1')
        independent=root/'independent_completed_audit-20261014-v1.json'
        original=json.loads(independent.read_text())
        result=audit(root)
        if original['valid'] is not True or result['valid'] is not True:
            raise RuntimeError('Incomplete/failed model must retain its own evidence: '+name)
        if result['interval_count']!=362 or result['points']['matched_values']!=1448 or result['points']['missing']:
            raise RuntimeError('Full matched telemetry coverage required')
        reports[name]=result
        for path in (root/'status.json',independent,root/'source_bindings.json',root/'worker_source.py',
            root/'control/layerwise_cache.json',root/'candidate/layerwise_cache.json'):
            artifacts[str(path)]=sha(path)
        print(name,'fresh complete audit True',flush=True)
    summary=dict(valid=True,reports=reports,interval_count=sum(r['interval_count'] for r in reports.values()),
        independently_matched_point_values=sum(r['points']['matched_values'] for r in reports.values()),
        artifact_sha256=artifacts,summary_source_sha256=sha(__file__),
        scope='two raw311 synthetic long-input pairs, same accepted QA method plus batch-lifetime whole-head host-mapped storage; not formal QA/REAL-Q or server capacity',
        native_cuda_counters_include_mapped_host=True,nvml_is_sampled=True,formal_enabled=False)
    with args.output.open('x') as stream: json.dump(summary,stream,indent=2)
    print(json.dumps(dict(valid=True,nvml={name:r['nvml_peak_mib'] for name,r in reports.items()},
        intervals=summary['interval_count'],points=summary['independently_matched_point_values']),indent=2))


if __name__=='__main__': main()
