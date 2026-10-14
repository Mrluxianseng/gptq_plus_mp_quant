"""Full QA requires direct after-dispatch identity and complete QA sampling."""
import csv
import json
from pathlib import Path
from tools.audit_qa_server_mapped_v2 import audit as prior_audit
from tools.qa_server_mapped_validation import sha


def audit(control,candidate,expected_tasks,bindings,*,expected_devices=4):
    result=prior_audit(control,candidate,expected_tasks,bindings,expected_devices=expected_devices)
    checks={};nvml=[];identities=[];host=[]
    for side,root in zip(('control','candidate'),map(Path,(control,candidate))):
        identity=json.loads((root/'qa_parameter_identity_rank0.json').read_text())
        identities.append(identity)
        window=json.loads((root/'qa_telemetry_window.json').read_text())
        checks[side+'_post_dispatch_parameter_identity']=identity['completed'] and identity['parameter_identity_during_QA']
        checks[side+'_QA_window_complete']=window['completed'] and window['end_unix']>=window['start_unix']
        samples={}
        with (root/'qa_all_devices_nvml.csv').open() as f:
            for row in csv.reader(f):
                if len(row)!=4:continue
                value=float(row[1]);index=row[2].strip();uuid=row[3].strip()
                slot=samples.setdefault(index,dict(uuid=uuid,peak_mib=0,samples=0))
                if slot['uuid']!=uuid:raise RuntimeError('QA NVML identity changed')
                slot['peak_mib']=max(slot['peak_mib'],value);slot['samples']+=1
        nvml.append(samples)
        summary=json.loads((root/'qa_host_tree.summary.json').read_text())
        checks[side+'_host_window']=summary['completed'] and summary['root_pid']==window['pid'] and summary['jsonl_sha256']==sha(root/'qa_host_tree.jsonl')
        host.append(summary)
    checks['same_parameter_alias_topology']=identities[0]['parameter_names']==identities[1]['parameter_names'] and identities[0]['shared_parameter_groups']==identities[1]['shared_parameter_groups']
    checks['QA_NVML_all_devices']=set(nvml[0])==set(nvml[1])=={str(i) for i in range(expected_devices)} and all(nvml[0][i]['uuid']==nvml[1][i]['uuid'] for i in nvml[0])
    checks['global_QA_NVML']=max(v['peak_mib'] for v in nvml[1].values())<=max(v['peak_mib'] for v in nvml[0].values())
    result.update(independent_formal_window_checks=checks,QA_nvml=nvml,QA_host_samples=host)
    result['valid']=result['valid'] and all(checks.values())
    return result
