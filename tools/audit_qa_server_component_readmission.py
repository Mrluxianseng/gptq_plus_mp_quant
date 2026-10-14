"""Readmit recording-only pre-dispatch identity rejection without GPU replay."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_server_mapped_component_v3 import admission
from tools.qa_server_mapped_validation import sha


def readmit(root):
    root=Path(root);original=json.loads((root/'status.json').read_text())
    failures=[k for k,v in original['checks'].items() if not v]
    if original['valid'] or failures!=['actual_server_backend']:
        raise RuntimeError('Only the known pre-dispatch control identity metadata rejection is eligible')
    control=json.loads((root/'control.json').read_text());candidate=json.loads((root/'candidate.json').read_text())
    if control['parameter_identity'] is not False or candidate['parameter_identity'] is not True:
        raise RuntimeError('Known pre-dispatch observation scope mismatch')
    result=admission(root)
    result.update(original_status_sha256=sha(root/'status.json'),original_valid=False,
        control_pre_dispatch_identity=False,control_during_QA_all_parameter_identity='not directly captured in this old component',
        disposition='Numerics/full-state/head-Parameter/official-dispatch/source/resource gates rechecked; pre-dispatch control ID is diagnostic. Formal QA separately requires post-dispatch all-Parameter identity.',
        no_new_gpu_execution=True,readmission_source_sha256=sha(__file__))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=readmit(a.root)
    with a.output.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(dict(valid=r['valid'],checks=r['checks']),indent=2))
    if not r['valid']:raise SystemExit(1)
