"""Read-only completion audit, including NVIDIA CSV memory units."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.storage_process_device_peaks import valid_record


def memory_mib(value):
    match=re.fullmatch(r'\s*(\d+)\s*(?:MiB)?\s*',value)
    if not match: raise ValueError('Invalid NVML memory value: '+repr(value))
    return int(match.group(1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    c,n=[json.loads((root/(side+'.json')).read_text()) for side in ('control','candidate')]
    bindings=json.loads((root/'source_bindings.json').read_text())
    checks={'sources_unchanged':all(sha(path)==digest for path,digest in bindings.items()),
            'arm_identity':c['arm']=='control' and n['arm']=='candidate',
            'component_coverage':all(x['passages']==32 and x['requests']==64 and x['batch_size']==16 and len(x['scores'])==64 for x in (c,n)),
            'device_peak_complete':all(valid_record(x['memory'],1) for x in (c,n))}
    for key in ('request_sha256','scores','likelihood_source_sha256','model_use_cache','requests','request_lengths','model_state_sha256','model','vocab_size','scope'):
        checks[key+'_equal']=c[key]==n[key]
    checks['actual_library_source']=all(sha(root/(side+'_likelihood_source.py'))==c['likelihood_source_sha256'] for side in ('control','candidate'))
    peaks=[]; counts=[]
    for side in ('control','candidate'):
        with (root/(side+'_nvml.csv')).open() as stream:
            values=[memory_mib(row[1]) for row in csv.reader(stream) if len(row)>1]
        peaks.append(max(values));counts.append(len(values))
    for key in ('peak_allocated_bytes','peak_reserved_bytes'):
        checks[key+'_nonincreasing']=n['memory'][key]<=c['memory'][key]
    checks['sampled_nvml_nonincreasing']=peaks[1]<=peaks[0]
    evidence={path.name:sha(path) for path in root.iterdir() if path.is_file()}
    return dict(valid=all(checks.values()),checks=checks,scope=c['scope'],
                control=c['memory'],candidate=n['memory'],nvml_peak_mib=peaks,
                nvml_sample_count=counts,nvml_is_sampled=True,
                seconds=[c['seconds'],n['seconds']],evidence_sha256=evidence)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
    if not result['valid']: raise SystemExit(1)
