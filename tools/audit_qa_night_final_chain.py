"""Serial fresh CPU independent audits for final reporting; no GPU replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def jobs():
    result=[]
    for model in ('qwen06','qwen17'):
        result.extend([
            (model+'_long','audit_qa_mapped_long_batch_completed.py',f'qa-mapped-long-batch-{model}-ctx1270-20261014-v1',True,362,1448),
            (model+'_natural','audit_qa_mapped_natural_tail_completed.py',f'qa-mapped-natural-tail-{model}-ctx128-20261014-v1',True,481,1924),
            (model+'_native','audit_qa_mapped_observed_selected_methods.py',f'qa-mapped-observed-native-{model}-20261014-v1',True,593,2372),
            (model+'_align','audit_qa_control_align_completed_v2.py',f'qa-control-align-{model}-20261014-v1',True,593,2372),
        ])
    result.append(('retained_single_pool_rejection','audit_qa_mapped_long_completed.py',
        'qa-mapped-long-qwen06-ctx1270-20261014-v1',False,356,1424))
    return result


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def completed_expected(row,expected,intervals,points):
    if (row.get('valid') is not expected or row.get('interval_count')!=intervals
            or row.get('points',{}).get('matched_values')!=points or row['points']['missing']):
        return False
    if expected:
        return row['points']['valid'] and all(row['checks'].values()) and all(
            all(row.get(key,{}).values()) for key in ('control_align_checks','independent_composition_checks','independent_selected_method_checks'))
    # False must be the already observed resource rejection, not an audit crash
    # or incomplete/incorrect numerics that also happen to produce False.
    failures={key for key,value in row['checks'].items() if not value}
    return failures=={'global_reserved'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-directory',type=Path,required=True);args=parser.parse_args()
    out=args.output_directory;out.mkdir(parents=True,exist_ok=False)
    bindings={str(ROOT/'tools'/script):sha(ROOT/'tools'/script) for _,script,*_ in jobs()}
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    (out/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    reports={};valid=False
    try:
        for name,script,folder,expected,interval_count,point_count in jobs():
            result_path=out/(name+'.json');log_path=out/(name+'.log')
            print(name,'fresh CPU independent audit starting',flush=True)
            with log_path.open('x') as log:
                process=subprocess.run([sys.executable,str(ROOT/'tools'/script),'--root',str(ROOT/'outputs'/folder),
                    '--output',str(result_path.resolve())],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
            if not result_path.exists():raise RuntimeError('Audit failed before complete result: '+name)
            row=json.loads(result_path.read_text())
            if process.returncode!=(0 if expected else 1) or not completed_expected(row,expected,interval_count,point_count):
                raise RuntimeError('Fresh complete expected acceptance/rejection not reproduced: '+name)
            reports[name]=dict(root=folder,expected_valid=expected,observed_valid=row['valid'],
                interval_count=interval_count,points=point_count,nvml_peak_mib=row['nvml_peak_mib'],
                strict_local_nonincrease=row['strict_local_nonincrease'],interval_regressions=row['interval_regressions'],
                fresh_audit_sha256=sha(result_path),log_sha256=sha(log_path),parent_status_sha256=sha(ROOT/'outputs'/folder/'status.json'))
            print(name,'complete; expected valid=',expected,flush=True)
        valid=all(sha(path)==value for path,value in bindings.items())
        if not valid:raise RuntimeError('Final auditor source changed')
    finally:
        result=dict(valid=valid,formal_enabled=False,reports=reports,source_sha256=bindings,
            accepted_pair_count=sum(r['expected_valid'] for r in reports.values()),
            accepted_intervals=sum(r['interval_count'] for r in reports.values() if r['expected_valid']),
            accepted_points=sum(r['points'] for r in reports.values() if r['expected_valid']),
            new_gpu_execution=False,
            scope='Fresh completed component audits only; original rejection preserved; raw311 or unrotated535/torch_fallback/synthetic/onevisible, NOT fullQA/REALQ/multidevice')
        with (out/'status.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
