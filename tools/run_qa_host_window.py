"""Gracefully stop read-only host sampling at the actual QA window boundary."""
import argparse
import json
import time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.qa_host_memory_observer_v3 import HostTreeObserver,bounded_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop',type=Path,required=True);a=p.parse_args()
    summary=a.output.with_suffix('.summary.json')
    if a.output.exists() or summary.exists() or a.stop.exists():raise FileExistsError('New host window required')
    files=[Path(__file__),ROOT/'tools/qa_host_memory_observer.py',ROOT/'tools/qa_host_memory_observer_v2.py',ROOT/'tools/qa_host_memory_observer_v3.py']
    bindings={str(x):bounded_sha(x) for x in files};observer=HostTreeObserver(a.pid)
    row=dict(samples=0,complete_samples=0,partial_samples=0,max_complete_sample_totals_bytes={},source_bindings=bindings,
        root_pid=a.pid,scope='Sequential sampled process-tree accounting, not total physical host or continuous peak')
    try:
        with a.output.open('x') as f:
            while not a.stop.exists():
                sample=observer.sample();f.write(json.dumps(sample)+'\n');f.flush();row['samples']+=1
                if sample['complete']:
                    row['complete_samples']+=1
                    for k,v in sample['totals_bytes'].items():row['max_complete_sample_totals_bytes'][k]=max(v,row['max_complete_sample_totals_bytes'].get(k,0))
                else:row['partial_samples']+=1
                if sample.get('termination'):
                    row['termination']=sample['termination'];break
                time.sleep(0.5)
            else:row['termination']='explicit_QA_window_closed'
    except BaseException as exc:
        row.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        row['sources_unchanged']=all(bounded_sha(x)==v for x,v in bindings.items())
        row['jsonl_sha256']=bounded_sha(a.output) if a.output.exists() else None
        row['completed']=row.get('termination')=='explicit_QA_window_closed' and row['sources_unchanged'] and row['complete_samples']>0 and 'error' not in row
        with summary.open('x') as f:json.dump(row,f,indent=2)


if __name__=='__main__':main()
