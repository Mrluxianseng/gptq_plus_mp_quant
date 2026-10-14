"""Versioned standalone CLI bootstrap for the frozen CPU host observer."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.qa_host_memory_observer_v2 import collect,bounded_sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--interval',type=float,default=0.5)
    parser.add_argument('--seconds',type=float);args=parser.parse_args()
    launch=args.output.with_suffix('.launch.json')
    if launch.exists() or args.output.exists() or args.output.with_suffix('.summary.json').exists():
        raise FileExistsError('Existing host telemetry must be preserved')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'tools/qa_host_memory_observer_v2.py',ROOT/'tools/qa_host_memory_observer.py']
    bindings={str(p):bounded_sha(p) for p in paths}
    record=dict(formal_enabled=False,completed=False,source_sha256=bindings,python=sys.executable,
        requested_pid=args.pid,interval_seconds=args.interval,requested_window_seconds=args.seconds,
        scope='Standalone read-only CPU process-tree telemetry; no CUDA/model execution or server capacity claim')
    try:
        record['summary']=collect(os.getpid() if args.pid=='self' else int(args.pid),args.output,args.interval,args.seconds)
        record['completed']=record['summary']['complete_samples']>0
        print(json.dumps(record['summary'],indent=2))
    except BaseException as exc:
        record.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        record['sources_unchanged']=all(bounded_sha(p)==h for p,h in bindings.items())
        record['completed']=record['completed'] and record['sources_unchanged']
        with launch.open('x') as stream:json.dump(record,stream,indent=2)


if __name__=='__main__':main()
