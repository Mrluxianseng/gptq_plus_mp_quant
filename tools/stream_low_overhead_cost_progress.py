"""Append the authoritative cost log to the existing research progress view.

Reads CPU files only, does not control experiments or retain CUDA tensors.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();position=0;seen=None
    with args.output.open('a',encoding='utf8') as dest:
        def write(value):dest.write(value);dest.flush()
        write(f'\n[cost-progress {time.strftime("%F %T")}] CPU reader for {args.queue}\n')
        while True:
            active=args.queue/'active_run.json'
            if active.exists():
                try:row=json.loads(active.read_text())
                except json.JSONDecodeError:row=None
                if row is not None and row!=seen:
                    write('\n[cost-active] '+json.dumps(row)+'\n');seen=row
            source=args.queue/'combined.log'
            if source.exists():
                with source.open(encoding='utf8',errors='replace') as stream:
                    stream.seek(position);chunk=stream.read();position=stream.tell()
                if chunk:write(f'\n[cost-log-source {source}]\n'+chunk)
            status=args.queue/'status.json'
            if status.exists():
                try:final=json.loads(status.read_text())
                except json.JSONDecodeError:time.sleep(1);continue
                write('\n[cost-controller-completed] '+json.dumps(final)+'\n');break
            time.sleep(1)


if __name__=='__main__':main()
