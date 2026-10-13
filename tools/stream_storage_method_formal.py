"""Follow an already running formal plan in one terminal/log, with no GPU work.

Standalone: copy this single file to /tmp on a live server. It does not require
pulling/altering the source files bound by the running experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(directory, *, interval=0.5, destination=None, follow_once=False, controller_log=None):
    directory = Path(directory).resolve()
    plan = read(directory/'plan.json')
    if plan.get('aggregate_worker_stdout'):
        raise ValueError('This new plan already forwards every worker to its nohup/console log; follow that one log directly.')
    destination = destination if destination is not None else sys.stdout.buffer
    offsets = {}
    announced = set()
    def announce(message):
        destination.write((message+'\n').encode())
        destination.flush()
    while True:
        jobs = ([dict(tag='formal-controller', controller_only=True)] if controller_log is not None else []) + plan['jobs']
        for job in jobs:
            # The plan records absolute server paths. Keep arm/stage identity.
            if job.get('controller_only'):
                files = [(Path(controller_log), 'controller')]
            else:
                files = [(directory/(job['tag']+'.log'), 'pipeline')]
                files += [(Path(job['result'])/(stage+'.log'), stage) for stage in ('stage0', 'stage2')]
            for path, stage in files:
                try:
                    stream = path.open('rb')
                except FileNotFoundError:
                    continue
                with stream:
                    previous = offsets.get(str(path), 0)
                    stream.seek(0, 2)
                    if stream.tell() < previous:
                        raise RuntimeError('Log truncated: '+str(path))
                    stream.seek(previous)
                    while True:
                        data = stream.read(65536)
                        if not data:
                            break
                        key = (job['tag'], stage)
                        if key not in announced:
                            announce('[formal-stream] '+job['tag']+' '+stage+' '+str(path))
                            announced.add(key)
                        destination.write(data.replace(b'\r\n', b'\n').replace(b'\r', b'\n'))
                        destination.flush()
                    offsets[str(path)] = stream.tell()
        status = directory/'status.json'
        if status.exists():
            # It is written once, exclusively, after the controller finishes.
            # Concurrent reads can see partial JSON: retry instead of faking a status.
            try:
                result = read(status)
            except json.JSONDecodeError:
                result = None
            if result is not None:
                announce('[formal-stream] controller finished: '+json.dumps(result, ensure_ascii=False))
                return result
        if follow_once:
            return None
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--interval-seconds', type=float, default=0.5)
    parser.add_argument('--controller-log', type=Path, help='Original nohup log, to include top-level controller output too.')
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error('interval must be positive')
    run(args.run_directory, interval=args.interval_seconds, controller_log=args.controller_log)


if __name__ == '__main__':
    main()
