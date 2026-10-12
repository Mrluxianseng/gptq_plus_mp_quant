"""Serial GPU followups after the current baseline worker exits."""
import json
from pathlib import Path
import subprocess
import time

root=Path('outputs/overnight_20260927_0143')
deadline=time.time()+3600
def baseline_running():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            raw=path.read_bytes().replace(b'\x00',b' ')
        except (PermissionError,FileNotFoundError,ProcessLookupError):
            continue
        if b'ptq.py' in raw and b'overnight_20260927_baseline' in raw:
            return True
    return False

while baseline_running():
    if time.time()>deadline:
        raise TimeoutError('baseline did not exit within queue wait budget')
    time.sleep(10)
baseline=(root/'baseline.log').read_text(errors='replace')
if 'Exact KL&PPL on wikitext2:' not in baseline:
    raise RuntimeError('baseline did not successfully reach evaluation; stop queue')
for mode in ('cross-audit','prefix-verify'):
    print('QUEUE_START',mode,time.strftime('%F %T'),flush=True)
    result=subprocess.run(['bash','scripts/overnight_20260927.sh',mode],
                          stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    row={'mode':mode,'returncode':result.returncode,'finished':time.strftime('%F %T')}
    with (root/'queue_results.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print('QUEUE_END',row,flush=True)
    # A failed audit is not a negative research result. Stop for investigation.
    if result.returncode:
        raise RuntimeError(f'{mode} failed; investigate before further GPU runs')
