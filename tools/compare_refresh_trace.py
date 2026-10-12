"""Read-only trajectory diagnostic; partial equality is not final-state proof."""
import argparse
import json
from pathlib import Path
import re
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='outputs/resume_20260927_1216')
args = parser.parse_args()
root = Path(args.root)

def read(arm):
    folder = root/arm
    text = (folder/'run.log').read_text(errors='replace')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    rows = re.findall(r'\| INFO\] (block-metrics [^\r\n]+)', text)
    # Only count complete newline-terminated records from a live writer.
    if text and not text.endswith('\n') and rows and text.endswith(rows[-1]):
        rows.pop()
    status = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else None
    return rows, status, text

a, sa, baseline_log = read('baseline-profile')
b, sb, log = read('prefix-fast')
n = min(len(a),len(b))
first = next(({'index':i,'baseline':a[i],'replay':b[i]} for i in range(n) if a[i]!=b[i]), None)
cache = re.findall(r'OVERNIGHT_PREFIX hits=(\d+) misses=(\d+) bytes=(\d+) verified=(\d+)', log)
result = {'baseline_records':len(a),'replay_records':len(b),'compared_records':n,
          'first_text_difference':first,'baseline_status':sa,'replay_status':sb,
          'last_cache_counters':dict(zip(('hits','misses','bytes','verified'),map(int,cache[-1]))) if cache else None,
          'scope':'Rounded logged block metrics only; no replacement for exact final state or gradient checks.'}

def phases(text):
    starts = {}
    down = {}
    for stamp, kind, number in re.findall(
            r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| INFO\] '
            r'(?:(Layer) (\d+) grad_lr schedule)', text):
        starts[int(number)] = datetime.fromisoformat(stamp)
    for stamp, number in re.findall(
            r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| INFO\] '
            r'Applying block_gd config for layer=(\d+) module=mlp.down_proj:', text):
        down[int(number)] = datetime.fromisoformat(stamp)
    return {i:{'before_down_seconds':(down[i]-starts[i]).total_seconds(),
               'down_to_next_layer_seconds':(starts[i+1]-down[i]).total_seconds()}
            for i in starts if i in down and i+1 in starts}

pa, pb = phases(baseline_log), phases(log)
common = sorted(pa.keys() & pb.keys())
result['completed_layer_phase_windows'] = [dict(layer=i,baseline=pa[i],replay=pb[i]) for i in common]
result['phase_window_limit'] = 'Wall-clock log timestamps rounded to seconds, not synchronized kernel profiles. Before-down covers code outside the reused prefix path.'
print(json.dumps(result,indent=2))
