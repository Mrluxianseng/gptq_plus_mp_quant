"""Summarize completed paired refresh diagnostics without dropping cold builds."""
import argparse
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
args = parser.parse_args()
root = args.directory
complete = json.loads((root/'paired_refresh_complete.json').read_text())
rows = [json.loads(s) for s in (root/'paired_refresh.jsonl').read_text().splitlines() if s.strip()]
assert len(rows)==complete['verified_pairs'] and rows
assert all(r['gradient_exact'] for r in rows)

def summary(items):
    if not items:
        return None
    a = sum(r['seconds']['baseline'] for r in items)
    b = sum(r['seconds']['replay'] for r in items)
    return {'pairs':len(items),'baseline_seconds':a,'replay_seconds':b,
            'aggregate_ratio':a/b,
            'median_paired_ratio':statistics.median(r['seconds']['baseline']/r['seconds']['replay'] for r in items),
            'replay_faster_pairs':sum(r['seconds']['replay']<r['seconds']['baseline'] for r in items)}

result = {'all_including_cold':summary(rows),
          'cold_or_partial_miss':summary([r for r in rows if r['new_misses']]),
          'fully_warm':summary([r for r in rows if not r['new_misses']]),
          'baseline_first':summary([r for r in rows if r['order'][0]=='baseline']),
          'replay_first':summary([r for r in rows if r['order'][0]=='replay']),
          'by_layer':{str(i):summary([r for r in rows if r['layer']==i]) for i in sorted({r['layer'] for r in rows},key=str)},
          'limits':['Real refresh function only, not whole PTQ runtime.',
                    'Sequential calls share allocator/cache and retain the first gradient until comparison.',
                    'Calls along one trajectory are correlated; no statistical confidence claim.',
                    'Each pair has identical data/weights and exact gradient/loss/count checks; order alternates.']}
(root/'paired_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
