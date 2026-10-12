"""Compare quantization block metrics between runs (whole model is stronger)."""
import argparse, json, re
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
a=p.parse_args()
def rows(path):
    text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',path.read_text(errors='replace'))
    return re.findall(r'\| INFO\] (block-metrics [^\r\n]+)',text)
x,y=rows(a.baseline),rows(a.candidate)
n=min(len(x),len(y))
d=next((i for i in range(n) if x[i]!=y[i]),None)
print(json.dumps({'baseline_records':len(x),'candidate_records':len(y),
                  'compared':n,'exact_logged_trace_match':d is None and len(x)==len(y),
                  'first_difference_index':d,
                  'baseline_first_difference':x[d] if d is not None else None,
                  'candidate_first_difference':y[d] if d is not None else None,
                  'scope':'Rounded logged block metrics only.'},indent=2))
