"""Small CUDA byte-equivalence probe; not a QA/model capacity experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.qa_tensor_fingerprint_stream import tensor_sha256
from tools.audit_qa_memory_rows_completed import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    bindings={str(f):sha(f) for f in (Path(__file__).resolve(),ROOT/'tools/qa_tensor_fingerprint_stream.py')}
    (args.output/'source_bindings.json').write_text(json.dumps(bindings,indent=2))
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(x.strip().isdigit() for x in active.splitlines()): raise RuntimeError('Another GPU process is active')
    import torch
    value=torch.arange(2*131*151936,device='cuda:0').reshape(2,131,151936).bfloat16()
    rows=[]
    for name,tensor in (('contiguous',value),('transposed',value.transpose(1,2)),
        ('empty',value[:,:0,:]),('scalar',value[0,0,0])):
        old=hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        new=tensor_sha256(tensor,max_bytes=8*1024*1024)
        rows.append(dict(name=name,shape=list(tensor.shape),stride=list(tensor.stride()),old=old,new=new,equal=old==new))
    torch.cuda.synchronize()
    result=dict(valid=all(x['equal'] for x in rows) and all(sha(f)==h for f,h in bindings.items()),
        checks=rows,scope='CUDA tensor fingerprint equivalence only; no QA/model memory benefit claim')
    with (args.output/'status.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if not result['valid']: raise SystemExit(1)


if __name__=='__main__': main()
