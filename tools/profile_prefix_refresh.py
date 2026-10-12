"""Paired real-refresh diagnostic, not an end-to-end speed benchmark.

Run through the existing PTQ CLI. Each eligible refresh evaluates both paths
at identical weights/data, alternates order, verifies equality, and returns
the baseline result to preserve the quantization trajectory.
"""
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from gptq_utils import gptq_plus_utils as impl
from gptq_utils.frozen_prefix_replay import FrozenDownPrefix

original = impl.collect_true_weight_gradient
signature = inspect.signature(original)
cache = None
counter = 0
output = Path(os.environ['RESEARCH_RUN_DIR'])
output.mkdir(parents=True, exist_ok=True)

def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter()-start

def paired(*args, **kwargs):
    global cache, counter
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    p = bound.arguments
    eligible = (p['module_name']=='mlp.down_proj'
                and not p['joint_targets'] and not p['extra_weight_overrides']
                and not p['global_shuffle'] and not p['grad_sq_accum']
                and p['refresh_loss_type']=='fisher_diag_mse'
                and p['next_layer'] is None and p['layer_recorder'] is None)
    if not eligible:
        cache = None
        return original(*args, **kwargs)
    if cache is None or cache.layer is not p['layer']:
        cache = FrozenDownPrefix(p['layer'])
    indices = p['sample_indices']
    if indices is None:
        indices = list(range(p['inps'].shape[0]))
    indices = list(indices)
    keys = [tuple(indices[i:i+p['bsz']]) for i in range(0,len(indices),p['bsz'])]
    if not keys:
        return original(*args, **kwargs)

    def replay():
        with cache.activate(keys):
            return original(*args, **kwargs)

    before = (cache.hits, cache.misses)
    order = ('baseline','replay') if counter % 2 == 0 else ('replay','baseline')
    results, seconds = {}, {}
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    for name in order:
        call = replay if name=='replay' else lambda: original(*args, **kwargs)
        results[name], seconds[name] = timed(call)
        if not torch.equal(cpu_rng,torch.random.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):
            raise AssertionError('refresh consumes randomness; paired protocol needs explicit RNG replay')
    a, b = results['baseline'], results['replay']
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    assert a[1:3] == b[1:3], 'count/loss mismatch'
    # Supported path should return only scalar/empty diagnostics; fail if it
    # acquires tensor-valued side effects instead of silently ignoring them.
    assert a[3] == b[3], 'extras mismatch'
    row = {'pair':counter,'layer':getattr(p['layer'].self_attn,'layer_idx',None),
           'order':order,'seconds':seconds,'gradient_exact':True,
           'new_hits':cache.hits-before[0],'new_misses':cache.misses-before[1],
           'retained_bytes':cache.bytes,'samples':len(indices),'batch_size':p['bsz'],
           'scope':'paired refresh only; includes cold build/lookup but excludes PTQ phases outside refresh'}
    with (output/'paired_refresh.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    counter += 1
    return a

impl.collect_true_weight_gradient = paired
sys.argv[0] = 'ptq.py'
runpy.run_path('ptq.py',run_name='__main__')
if counter == 0:
    raise RuntimeError('no eligible paired refreshes ran; diagnostic is inconclusive')
(output/'paired_refresh_complete.json').write_text(json.dumps({'verified_pairs':counter}))
