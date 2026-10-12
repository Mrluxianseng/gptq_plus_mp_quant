"""Opt-in runtime auditing/repair, leaving the user's baseline source intact."""
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import time
import inspect
import hashlib
import importlib.util

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from gptq_utils import gptq_plus_utils as impl

if os.environ.get('REPLAY_IMPL_PATH'):
    spec=importlib.util.spec_from_file_location('gptq_utils.frozen_prefix_replay',os.environ['REPLAY_IMPL_PATH'])
    replay_module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=replay_module
    spec.loader.exec_module(replay_module)

mode = sys.argv.pop(1)
assert mode in ('cross-audit', 'cross-fixed', 'ts-fixed', 'prefix-verify', 'prefix-fast', 'baseline-profile')
root = Path(os.environ.get('RESEARCH_RUN_DIR','outputs/overnight_20260927_0143'))
root.mkdir(parents=True,exist_ok=True)
original_cross = impl.compute_cross_module_compensation
original_push = impl._two_sided_block_push
original_gradient = impl.collect_true_weight_gradient
counter = 0

def cross_probe(*args, **kwargs):
    global counter
    assert not args, 'pipeline should use explicit keyword arguments'
    counter += 1
    layer = kwargs['layer']
    _, target = impl.resolve_quant_module(kwargs['full'], kwargs['target_name'])
    captures = {'outputs': []}
    def input_hook(module, inputs, output):
        captures['h'] = inputs[0].detach().float()
    def output_hook(module, inputs, output):
        if isinstance(output, (tuple,list)):
            output = output[0]
        captures['outputs'].append(output.detach())
    handles = [target.register_forward_hook(input_hook), layer.register_forward_hook(output_hook)]
    start = time.monotonic()
    try:
        old = original_cross(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    assert len(captures['outputs']) == 2
    # Preserve the existing BF16 subtraction BEFORE conversion to FP32.
    # Otherwise an audit/repair would inadvertently also change its arithmetic.
    d = (captures['outputs'][1]-captures['outputs'][0]).float().reshape(-1,old.shape[0])
    h = captures['h'].reshape(-1,old.shape[1])
    H = h.T@h
    lam = kwargs.get('damping',0.01)*H.diagonal().mean()
    H.diagonal().add_(lam)
    rhs = d.T@h
    fixed = torch.linalg.solve(H, -rhs.T).T
    F = kwargs.get('layer_output_fisher')
    def objective(delta):
        residual = d+h@delta.T
        return {'mse':float(residual.square().mean()),
                'fisher':None if F is None else float(((residual@F.to(residual))*residual).sum(-1).mean())}
    row = {'mode':mode,'pair_index':counter,'feeder':kwargs['feeder_name'],
           'target':kwargs['target_name'],'F_is_none':F is None,
           'F_shape':None if F is None else list(F.shape),
           'F_norm':None if F is None else float(F.float().norm()),
           'old_norm':float(old.norm()),'fixed_norm':float(fixed.norm()),
           'old_relative_stationarity':float((old@H+rhs).norm()/rhs.norm().clamp_min(1e-30)),
           'fixed_relative_stationarity':float((fixed@H+rhs).norm()/rhs.norm().clamp_min(1e-30)),
           'before':objective(torch.zeros_like(old)),'legacy':objective(old),'fixed':objective(fixed),
           'metrics_are_linear_predictions':True,
           'structurally_linear_target':kwargs['target_name']=='mlp.down_proj',
           'seconds':time.monotonic()-start}
    logging.info('OVERNIGHT_CROSS_AUDIT %s',json.dumps(row))
    with (root/(mode+'.jsonl')).open('a') as file:
        file.write(json.dumps(row)+'\n')
    return fixed if mode=='cross-fixed' else old

fixed_push_calls = 0
def fixed_push(W1,U,A,scale,q_lo,maxq):
    global fixed_push_calls
    fixed_push_calls += 1
    Q,Qi,Err = original_push(W1,U,A,scale,q_lo,maxq)
    # Cross-block GPTQ compensation needs actual W-Q in the input Cholesky
    # basis. Row feedback means raw per-entry errors are not that residual.
    An = A/torch.diagonal(A)[:,None]
    logging.info('TWO_SIDED_FIXED_PUSH calls=%d rows=%d cols=%d',
                 fixed_push_calls,Err.shape[0],Err.shape[1])
    return Q,Qi,An.T@Err

prefix_cache = None
gradient_signature = inspect.signature(original_gradient)
def prefix_gradient(*args,**kwargs):
    global prefix_cache
    bound = gradient_signature.bind(*args,**kwargs)
    bound.apply_defaults()
    p = bound.arguments
    eligible = (p['module_name']=='mlp.down_proj'
                and not p['joint_targets'] and not p['extra_weight_overrides']
                and not p['global_shuffle'] and not p['grad_sq_accum']
                and p['refresh_loss_type']=='fisher_diag_mse'
                and p['next_layer'] is None)
    if not eligible:
        prefix_cache=None
        return original_gradient(*args,**kwargs)
    from gptq_utils.frozen_prefix_replay import FrozenDownPrefix
    if prefix_cache is None or prefix_cache.layer is not p['layer']:
        prefix_cache=FrozenDownPrefix(p['layer'])
    indices=p['sample_indices']
    if indices is None: indices=list(range(p['inps'].shape[0]))
    indices=list(indices)
    keys=[tuple(indices[i:i+p['bsz']]) for i in range(0,len(indices),p['bsz'])]
    if not keys: return original_gradient(*args,**kwargs)
    with prefix_cache.activate(keys,verify=mode=='prefix-verify'):
        result=original_gradient(*args,**kwargs)
    if mode=='prefix-verify':
        reference=original_gradient(*args,**kwargs)
        torch.testing.assert_close(result[0],reference[0],rtol=0,atol=0)
        assert result[1:3]==reference[1:3], 'count/loss mismatch'
        logging.info('OVERNIGHT_PREFIX_GRADIENT_EXACT_OK')
    logging.info('OVERNIGHT_PREFIX hits=%d misses=%d bytes=%d verified=%d',
        prefix_cache.hits,prefix_cache.misses,prefix_cache.bytes,prefix_cache.verified_hits)
    return result

if mode.startswith('prefix'):
    impl.collect_true_weight_gradient=prefix_gradient
elif mode.startswith('cross'):
    impl.compute_cross_module_compensation = cross_probe
elif mode=='ts-fixed':
    impl._two_sided_block_push = fixed_push
    _original_ts_fisher_for = impl._ts_fisher_for
    def trace_ts_fisher_for(layer_idx, module_name, static_fisher_by_layer, args):
        value = _original_ts_fisher_for(layer_idx, module_name, static_fisher_by_layer, args)
        logging.info('TS_FISHER_TRACE layer=%s module=%s selected=%s shape=%s blockwise_env=%s',
                     layer_idx, module_name, value is not None,
                     None if value is None else tuple(value.shape), os.environ.get('TS_BLOCKWISE'))
        return value
    impl._ts_fisher_for = trace_ts_fisher_for

# Identical lightweight phase timing and state fingerprints in both performance
# arms. Hashing is outside the timed quantization region; it is not speedup.
original_fwrd=impl.gptq_fwrd
def measured_fwrd(args,analyzer,dataloader,dev):
    torch.cuda.synchronize()
    start=time.monotonic()
    result=original_fwrd(args,analyzer,dataloader,dev)
    torch.cuda.synchronize()
    seconds=time.monotonic()-start
    h=hashlib.sha256()
    count=0
    for name,t in sorted(analyzer.model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(t.shape),t.dtype)).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        count+=1
    row={'mode':mode,'quantization_seconds':seconds,'state_sha256':h.hexdigest(),
         'state_tensors':count,'peak_allocated':torch.cuda.max_memory_allocated(),
         'peak_reserved':torch.cuda.max_memory_reserved()}
    (root/(mode+'_metrics.json')).write_text(json.dumps(row,indent=2))
    logging.info('OVERNIGHT_METRICS %s',json.dumps(row))
    return result
impl.gptq_fwrd=measured_fwrd
sys.argv[0]='ptq.py'
runpy.run_path('ptq.py', run_name='__main__')
if mode=='ts-fixed' and os.environ.get('TS_BLOCKWISE')=='1':
    if fixed_push_calls <= 0:
        raise RuntimeError('TS_BLOCKWISE was requested but fixed row propagation never ran')
