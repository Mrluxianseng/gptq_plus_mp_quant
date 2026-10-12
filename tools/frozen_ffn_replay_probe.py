"""REAL-Q probe with optional exact-replay checks for the frozen FFN prefix."""
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from utils import eval_utils
from gptq_utils import gptq_plus_utils as impl
from gptq_utils.frozen_ffn_replay import FrozenFFNPrefix

root = Path(os.environ['RESEARCH_RUN_DIR'])
root.mkdir(parents=True, exist_ok=True)
original_static = impl.collect_static_end_to_end_saliency_and_fisher
original_fwrd = impl.gptq_fwrd
original_gradient = impl.collect_true_weight_gradient
signature = inspect.signature(original_gradient)
original_tokenwise_kl = eval_utils.tokenwise_kl_from_logits
enabled = os.environ.get('FROZEN_FFN_REPLAY') == '1'
verify = os.environ.get('FROZEN_FFN_VERIFY') == '1'
cache = None
cache_totals = {'instances': 0, 'hits': 0, 'misses': 0,
                'verified_hits': 0, 'peak_bytes': 0}
eval_window_kls = []


def capture_window_kl(student_logits, reference_logits):
    value = original_tokenwise_kl(student_logits, reference_logits)
    eval_window_kls.append(float(value.detach().float().mean().cpu()))
    return value


def hash_tree(obj, digest):
    if isinstance(obj, torch.Tensor):
        digest.update(str((obj.shape, obj.dtype)).encode())
        digest.update(obj.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(obj, dict):
        for key in sorted(obj):
            digest.update(str(key).encode()); hash_tree(obj[key], digest)
    elif isinstance(obj, (tuple, list)):
        for value in obj:
            hash_tree(value, digest)
    else:
        digest.update(str(obj).encode())


def measured_static(**kwargs):
    torch.cuda.synchronize(); start = time.monotonic()
    result = original_static(**kwargs)
    torch.cuda.synchronize()
    hashes = []
    for obj in result[:2]:
        digest = hashlib.sha256(); hash_tree(obj, digest); hashes.append(digest.hexdigest())
    (root / 'static_stats.json').write_text(json.dumps({'saliency': hashes[0], 'fisher': hashes[1]}, indent=2))
    return result


def cache_gradient(*args, **kwargs):
    global cache
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    p = bound.arguments
    module_name = p['module_name']
    eligible = (enabled and module_name in ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
                and not p['joint_targets'] and not p['extra_weight_overrides']
                and not p['global_shuffle'] and not p['grad_sq_accum']
                and p['refresh_loss_type'] == 'fisher_diag_mse'
                and p['next_layer'] is None and p['layer_recorder'] is None)
    if not eligible:
        _release_cache()
        cache = None
        return original_gradient(*args, **kwargs)
    if cache is not None and cache.layer is not p['layer']:
        _release_cache()
        cache = None
    if cache is None:
        cache = FrozenFFNPrefix(p['layer'])
        cache_totals['instances'] += 1
    indices = p['sample_indices']
    if indices is None:
        indices = list(range(p['inps'].shape[0]))
    indices = list(indices)
    keys = [tuple(indices[i:i + p['bsz']]) for i in range(0, len(indices), p['bsz'])]
    if not keys:
        return original_gradient(*args, **kwargs)
    with cache.activate(keys, module_name, verify=verify):
        result = original_gradient(*args, **kwargs)
    cache_totals['peak_bytes'] = max(cache_totals['peak_bytes'], cache.bytes)
    return result


def _release_cache():
    global cache
    if cache is None:
        return
    cache_totals['hits'] += cache.hits
    cache_totals['misses'] += cache.misses
    cache_totals['verified_hits'] += cache.verified_hits
    cache_totals['peak_bytes'] = max(cache_totals['peak_bytes'], cache.bytes)
    cache.entries.clear()
    cache.bytes = 0


@torch.no_grad()
def analysis_hook(event, p):
    if event != 'after_layer_quantized':
        return False
    outputs = []
    for j in range(min(4, len(p['inps']))):
        out = p['layer'](p['inps'][j].unsqueeze(0).to(p['dev']),
                         attention_mask=p['attention_mask'], position_ids=p['position_ids'],
                         position_embeddings=p['position_embeddings'])
        if isinstance(out, (tuple, list)):
            out = out[0]
        outputs.append(out.detach().cpu())
    output = torch.cat(outputs)
    target = p['fp_inps'][:len(output)].detach().cpu()
    fisher = p['static_fisher_by_layer'][p['layer_idx']].float().cpu()
    delta = (output - target).float()
    flat = delta.reshape(-1, delta.shape[-1])
    row = {'layer': p['layer_idx'], 'mse': delta.square().mean().item(),
           'aggregate_loss': (0.5 * ((flat @ fisher) * flat).sum(-1).mean()).item()}
    torch.save({'output': output, 'target': target}, root / f"layer_{p['layer_idx']}_outputs.pt")
    with (root / 'layer_metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    return False


def measured_fwrd(args, analyzer, dataloader, dev):
    args._analysis_hook = analysis_hook
    torch.cuda.synchronize(); start = time.monotonic()
    result = original_fwrd(args, analyzer, dataloader, dev)
    torch.cuda.synchronize(); elapsed = time.monotonic() - start
    _release_cache()
    digest = hashlib.sha256(); hash_tree(analyzer.model.state_dict(), digest)
    row = {'quantization_seconds': elapsed, 'state_sha256': digest.hexdigest(),
           'peak_allocated': torch.cuda.max_memory_allocated(),
           'peak_reserved': torch.cuda.max_memory_reserved(), 'cache_enabled': enabled,
           'cache_verify': verify, 'cache_instances': cache_totals['instances'],
           'cache_hits': cache_totals['hits'], 'cache_misses': cache_totals['misses'],
           'cache_verified_hits': cache_totals['verified_hits'],
           'cache_peak_bytes': cache_totals['peak_bytes']}
    (root / 'metrics.json').write_text(json.dumps(row, indent=2))
    logging.info('FROZEN_FFN_METRICS %s', json.dumps(row))
    return result


impl.collect_static_end_to_end_saliency_and_fisher = measured_static
impl.collect_true_weight_gradient = cache_gradient
impl.gptq_fwrd = measured_fwrd
eval_utils.tokenwise_kl_from_logits = capture_window_kl
sys.argv[0] = 'ptq.py'
runpy.run_path('ptq.py', run_name='__main__')
(root / 'eval_window_kl.json').write_text(json.dumps(eval_window_kls, indent=2))
