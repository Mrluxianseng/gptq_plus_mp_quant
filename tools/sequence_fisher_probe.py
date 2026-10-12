"""Identical diagnostics for all three sequence-Fisher experiment arms."""
import hashlib
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

root = Path(os.environ['RESEARCH_RUN_DIR'])
root.mkdir(parents=True, exist_ok=True)
original_static, original_fwrd = impl.collect_static_end_to_end_saliency_and_fisher, impl.gptq_fwrd
original_tokenwise_kl = eval_utils.tokenwise_kl_from_logits
diagnostic_seconds = 0.
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
    torch.cuda.synchronize(); elapsed = time.monotonic() - start
    hashes = []
    for obj in result[:2]:
        digest = hashlib.sha256(); hash_tree(obj, digest); hashes.append(digest.hexdigest())
    (root / 'static_stats.json').write_text(json.dumps(dict(seconds=elapsed, saliency=hashes[0], fisher=hashes[1]), indent=2))
    return result


@torch.no_grad()
def analysis_hook(event, p):
    global diagnostic_seconds
    if event != 'after_layer_quantized':
        return False
    torch.cuda.synchronize(); start = time.monotonic()
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
    delta = (output - target).float()
    fisher = p['static_fisher_by_layer'][p['layer_idx']].float().cpu()
    flat = delta.reshape(-1, delta.shape[-1])
    row = dict(layer=p['layer_idx'], mse=delta.square().mean().item(),
               aggregate_loss=(0.5 * ((flat @ fisher) * flat).sum(-1).mean()).item())
    torch.save(dict(output=output, target=target), root / f"layer_{p['layer_idx']}_outputs.pt")
    with (root / 'layer_metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    torch.cuda.synchronize(); diagnostic_seconds += time.monotonic() - start
    return False


def measured_fwrd(args, analyzer, dataloader, dev):
    args._analysis_hook = analysis_hook
    torch.cuda.synchronize(); start = time.monotonic()
    result = original_fwrd(args, analyzer, dataloader, dev)
    torch.cuda.synchronize(); elapsed = time.monotonic() - start
    digest = hashlib.sha256(); hash_tree(analyzer.model.state_dict(), digest)
    row = dict(quantization_seconds=elapsed, diagnostic_seconds=diagnostic_seconds,
               state_sha256=digest.hexdigest(), peak_allocated=torch.cuda.max_memory_allocated(),
               peak_reserved=torch.cuda.max_memory_reserved())
    (root / 'metrics.json').write_text(json.dumps(row, indent=2))
    logging.info('SEQUENCE_FISHER_METRICS %s', json.dumps(row))
    return result


impl.collect_static_end_to_end_saliency_and_fisher = measured_static
impl.gptq_fwrd = measured_fwrd
eval_utils.tokenwise_kl_from_logits = capture_window_kl
sys.argv[0] = 'ptq.py'
runpy.run_path('ptq.py', run_name='__main__')
(root / 'eval_window_kl.json').write_text(json.dumps(eval_window_kls, indent=2))
