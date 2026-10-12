"""State ownership and evaluation evidence for the formal CPU-master runner.

Non-owner ranks deliberately release layers to meta. Their layout records are
not weight hashes; only the CPU master supplies the complete numerical digest.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


SCHEMA = 'realq-state-ownership-v1'


def state_record(model, *, cpu_master, rank, world_size):
    import torch

    state = sorted(model.state_dict().items())
    layout = [(name, list(tensor.shape), str(tensor.dtype)) for name, tensor in state]
    layout_sha = hashlib.sha256(json.dumps(layout, separators=(',', ':')).encode()).hexdigest()
    authoritative = not cpu_master or rank == 0
    meta_count = sum(tensor.is_meta for _, tensor in state)
    if authoritative and meta_count:
        raise RuntimeError('Authoritative state contains meta tensors; full state cannot be verified')
    digest = hashlib.sha256()
    if authoritative:
        for name, tensor in state:
            # Preserve the established canonical byte-hash format exactly.
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(state_audit_schema=SCHEMA, rank=rank, world_size=world_size,
                state_audit_cpu_master=bool(cpu_master), state_audit_authoritative=authoritative,
                state_owner_rank=0 if cpu_master else rank,
                state_sha256=digest.hexdigest() if authoritative else None,
                state_tensors=len(state) if authoritative else 0,
                state_layout_sha256=layout_sha, state_layout_tensors=len(state),
                state_meta_tensors=meta_count,
                state_scope='complete_weight_bytes' if authoritative else 'layout_only_no_weight_bytes')


def install_entry_audit(source):
    """Replace only the frozen entry's post-quantization hash block."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'measured_fwrd']
    if len(functions) != 1:
        raise RuntimeError('Expected exactly one measured_fwrd')
    fn = functions[0]
    begin = next(n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'digest' for t in n.targets))
    end = next(n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'row' for t in n.targets))
    lines = source.splitlines(keepends=True)
    original = ''.join(lines[begin.lineno-1:end.end_lineno])
    if 'tensor.detach().cpu()' not in original or 'state_tensors' not in original:
        raise RuntimeError('Canonical state hash block changed')
    replacement = '''    from tools.storage_method_distributed_audit import state_record as _state_record
    _cpu_master = bool(getattr(args, 'stage2_cpu_master', False))
    if _cpu_master != bool(getattr(analyzer.model, '_gptqplus_stage2_cpu_master', False)):
        raise RuntimeError('CPU master argument/model ownership disagree')
    row = {"mode": mode, "quantization_seconds": elapsed,
           "peak_allocated": torch.cuda.max_memory_allocated(),
           "peak_reserved": torch.cuda.max_memory_reserved()}
    row.update(_state_record(analyzer.model, cpu_master=_cpu_master,
                            rank=int(os.environ.get('RANK', '0')),
                            world_size=int(os.environ.get('WORLD_SIZE', '1'))))
'''
    lines[begin.lineno-1:end.end_lineno] = [replacement]
    result = ''.join(lines)
    compile(result, 'formal_owned_state_entry', 'exec')
    return result


def validate_records(records, traces, *, ranks, cpu_master, tensors=507, layers=28):
    """Freshly validate raw records, with no synthesized non-owner evidence."""
    errors = []
    indexed = {r.get('rank'): r for r in records}
    if len(records) != ranks or set(indexed) != set(range(ranks)):
        errors.append('Missing/duplicate state rank')
    for rank, row in indexed.items():
        authoritative = not cpu_master or rank == 0
        expected = dict(state_audit_schema=SCHEMA, world_size=ranks,
                        state_audit_cpu_master=cpu_master, state_audit_authoritative=authoritative,
                        state_owner_rank=0 if cpu_master else rank,
                        state_layout_tensors=tensors,
                        state_tensors=tensors if authoritative else 0,
                        state_scope='complete_weight_bytes' if authoritative else 'layout_only_no_weight_bytes')
        if any(row.get(k) != v for k, v in expected.items()):
            errors.append(f'rank{rank}: ownership/count/schema mismatch')
        layout = row.get('state_layout_sha256')
        digest = row.get('state_sha256')
        if not isinstance(layout, str) or len(layout) != 64:
            errors.append(f'rank{rank}: missing layout digest')
        if authoritative:
            if row.get('state_meta_tensors') != 0 or not isinstance(digest, str) or len(digest) != 64:
                errors.append(f'rank{rank}: missing complete numerical state')
        elif digest is not None or not isinstance(row.get('state_meta_tensors'), int) or not 0 <= row['state_meta_tensors'] <= tensors:
            errors.append(f'rank{rank}: non-owner claims numerical state or invalid meta count')
    if len({row.get('state_layout_sha256') for row in records}) != 1:
        errors.append('State layouts differ across ranks')
    if not cpu_master and len({row.get('state_sha256') for row in records}) != 1:
        errors.append('Replicated numerical states differ across ranks')
    eval_ranks = [0] if cpu_master else list(range(ranks))
    if set(traces) != set(eval_ranks):
        errors.append('Missing/unexpected evaluation rank')
    for rank, trace in traces.items():
        if (len(trace) != layers or {r.get('layer') for r in trace} != set(range(layers))
                or any(not isinstance(r.get('forward_count'), int) or r['forward_count'] <= 0
                       or not isinstance(r.get('shapes'), list) or len(r['shapes']) != r['forward_count']
                       or not isinstance(r.get('sha256'), str) or len(r['sha256']) != 64 for r in trace)):
            errors.append(f'rank{rank}: incomplete/nonempty layer evidence')
    if not cpu_master and traces and any(t != traces.get(0) for t in traces.values()):
        errors.append('Replicated evaluation traces differ')
    return dict(valid=not errors, errors=errors, schema=SCHEMA,
                numerical_state_ranks=eval_ranks, evaluation_ranks=eval_ranks,
                layout_record_ranks=sorted(indexed), cpu_master=cpu_master)


def audit_directory(directory, *, ranks, cpu_master):
    directory = Path(directory)
    records = [json.loads(p.read_text()) for p in sorted(directory.glob('baseline-profile_metrics_rank*.json'))]
    # File identity and embedded rank must agree; duplicated/misnamed files fail.
    files_ok = all(json.loads(p.read_text()).get('rank') == int(p.stem.rsplit('rank', 1)[1])
                   for p in directory.glob('baseline-profile_metrics_rank*.json'))
    traces = {int(p.stem.rsplit('rank', 1)[1]): json.loads(p.read_text())
              for p in directory.glob('layer_output_fingerprints_rank*.json')}
    report = validate_records(records, traces, ranks=ranks, cpu_master=cpu_master)
    canonical = directory/'layer_output_fingerprints.json'
    metrics = directory/'baseline-profile_metrics.json'
    owner = next((r for r in records if r.get('rank') == 0), None)
    if (not files_ok or owner is None or not metrics.exists() or json.loads(metrics.read_text()) != owner
            or not canonical.exists() or json.loads(canonical.read_text()) != traces.get(0)):
        report['errors'].append('Canonical owner files/rank identity missing or inconsistent')
        report['valid'] = False
    return report, owner
