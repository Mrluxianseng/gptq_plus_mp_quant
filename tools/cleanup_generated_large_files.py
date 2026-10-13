#!/usr/bin/env python3
"""Preview/apply deletion of generated server payloads, preserving models/data.

Only cache/ and outputs/ below a verified repository root are scanned. Original
models, datasets, environments, code, logs and numerical audit records remain.
Prepared weights are explicitly disposable; their _SUCCESS markers are removed
with them so a later run cannot mistake an incomplete model for a ready one.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat
import subprocess

PAYLOAD_EXTENSIONS = {'.pt', '.pth', '.npy', '.npz', '.pkl', '.pickle', '.dat', '.h5', '.hdf5'}
PREPARED_NAMES = {'_prepared_checkpoints', '_prepared_checkpoint', 'prepared_checkpoints', 'prepared_checkpoint'}
PROTECTED_NAMES = {'tokens', 'data', 'dataset', 'datasets', 'models', 'llmmodels', 'huggingface', 'hf_cache', 'env', '.venv'}
ACTIVE = re.compile(r'(?:^|[ /])(?:run_storage_method_formal\.(?:py|sh)|run_storage_method_pipeline\.py|'
                    r'ptq\.py|torchrun|torch\.distributed\.run)(?:\s|$)|\sbaseline-profile(?:\s|$)')


def active_experiments():
    result = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if ACTIVE.search(line)]


def repository_root(value):
    root = Path(value).resolve(strict=True)
    if not (root/'ptq.py').is_file() or not (root/'process_args.py').is_file():
        raise ValueError('Root must be the gptq_plus repository containing ptq.py and process_args.py')
    return root


def prepared_root(path, root):
    parts = path.relative_to(root).parts
    for index, part in enumerate(parts):
        if part in PREPARED_NAMES and index+1 < len(parts)-1:
            return root.joinpath(*parts[:index+2])
    return None


def eligible(path, root, minimum):
    parts = path.relative_to(root).parts
    if parts[0] not in {'cache', 'outputs'} or any(p.lower() in PROTECTED_NAMES for p in parts[1:-1]):
        return False
    prepared = prepared_root(path, root)
    if prepared and path.suffix.lower() in {'.safetensors', '.bin', '.pt', '.pth'}:
        return True
    # Model weights outside a prepared checkpoint are protected. Most raw
    # downloaded dataset formats are not payload extensions either.
    if path.name.lower().startswith(('model', 'pytorch_model', 'qmodel', 'checkpoint')):
        return False
    if (path.parent/'config.json').is_file():
        return False
    if (path.name.lower().startswith(('qwen', 'llama', 'mistral'))
            and not any(tag in path.name.lower() for tag in ('fisher', 'saliency', 'gradient', 'logits', 'hessian'))):
        return False
    if path.suffix.lower() not in PAYLOAD_EXTENSIONS:
        return False
    try:
        return path.stat().st_size >= minimum
    except FileNotFoundError:
        return False


def fingerprint(path):
    info = path.lstat()
    return dict(device=info.st_dev, inode=info.st_ino, bytes=info.st_size,
                modified_ns=info.st_mtime_ns, kind='symlink' if stat.S_ISLNK(info.st_mode) else 'file',
                target=os.readlink(path) if stat.S_ISLNK(info.st_mode) else None,
                links=info.st_nlink, allocated_bytes=getattr(info, 'st_blocks', 0)*512)


def safe_parent(path, root):
    path.relative_to(root)
    if '..' in path.parts or path.relative_to(root).parts[0] not in {'cache', 'outputs'}:
        raise ValueError('Deletion path outside allowed cache/outputs scope')
    for parent in path.parents:
        if parent == root:
            return
        if parent.is_symlink():
            raise ValueError('Refusing a directory symlink in deletion path: '+str(parent))
    raise ValueError('Deletion path escaped repository root')


def make_plan(root, minimum):
    root = repository_root(root)
    selected = {}
    errors = []
    def onerror(exc):
        errors.append(str(exc))
    for name in ('cache', 'outputs'):
        base = root/name
        if not base.is_dir() or base.is_symlink():
            continue
        for current, dirs, files in os.walk(base, followlinks=False, onerror=onerror):
            dirs[:] = [d for d in dirs if d.lower() not in PROTECTED_NAMES and not (Path(current)/d).is_symlink()]
            for name in files:
                path = Path(current)/name
                if eligible(path, root, minimum):
                    safe_parent(path, root)
                    row = dict(path=str(path), **fingerprint(path), reason='generated_large_payload')
                    selected[str(path)] = row
                    checkpoint = prepared_root(path, root)
                    if checkpoint and (checkpoint/'_SUCCESS').is_file():
                        marker = checkpoint/'_SUCCESS'
                        selected[str(marker)] = dict(path=str(marker), **fingerprint(marker), reason='invalidate_removed_prepared_weights')
    if errors:
        raise RuntimeError('Incomplete filesystem scan; refusing cleanup: '+repr(errors))
    groups = Counter((row['device'], row['inode']) for row in selected.values() if row['kind'] == 'file')
    freed, counted = 0, set()
    for row in selected.values():
        key = row['device'], row['inode']
        if row['kind'] == 'file' and key not in counted and groups[key] >= row['links']:
            freed += row['allocated_bytes']
            counted.add(key)
    return dict(schema=1, root=str(root), minimum_bytes=minimum,
                created=datetime.now().isoformat(), files=sorted(selected.values(), key=lambda r: (-r['bytes'], r['path'])),
                estimated_reclaimed_bytes=freed,
                note='Hardlink bytes are reclaimed only when all stored names are removed; original/non-prepared models, datasets and logs are protected.')


def apply_plan(plan, process_check=active_experiments):
    running = process_check()
    if running:
        raise RuntimeError('Experiments are still running; nothing deleted:\n'+'\n'.join(running))
    root = repository_root(plan['root'])
    current = {row['path']: row for row in make_plan(root, plan['minimum_bytes'])['files']}
    signature = ('device', 'inode', 'bytes', 'modified_ns', 'kind', 'target')
    # Check every target before deleting anything. Newly created files that
    # were absent from the saved plan are never added to this apply operation.
    for row in plan['files']:
        path = Path(row['path'])
        safe_parent(path, root)
        if row['path'] not in current or any(row.get(k) != current[row['path']].get(k) for k in signature):
            raise RuntimeError('Planned file changed or no longer eligible; nothing deleted: '+str(path))
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    audit = root/f'cleanup-generated-result-{stamp}.jsonl'
    running = process_check()
    if running:
        raise RuntimeError('An experiment started during preflight; nothing deleted')
    with audit.open('x', encoding='utf-8') as stream:
        # Markers first: even an interrupted cleanup cannot leave a deleted
        # prepared model incorrectly marked complete.
        rows = sorted(plan['files'], key=lambda r: r['reason'] != 'invalidate_removed_prepared_weights')
        for row in rows:
            path = Path(row['path'])
            safe_parent(path, root)
            actual = fingerprint(path)
            if any(actual.get(k) != row.get(k) for k in signature):
                raise RuntimeError('File changed during apply; cleanup stopped: '+str(path))
            path.unlink()  # Unlink one checked file; never a recursive directory removal.
            stream.write(json.dumps(dict(path=str(path), deleted=True, bytes=row['bytes']))+'\n')
            stream.flush()
            print('DELETED '+str(path), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--min-mib', type=int, default=100,
                        help='Minimum size of generated payloads; prepared weights are always selected.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', type=Path, help='Write a new preview plan; no files are deleted.')
    mode.add_argument('--apply', type=Path, help='Delete only unchanged eligible files in this saved plan.')
    args = parser.parse_args()
    if args.min_mib < 1:
        parser.error('min-mib must be positive')
    if args.plan:
        plan = make_plan(args.root, args.min_mib*1024**2)
        with args.plan.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, indent=2)
        for row in plan['files']:
            print(f"{row['bytes']/1024**3:.3f} GiB  {row['path']}  [{row['reason']}]")
        print(f"PLAN {args.plan}; {len(plan['files'])} files; estimated reclaim {plan['estimated_reclaimed_bytes']/1024**3:.2f} GiB; nothing deleted")
    else:
        plan = json.loads(args.apply.read_text(encoding='utf-8'))
        if repository_root(args.root) != repository_root(plan['root']):
            raise ValueError('Command root differs from saved plan root')
        print('Audit: '+str(apply_plan(plan)))


if __name__ == '__main__':
    main()
