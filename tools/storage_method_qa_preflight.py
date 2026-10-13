"""Load the exact paper QA tasks on CPU before a formal GPU experiment.

Default: verify the same offline environment used by the worker. --download
explicitly populates shared HF caches online; run offline verification after it.
No model is loaded, and GPU devices are hidden before importing dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def dataset_cache_directory(root, environment):
    """Choose a path hint only; task loaders must still verify all QA data."""
    if environment.get('HF_DATASETS_CACHE'):
        return str(Path(environment['HF_DATASETS_CACHE']).resolve())
    root = Path(root)
    for candidate in (root.parent/'hf_cache/datasets', root/'hf_cache/datasets'):
        if candidate.is_dir() and (candidate/'piqa').is_dir():
            return str(candidate.resolve())
    return None


def task_objects(value):
    if hasattr(value, 'dataset'):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from task_objects(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from task_objects(child)


def check_datasets(manager, names, load_task):
    records, errors = [], []
    for name in names:
        print(f'[qa-preflight] loading {name}: datasets only, no model/GPU', flush=True)
        try:
            tasks = list(task_objects(load_task([name], task_manager=manager)))
            if not tasks:
                raise RuntimeError('Task resolved to no dataset-bearing leaf tasks')
            leaves = []
            for task in tasks:
                dataset = task.dataset
                if not dataset:
                    raise RuntimeError('Task has no loaded dataset')
                splits = {split: len(rows) for split, rows in dataset.items()}
                if not splits or any(rows <= 0 for rows in splits.values()):
                    raise RuntimeError(f'Empty dataset splits: {splits}')
                if task.has_test_docs():
                    docs = task.test_docs()
                elif task.has_validation_docs():
                    docs = task.validation_docs()
                else:
                    raise RuntimeError('Task has no evaluation documents')
                if not len(docs):
                    raise RuntimeError('Task has no nonempty evaluation documents')
                leaves.append(dict(splits=splits, evaluation_documents=len(docs),
                                   dataset_fingerprints={k: getattr(v, '_fingerprint', None) for k, v in dataset.items()}))
            records.append(dict(task=name, leaves=leaves))
        except Exception as exc:
            errors.append(dict(task=name, error=repr(exc)))
            print(f'[qa-preflight] FAILED {name}: {exc}', flush=True)
    return dict(valid=not errors and len(records) == len(names), tasks=records, errors=errors,
                scope='exact paper QA datasets loaded before GPU; not QA scores')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Explicitly allow downloading QA data; offline is the default.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '0' if args.download else '1'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    selected = dataset_cache_directory(root, os.environ)
    if selected:
        os.environ['HF_DATASETS_CACHE'] = selected
    print(f'[qa-preflight] HF_DATASETS_CACHE={selected or "library default"}', flush=True)
    try:
        from lm_eval import tasks, utils
        from utils.eval_utils import _resolve_paper_qa_tasks
        custom = root/'datasets/lm_eval_configs/tasks'
        options = dict(include_defaults=True)
        if custom.is_dir():
            options['include_path'] = str(custom)
        manager = tasks.TaskManager(**options)
        names = _resolve_paper_qa_tasks(utils.pattern_match, manager.all_tasks)
        report = check_datasets(manager, names, tasks.get_task_dict)
    except Exception as exc:
        report = dict(valid=False, tasks=[], errors=[dict(error=repr(exc))])
    report.update(mode='download' if args.download else 'offline', gpu_devices_hidden=True,
                  hf_datasets_cache=selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if not report['valid']:
        raise SystemExit('QA data preflight failed; GPU experiment must not start. Populate the exact task caches, then recheck offline.')


if __name__ == '__main__':
    main()
