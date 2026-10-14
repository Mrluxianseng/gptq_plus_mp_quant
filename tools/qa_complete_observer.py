"""Prepared streaming QA evidence observer; no placement or formal launcher.

Enter inside the evaluator, after selecting its scoring/cache methods. Task
boundaries must wrap the existing simple_evaluate calls. CPU fixtures do not
constitute full QA or multi-device memory validation.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
import marshal
from pathlib import Path
import torch
from tools.qa_tensor_fingerprint_stream import update_tensor_bytes, tensor_sha256


def state_digest(model):
    digest = hashlib.sha256()
    layout = {}
    for name, value in model.state_dict().items():
        if value.is_meta:
            raise RuntimeError('Complete state requires actual tensor bytes')
        layout[name] = [list(value.shape), str(value.dtype)]
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        update_tensor_bytes(digest, value)
    return dict(sha256=digest.hexdigest(), count=len(layout), layout=layout)


def tensor_record(value):
    if not isinstance(value, torch.Tensor) or value.is_meta or not value.numel():
        raise RuntimeError('Expected complete nonempty tensor output')
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                stride=list(value.stride()), sha256=tensor_sha256(value))


def method_record(method):
    # AST-built selected methods have no inspectable source file. Preserve their
    # actual code separately, and do not mislabel bytecode as a source digest.
    selected = inspect.unwrap(method)
    source_sha = None
    if Path(selected.__code__.co_filename).is_file():
        try:
            source_sha = hashlib.sha256(inspect.getsource(selected).encode()).hexdigest()
        except (OSError, TypeError):
            pass
    return dict(source_sha256=source_sha,
                code_sha256=hashlib.sha256(marshal.dumps(method.__code__)).hexdigest(),
                code_filename=method.__code__.co_filename,
                selected_code_sha256=hashlib.sha256(marshal.dumps(selected.__code__)).hexdigest(),
                selected_code_filename=selected.__code__.co_filename)


class CompleteQAObserver:
    def __init__(self, model, hflm_class, destination, *, mark=lambda phase: None):
        if model.training or model.config.use_cache is not True or len(model.model.layers) != 28:
            raise RuntimeError('Requires eval, original enabled KV, complete 28 layers')
        if getattr(hflm_class, '_complete_qa_observer', False):
            raise RuntimeError('Observer already active')
        self.model, self.cls = model, hflm_class
        self.root = Path(destination)
        self.root.mkdir(parents=True, exist_ok=True)
        if any((self.root/name).exists() for name in ('qa_complete_events.jsonl', 'qa_complete_summary.json')):
            raise RuntimeError('Evidence exists; use a fresh directory')
        self.mark = mark
        self.original_call = hflm_class._model_call
        self.original_score = hflm_class._loglikelihood_tokens
        self.task_name = None
        self.forward = 0
        self.active_forward = None
        self.active_layers = set()
        self.layer_counts = [0]*28
        self.tasks = []
        self.handles = []
        self.completed = False
        self.events = hashlib.sha256()
        self.event_count = 0
        self.entered = False

    def emit(self, kind, **fields):
        row = dict(kind=kind, task=self.task_name, **fields)
        data = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n'
        self.stream.write(data)
        self.stream.flush()
        self.events.update(data.encode())
        self.event_count += 1

    def __enter__(self):
        if self.entered or getattr(self.cls, '_complete_qa_observer', False):
            raise RuntimeError('Observer cannot be entered twice')
        self.before = state_digest(self.model)
        self.stream = (self.root/'qa_complete_events.jsonl').open('x', encoding='utf-8', newline='\n')
        self.entered = True
        observer = self
        def call(instance, *args, **kwargs):
            if observer.task_name is None or observer.active_forward is not None:
                raise RuntimeError('Missing task boundary or nested model call')
            observer.forward += 1
            number = observer.forward
            observer.active_forward = number
            observer.active_layers = set()
            observer.mark(f'qa_forward{number}_model_start')
            try:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError('Requires the original tensor model-call input')
                observer.emit('model_input', forward=number, **tensor_record(args[0]))
                result = observer.original_call(instance, *args, **kwargs)
                if observer.active_layers != set(range(28)):
                    raise RuntimeError('Incomplete per-forward decoder coverage')
                observer.mark(f'qa_forward{number}_logits_ready')
                observer.emit('logits', forward=number, **tensor_record(result))
                observer.mark(f'qa_forward{number}_scoring')
                return result
            finally:
                observer.active_forward = None
        def score(instance, requests, *args, **kwargs):
            if observer.task_name is None:
                raise RuntimeError('Missing task boundary')
            request_bytes = json.dumps(requests, ensure_ascii=False, allow_nan=False).encode()
            observer.emit('requests', count=len(requests), sha256=hashlib.sha256(request_bytes).hexdigest())
            result = observer.original_score(instance, requests, *args, **kwargs)
            if len(result) != len(requests) or not result:
                raise RuntimeError('Incomplete request scores')
            for item in result:
                if (not isinstance(item, (tuple, list)) or len(item) != 2
                        or type(item[0]) not in (int, float) or type(item[1]) is not bool):
                    raise RuntimeError('Requires full logprob and greedy score pairs')
            data = json.dumps(result, allow_nan=False).encode()
            observer.emit('scores', count=len(result), sha256=hashlib.sha256(data).hexdigest())
            observer.tasks[-1]['scores'] += len(result)
            return result
        def capture(index):
            def hook(module, inputs, output):
                if observer.active_forward is None:
                    raise RuntimeError('Decoder output outside observed model call')
                if index in observer.active_layers:
                    raise RuntimeError('Repeated decoder layer in one forward')
                value = output[0] if isinstance(output, (tuple, list)) else output
                observer.emit('layer', forward=observer.active_forward, layer=index, **tensor_record(value))
                observer.layer_counts[index] += 1
                observer.active_layers.add(index)
                observer.mark(f'qa_forward{observer.active_forward}_layer{index}_observed')
            return hook
        try:
            for index, layer in enumerate(self.model.model.layers):
                self.handles.append(layer.register_forward_hook(capture(index)))
            self.cls._model_call, self.cls._loglikelihood_tokens = call, score
            self.cls._complete_qa_observer = True
        except BaseException:
            for handle in self.handles:
                handle.remove()
            self.stream.close()
            raise
        return self

    @contextmanager
    def task(self, name):
        if not self.entered or self.task_name is not None or name in [row['name'] for row in self.tasks]:
            raise RuntimeError('Invalid, nested or duplicate task boundary')
        self.task_name = str(name)
        row = dict(name=str(name), scores=0, forwards_before=self.forward, completed=False)
        self.tasks.append(row)
        self.mark(f'qa_task{len(self.tasks)}_start')
        try:
            yield
            if row['scores'] <= 0 or self.forward <= row['forwards_before']:
                raise RuntimeError('Task has no complete observed scoring forwards')
            row['completed'] = True
        finally:
            row['forwards'] = self.forward-row['forwards_before']
            self.mark(f'qa_task{len(self.tasks)}_end')
            self.task_name = None

    def __exit__(self, exc_type, exc, traceback):
        self.cls._model_call, self.cls._loglikelihood_tokens = self.original_call, self.original_score
        self.cls._complete_qa_observer = False
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.stream.close()
        error = None
        try:
            after = state_digest(self.model)
            complete = (exc is None and self.task_name is None and self.forward > 0
                and all(row['completed'] for row in self.tasks)
                and self.layer_counts == [self.forward]*28 and self.before == after
                and self.model.config.use_cache is True)
            record = dict(schema='qa-complete-stream-observer-v1', completed=complete,
                before=self.before, after=after, tasks=self.tasks, forward_count=self.forward,
                layer_counts=self.layer_counts, events_sha256=self.events.hexdigest(),
                event_count=self.event_count, config_use_cache=self.model.config.use_cache,
                selected_call=method_record(self.original_call),
                selected_score=method_record(self.original_score),
                scope='prepared complete byte observations; no independent QA or memory acceptance')
        except BaseException as failure:
            error = failure
            record = dict(schema='qa-complete-stream-observer-v1', completed=False,
                          recording_error_type=type(failure).__name__, recording_error=str(failure))
        if exc is not None:
            record.update(completed=False, error_type=type(exc).__name__, error=str(exc))
        with (self.root/'qa_complete_summary.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        if exc is None and (error is not None or not record['completed']):
            raise RuntimeError('Incomplete QA evidence') from error
        return False


def validate_observer_pair(control_root, candidate_root, expected_tasks):
    """Numerical-observation gate only. Caller still owes method/source/memory audits."""
    records = []
    for root in (control_root, candidate_root):
        root = Path(root)
        summary = json.loads((root/'qa_complete_summary.json').read_text())
        digest = hashlib.sha256()
        count = 0
        forwards = {}
        scores, requests = {}, {}
        with (root/'qa_complete_events.jsonl').open('rb') as stream:
            for line in stream:
                digest.update(line); count += 1
                row = json.loads(line)
                task = row['task']
                if row['kind'] in ('layer', 'logits', 'model_input'):
                    number = row['forward']
                    key = ('layer', row['layer']) if row['kind']=='layer' else (row['kind'],)
                    forward = forwards.setdefault(number, dict(task=task, keys=set()))
                    if forward['task'] != task or key in forward['keys']:
                        raise RuntimeError('Duplicate or inconsistent forward evidence')
                    forward['keys'].add(key)
                    if (not row['shape'] or any(type(x) is not int or x<=0 for x in row['shape'])
                            or len(row['stride']) != len(row['shape']) or len(row['sha256']) != 64):
                        raise RuntimeError('Incomplete output tensor evidence')
                elif row['kind'] in ('scores', 'requests'):
                    if type(row['count']) is not int or row['count']<=0 or len(row['sha256']) != 64:
                        raise RuntimeError('Incomplete score/input evidence')
                    table = scores if row['kind']=='scores' else requests
                    table[task] = table.get(task,0)+row['count']
                else:
                    raise RuntimeError('Unknown event evidence')
        if digest.hexdigest() != summary['events_sha256'] or count != summary['event_count']:
            raise RuntimeError('Event evidence SHA/count mismatch')
        expected = {('layer', index) for index in range(28)} | {('logits',), ('model_input',)}
        if (set(forwards) != set(range(1,summary['forward_count']+1))
                or any(row['keys'] != expected for row in forwards.values())
                or scores != requests
                or scores != {row['name']:row['scores'] for row in summary['tasks']}
                or any(sum(f['task']==row['name'] for f in forwards.values()) != row['forwards'] for row in summary['tasks'])):
            raise RuntimeError('Incomplete per-task/forward evidence')
        records.append(summary)
    c, n = records
    checks = dict(completed=c['completed'] is True and n['completed'] is True,
                  task_coverage=[r['name'] for r in c['tasks']] == list(expected_tasks)
                    == [r['name'] for r in n['tasks']],
                  cache_enabled=c['config_use_cache'] is True and n['config_use_cache'] is True)
    for key in ('before', 'after', 'tasks', 'forward_count', 'layer_counts', 'events_sha256', 'event_count'):
        checks[key] = c[key] == n[key]
    checks['states_unchanged'] = c['before'] == c['after'] == n['before'] == n['after']
    checks['complete_layers'] = c['forward_count'] > 0 and c['layer_counts'] == [c['forward_count']]*28
    return dict(valid=all(checks.values()), checks=checks,
                scope='numerical observer only; method/source/all-device telemetry still required')
