"""Prepared task-boundary adapter; delegates the original evaluator unchanged."""


class TaskBoundaries:
    def __init__(self, observer, evaluation_module, expected_tasks):
        self.observer = observer
        self.module = evaluation_module
        self.expected = tuple(expected_tasks)
        if not self.expected or len(set(self.expected)) != len(self.expected):
            raise RuntimeError('Requires explicit unique full task sequence')
        self.original = evaluation_module.simple_evaluate
        self.seen = []

    def __enter__(self):
        if getattr(self.module, '_qa_task_boundary_observer', False):
            raise RuntimeError('Task boundary observer already installed')
        adapter = self
        def evaluate(hflm, *args, **kwargs):
            tasks = kwargs.get('tasks')
            if (args or not isinstance(tasks, (tuple, list)) or len(tasks) != 1
                    or tuple(adapter.seen+[tasks[0]]) != adapter.expected[:len(adapter.seen)+1]):
                raise RuntimeError('Requires original one-task evaluator sequence')
            if kwargs.get('limit') is not None or kwargs.get('samples') is not None:
                raise RuntimeError('Full QA cannot use a reduced evaluation limit/sample set')
            if hflm.model is not adapter.observer.model or hflm.batch_size_per_gpu != 16:
                raise RuntimeError('Requires the observed model and unchanged batch16')
            with adapter.observer.task(tasks[0]):
                result = adapter.original(hflm, *args, **kwargs)
            adapter.seen.append(tasks[0])
            return result
        self.module.simple_evaluate = evaluate
        self.module._qa_task_boundary_observer = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.module.simple_evaluate = self.original
        self.module._qa_task_boundary_observer = False
        if exc is None and tuple(self.seen) != self.expected:
            raise RuntimeError('Incomplete full task sequence')
        return False
