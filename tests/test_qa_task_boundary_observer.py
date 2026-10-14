from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from tools.qa_task_boundary_observer import TaskBoundaries


class Observer:
    def __init__(self): self.model=object();self.tasks=[]
    @contextmanager
    def task(self,name):
        self.tasks.append(name)
        yield


def setup():
    observer=Observer();lm=SimpleNamespace(model=observer.model,batch_size_per_gpu=16)
    calls=[]
    def evaluate(model,**kwargs):
        calls.append((model,kwargs));return dict(results={kwargs['tasks'][0]:dict(acc=0.5)})
    return observer,lm,SimpleNamespace(simple_evaluate=evaluate),calls


def test_all_tasks_delegated_unchanged_and_original_restored():
    observer,lm,module,calls=setup();original=module.simple_evaluate
    with TaskBoundaries(observer,module,['a','b']):
        for name in ['a','b']:
            result=module.simple_evaluate(lm,tasks=[name],task_manager='original')
            assert result==dict(results={name:dict(acc=0.5)})
    assert observer.tasks==['a','b'] and module.simple_evaluate is original
    assert [kwargs for model,kwargs in calls]==[dict(tasks=[name],task_manager='original') for name in ['a','b']]


def test_reduced_batch_limit_wrong_order_and_incomplete_rejected():
    for args in (dict(tasks=['b']),dict(tasks=['a'],limit=2),dict(tasks=['a'],samples={'a':[1]})):
        observer,lm,module,_=setup();original=module.simple_evaluate
        with pytest.raises(RuntimeError):
            with TaskBoundaries(observer,module,['a','b']): module.simple_evaluate(lm,**args)
        assert module.simple_evaluate is original
    observer,lm,module,_=setup();lm.batch_size_per_gpu=8
    with pytest.raises(RuntimeError,match='batch16'):
        with TaskBoundaries(observer,module,['a']): module.simple_evaluate(lm,tasks=['a'])
    observer,lm,module,_=setup()
    with pytest.raises(RuntimeError,match='Incomplete'):
        with TaskBoundaries(observer,module,['a']): pass


def test_original_evaluation_exception_not_replaced():
    observer,lm,module,_=setup()
    def fail(*args,**kwargs): raise ValueError('native evaluator failed')
    module.simple_evaluate=fail
    with pytest.raises(ValueError,match='native evaluator failed'):
        with TaskBoundaries(observer,module,['a']): module.simple_evaluate(lm,tasks=['a'])
    assert module.simple_evaluate is fail
