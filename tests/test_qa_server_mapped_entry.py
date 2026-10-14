import ast
from types import SimpleNamespace
import pytest
from tools.qa_server_runtime_compat import SnapshotCUDA,probe_source
from tools.run_storage_formal_qa_mapped import entry


def test_snapshot_compat_delegates_native_snapshot_without_changing_segments():
    segments=[dict(address=1,active_size=64,total_size=64)]
    class Pool:
        def snapshot(self):return segments
    facade=SnapshotCUDA(SimpleNamespace(MemPool=Pool,memory_snapshot=lambda:segments))
    assert facade.MemPool().snapshot(include_traces=False) is segments
    assert facade.memory_snapshot(include_traces=False) is segments
    with pytest.raises(RuntimeError):facade.memory_snapshot(include_traces=True)


def test_formal_entry_keeps_original_arm_capture_and_ptq_execution():
    original="import runpy\n_hinv_release='--research_hinv_solver_workspace' in __import__('sys').argv\nrunpy.run_path('ptq.py', run_name='__main__')\n"
    result=entry(original);tree=ast.parse(result)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='runpy.run_path']
    assert len(calls)==1 and ast.dump(calls[0])==ast.dump(ast.parse(original).body[-1].value)
    assert 'enabled=_hinv_release' in result and 'PAPER_QA_TASKS' in result
    assert result.index('_qa_mapped_runtime_library')<result.index('import runpy')
    with pytest.raises(RuntimeError):entry(original.replace('_hinv_release=','wrong='))


def test_probe_only_adds_snapshot_facade_and_provenance():
    result=probe_source()
    assert "tensor.fill_(7)" in result and 'torch.nn.functional.linear(inputs,weight)' in result
    assert 'qa_server_runtime_compat.py' in result
    compile(result,'server_probe_test','exec')
