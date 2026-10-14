import ast
import hashlib
import sys
import pytest
from tools.qa_mapped_early_runtime_entry import build_entry,SETUP


def test_actual_formal_entry_math_and_all_future_imports_preserved():
    from tools.run_storage_method_formal import build_entry as actual_entry
    original=actual_entry();generated,metadata=build_entry(original,validation_only=True)
    tree=ast.parse(generated);old=ast.parse(original)
    index=0
    if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str): index=1
    while isinstance(tree.body[index],ast.ImportFrom) and tree.body[index].module=='__future__': index+=1
    del tree.body[index:index+len(ast.parse(SETUP).body)]
    assert ast.dump(tree,include_attributes=False)==ast.dump(old,include_attributes=False)
    compile(generated,'actual_formal_validation_only','exec')
    assert metadata['formal_enabled'] is False


def test_only_original_qa_rank0_prepares_before_original_body(tmp_path,monkeypatch):
    from tools import qa_mapped_head_storage_prototype as helper
    binary=tmp_path/'runtime.so';binary.write_bytes(b'CPU fake binary, not executable')
    monkeypatch.setenv('QA_MAPPED_HEAD_LIBRARY',str(binary))
    monkeypatch.setenv('QA_MAPPED_HEAD_LIBRARY_SHA256',hashlib.sha256(binary.read_bytes()).hexdigest())
    monkeypatch.setenv('REALQ_QA_STORAGE_VALIDATION','1')
    trace=[]
    monkeypatch.setattr(helper,'prepare_runtime',lambda path:(trace.append('prepare') or object(),dict(can_map_host_memory=1,unified_addressing=1)))
    original="import runpy\ntrace.append('original')\nrunpy.run_path('ptq.py', run_name='__main__')\n"
    generated,_=build_entry(original,validation_only=True)
    import runpy
    monkeypatch.setattr(runpy,'run_path',lambda *args,**kwargs:trace.append('ptq'))
    for rank,local,argv,selected in [('0','0',['entry','--lm_eval'],True),('1','1',['entry','--lm_eval'],False),
        ('0','0',['entry','--lm_eval','--exit_after_precompute'],False),('0','0',['entry','--lm_eval','--skip_eval'],False)]:
        trace.clear();monkeypatch.setenv('RANK',rank);monkeypatch.setenv('LOCAL_RANK',local);monkeypatch.setattr(sys,'argv',argv)
        scope=dict(trace=trace);exec(generated,scope)
        assert trace==(['prepare'] if selected else [])+['original','ptq']
        assert scope['_qa_mapped_early_runtime']['prepared'] is selected
    import torch
    assert not torch.cuda.is_initialized()


def test_changed_binary_or_nonvalidation_entry_rejected(tmp_path,monkeypatch):
    original="import runpy\nrunpy.run_path('ptq.py',run_name='__main__')\n"
    with pytest.raises(RuntimeError,match='validation only'): build_entry(original,validation_only=False)
    with pytest.raises(RuntimeError,match='unique'): build_entry('pass',validation_only=True)
    generated,_=build_entry(original,validation_only=True)
    with pytest.raises(RuntimeError,match='already installed'): build_entry(generated,validation_only=True)
    binary=tmp_path/'runtime.so';binary.write_bytes(b'CPU fake')
    monkeypatch.setenv('RANK','0');monkeypatch.setenv('LOCAL_RANK','0');monkeypatch.setattr(sys,'argv',['entry','--lm_eval'])
    monkeypatch.setenv('QA_MAPPED_HEAD_LIBRARY',str(binary));monkeypatch.setenv('QA_MAPPED_HEAD_LIBRARY_SHA256','wrong')
    monkeypatch.setenv('REALQ_QA_STORAGE_VALIDATION','1')
    with pytest.raises(RuntimeError,match='binary changed'): exec(generated,{})
    monkeypatch.setenv('REALQ_QA_STORAGE_VALIDATION','0')
    with pytest.raises(RuntimeError,match='explicit validation-only'): exec(generated,{})
