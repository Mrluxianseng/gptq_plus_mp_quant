import ast
from copy import deepcopy
from tools.run_qa_mapped_long_batch_component import source,namespace,expected_phases
from tools.run_qa_mapped_long_component import source as old_source
from tools.audit_qa_mapped_long_batch_completed import audit,batch_checks


def test_actual_raw_entry_preserves_every_original_forward_observer():
    old,new=ast.parse(old_source()),ast.parse(source())
    def selected(tree,name):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    for name in ('state','capture','measured_call','observe_forward','observe_output'):
        assert selected(old,name)==selected(new,name) and len(selected(new,name))==1
    assert source().count('build_batch_method(original_score,_head_storage.release_batch)')==1
    assert len(expected_phases(3))==362
    assert namespace()['worker'] and audit


def test_missing_batch_or_live_host_cannot_be_accepted():
    row=dict(mapped_head=dict(calls=3,allocations=3,releases=3,batch_releases=[dict(call=i,enabled=True,
        active_before_release=0,host_live_after_release=0,allocations=i,releases=i) for i in (1,2,3)]))
    assert all(batch_checks(row,True).values())
    for which in ('missing','active','host','identity'):
        bad=deepcopy(row)
        if which=='missing': bad['mapped_head']['batch_releases'].pop()
        elif which=='identity': bad['mapped_head']['batch_releases'][2]['call']=2
        else: bad['mapped_head']['batch_releases'][0]['active_before_release' if which=='active' else 'host_live_after_release']=1
        assert not all(batch_checks(bad,True).values())
