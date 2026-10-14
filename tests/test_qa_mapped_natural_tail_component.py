import ast
from copy import deepcopy
from tools.run_qa_mapped_natural_tail_component import source,namespace,expected_phases
from tools.run_qa_mapped_long_batch_component import source as old_source
from tools.audit_qa_mapped_natural_tail_completed import natural_batch_checks,geometry_checks


def test_actual_raw_source_preserves_math_and_all_original_requests():
    old,new=ast.parse(old_source()),ast.parse(source())
    def selected(tree,name):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    for name in ('state','capture','measured_call','observe_forward','observe_output'):
        assert selected(old,name)==selected(new,name) and len(selected(new,name))==1
    worker=namespace()['worker']
    assert worker and source().count("requests.append((('natural-tail-unique'")==1
    assert "range(32)" in source() and "range(2)" in source()
    assert source().count('NaturalBatchMappedHeadStorage as MappedHeadStorage')==1
    assert len(expected_phases(4))==481


def test_configured_batch_and_actual_partial_request_coverage_required():
    row=dict(batch_size=16,scores=[[0.0,True]]*65,
        logits=[dict(shape=[n,3,151936]) for n in (16,16,16,1)],
        mapped_head=dict(configured_batch_size=16,actual_batch_sizes=[16,16,16,1]))
    assert all(natural_batch_checks(row).values())
    for kind in ('lost_request','fake_full_tail','reduce_config'):
        bad=deepcopy(row)
        if kind=='lost_request': bad['scores'].pop()
        elif kind=='fake_full_tail': bad['logits'][-1]['shape'][0]=16
        else: bad['batch_size']=1
        assert not all(natural_batch_checks(bad).values())
