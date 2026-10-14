from tools.audit_qa_night_final_chain import jobs,completed_expected


def test_incomplete_or_wrong_false_cannot_pass_final_chain():
    row=dict(valid=False,interval_count=356,points=dict(matched_values=1424,missing=[],valid=False),checks=dict(global_reserved=False,numerics=True))
    assert completed_expected(row,False,356,1424)
    row['checks']['numerics']=False
    assert not completed_expected(row,False,356,1424)
    row['checks']['numerics']=True;row['points']['missing']=['one']
    assert not completed_expected(row,False,356,1424)


def test_complete_acceptance_requires_all_supplemental_gates_and_full_scope():
    row=dict(valid=True,interval_count=593,points=dict(matched_values=2372,missing=[],valid=True),
             checks=dict(numerical=True),control_align_checks=dict(actual_hook=False))
    assert not completed_expected(row,True,593,2372)
    row['control_align_checks']['actual_hook']=True
    assert completed_expected(row,True,593,2372)
    assert len(jobs())==9 and sum(j[3] for j in jobs())==8
