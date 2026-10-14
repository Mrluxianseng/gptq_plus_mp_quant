from tools.run_qa_control_align_component import source as ancestor_source
from tools.run_qa_control_align_component_v2 import source,namespace


def test_entry_changes_auditor_and_binding_only_not_worker():
    text=source()
    restored=text.replace('from tools.audit_qa_control_align_completed_v2 import audit as _complete_audit',
        'from tools.audit_qa_control_align_completed import audit as _complete_audit').replace(
        "ROOT/'tools/audit_qa_control_align_completed.py',ROOT/'tools/audit_qa_control_align_completed_v2.py',ROOT/'tools/run_qa_control_align_component.py',",
        "ROOT/'tools/audit_qa_control_align_completed.py',")
    assert restored==ancestor_source()
    assert callable(namespace()['worker'])
