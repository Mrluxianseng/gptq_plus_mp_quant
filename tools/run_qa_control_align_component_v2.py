"""Preserved CUDA math/observation entry with the repaired expected-SHA auditor."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_control_align_component import source as ancestor_source


def source():
    text=ancestor_source()
    changes={
        'from tools.audit_qa_control_align_completed import audit as _complete_audit':
        'from tools.audit_qa_control_align_completed_v2 import audit as _complete_audit',
        "ROOT/'tools/audit_qa_control_align_completed.py',":
        "ROOT/'tools/audit_qa_control_align_completed.py',ROOT/'tools/audit_qa_control_align_completed_v2.py',ROOT/'tools/run_qa_control_align_component.py',",
    }
    for old,new in changes.items():
        if text.count(old)!=1: raise RuntimeError('Frozen control-entry audit anchor changed')
        text=text.replace(old,new)
    compile(text,'qa_control_align_fixture_v2','exec')
    return text


def namespace():
    scope=dict(__name__='qa_control_align_fixture_v2',__file__=__file__,_generator=source)
    exec(compile(source(),'qa_control_align_fixture_v2','exec'),scope)
    return scope


if __name__=='__main__': namespace()['main']()
