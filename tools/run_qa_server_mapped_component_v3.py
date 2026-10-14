"""Add after-placement identities; preserve pre-dispatch v2 failure artifact."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_server_mapped_component_v2 import source as prior_source


def source():
    text=prior_source()
    changes={
        'from tools.qa_server_mapped_validation import install':'from tools.qa_server_mapped_telemetry import install',
        "original=lm_eval.simple_evaluate;lm_eval.simple_evaluate=simple":"original=lm_eval.simple_evaluate;lm_eval.simple_evaluate=simple",
        "before={n:id(p) for n,p in model.named_parameters(remove_duplicate=False)}":"before=None",
        '        dist_utils.distribute_model(model)':'        dist_utils.distribute_model(model)\n        before={n:id(p) for n,p in model.named_parameters(remove_duplicate=False)}',
        "'actual_server_backend']=all(x['backend']=='fast_hadamard_transform' and x['parameter_identity'] for x in rows)":"'actual_server_backend']=all(x['backend']=='fast_hadamard_transform' for x in rows) and rows[1]['parameter_identity']",
        "ROOT/'tools/run_qa_server_mapped_component.py',ROOT/'tools/audit_qa_server_mapped_v2.py',":"ROOT/'tools/run_qa_server_mapped_component.py',ROOT/'tools/run_qa_server_mapped_component_v2.py',ROOT/'tools/run_qa_host_window.py',ROOT/'tools/audit_qa_server_mapped_v2.py',",
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError('Frozen component normalization changed: '+old)
        text=text.replace(old,new)
    return text


scope=dict(__name__='qa_server_component_v3',__file__=__file__)
exec(compile(source(),'qa_server_component_v3','exec'),scope)
admission=scope['admission']
if __name__=='__main__':scope['main']()
