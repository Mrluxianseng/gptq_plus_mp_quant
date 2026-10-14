"""Complete formal QA identity/telemetry gates, with CPU-only old readmission."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_storage_formal_qa_mapped_v3 import source as prior_source


def source():
    text=prior_source()
    changes={
        'from tools.run_qa_server_mapped_component_v2 import admission':'from tools.run_qa_server_mapped_component_v3 import admission\n    from tools.audit_qa_server_component_readmission import readmit',
        "if not previous['valid'] or not admission(root)['valid']:":"fresh=admission(root) if previous['valid'] else readmit(root)\n        if not fresh['valid']:",
        'from tools.audit_qa_server_mapped_v2 import audit':'from tools.audit_qa_server_mapped_v3 import audit',
        "extra=[ROOT/'tools/run_storage_formal_qa_mapped_v2.py',":"extra=[ROOT/'tools/run_storage_formal_qa_mapped_v3.py',ROOT/'tools/run_storage_formal_qa_mapped_v2.py',ROOT/'tools/run_qa_server_mapped_component_v3.py',ROOT/'tools/audit_qa_server_component_readmission.py',ROOT/'tools/audit_qa_server_mapped_v3.py',ROOT/'tools/run_qa_host_window.py',",
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError('Frozen formal admission anchor changed: '+old)
        text=text.replace(old,new)
    return text


scope=dict(__name__='qa_server_formal_v4',__file__=__file__)
exec(compile(source(),'qa_server_formal_v4','exec'),scope)
if __name__=='__main__':scope['main']()
