"""Versioned formal entry with complete server phase/payload admission."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def source():
    text=(ROOT/'tools/run_storage_formal_qa_mapped.py').read_text()
    changes={
        'from tools.run_qa_server_mapped_component import admission':'from tools.run_qa_server_mapped_component_v2 import admission',
        'from tools.audit_qa_server_mapped import audit':'from tools.audit_qa_server_mapped_v2 import audit',
        "extra=list(ROOT.glob('tools/qa_*.py'))+":"extra=[ROOT/'tools/run_storage_formal_qa_mapped.py',ROOT/'tools/run_qa_server_mapped_component_v2.py',ROOT/'tools/audit_qa_server_mapped_v2.py',*list(ROOT.glob('tools/qa_*.py'))]+",
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError('Frozen formal entry changed: '+old)
        text=text.replace(old,new)
    return text


scope=dict(__name__='qa_server_formal_v2',__file__=__file__)
exec(compile(source(),'qa_server_formal_v2','exec'),scope)
if __name__=='__main__':scope['main']()
