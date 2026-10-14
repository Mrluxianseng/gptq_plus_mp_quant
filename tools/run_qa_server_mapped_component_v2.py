"""Preserve the v1 source-path prelaunch error; fix only path normalization."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def source():
    text=(ROOT/'tools/run_qa_server_mapped_component.py').read_text()
    changes={
        'return {str(p.resolve()):sha(p) for p in files}':'return {str(Path(p).resolve()):sha(p) for p in files}',
        'from tools.audit_qa_server_mapped import audit':'from tools.audit_qa_server_mapped_v2 import audit',
        "files = [*ROOT.glob('tools/qa_*.py'),":"files = [ROOT/'tools/run_qa_server_mapped_component.py',ROOT/'tools/audit_qa_server_mapped_v2.py',*ROOT.glob('tools/qa_*.py'),",
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError('Frozen component entry changed: '+old)
        text=text.replace(old,new)
    return text


scope=dict(__name__='qa_server_component_v2',__file__=__file__)
exec(compile(source(),'qa_server_component_v2','exec'),scope)
admission=scope['admission']
if __name__=='__main__':scope['main']()
