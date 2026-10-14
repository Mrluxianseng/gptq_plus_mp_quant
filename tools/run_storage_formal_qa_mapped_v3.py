"""Formal opt-in adapter with dedicated QA physical/host telemetry."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_storage_formal_qa_mapped_v2 import source as ancestor_source


def source():
    text=ancestor_source()
    old='from tools.qa_server_mapped_validation import install as _qa_server_install'
    if text.count(old)!=1:raise RuntimeError('Frozen formal storage setup changed')
    text=text.replace(old,'from tools.qa_server_mapped_telemetry import install as _qa_server_install')
    old="extra=[ROOT/'tools/run_storage_formal_qa_mapped.py',"
    if text.count(old)!=1:raise RuntimeError('Frozen formal binding setup changed')
    text=text.replace(old,"extra=[ROOT/'tools/run_storage_formal_qa_mapped_v2.py',ROOT/'tools/run_storage_formal_qa_mapped.py',ROOT/'tools/run_qa_host_memory_observer.py',ROOT/'tools/run_qa_host_memory_observer_v2.py',")
    return text


scope=dict(__name__='qa_server_formal_v3',__file__=__file__)
exec(compile(source(),'qa_server_formal_v3','exec'),scope)
if __name__=='__main__':scope['main']()
