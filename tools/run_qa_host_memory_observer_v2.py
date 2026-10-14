"""Standalone bootstrap with stable root identity and unchanged byte accounting."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_qa_host_memory_observer as ancestor


def source():
    text=Path(ancestor.__file__).read_text()
    changes={
        'from tools.qa_host_memory_observer_v2 import collect,bounded_sha':
        'from tools.qa_host_memory_observer_v3 import collect,bounded_sha',
        "ROOT/'tools/qa_host_memory_observer_v2.py',ROOT/'tools/qa_host_memory_observer.py'":
        "ROOT/'tools/run_qa_host_memory_observer.py',ROOT/'tools/qa_host_memory_observer_v3.py',ROOT/'tools/qa_host_memory_observer_v2.py',ROOT/'tools/qa_host_memory_observer.py'",
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError('Frozen standalone host entry anchor changed')
        text=text.replace(old,new)
    return text


if __name__=='__main__':
    scope=dict(__name__='qa_host_stable_cli_v2',__file__=__file__)
    exec(compile(source(),'qa_host_stable_cli_v2','exec'),scope)
    scope['main']()
