"""Bind actual server FHT implementation before the explicit formal run."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_storage_formal_qa_mapped_v4 import source as prior_source


def source():
    text=prior_source()
    anchor='    return binary,bindings\n'
    if text.count(anchor)!=1:raise RuntimeError('Frozen runtime admission return changed')
    text=text.replace(anchor,'''    import importlib.util
    for name in ('fast_hadamard_transform','fast_hadamard_transform_cuda'):
        module=importlib.util.find_spec(name)
        if module is None or module.origin is None:
            raise RuntimeError('Actual server FHT implementation missing: '+name)
        path=Path(module.origin).resolve()
        bindings[str(path)]=sha(path)
    return binary,bindings
''')
    anchor="extra=[ROOT/'tools/run_storage_formal_qa_mapped_v3.py',"
    if text.count(anchor)!=1:raise RuntimeError('Frozen source manifest anchor changed')
    return text.replace(anchor,"extra=[ROOT/'tools/run_storage_formal_qa_mapped_v4.py',ROOT/'tools/run_storage_formal_qa_mapped_v3.py',")


scope=dict(__name__='qa_server_formal_v5',__file__=__file__)
exec(compile(source(),'qa_server_formal_v5','exec'),scope)
if __name__=='__main__':scope['main']()
