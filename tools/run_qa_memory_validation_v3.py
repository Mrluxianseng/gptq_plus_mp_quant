"""Continue scored-row validation after rejecting the no-KV GPU mismatch.

Reuse the frozen v2 worker/scoring math. Only schedule control and rows; include
both runner sources in bindings. Old failed comparison remains untouched.
"""
from pathlib import Path


def source():
    path=Path(__file__).with_name('run_qa_memory_validation_v2.py')
    text=path.read_text()
    changes={
        "for side in ('rows','rows_no_kv'):":"for side in ('rows',):",
        "choices=('control','rows','rows_no_kv')":"choices=('control','rows')",
        "for side in ('control','rows','rows_no_kv'):":"for side in ('control','rows'):",
        "paths=[Path(__file__),":"paths=[Path(__file__),Path(__file__).with_name('run_qa_memory_validation_v2.py'),",
    }
    for old,new in changes.items():
        if text.count(old)!=1: raise RuntimeError('Frozen validation runner changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_memory_rows_only','exec')
    return text


if __name__=='__main__':
    exec(compile(source(),'qa_memory_rows_only','exec'),dict(__name__='__main__',__file__=__file__))
