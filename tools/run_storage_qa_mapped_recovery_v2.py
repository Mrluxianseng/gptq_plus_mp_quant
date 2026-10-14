"""Add independent raw pointer journal gate to the frozen recovery runner."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_storage_qa_mapped_recovery as ancestor


def source():
    text = Path(ancestor.__file__).read_text()
    changes = {
        "extra = [Path(__file__), ROOT/'tools/qa_server_runtime_compat_v2.py',":
            "extra = [Path(__file__), Path(ancestor.__file__), ROOT/'tools/audit_qa_head_accounting_journal.py', ROOT/'tools/qa_server_runtime_compat_v2.py',",
        "pair['qa_whole_mapped_storage'] = qa":
            "from tools.audit_qa_head_accounting_journal import audit as audit_journal\n        journal = audit_journal(result_root)\n        pair['mapped_pointer_journal'] = journal\n        pair['checks']['independent_complete_raw_pointer_journal'] = journal['valid']\n        pair['qa_whole_mapped_storage'] = qa",
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen recovery audit contract changed: ' + old)
        text = text.replace(old, new)
    return text


if __name__ == '__main__':
    exec(compile(source(), 'qa_safe_snapshot_recovery_v2', 'exec'),
         dict(__name__='__main__', __file__=__file__, ancestor=ancestor))
