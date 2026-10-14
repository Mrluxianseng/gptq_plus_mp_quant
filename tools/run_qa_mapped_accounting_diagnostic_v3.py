"""Fixed snapshot binding stress: two real geometry cycles and full-byte probes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_qa_mapped_accounting_diagnostic_v2 as ancestor_v2
from tools import run_qa_mapped_accounting_diagnostic as ancestor


def source():
    text = ancestor_v2.source()
    changes = {
        'from tools.qa_server_runtime_compat import SnapshotCUDA':
            'from tools.qa_server_runtime_compat_v2 import SnapshotCUDA\n    from tools.qa_tensor_fingerprint_stream import tensor_sha256',
        'inputs = shapes(args.events)[:args.max_calls]':
            'inputs = (shapes(args.events) * 2)[:args.max_calls]',
        "Path('tools/qa_server_runtime_compat.py')":
            "Path(ancestor_v2.__file__), Path('tools/qa_tensor_fingerprint_stream.py'), Path('tools/qa_server_runtime_compat_v2.py'), Path('tools/qa_server_runtime_compat.py')",
        '    started = time.monotonic()':
            '    numerical_probes = []\n    torch.manual_seed(19)\n    started = time.monotonic()',
        '                value = torch.zeros((*row[\'shape\'], 1024),':
            '                value = torch.randn((*row[\'shape\'], 1024),',
        '                del output, value':
            '''                if ticket.calls in (1, 255, 256, 257, 258, 513, 514):
                    native_output = torch.nn.functional.linear(value, head.weight)
                    probe = dict(call=ticket.calls, shape=list(output.shape),
                        mapped_sha256=tensor_sha256(output),
                        native_sha256=tensor_sha256(native_output),
                        payload_bytes=output.numel()*output.element_size())
                    probe['exact'] = probe['mapped_sha256'] == probe['native_sha256']
                    numerical_probes.append(probe)
                    del native_output
                    if not probe['exact']:
                        raise RuntimeError('Whole GEMM full byte probe differs')
                del output, value''',
        "    report.update(wall_seconds=":
            "    report['numerical_probes'] = numerical_probes\n    report['snapshot_binding'] = 'unfiltered-native-then-exact-pool-id-filter'\n    report.update(wall_seconds=",
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen diagnostic contract changed: ' + old)
        text = text.replace(old, new)
    return text


if __name__ == '__main__':
    exec(compile(source(), 'qa_head_history_safe_snapshot_v3', 'exec'),
         dict(__name__='__main__', __file__=__file__, ancestor=ancestor, ancestor_v2=ancestor_v2))
