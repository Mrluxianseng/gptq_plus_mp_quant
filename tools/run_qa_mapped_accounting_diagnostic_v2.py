"""Head-only history probe after native GEMM initializes its library workspace.

The formal worker had already run quantization and PPL before mapped QA. The
fresh v1 probe instead captured cuBLAS persistent workspace in its private
pool on the first call; that failed release is preserved, not QA evidence.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_qa_mapped_accounting_diagnostic as ancestor


def source():
    text = Path(ancestor.__file__).read_text()
    anchor = '    source_paths = [Path(__file__),'
    if text.count(anchor) != 1:
        raise RuntimeError('Diagnostic native head setup contract changed')
    warmup = '''    # Native initialization precedes mapped QA in the actual formal run.
    with torch.inference_mode():
        native_input = torch.zeros((*inputs[0]['shape'], 1024),
            device=f'cuda:{args.device}', dtype=torch.bfloat16)
        native_output = head(native_input)
        torch.cuda.synchronize(args.device)
        del native_output, native_input
    gc.collect()
    torch.cuda.synchronize(args.device)
    if library.qa_live_bytes() or library.qa_allocation_count():
        raise RuntimeError('Native warmup must not use mapped storage')
'''
    text = text.replace(anchor, warmup + anchor)
    text = text.replace("Path('tools/qa_mapped_head_storage_prototype.py'),",
                        "Path(ancestor.__file__), Path('tools/qa_mapped_head_storage_prototype.py'),")
    text = text.replace("completed=False, guard_passed=False)",
                        "completed=False, guard_passed=False, native_workspace_initialized=True)")
    return text


if __name__ == '__main__':
    exec(compile(source(), 'qa_head_history_native_initialized_v2', 'exec'),
         dict(__name__='__main__', __file__=__file__, ancestor=ancestor))
