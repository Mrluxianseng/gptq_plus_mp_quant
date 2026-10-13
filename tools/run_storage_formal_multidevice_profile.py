"""Reuse the frozen profile pipeline; collect post-QA peaks on all devices."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_storage_method_pipeline as original


def footer(source):
    anchor='    _allocated, _reserved = _process_peaks()'
    update="""    from tools.storage_process_device_peaks import collect_peaks as _collect_device_peaks
    _all_device_peaks = _collect_device_peaks(torch.cuda, _process_peaks)
    _allocated, _reserved = _all_device_peaks['peak_allocated_bytes'], _all_device_peaks['peak_reserved_bytes']"""
    assert source.count(anchor)==1
    source=source.replace(anchor,update)
    anchor='    (_destination / f"process_allocator_peak_{_stage}_rank{_rank}.json").write_text(json.dumps(_record, indent=2))'
    assert source.count(anchor)==1
    source=source.replace(anchor,'    _record.update(_all_device_peaks)\n'+anchor)
    compile(source,'all_device_footer','exec')
    return source


def main():
    original.PROCESS_PEAK_FOOTER=footer(original.PROCESS_PEAK_FOOTER)
    original.TRACKED += ('tools/storage_process_device_peaks.py','tools/run_storage_formal_multidevice_profile.py')
    original.main()


if __name__=='__main__': main()
