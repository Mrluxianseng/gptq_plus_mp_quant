"""Formal runner with the user-authorized global-capacity audit adapter.

Original formal math, manifests, entry, cache and worker execution are reused
unchanged. The original per-GPU gate report is retained inside each new audit.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_storage_method_formal as formal
from tools.audit_storage_formal_global_capacity import audit_pair


def main():
    formal.FORMAL_TOOL_SOURCES += ('tools/audit_storage_formal_global_capacity.py',
                                   'tools/run_storage_formal_global_capacity.py',
                                   'tools/storage_process_device_peaks.py',
                                   'tools/run_storage_formal_multidevice_profile.py')
    formal.audit_pair = audit_pair
    original_save = formal.save
    def save(path, value):
        if Path(path).name == 'plan.json':
            for job in [*value['jobs'], *value['optional_precompute_jobs'].values()]:
                launcher=job['launcher']
                launcher[2]=str(ROOT/'tools/run_storage_formal_multidevice_profile.py')
            value = dict(value, acceptance_policy='user_global_process_and_device_maxima_v1',
                         per_gpu_peak_policy='retain as diagnostic; physical identity still required')
        original_save(path, value)
    formal.save = save
    formal.main()


if __name__ == '__main__':
    main()
