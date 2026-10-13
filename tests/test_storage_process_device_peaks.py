from types import SimpleNamespace
from copy import deepcopy
import pytest

from tools.storage_process_device_peaks import collect_peaks, valid_record
from tools.run_storage_formal_multidevice_profile import footer
from tools.run_storage_method_pipeline import PROCESS_PEAK_FOOTER


def test_qa_other_device_peak_included_and_unused_devices_not_synchronized(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3')
    sync=[]
    cuda=SimpleNamespace(device_count=lambda:4,synchronize=sync.append)
    values={'cuda:0':(6,9),'cuda:1':(18,19),'cuda:2':(0,0),'cuda:3':(2,3)}
    row=collect_peaks(cuda,lambda device:values[device])
    assert row['peak_allocated_bytes']==18 and row['peak_reserved_bytes']==19
    assert sync==[0,1,3] and valid_record(row,4)
    assert [x['physical_selector'] for x in row['devices']]==['0','1','2','3']


@pytest.mark.parametrize('fault',['old_scope','missing','duplicate','aggregate','negative'])
def test_partial_or_forged_device_evidence_rejected(fault):
    row=collect_peaks(SimpleNamespace(device_count=lambda:2,synchronize=lambda _:None),lambda _:(6,9))
    if fault=='old_scope': row.pop('device_peak_schema')
    elif fault=='missing': row['devices'].pop()
    elif fault=='duplicate': row['devices'][1]['device_index']=0
    elif fault=='aggregate': row['peak_reserved_bytes']=1
    elif fault=='negative': row['devices'][1]['peak_allocated_bytes']=-1
    assert not valid_record(row,2)


def test_actual_footer_keeps_owner_software_trace_and_cleanup():
    changed=footer(PROCESS_PEAK_FOOTER)
    assert '_record.update(_all_device_peaks)' in changed
    assert 'software_environment_' in changed and 'destroy_process_group' in changed
    assert 'layer_output_fingerprints.json' in changed
    with pytest.raises(AssertionError):footer(changed)
