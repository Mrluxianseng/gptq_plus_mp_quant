import json
from pathlib import Path
import pytest
from tools.qa_visible_device_phase_profile import VisibleDevicePhases
from tools.audit_memory_boundary_points import boundaries, compare
from tools.audit_completed_memory_pair_capacity import intervals


class FakeCUDA:
    def __init__(self):
        self.rows = [[0,0,0,0] for _ in range(4)]
        self.synchronized = []; self.reset = []
    def device_count(self): return 4
    def memory_allocated(self,i): return self.rows[i][0]
    def memory_reserved(self,i): return self.rows[i][1]
    def max_memory_allocated(self,i): return self.rows[i][2]
    def max_memory_reserved(self,i): return self.rows[i][3]
    def synchronize(self,i): self.synchronized.append(i)
    def reset_peak_memory_stats(self,i):
        self.reset.append(i); self.rows[i][2:]=self.rows[i][:2]


def test_transient_peak_idle_devices_identity_and_independent_points(tmp_path):
    cuda=FakeCUDA()
    profile=VisibleDevicePhases(cuda,tmp_path/'qa_stage2/memory_profile',exclusive_peak_owner=True,rank=3)
    assert cuda.synchronized==cuda.reset==[]
    cuda.rows[2]=[0,0,900,1200] # Allocation already released before the boundary.
    profile.mark('qa_setup') # Repeated phase identity must retain occurrences.
    cuda.rows[1]=[30,40,60,80]
    report=profile.finish('completed')
    assert set(cuda.synchronized)==set(cuda.reset)=={1,2}
    assert report['visible_device_count']==4
    assert report['peak_allocated_bytes']==900 and report['peak_reserved_bytes']==1200
    paths=[Path(p) for p in report['memory_profile_files']]
    rows=[json.loads(line) for p in paths for line in p.read_text().splitlines()]
    assert len(rows)==8 and {row['device_index'] for row in rows}==set(range(4))
    assert len(intervals(paths))==8
    points=boundaries(paths)
    assert len(points)==32 and compare(points,points)['valid']
    assert len({key[3] for key in intervals(paths)})==4
    with pytest.raises(RuntimeError,match='already closed'): profile.mark('invalid')


def test_prior_process_maxima_carried_and_failure_labeled(tmp_path):
    cuda=FakeCUDA();cuda.rows[0]=[10,20,80,90]
    profile=VisibleDevicePhases(cuda,tmp_path/'qa_stage2',exclusive_peak_owner=True,
        prior_process_peak_getter=lambda device:(200,300))
    cuda.rows[0]=[5,15,12,22]
    result=profile.finish('failed')
    assert result['status']=='failed' and result['peak_allocated_bytes']==200
    assert result['peak_reserved_bytes']==300
    assert 'qa_failed' in Path(result['memory_profile_files'][0]).read_text()
    with pytest.raises(FileExistsError):
        VisibleDevicePhases(cuda,tmp_path/'qa_stage2',exclusive_peak_owner=True)


def test_exclusive_owner_and_visible_devices_required(tmp_path):
    with pytest.raises(RuntimeError,match='exclusive'):
        VisibleDevicePhases(FakeCUDA(),tmp_path/'no',exclusive_peak_owner=False)
    empty=FakeCUDA();empty.device_count=lambda:0
    with pytest.raises(RuntimeError,match='visible CUDA'):
        VisibleDevicePhases(empty,tmp_path/'empty',exclusive_peak_owner=True)
