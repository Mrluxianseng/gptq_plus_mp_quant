"""Validate new observers on the frozen online-wrapper GPU fixture.

This retains unrotated base weights and the existing torch_fallback backend;
it is not correctly rotated REAL-Q quality or formal benchmark validation.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_rotation_placement_component_v2 import source as prior_source


ADAPTER = '''
class _QADeviceProfileAdapter:
    def __init__(self, folder, device, tag):
        import torch
        from tools.qa_visible_device_phase_profile import VisibleDevicePhases
        from utils.memory_utils import get_process_cuda_memory_peaks
        if torch.cuda.device_count()!=1:
            raise RuntimeError('Local observer component requires exactly one visible device')
        self.group=VisibleDevicePhases(torch.cuda,folder,exclusive_peak_owner=True,
            prior_process_peak_getter=get_process_cuda_memory_peaks)
        self.path=self.group.paths[0]
    def mark(self, phase): self.group.mark(phase)
    def finish(self, status): self.group.finish(status)

'''


def source():
    text=prior_source()
    old='''    try:
        scores=hflm._loglikelihood_tokens(requests,disable_tqdm=True)
        profile.mark('qa_scoring_finished')
        elapsed=time.monotonic()-start'''
    new='''    from tools.qa_complete_observer import CompleteQAObserver
    try:
        with CompleteQAObserver(model,HFLM,args.output/(args.arm+'_complete'),mark=profile.mark) as complete_observer:
            with complete_observer.task('synthetic_component'):
                scores=hflm._loglikelihood_tokens(requests,disable_tqdm=True)
        profile.mark('qa_scoring_finished')
        elapsed=time.monotonic()-start'''
    replacements={
        'def worker(args):':ADAPTER+'def worker(args):',
        "    profile=CudaPhaseMemoryProfiler(str(args.output/(args.arm+'_stage2/memory_profile')),'cuda:0',tag='evaluation')":
        "    profile=_QADeviceProfileAdapter(str(args.output/(args.arm+'_stage2/memory_profile')),'cuda:0',tag='evaluation')",
        old:new,
        '    memory=collect_peaks(torch.cuda,get_process_cuda_memory_peaks)':
        '''    def _owned_peaks(device):
        current=get_process_cuda_memory_peaks(device)
        index=torch.device(device).index
        return tuple(max(current[i],profile.group.peaks[index][i]) for i in (0,1))
    memory=collect_peaks(torch.cuda,_owned_peaks)''',
        "    source += [x for x in Path(args.model).iterdir()":
        "    source += [ROOT/'tools/qa_complete_observer.py',ROOT/'tools/qa_tensor_fingerprint_stream.py',ROOT/'tools/qa_visible_device_phase_profile.py',ROOT/'tools/run_qa_rotation_placement_component_v2.py']\n    source += [x for x in Path(args.model).iterdir()",
    }
    for old,new in replacements.items():
        if text.count(old)!=1: raise RuntimeError('Frozen observer integration anchor changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_complete_observer_fixture','exec')
    return text


if __name__=='__main__':
    exec(compile(source(),'qa_complete_observer_fixture','exec'),dict(__name__='__main__',__file__=__file__))
