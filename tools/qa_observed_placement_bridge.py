"""Prepared validation-only composition; does not enable a formal launcher.

Placement precedes dispatch. Full-byte/task observation is installed after the
selected QA methods. Exact QA counter maxima are carried into the existing
CPU-only process accumulator, including on failures and after cleanup.
"""
import json
from pathlib import Path
from tools.qa_layerwise_placement_bridge import PlacementBridge
from tools.qa_complete_observer import CompleteQAObserver
from tools.qa_task_boundary_observer import TaskBoundaries
from tools.qa_visible_device_phase_profile import VisibleDevicePhases


class PeakCarry:
    @staticmethod
    def validate_owner(memory_module):
        owner=getattr(memory_module,'_process_profile_peaks',None)
        if not isinstance(owner,dict):
            raise RuntimeError('Unsupported original process-peak accumulator protocol')
        if any(type(index) is not int or index<0 or not isinstance(row,(tuple,list)) or len(row)!=2
               or any(type(v) is not int or v<0 for v in row) for index,row in owner.items()):
            raise RuntimeError('Invalid original process-peak accumulator values')

    def __init__(self, memory_module, group):
        self.validate_owner(memory_module)
        self.memory_module, self.group = memory_module, group
        self.publish()

    def publish(self):
        owner=self.memory_module._process_profile_peaks
        for index, (allocated,reserved) in enumerate(self.group.peaks):
            previous=owner.get(index,(0,0))
            if (not isinstance(previous,(tuple,list)) or len(previous)!=2
                    or any(type(v) is not int or v<0 for v in previous)):
                raise RuntimeError('Invalid original process-peak accumulator values')
            owner[index]=(max(previous[0],int(allocated)),max(previous[1],int(reserved)))

    def mark(self, phase):
        self.group.mark(phase)
        self.publish()

    def finish(self,status):
        try:
            return self.group.finish(status)
        finally:
            # Even failed QA must not erase earlier quantization/QA maxima.
            self.publish()


class ObservedPlacementBridge:
    def __init__(self, eval_module, dist_module, evaluation_module, memory_module, cuda,
                 *, enabled, destination, expected_tasks, validation_only,
                 device='cuda:0', exclusive_peak_owner=False):
        if validation_only is not True:
            raise RuntimeError('Prepared interface is validation-only; formal launch is not enabled')
        if exclusive_peak_owner is not True:
            raise RuntimeError('Requires exclusive QA peak-counter ownership')
        PeakCarry.validate_owner(memory_module)
        self.root=Path(destination)
        self.report_path=self.root/'qa_observed_placement_rank0.json'
        if self.report_path.exists(): raise RuntimeError('Evidence exists; select a fresh directory')
        self.eval_module, self.memory_module = eval_module, memory_module
        self.evaluation_module, self.cuda = evaluation_module, cuda
        self.expected_tasks=tuple(expected_tasks)
        if not self.expected_tasks or len(set(self.expected_tasks))!=len(self.expected_tasks):
            raise RuntimeError('Requires explicit unique complete task sequence')
        self.original_eval=eval_module.qa_eval
        self.active=None
        self.running=False
        self.control_handles=[]
        eval_module.qa_eval=self.observed_eval
        try:
            self.bridge=PlacementBridge(eval_module,dist_module,enabled=enabled,
                destination=self.root,device=device,mark=self.mark)
        except BaseException:
            eval_module.qa_eval=self.original_eval
            raise
        eval_module.qa_eval=self.qa_eval

    def mark(self,phase):
        if self.active is None: raise RuntimeError('QA marker outside owned profiling interval')
        self.active.mark(phase)

    def observed_eval(self,model,tokenizer,*args,**kwargs):
        from lm_eval.models.huggingface import HFLM
        with CompleteQAObserver(model,HFLM,self.root/'qa_complete',mark=self.mark) as observer:
            with TaskBoundaries(observer,self.evaluation_module,self.expected_tasks):
                return self.original_eval(model,tokenizer,*args,**kwargs)

    def install_control_markers(self,model):
        # Keep original accelerate dispatch and forward methods. Only the
        # corresponding phase boundaries are added; no weights are moved.
        calls=[0]*28
        def pre(index):
            def hook(module,inputs):
                calls[index]+=1
                self.mark(f'forward{calls[index]}_layer{index}_load_start')
                self.mark(f'forward{calls[index]}_layer{index}_compute_start')
            return hook
        def post(index):
            def hook(module,inputs,output):
                self.mark(f'forward{calls[index]}_layer{index}_offload_start')
                self.mark(f'forward{calls[index]}_layer{index}_end')
            return hook
        for index,layer in enumerate(model.model.layers):
            self.control_handles.extend((layer.register_forward_pre_hook(pre(index)),
                                         layer.register_forward_hook(post(index))))

    def remove_control_markers(self):
        for handle in self.control_handles: handle.remove()
        self.control_handles=[]

    def qa_eval(self,model,tokenizer,*args,**kwargs):
        if self.running or self.report_path.exists():
            raise RuntimeError('QA validation already running or recorded')
        self.running=True
        record=dict(schema='qa-observed-placement-validation-v1',completed=False,
            formal_enabled=False,expected_tasks=list(self.expected_tasks),
            scope='validation-only composition; full numerical/source/method/NVML acceptance still external')
        primary_error=None
        try:
            group=VisibleDevicePhases(self.cuda,self.root/'qa_stage2/memory_profile',
                exclusive_peak_owner=True,prior_process_peak_getter=self.memory_module.get_process_cuda_memory_peaks)
            self.active=PeakCarry(self.memory_module,group)
            if not self.bridge.enabled:
                self.install_control_markers(model)
            result=self.bridge.qa_eval(model,tokenizer,*args,**kwargs)
            self.remove_control_markers()
            # Inner bridge has now completed its finally block and removed
            # placement/scoring hooks. Keep that cleanup inside the QA interval.
            self.mark('qa_bridge_cleanup_complete')
            record['memory']=self.active.finish('completed')
            record.update(completed=True,metrics=result,
                          process_accumulator={str(k):list(v) for k,v in self.memory_module._process_profile_peaks.items()})
            return result
        except BaseException as exc:
            primary_error=exc
            record.update(error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            self.remove_control_markers()
            if self.active is not None and not self.active.group.closed:
                try:
                    record['memory']=self.active.finish('failed')
                except BaseException as failure:
                    record.update(profiling_cleanup_error_type=type(failure).__name__,
                                  profiling_cleanup_error=str(failure),completed=False)
                    for stream in self.active.group.files:
                        stream.close()
                    self.active.group.closed=True
                    if primary_error is None: primary_error=failure
            self.active=None;self.running=False
            self.root.mkdir(parents=True,exist_ok=True)
            with self.report_path.open('x') as stream: json.dump(record,stream,indent=2,allow_nan=False)
            if record.get('profiling_cleanup_error') and record.get('error_type') is None:
                raise primary_error

    def restore(self):
        if self.running: raise RuntimeError('Cannot restore an active QA observer')
        self.bridge.restore()
        self.eval_module.qa_eval=self.original_eval
