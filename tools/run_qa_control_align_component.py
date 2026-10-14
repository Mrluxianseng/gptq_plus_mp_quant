"""Single-CUDA-device compatibility screen of the genuine control dispatch hook.

Same 65-request workload and full GEMMs. This tests the new pure control
observer, not a new algorithm or formal multi-device capacity.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_mapped_observed_bridge_component import source as ancestor_source


def source():
    text = ancestor_source()
    changes = {
        'from tools.qa_mapped_observed_placement_bridge import MappedObservedPlacementBridge as ObservedPlacementBridge':
        'from tools.qa_mapped_observed_placement_bridge_v2 import MappedObservedPlacementBridge as ObservedPlacementBridge',
        '        dist_utils.distribute_model(model)\n        metrics=ev.qa_eval(model,tokenizer,16)':
        '''        dist_utils.distribute_model(model)
        _control_hook=None
        if args.arm=='control':
            from accelerate.hooks import add_hook_to_module,AlignDevicesHook
            if hasattr(model.lm_head,'_hf_hook'): raise RuntimeError('Unexpected preexisting single-device head hook')
            add_hook_to_module(model.lm_head,AlignDevicesHook(execution_device='cuda:0',io_same_device=True))
            _control_hook=(model.lm_head.forward,model.lm_head._hf_hook,id(model.lm_head.weight))
        metrics=ev.qa_eval(model,tokenizer,16)
        _hook_preserved=(model.lm_head.forward is _control_hook[0] and model.lm_head._hf_hook is _control_hook[1]
            and id(model.lm_head.weight)==_control_hook[2]) if _control_hook else not hasattr(model.lm_head,'_hf_hook')''',
        '        mapped_capabilities=_mapped_capabilities,parameter_identity_preserved=spec==identity,':
        '        control_hook_preserved=_hook_preserved,mapped_capabilities=_mapped_capabilities,parameter_identity_preserved=spec==identity,',
        "    files=[Path(__file__),args.output/'worker_source.py',":
        "    from accelerate import hooks as _accelerate_hooks\n    files=[Path(__file__),Path(inspect.getfile(_accelerate_hooks)),ROOT/'tools/run_qa_mapped_observed_bridge_component.py',ROOT/'tools/qa_control_head_observer.py',ROOT/'tools/qa_mapped_observed_placement_bridge_v2.py',ROOT/'tools/audit_qa_control_align_completed.py',args.output/'worker_source.py',",
        '    from tools.audit_qa_mapped_observed_bridge_completed import audit as _complete_audit':
        '    from tools.audit_qa_control_align_completed import audit as _complete_audit',
    }
    for old,new in changes.items():
        if text.count(old)!=1:
            raise RuntimeError('Frozen CUDA control compatibility anchor changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_control_align_fixture','exec')
    return text


def namespace():
    scope=dict(__name__='qa_control_align_fixture',__file__=__file__,_generator=source)
    exec(compile(source(),'qa_control_align_fixture','exec'),scope)
    return scope


if __name__=='__main__': namespace()['main']()
