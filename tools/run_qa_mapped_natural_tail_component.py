"""New 65-request fixture: configured batch16 with an original natural tail."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_mapped_long_batch_component import module_source as previous_module


def module_source():
    text=previous_module()
    changes={
        'from tools.qa_mapped_head_batch_lifetime import prepare_runtime,BatchMappedHeadStorage as MappedHeadStorage,build_batch_method':
        'from tools.qa_mapped_head_batch_lifetime import prepare_runtime\n    from tools.qa_mapped_head_natural_batch import NaturalBatchMappedHeadStorage as MappedHeadStorage,build_batch_method',
        "    compile(text,'qa_mapped_long_batch_component','exec')\n    return text":'''    anchor="    profile.mark('qa_scoring_start');start=time.monotonic()"
    if text.count(anchor)!=1: raise RuntimeError('Original request/score boundary changed')
    text=text.replace(anchor,"    requests.append((('natural-tail-unique','choice'),[11,12,13],[14]))\\n"+anchor)
    compile(text,'qa_mapped_natural_tail_component','exec')
    return text''',
        "p.add_argument('--length',type=int,choices=(1024,1270),default=1270)":
        "p.add_argument('--length',type=int,choices=(128,),default=128)",
        "reference['model']!=args.model or reference['max_context_tokens']!=args.length":
        "reference['model']!=args.model or reference['max_context_tokens']!=1270",
        "same_observer_map_flag_and_phase_identity=True,formal_enabled=False":
        "same_observer_map_flag_and_phase_identity=True,formal_enabled=False,reference_scope='prior source/method admission only; different requests, not a numerical baseline'",
        "ROOT/'tools/audit_qa_mapped_long_batch_completed.py',":
        "ROOT/'tools/audit_qa_mapped_natural_tail_completed.py',ROOT/'tools/run_qa_mapped_long_batch_component.py',ROOT/'tools/audit_qa_mapped_long_batch_completed.py',ROOT/'tools/qa_mapped_head_natural_batch.py',",
        '    from tools.audit_qa_mapped_long_batch_completed import audit':
        '    from tools.audit_qa_mapped_natural_tail_completed import audit',
    }
    for old,new in changes.items():
        if text.count(old)!=1: raise RuntimeError('Frozen natural-tail runner anchor changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_mapped_natural_tail_runner','exec')
    return text


_scope=dict(__name__='qa_mapped_natural_tail_runner',__file__=__file__)
exec(compile(module_source(),'qa_mapped_natural_tail_runner','exec'),_scope)
source=_scope['source'];namespace=_scope['namespace'];expected_phases=_scope['expected_phases']
NUMERICAL_KEYS=_scope['NUMERICAL_KEYS'];main=_scope['main']
if __name__=='__main__': main()
