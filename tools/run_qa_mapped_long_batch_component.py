"""Versioned last-consumer pool lifetime; frozen long runner remains intact."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def module_source():
    text=(ROOT/'tools/run_qa_mapped_long_component.py').read_text()
    changes={
        'from tools.qa_mapped_head_storage_prototype import prepare_runtime,MappedHeadStorage':
        'from tools.qa_mapped_head_batch_lifetime import prepare_runtime,BatchMappedHeadStorage as MappedHeadStorage,build_batch_method',
        "        _mapped_pool=torch.cuda.MemPool(_mapped_allocator.allocator())":
        '        _mapped_pool=None',
        '        pool=_mapped_pool,mark=profile.mark)':
        '        pool_factory=lambda:torch.cuda.MemPool(_mapped_allocator.allocator()),mark=profile.mark)',
        "    compile(text,'qa_mapped_long_component','exec')\n    return text":
        '''    anchor="    if args.arm!='control': HFLM._loglikelihood_tokens=rows_method(original_score)"
    if text.count(anchor)!=1: raise RuntimeError('Frozen original score installation changed')
    text=text.replace(anchor,"    if args.arm!='control': HFLM._loglikelihood_tokens=build_batch_method(original_score,_head_storage.release_batch)")
    compile(text,'qa_mapped_long_batch_component','exec')
    return text''',
        "        phases.update(f'head{call}_storage_{suffix}' for suffix in ('start','end'))":
        "        phases.update(f'head{call}_storage_{suffix}' for suffix in ('start','end','release_start','release_end'))",
        "ROOT/'tools/audit_qa_mapped_long_completed.py',":
        "ROOT/'tools/audit_qa_mapped_long_batch_completed.py',ROOT/'tools/run_qa_mapped_long_component.py',ROOT/'tools/audit_qa_mapped_long_completed.py',ROOT/'tools/qa_mapped_head_batch_lifetime.py',",
        '    from tools.audit_qa_mapped_long_completed import audit':
        '    from tools.audit_qa_mapped_long_batch_completed import audit',
    }
    for old,new in changes.items():
        if text.count(old)!=1: raise RuntimeError('Frozen batch runner anchor changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_mapped_long_batch_runner','exec')
    return text


_scope=dict(__name__='qa_mapped_long_batch_runner',__file__=__file__)
exec(compile(module_source(),'qa_mapped_long_batch_runner','exec'),_scope)
source=_scope['source']
namespace=_scope['namespace']
expected_phases=_scope['expected_phases']
NUMERICAL_KEYS=_scope['NUMERICAL_KEYS']
main=_scope['main']
if __name__=='__main__': main()
