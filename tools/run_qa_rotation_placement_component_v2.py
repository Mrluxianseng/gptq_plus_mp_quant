"""Keep v1 mathematics; correctly bind and observe its optional Hadamard backend."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_rotation_placement_component import source as previous_source


def source():
    text=previous_source()
    old="""    import fast_hadamard_transform
    source += [Path(fast_hadamard_transform.__file__)]
    source += [Path(module.__file__) for name,module in list(sys.modules.items()) if name.startswith('fast_hadamard') and getattr(module,'__file__',None) and module.__file__.endswith('.so')]"""
    new="""    from utils import hadamard_utils as _backend_utils
    _backend_name='fast_hadamard_transform' if _backend_utils.fast_hadamard_transform is not None else 'torch_fallback'
    _backend_contract=args.output/'runtime_backend_contract.json'
    _backend_contract.write_text(json.dumps(dict(selected=_backend_name,scope='existing installed backend; no dependency or math change'),indent=2))
    source += [_backend_contract,ROOT/'utils/had_mat.pt',Path(__file__).with_name('run_qa_rotation_placement_component.py')]
    if _backend_utils.fast_hadamard_transform is not None:
        source += [Path(_backend_utils.fast_hadamard_transform.__file__)]
        source += [Path(module.__file__) for name,module in list(sys.modules.items()) if name.startswith('fast_hadamard') and getattr(module,'__file__',None) and module.__file__.endswith('.so')]"""
    replacements={
        old:new,
        "    return model":"    from utils import hadamard_utils\n    model._qa_fixture_hadamard_backend='fast_hadamard_transform' if hadamard_utils.fast_hadamard_transform is not None else 'torch_fallback'\n    return model",
        "        weight_records=ticket.records,":"        hadamard_backend=model._qa_fixture_hadamard_backend,weight_records=ticket.records,",
        "        checks['states_unchanged']=":"        backend=json.loads((root/'runtime_backend_contract.json').read_text())\n        from utils import hadamard_utils\n        installed_backend='fast_hadamard_transform' if hadamard_utils.fast_hadamard_transform is not None else 'torch_fallback'\n        checks['actual_hadamard_backend']=c['hadamard_backend']==n['hadamard_backend']==backend['selected']==installed_backend\n        checks['states_unchanged']=",
    }
    for old,new in replacements.items():
        if text.count(old)!=1: raise RuntimeError('Frozen optional-backend contract changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_rotation_placement_v2','exec')
    return text


if __name__=='__main__':
    import argparse,json
    if '--audit-root' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--audit-root',type=Path,required=True);p.add_argument('--audit-output',type=Path,required=True)
        args=p.parse_args();namespace=dict(__name__='qa_rotation_placement_audit',__file__=__file__)
        exec(compile(source(),'qa_rotation_placement_v2','exec'),namespace)
        bindings=json.loads((args.audit_root/'source_bindings.json').read_text())
        result=namespace['audit'](args.audit_root,bindings)
        with args.audit_output.open('x') as f: json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
        if not result['valid']: raise SystemExit(1)
    else:
        exec(compile(source(),'qa_rotation_placement_v2','exec'),dict(__name__='__main__',__file__=__file__))
