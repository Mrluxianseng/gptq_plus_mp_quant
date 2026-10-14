"""Test placement of real activation-rotation wrappers, not REAL-Q quality.

Base weights remain unrotated: both arms share an explicit wrapper fixture.
This cannot be reported as an accurate rotated/quantized benchmark model.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_layerwise_component_v2 import source as previous_source


WRAP = '''
def fixture_wrappers(model):
    from utils import model_utils,rotation_utils
    analyzer=model_utils.ModelAnalyzer.__new__(model_utils.ModelAnalyzer)
    analyzer.model=model;analyzer.model_arch='Qwen3ForCausalLM'
    rotation_utils.add_activation_quant_wrappers_for_rotation(analyzer)
    return model

'''


def source():
    text=previous_source()
    replacements={
        "SIDES=('control','rows','layerwise','layerwise_cache')":"SIDES=('control','layerwise_cache')",
        "def state(model):":WRAP+"def state(model):",
        "    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)":"    fixture_wrappers(model)\n    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)",
        "    expected_layout={k:[list(v.shape),str(v.dtype)] for k,v in expected.state_dict().items()}":"    fixture_wrappers(expected)\n    expected_layout={k:[list(v.shape),str(v.dtype)] for k,v in expected.state_dict().items()}",
        "bounded synthetic-token real Qwen3 scoring; not full benchmark/PTQ":"activation-rotation wrapper placement fixture with unrotated base weights; not REAL-Q or QA accuracy",
        "source=[Path(__file__),Path(__file__).with_name('run_qa_layerwise_component.py'),":"source=[Path(__file__),Path(__file__).with_name('run_qa_layerwise_component_v2.py'),Path(__file__).with_name('run_qa_layerwise_component.py'),ROOT/'utils/rotation_utils.py',ROOT/'utils/quant_utils.py',ROOT/'utils/hadamard_utils.py',ROOT/'utils/model_utils.py',",
        "    source += [x for x in Path(args.model).iterdir()":"    import fast_hadamard_transform\n    source += [Path(fast_hadamard_transform.__file__)]\n    source += [Path(module.__file__) for name,module in list(sys.modules.items()) if name.startswith('fast_hadamard') and getattr(module,'__file__',None) and module.__file__.endswith('.so')]\n    source += [x for x in Path(args.model).iterdir()",
    }
    for old,new in replacements.items():
        if text.count(old)!=1: raise RuntimeError('Frozen wrapper component source changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_rotation_placement_component','exec')
    return text


if __name__=='__main__':
    import argparse,json
    if '--audit-root' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--audit-root',type=Path,required=True);p.add_argument('--audit-output',type=Path,required=True)
        args=p.parse_args();namespace=dict(__name__='qa_rotation_placement_audit',__file__=__file__)
        exec(compile(source(),'qa_rotation_placement_audit','exec'),namespace)
        bindings=json.loads((args.audit_root/'source_bindings.json').read_text())
        result=namespace['audit'](args.audit_root,bindings)
        with args.audit_output.open('x') as f: json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
        if not result['valid']: raise SystemExit(1)
    else:
        exec(compile(source(),'qa_rotation_placement_component','exec'),dict(__name__='__main__',__file__=__file__))
