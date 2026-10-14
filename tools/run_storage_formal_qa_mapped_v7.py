"""Admit model config identities by resolved file SHA, including HF symlinks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_storage_formal_qa_mapped_v6 import source as prior_source


def source():
    text=prior_source()
    changes={
        'def require_admission(runtime_root, component_roots):':'def require_admission(runtime_root, component_roots, model_configs):',
        "seen.update(Path(p).parent.name for p in row if Path(p).name=='config.json')":
        "seen.update(name for name,path in model_configs.items() if str(path.resolve()) in row and row[str(path.resolve())]==sha(path))",
        'binary,bindings=require_admission(known.runtime_root,known.component_root)':
        "formal_args=formal.parser().parse_args(remaining)\n    model_configs={'Qwen3-0.6B':formal_args.model06/'config.json','Qwen3-1.7B':formal_args.model17/'config.json'}\n    binary,bindings=require_admission(known.runtime_root,known.component_root,model_configs)",
        "extra=[ROOT/'tools/run_storage_formal_qa_mapped_v5.py',":"extra=[ROOT/'tools/run_storage_formal_qa_mapped_v6.py',ROOT/'tools/run_storage_formal_qa_mapped_v5.py',",
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError('Frozen config admission anchor changed: '+old)
        text=text.replace(old,new)
    return text


scope=dict(__name__='qa_server_formal_v7',__file__=__file__)
exec(compile(source(),'qa_server_formal_v7','exec'),scope)
if __name__=='__main__':scope['main']()
