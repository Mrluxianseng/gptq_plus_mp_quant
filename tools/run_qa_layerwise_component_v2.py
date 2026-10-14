"""Frozen v1 worker/math plus explicit K/V and library source observation."""
from pathlib import Path


def source():
    text=Path(__file__).with_name('run_qa_layerwise_component.py').read_text()
    changes={
        "dtype=str(layer.keys.dtype),stride=list(layer.keys.stride()))":
        "dtype=str(layer.keys.dtype),stride=list(layer.keys.stride()),value_device=str(layer.values.device),value_shape=list(layer.values.shape),value_dtype=str(layer.values.dtype),value_stride=list(layer.values.stride()))",
        "all(layer['device']==('cpu' if side=='layerwise_cache' else 'cuda:0') for layer in output)":
        "all(layer['device']==('cpu' if side=='layerwise_cache' and index!=0 else 'cuda:0') and layer['value_device']==layer['device'] for index,layer in enumerate(output))",
        "source=[Path(__file__),":
        "source=[Path(__file__),Path(__file__).with_name('run_qa_layerwise_component.py'),",
        "source += [x for x in Path(args.model).iterdir()":
        "from transformers.cache_utils import Cache\n    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model\n    from lm_eval.models.huggingface import HFLM\n    source += [Path(inspect.getfile(Cache)),Path(inspect.getfile(Qwen3Model)),Path(inspect.getfile(HFLM))]\n    source += [x for x in Path(args.model).iterdir()",
    }
    for old,new in changes.items():
        if text.count(old)!=1: raise RuntimeError('Frozen QA component source changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_layerwise_observation_v2','exec')
    return text


if __name__=='__main__':
    exec(compile(source(),'qa_layerwise_observation_v2','exec'),dict(__name__='__main__',__file__=__file__))
