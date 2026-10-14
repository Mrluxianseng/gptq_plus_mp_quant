"""Frozen long-input mathematics with bounded observation buffers only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_qa_layerwise_long_feasibility import source as previous_source


def source():
    text=previous_source()
    old="""        value=result.detach().cpu().contiguous()
        logits.append(dict(shape=list(value.shape),dtype=str(value.dtype),stride=list(result.stride()),
            sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()))"""
    new="""        logits.append(dict(shape=list(result.shape),dtype=str(result.dtype),stride=list(result.stride()),
            sha256=_stream_tensor_sha256(result)))"""
    changes={
        "from tools.audit_qa_memory_rows_completed import sha":"from tools.audit_qa_memory_rows_completed import sha\nfrom tools.qa_tensor_fingerprint_stream import update_tensor_bytes as _stream_tensor_bytes,tensor_sha256 as _stream_tensor_sha256",
        "        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())":"        _stream_tensor_bytes(digest,value)",
        "        row['digest'].update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())":"        _stream_tensor_bytes(row['digest'],value)",
        old:new,
        "source=[Path(__file__),":"source=[Path(__file__),Path(__file__).with_name('run_qa_layerwise_long_feasibility.py'),ROOT/'tools/qa_tensor_fingerprint_stream.py',",
    }
    for old,new in changes.items():
        if text.count(old)!=1: raise RuntimeError('Frozen observer contract changed: '+old)
        text=text.replace(old,new)
    compile(text,'qa_stream_fingerprint_observation','exec')
    return text


if __name__=='__main__':
    exec(compile(source(),'qa_stream_fingerprint_observation','exec'),dict(__name__='__main__',__file__=__file__))
