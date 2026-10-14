"""Exact C-order tensor byte digests with bounded temporary CPU storage.

Observation only: no arithmetic or GPU tensor layout is changed.
"""
import hashlib
import torch


def update_tensor_bytes(digest, tensor, *, max_bytes=16*1024*1024):
    if max_bytes < tensor.element_size():
        raise ValueError('Byte limit must hold at least one tensor element')
    value=tensor.detach()
    if value.numel()*value.element_size() <= max_bytes:
        block=value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(block)
        return
    if value.shape[0] == 1:
        return update_tensor_bytes(digest,value[0],max_bytes=max_bytes)
    row_bytes=value.numel()*value.element_size()//value.shape[0]
    rows=max(1,max_bytes//row_bytes)
    for start in range(0,value.shape[0],rows):
        update_tensor_bytes(digest,value[start:start+rows],max_bytes=max_bytes)


def tensor_sha256(tensor, *, max_bytes=16*1024*1024):
    digest=hashlib.sha256()
    update_tensor_bytes(digest,tensor,max_bytes=max_bytes)
    return digest.hexdigest()
