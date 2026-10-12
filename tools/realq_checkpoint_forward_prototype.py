"""Prototype preserving module names while checkpointing a block's body.

Not integrated into REAL-Q or validated with FSDP/GPU yet. Keeping the
module object/tree unchanged avoids renaming saliency cache module keys.
FSDP pre-backward parameter availability requires a real separate test.
"""
from functools import wraps
import torch
from torch.utils.checkpoint import checkpoint


def wrap_block_forward(block):
    original = block.forward
    if getattr(block, "_realq_checkpoint_prototype_original", None) is not None:
        raise ValueError("block is already checkpointed")

    @wraps(original)
    def forward(*args, **kwargs):
        if not torch.is_grad_enabled():
            return original(*args, **kwargs)
        return checkpoint(original, *args, use_reentrant=False, **kwargs)

    block._realq_checkpoint_prototype_original = original
    block.forward = forward


def unwrap_block_forward(block):
    original = getattr(block, "_realq_checkpoint_prototype_original", None)
    if original is None:
        raise ValueError("block is not checkpointed")
    block.forward = original
    delattr(block, "_realq_checkpoint_prototype_original")
