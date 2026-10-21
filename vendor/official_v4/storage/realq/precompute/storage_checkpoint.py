"""Scope non-reentrant block checkpointing to the static forward/backward loop."""
from contextlib import contextmanager
from functools import wraps

import torch
from torch.utils.checkpoint import checkpoint


def _wrapped(original):
    @wraps(original)
    def forward(*values, **options):
        if not torch.is_grad_enabled():
            return original(*values, **options)
        return checkpoint(original, *values, use_reentrant=False, **options)
    return forward


@contextmanager
def block_checkpoint(layers):
    originals = []
    try:
        for layer in layers:
            had_instance_forward = 'forward' in layer.__dict__
            prior_instance_forward = layer.__dict__.get('forward')
            originals.append((layer, had_instance_forward, prior_instance_forward))
            layer.forward = _wrapped(layer.forward)
        yield
    finally:
        for layer, had_instance_forward, prior_instance_forward in reversed(originals):
            if had_instance_forward:
                layer.forward = prior_instance_forward
            else:
                del layer.forward
