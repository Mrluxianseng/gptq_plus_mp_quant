"""Distributed environment bootstrap.

The single source of truth for: rank pinning to local GPU, NCCL init, and
re-exporting the rank/world helpers used elsewhere in RealQ. Everything else
in `realq/parallel/` builds on top of this — keep it minimal.
"""
import atexit
import os

import torch
import torch.distributed as dist

from utils import dist_utils


def _atexit_destroy() -> None:
    """Avoid the `destroy_process_group() was not called` warning on exit.

    Wrapped in try/except because the cpu_master eval flow may have already
    called ``destroy_process_group()`` before lm_eval (mirrors ptq.py:198-199).
    A second destroy raises; swallow it.
    """
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def init() -> None:
    """Pin this rank to its local GPU and initialise NCCL.

    Order matters: torch.cuda.set_device() MUST run before any CUDA op or NCCL
    init, otherwise rank>0 defaults to cuda:0 and every collective deadlocks.

    No-op when launched without torchrun (no RANK env var) — single-process
    runs stay supported.
    """
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if "RANK" not in os.environ:
        # Bare `python -m realq.ptq` — distributed APIs become no-ops via the
        # is_dist_available_and_initialized() guards in utils.dist_utils.
        return
    dist_utils.init_process_group()
    atexit.register(_atexit_destroy)


# Re-export the helpers most callers need so they can `from realq.parallel
# import env` and stay inside the realq namespace.
get_rank = dist_utils.get_rank
get_world_size = dist_utils.get_world_size
is_main = dist_utils.is_main
is_dist_available_and_initialized = dist_utils.is_dist_available_and_initialized
barrier = dist_utils.barrier
